"""Recompute the SEOBNRv5PHM dynamics for a GW250114 posterior sample,
and express it in a frame tied to the observer.

The waveform call reproduces exactly what bilby (`gwsignal_binary_black_hole`)
did during parameter estimation, so the dynamics are the ones that generated
the waveform used in the likelihood.

Frames
------
I-frame (LAL "L0" frame): z along the Newtonian orbital angular momentum at
f_ref, x along the separation vector n = (x_1 - x_2)/r at f_ref.
pyseobnr builds the inertial-frame orbit as n(t) = R(t) (cos phi, sin phi, 0),
with R = quatI2J * quatJ2P(t) equal to the identity at t_ref.

Sky frame: X = west, Y = north, Z = towards the observer (right-handed).
A camera at +Z looking towards -Z with Y up sees the sky as seen from Earth.
bilby's polarization basis is m = (cos psi, sin psi, 0), n = (-sin psi, cos psi, 0)
in this frame, and the propagation direction m x n = +Z.
"""

from pathlib import Path

import h5py
import lal
import numpy as np
import quaternion
from bilby.gw import utils as gwutils
from bilby.gw.conversion import bilby_to_lalsimulation_spins
from pyseobnr.eob.dynamics.pn_evolution_opt import (
    build_splines_PN,
    compute_quasiprecessing_PNdynamics_opt,
)
from pyseobnr.eob.utils.utils_precession_opt import (
    SEOBBuildJframeVectors,
    inspiral_merger_quaternion_angles,
)
from pyseobnr.generate_waveform import GenerateWaveform

POSTERIOR_FILE = (
    Path(__file__).parent.parent / "GW250114_posteriors" / "SEOB_posterior_samples.h5"
)
POSTERIOR_LABEL = "bilby-SEOBNRv5PHM-reweighted"

# settings of the SEOBNRv5PHM PE run (from the config stored in the posterior file)
REFERENCE_FREQUENCY = 20.0
WAVEFORM_MINIMUM_FREQUENCY = 10.0
MAXIMUM_FREQUENCY = 896.0
DURATION = 8.0
LMAX_NYQUIST = 1
RUN_SETTINGS = dict(
    reference_frequency=REFERENCE_FREQUENCY,
    minimum_frequency=WAVEFORM_MINIMUM_FREQUENCY,
    maximum_frequency=MAXIMUM_FREQUENCY,
    duration=DURATION,
    lmax_nyquist=LMAX_NYQUIST,
)

# km per solar mass, G M_sun / c^2
KM_PER_MSUN = lal.MRSUN_SI / 1e3


def load_posterior():
    with h5py.File(POSTERIOR_FILE, "r") as f:
        return f[POSTERIOR_LABEL]["posterior_samples"][()]


def sample_as_dict(posterior, index):
    return {name: float(posterior[name][index]) for name in posterior.dtype.names}


def pyseobnr_parameters(sample, settings=None):
    """Same parameters bilby's gwsignal_binary_black_hole passes on to pyseobnr.

    settings: the bilby run's waveform settings (keys of RUN_SETTINGS);
    defaults to those of the GW250114 run.
    """
    settings = {**RUN_SETTINGS, **(settings or {})}
    iota, s1x, s1y, s1z, s2x, s2y, s2z = bilby_to_lalsimulation_spins(
        theta_jn=sample["theta_jn"],
        phi_jl=sample["phi_jl"],
        tilt_1=sample["tilt_1"],
        tilt_2=sample["tilt_2"],
        phi_12=sample["phi_12"],
        a_1=sample["a_1"],
        a_2=sample["a_2"],
        mass_1=sample["mass_1"] * gwutils.solar_mass,
        mass_2=sample["mass_2"] * gwutils.solar_mass,
        reference_frequency=settings["reference_frequency"],
        phase=sample["phase"],
    )
    return dict(
        mass1=sample["mass_1"],
        mass2=sample["mass_2"],
        spin1x=s1x,
        spin1y=s1y,
        spin1z=s1z,
        spin2x=s2x,
        spin2y=s2y,
        spin2z=s2z,
        deltaF=1 / settings["duration"],
        f22_start=settings["minimum_frequency"],
        f_max=settings["maximum_frequency"],
        f_ref=settings["reference_frequency"],
        phi_ref=sample["phase"],
        distance=sample["luminosity_distance"],
        inclination=iota,
        approximant="SEOBNRv5PHM",
        lmax_nyquist=settings["lmax_nyquist"],
        gwsignal_environment=True,
        conditioning=2,
    )


def run_model(sample, with_modes=False, settings=None):
    """Run SEOBNRv5PHM through the same code path used in PE.

    Returns the pyseobnr model object (holding the dynamics) and the generator.
    `generate_fd_polarizations` lowers f22_start for conditioning, so the
    dynamics start somewhat below 10 Hz.

    with_modes: build the polarizations from the inertial modes instead of
    directly from the co-precessing ones, so that the modes in the final-J frame
    are stored (`model.wavefom_modesJ`). The dynamics are unchanged.
    settings: see `pyseobnr_parameters`.
    """
    params = pyseobnr_parameters(sample, settings)
    if with_modes:
        params["polarizations_from_coprec"] = False
    generator = GenerateWaveform(dict(params))
    generator.generate_fd_polarizations()
    return generator.model, generator


def observer_direction_I(inclination, phi_ref):
    """Unit vector towards the observer in the I-frame (LAL convention),
    together with the polarization basis vectors P, Q such that
    h_+ = (P P - Q Q) : h / 2 and h_x = (P Q + Q P) : h / 2.
    The basis is verified numerically against the model in `validate_dynamics.py`.
    """
    theta, varphi = inclination, np.pi / 2 - phi_ref
    N = np.array(
        [np.sin(theta) * np.cos(varphi), np.sin(theta) * np.sin(varphi), np.cos(theta)]
    )
    e_theta = np.array(
        [np.cos(theta) * np.cos(varphi), np.cos(theta) * np.sin(varphi), -np.sin(theta)]
    )
    e_varphi = np.array([-np.sin(varphi), np.cos(varphi), 0.0])
    return N, e_varphi, -e_theta


def rotation_I_to_sky(inclination, phi_ref, psi):
    """Rotation matrix taking I-frame vectors to the sky frame (west, north, observer).

    It maps the polarization basis (P, Q) onto bilby's (m, n), and the
    direction of the observer onto +Z.
    """
    N, P, Q = observer_direction_I(inclination, phi_ref)
    m = np.array([np.cos(psi), np.sin(psi), 0.0])
    n = np.array([-np.sin(psi), np.cos(psi), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    return np.column_stack([m, n, z]) @ np.vstack([P, Q, N])


def coprecessing_quaternions(model):
    """Rotation from the co-precessing frame to the I-frame on the dynamics time grid,
    recomputed exactly as inside pyseobnr."""
    _, quatJ2P_dyn, quatI2J, *_ = inspiral_merger_quaternion_angles(
        model.dynamics[:, 0],
        model.dynamics[:, 6],
        model.t_attach,
        model.Lvec_hat_attach,
        model.Jfhat_attach,
        model.splines,
        t_ref=model.t_ref,
    )
    return quatI2J * quatJ2P_dyn


def inertial_dynamics(model, t_geom=None):
    """Orbit, spins and angular momenta in the I-frame, in units of the total
    (detector-frame) mass M.

    t_geom: times (in M, on the dynamics' time axis) at which to evaluate;
    defaults to the dynamics' own grid. Only times up to the end of the
    dynamics are meaningful.
    """
    from scipy.interpolate import CubicSpline

    dyn = model.dynamics
    t_dyn = dyn[:, 0]
    if t_geom is None:
        t_geom = t_dyn

    r = CubicSpline(t_dyn, dyn[:, 1])(t_geom)
    phi = CubicSpline(t_dyn, dyn[:, 2])(t_geom)
    omega = CubicSpline(t_dyn, dyn[:, 6])(t_geom)

    R = coprecessing_quaternions(model)
    R = quaternion.squad(R, t_dyn, t_geom)
    rot = quaternion.as_rotation_matrix(R)

    n_P = np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
    n_I = np.einsum("tij,tj->ti", rot, n_P)
    ez_I = rot[:, :, 2]

    X1, X2 = model.m_1, model.m_2  # pyseobnr masses are fractions of M
    pn = model.splines["everything"](omega)
    chi1 = pn[:, 4:7]
    chi2 = pn[:, 7:10]
    LN_hat = pn[:, 10:13]
    LN_hat = LN_hat / np.linalg.norm(LN_hat, axis=1)[:, None]
    L = pn[:, 13:16]
    J = L + X1**2 * chi1 + X2**2 * chi2

    return dict(
        t=t_geom,
        r=r,
        phi=phi,
        omega=omega,
        n=n_I,
        x1=X2 * r[:, None] * n_I,
        x2=-X1 * r[:, None] * n_I,
        ez_coprecessing=ez_I,
        chi1=chi1,
        chi2=chi2,
        LN_hat=LN_hat,
        L=L,
        J=J,
    )


def J_hat_at_past_infinity(model, f_low=1e-3):
    """Direction of the total angular momentum J = L + S1 + S2 in the I-frame,
    integrating pyseobnr's own PN spin-precession equations backwards from
    f_ref down to a GW frequency f_low [Hz].

    Returns J_hat at f_low and the array of (f_22, J_hat) along the way,
    so that convergence can be checked.
    """
    M_sec = model.M * lal.MTSUN_SI
    omega_low = np.pi * f_low * M_sec
    t, y = compute_quasiprecessing_PNdynamics_opt(
        model.omega_ref,
        omega_low,
        model.m_1,
        model.m_2,
        model.chi1_v,
        model.chi2_v,
    )
    # keep only the backward part (orbital frequency <= omega_ref)
    keep = y[:, -1] <= model.omega_ref * (1 + 1e-12)
    y = y[keep]
    splines = build_splines_PN(t[keep], y, model.m_1, model.m_2, omega_low)
    pn = splines["everything"](y[:, -1])
    J = pn[:, 13:16] + model.m_1**2 * pn[:, 4:7] + model.m_2**2 * pn[:, 7:10]
    J_hat = J / np.linalg.norm(J, axis=1)[:, None]
    f22 = y[:, -1] / (np.pi * M_sec)
    return J_hat[0], f22, J_hat


def geometric_to_seconds(model):
    """Conversion of dynamics time [M] to time [s] relative to the waveform's t=0,
    which bilby places at geocent_time (verified in `validate_dynamics.py`).

    pyseobnr shifts the waveform time axis (which starts at the first dynamics
    time) so that t=0 is the peak of the frame-invariant amplitude;
    `model.t` is already in seconds after `generate_fd_polarizations`.
    """
    M_sec = model.M * lal.MTSUN_SI
    t0_dyn = model.dynamics[0, 0]
    offset = model.t[0]
    return lambda t_geom: (t_geom - t0_dyn) * M_sec + offset


def seconds_to_geometric(model):
    """Inverse of `geometric_to_seconds`."""
    M_sec = model.M * lal.MTSUN_SI
    t0_dyn = model.dynamics[0, 0]
    offset = model.t[0]
    return lambda t_sec: (t_sec - offset) / M_sec + t0_dyn


def J_frame_basis(model):
    """pyseobnr's J-frame triad (e1J, e2J, e3J) in the I-frame; e3J is the
    direction of J at attachment, along which the remnant spin is placed."""
    return SEOBBuildJframeVectors(model.Jfhat_attach)


def kerr_qnm_frequency(ell, m, n, final_spin, final_mass):
    """Complex Kerr QNM frequency in units of 1/M (Re > 0, Im < 0, h ~ exp(-i omega t))."""
    import qnm

    if final_spin <= 0:
        raise NotImplementedError("remnant spin anti-aligned with J")
    omega, _, _ = qnm.modes_cache(s=-2, l=ell, m=m, n=n)(a=final_spin)
    return omega / final_mass


RINGDOWN_MODES = ((2, 2, 0), (2, 2, 1))


def fit_ringdown(model, fit_length=50.0):
    """Least-squares fit of the (2,2) mode in the final-J frame with the
    220 and 221 Kerr QNMs of the remnant, from the attachment time to
    `fit_length` M after it.

    Requires a model run with `with_modes=True`. Returns the complex QNM
    frequencies [1/M], the complex amplitudes at t_attach and the relative
    residual of the fit.
    """
    t_geom = seconds_to_geometric(model)(model.t)
    h = model.wavefom_modesJ["2,2"]
    sel = (t_geom >= model.t_attach) & (t_geom <= model.t_attach + fit_length)
    omegas = np.array([
        kerr_qnm_frequency(*lmn, model.final_spin, model.final_mass) for lmn in RINGDOWN_MODES
    ])
    basis = np.exp(-1j * np.outer(t_geom[sel] - model.t_attach, omegas))
    amplitudes, *_ = np.linalg.lstsq(basis, h[sel], rcond=None)
    residual = np.linalg.norm(basis @ amplitudes - h[sel]) / np.linalg.norm(h[sel])
    return dict(omegas=omegas, amplitudes=amplitudes, residual=residual)


def ringdown_h22(fit, t_geom, t_attach):
    """The fitted 220 + 221 (2,2) mode at times t_geom >= t_attach."""
    return np.exp(-1j * np.outer(t_geom - t_attach, fit["omegas"])) @ fit["amplitudes"]


KICK_EPOCH = -100.0  # [M] from the peak: NRSur7dq4Remnant's reference epoch
_nrsur_remnant = {}


def nrsur_kick(m_1, m_2, chi1, chi2, n, LN_hat):
    """Recoil velocity [c] from NRSur7dq4Remnant (surfinBH), given the spins and the orbit
    at KICK_EPOCH from the peak, in the frame of the input vectors. The fit takes the spins
    in the coorbital frame of that epoch (z along L, x along the separation from the lighter
    to the heavier body) and returns the kick in the same frame. Also returns the fit's
    1-sigma error estimate on the speed."""
    if "fit" not in _nrsur_remnant:
        import surfinBH

        _nrsur_remnant["fit"] = surfinBH.LoadFits("NRSur7dq4Remnant")
    if m_1 >= m_2:
        q, chiA, chiB, x = m_1 / m_2, chi1, chi2, n
    else:  # n points from body 2 to body 1
        q, chiA, chiB, x = m_2 / m_1, chi2, chi1, -n
    e3 = LN_hat / np.linalg.norm(LN_hat)
    e1 = x - (x @ e3) * e3
    e1 /= np.linalg.norm(e1)
    basis = np.column_stack([e1, np.cross(e3, e1), e3])
    _, _, vf, _, _, vf_err = _nrsur_remnant["fit"].all(
        q, basis.T @ chiA, basis.T @ chiB, allow_extrap=True)
    return basis @ vf, float(np.linalg.norm(vf_err))
