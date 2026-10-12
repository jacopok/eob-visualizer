"""SEOBNRv4PHM dynamics of GW190521 posterior samples (GWTC-2.1 release, LALInference run).

The waveform model is called as LALInference called it: SimInspiralChooseTDWaveform with
the Cartesian spins of the sample, iota, phase, f_min = 5 Hz (fmin-template) and
sampling rate 512 Hz. SEOBNRv4PHM ignores f_ref: the spins are those at f_min, and the
phase is defined at coalescence. ChooseTDWaveform calls SimIMRSpinPrecEOBWaveform, which
calls SimIMRSpinPrecEOBWaveformAll with fixed flags; the latter also returns the
dynamics, so we call it directly with the same flags (identified by reproducing the
ChooseTDWaveform polarizations to machine precision, `validate_v4phm.py`), and apply the
polarization rotation by -pi/2 (an overall sign) that ChooseTDWaveform applies.

Frames: the dynamics are integrated in the LAL L0 frame of f_min (z along L_N, x along
the separation vector at the start); `observer_direction_I` and `rotation_I_to_sky`
from the GW250114 module apply (checked in `validate_v4phm.py`).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import lal
import lalsimulation as ls
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).parent.parent / "gw250114"))
import eob_dynamics as ed  # noqa: E402  (observer frame, QNMs, PN backward evolution)

POSTERIOR_FILE = (
    Path(__file__).parent.parent
    / "IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_nocosmo.h5"
)
POSTERIOR_LABEL = "C01:SEOBNRv4PHM"

# settings of the LALInference run (config_file in the release)
WAVEFORM_MINIMUM_FREQUENCY = 5.0  # fmin-template
SAMPLING_RATE = 512.0
KM_PER_MSUN = ed.KM_PER_MSUN

N_DYNAMICS_VARIABLES = 26
SEOB_FLAGS = {
    "SEOBNRv4P_SpinAlignedEOBversion": 4,
    "SEOBNRv4P_SymmetrizehPlminusm": 1,
    "SEOBNRv4P_HamiltonianDerivative": 1,  # numerical (default)
    "SEOBNRv4P_euler_extension": 0,  # QNM simple precession
    "SEOBNRv4P_Zframe": 0,  # L
    "SEOBNRv4P_debug": 0,
    "ellMaxForNyquistCheck": 5,  # default EOBEllMaxForNyquistCheck
}
MODES = [(2, 2), (2, 1), (3, 3), (4, 4), (5, 5)]  # default v4PHM mode content


def load_posterior():
    with h5py.File(POSTERIOR_FILE, "r") as f:
        return f[POSTERIOR_LABEL]["posterior_samples"][()]


def sample_as_dict(posterior, index):
    return {name: float(posterior[name][index]) for name in posterior.dtype.names}


def _sph_harm_list(series):
    modes = {}
    while series is not None:
        modes[(series.l, series.m)] = np.array(series.mode.data.data)
        series = series.next
    return modes


@dataclass
class Model:
    """What `export_scene.py` needs, with the names of the pyseobnr model attributes
    where they exist. Times in units of the total detector-frame mass M."""

    M: float  # total detector-frame mass [M_sun]
    m_1: float  # mass fractions
    m_2: float
    dynamics: np.ndarray  # (n, 26): t, r_vec, p_vec, S1, S2, phiD, phi, v_vec, r, ...
    t_attach: float
    t_peak: float
    J_final: np.ndarray  # final J [M^2], the remnant spin direction
    final_mass: float  # [M]
    final_spin: float
    t_modes: np.ndarray  # [M], time axis of the modes and polarizations
    hJ: dict  # modes in the final-J frame
    hI: dict  # modes in the I-frame
    distance_in_M: float
    hpc: np.ndarray  # h_+ - i h_x as returned by ChooseTDWaveform
    epoch: float  # [s], time of the first sample relative to the waveform's t = 0
    M_sec: float  # M in seconds as used by the model (t_modes step / deltaT)
    chi1_start: np.ndarray
    chi2_start: np.ndarray

    @property
    def Jfhat_attach(self):
        return self.J_final / np.linalg.norm(self.J_final)

    def t_seconds(self, t_geom):
        """Model time [M] -> waveform time [s] relative to the waveform's t = 0."""
        return (np.asarray(t_geom) - self.t_modes[0]) * self.M_sec + self.epoch

    def t_geometric(self, t_sec):
        return (np.asarray(t_sec) - self.epoch) / self.M_sec + self.t_modes[0]


def run_model(sample, f_min=WAVEFORM_MINIMUM_FREQUENCY, delta_t=1 / SAMPLING_RATE):
    m1, m2 = sample["mass_1"], sample["mass_2"]
    S1 = [sample["spin_1x"], sample["spin_1y"], sample["spin_1z"]]
    S2 = [sample["spin_2x"], sample["spin_2y"], sample["spin_2z"]]
    modearray = ls.SimInspiralCreateModeArray()
    for ell, m in MODES:
        ls.SimInspiralModeArrayActivateMode(modearray, ell, m)
    flags = lal.CreateDict()
    for k, v in SEOB_FLAGS.items():
        lal.DictInsertINT4Value(flags, k, v)
    out = ls.SimIMRSpinPrecEOBWaveformAll(
        sample["phase"], delta_t, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, f_min,
        sample["luminosity_distance"] * 1e6 * lal.PC_SI, sample["iota"], *S1, *S2,
        modearray, flags,
    )
    hp, hc = out[0], out[1]
    dyn = np.array(out[6].data).reshape(N_DYNAMICS_VARIABLES, -1).T
    merger = np.array(out[21].data)
    # modes and polarizations share a uniform grid in M starting with the dynamics
    t_modes = float(out[3].mode.epoch) + out[3].mode.deltaT * np.arange(hp.data.length)
    M = m1 + m2
    return Model(
        M=M, m_1=m1 / M, m_2=m2 / M, dynamics=dyn,
        t_attach=merger[1], t_peak=merger[2], J_final=merger[3:6],
        final_mass=merger[6], final_spin=merger[7],
        t_modes=t_modes, hJ=_sph_harm_list(out[3]), hI=_sph_harm_list(out[2]),
        distance_in_M=sample["luminosity_distance"] * 1e6 * lal.PC_SI / (M * lal.MRSUN_SI),
        # ChooseTDWaveform rotates the polarizations by -pi/2, i.e. flips their sign
        hpc=-(np.array(hp.data.data) - 1j * np.array(hc.data.data)),
        epoch=float(hp.epoch), M_sec=hp.deltaT / (t_modes[1] - t_modes[0]),
        chi1_start=np.array(S1), chi2_start=np.array(S2),
    )


def polarizations(model, iota, phase):
    """h_+ - i h_x for another observer azimuth: the phase only enters v4PHM through the
    observer direction (iota, pi/2 - phase), as checked against the full model to
    machine precision; the sign is ChooseTDWaveform's."""
    return -sum(h * lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phase, -2, ell, m)
                for (ell, m), h in model.hI.items()) / model.distance_in_M


PHASES_FILE = Path(__file__).parent / "products" / "phase_reconstruction.npz"


def reconstructed_phases():
    """Phases drawn by `reconstruct_phase.py` (the release's are not usable, see there)."""
    if not PHASES_FILE.exists():
        raise FileNotFoundError("run reconstruct_phase.py first")
    d = np.load(PHASES_FILE)
    return dict(zip(d["indices"].tolist(), d["phase_drawn"]))


def sample_with_phase(posterior, index):
    """A posterior sample with its coalescence phase replaced by the reconstructed one."""
    s = sample_as_dict(posterior, index)
    s["phase_release"] = s["phase"]
    s["phase"] = float(reconstructed_phases()[index])
    return s


def inertial_dynamics(model, t_geom=None):
    """Orbit, spins and angular momenta in the I-frame (L0 frame of f_min), in units of M."""
    dyn = model.dynamics
    t_dyn = dyn[:, 0]
    if t_geom is None:
        t_geom = t_dyn
    spline = lambda cols: CubicSpline(t_dyn, dyn[:, cols])(t_geom)
    r_vec = spline(slice(1, 4))
    p_vec = spline(slice(4, 7))
    S1, S2 = spline(slice(7, 10)), spline(slice(10, 13))
    v_vec = spline(slice(15, 18))
    r = np.linalg.norm(r_vec, axis=1)
    n = r_vec / r[:, None]
    nu = model.m_1 * model.m_2
    L = nu * np.cross(r_vec, p_vec)  # canonical orbital angular momentum [M^2]
    LN = np.cross(r_vec, v_vec)
    # orbital phase: angle swept by n, accumulated along the orbit
    dphi = np.linalg.norm(np.cross(n[:-1], n[1:]), axis=1)
    phi = np.r_[0, np.cumsum(np.arcsin(np.clip(dphi, -1, 1)))] if len(n) > 1 else np.zeros(1)
    return dict(
        t=t_geom, r=r, n=n, phi=phi,
        omega=spline(22),
        x1=model.m_2 * r_vec, x2=-model.m_1 * r_vec,
        chi1=S1 / model.m_1**2, chi2=S2 / model.m_2**2,
        LN_hat=LN / np.linalg.norm(LN, axis=1)[:, None],
        L=L, J=L + S1 + S2,
    )


def orbital_phase(model, t_geom):
    """Orbital phase [rad] at arbitrary times, from the angle swept by n on a fine grid."""
    t_fine = np.linspace(model.dynamics[0, 0], model.dynamics[-1, 0], 200000)
    d = inertial_dynamics(model, t_fine)
    return np.interp(t_geom, t_fine, d["phi"])


def J_hat_at_past_infinity(model, f_low=1e-3):
    """J direction at t -> -inf, evolving the sample's spins at f_min backwards with
    pyseobnr's PN precession equations (the v4PHM dynamics start at f_min)."""
    from pyseobnr.eob.dynamics.pn_evolution_opt import (
        build_splines_PN,
        compute_quasiprecessing_PNdynamics_opt,
    )

    omega_start = model.dynamics[0, 22]
    omega_low = np.pi * f_low * model.M_sec
    # spins in the frame where L_N(f_min) = z: that is the I-frame
    t, y = compute_quasiprecessing_PNdynamics_opt(
        omega_start, omega_low, model.m_1, model.m_2, model.chi1_start, model.chi2_start
    )
    keep = y[:, -1] <= omega_start * (1 + 1e-12)
    y = y[keep]
    splines = build_splines_PN(t[keep], y, model.m_1, model.m_2, omega_low)
    pn = splines["everything"](y[:, -1])
    J = pn[:, 13:16] + model.m_1**2 * pn[:, 4:7] + model.m_2**2 * pn[:, 7:10]
    J_hat = J / np.linalg.norm(J, axis=1)[:, None]
    return J_hat[0], y[:, -1] / (np.pi * model.M_sec), J_hat


def fit_ringdown(model, fit_length=50.0):
    """220 + 221 fit of the (2,2) mode in the final-J frame from t_attach (as in GW250114)."""
    t = model.t_modes
    h = model.hJ[(2, 2)]
    sel = (t >= model.t_attach) & (t <= model.t_attach + fit_length)
    omegas = np.array([ed.kerr_qnm_frequency(*lmn, model.final_spin, model.final_mass)
                       for lmn in ed.RINGDOWN_MODES])
    basis = np.exp(-1j * np.outer(t[sel] - model.t_attach, omegas))
    amplitudes, *_ = np.linalg.lstsq(basis, h[sel], rcond=None)
    residual = np.linalg.norm(basis @ amplitudes - h[sel]) / np.linalg.norm(h[sel])
    return dict(omegas=omegas, amplitudes=amplitudes, residual=residual, n_points=int(sel.sum()))
