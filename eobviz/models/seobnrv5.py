"""SEOBNRv5PHM (pyseobnr) dynamics of a posterior sample, called as in bilby PE.

The waveform call reproduces exactly what bilby (`gwsignal_binary_black_hole`)
did during parameter estimation, so the dynamics are the ones that generated
the waveform used in the likelihood.

pyseobnr builds the inertial-frame orbit as n(t) = R(t) (cos phi, sin phi, 0),
with R = quatI2J * quatJ2P(t) equal to the identity at t_ref: the I-frame is LAL's
L0 frame at f_ref (`physics`).
"""

import lal
import numpy as np
import quaternion
from bilby.gw import utils as gwutils
from bilby.gw.conversion import bilby_to_lalsimulation_spins
from pyseobnr.eob.utils.utils_precession_opt import (
    SEOBBuildJframeVectors,
    inspiral_merger_quaternion_angles,
)
from pyseobnr.generate_waveform import GenerateWaveform

from .base import EOBRun

# default waveform settings: those of the GW250114 SEOBNRv5PHM PE run
# (from the config stored in the posterior file)
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
    dynamics start somewhat below f_min.

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


def geometric_to_seconds(model):
    """Conversion of dynamics time [M] to time [s] relative to the waveform's t=0,
    which bilby places at geocent_time (verified in `gw250114/validate_dynamics.py`).

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


class SEOBNRv5PHM(EOBRun):
    """pyseobnr run. The remnant: pyseobnr's final mass and spin, the spin along J at
    attachment (as pyseobnr models it); the ringdown is fitted on the (2,2) mode in the
    final-J frame. Spins are given at f_ref."""

    approximant = "SEOBNRv5PHM"

    def __init__(self, sample, settings=None, with_modes=True):
        """settings: see `pyseobnr_parameters`; with_modes: see `run_model` (needed for
        the ringdown fit)."""
        self.model, self.generator = run_model(sample, with_modes, settings)
        m, p = self.model, self.generator.parameters
        self.M, self.m_1, self.m_2 = m.M, m.m_1, m.m_2
        self.M_sec = m.M * lal.MTSUN_SI
        self.final_mass, self.final_spin = m.final_mass, m.final_spin
        self.t_attach = self.t_peak = m.t_attach
        self.t_start, self.t_eval_end = m.dynamics[0, 0], m.t_attach
        self.omega_ref, self.chi1_ref, self.chi2_ref = m.omega_ref, m.chi1_v, m.chi2_v
        self.inclination, self.phi_ref = p["inclination"], p["phi_ref"]
        self._to_geometric = seconds_to_geometric(m)
        self._to_seconds = geometric_to_seconds(m)

    def inertial_dynamics(self, t_geom=None):
        return inertial_dynamics(self.model, t_geom)

    def t_geometric(self, t_sec):
        return self._to_geometric(t_sec)

    def t_seconds(self, t_geom):
        return self._to_seconds(t_geom)

    @property
    def Jfhat_attach(self):
        return self.model.Jfhat_attach

    def J_reference(self):
        return self.inertial_dynamics(np.atleast_1d(self.model.t_ref))["J"][0]

    def ringdown_mode(self):
        return self.t_geometric(self.model.t), self.model.wavefom_modesJ["2,2"]
