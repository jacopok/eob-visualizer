"""TEOBResumS-Dali (eccentric, precessing) dynamics of a posterior sample, called as in
the GW150914 bilby run of Gamba et al., CQG 10.1088/1361-6382/adfe50 (f_min = f_ref = 15 Hz).

The PE run called TEOBResumS through bilby's `gwsignal_eccentric_binary_black_hole`
and gwsignal's `TEOBResumSDALI` generator (lalsuite `teobresums-dali` branch, 37da46cd),
with conditioning: for a model whose spins and eccentricity are defined at the start
frequency, gwsignal keeps f22_start = 15 Hz (no lowering), generates at
deltaT = 1 / 2048 s (896 Hz rounded up to a power-of-two multiple of 1/8 Hz), tapers the
start, zero-pads to 8 s and FFTs, with the maximum of h_+ at geocent_time
(`pe_time_origin`). `teob_parameters` reproduces the generator's parameter conversion
(mean -> true anomaly, coalescence_angle = pi/2 - phase). The PE polarizations are
TEOB's own h_+, h_x (all inertial modes, observer at azimuth `phase`, LAL tetrad sign),
as checked against the stored SNRs in `gw150914/validate_teob.py`.

TEOBResumS is at the paper's version (bitbucket tag 2505.21612). Besides the Python
outputs (co-precessing dynamics r, phi, ...; inertial modes; h_+, h_x) we ask it to write
its spin dynamics (spins, L_hat, Euler angles on the EOB time axis) and the co-precessing
modes to files; this leaves the waveform bit-identical.

Frames: the I-frame is TEOB's: z along L_hat at the start (15 Hz), spins given in it
(bilby_to_lalsimulation_spins at f_ref = 15 Hz, i.e. LAL's L0 frame). The co-precessing
frame has z along L_hat(t), Euler angles (alpha, beta, gamma) from the spin dynamics;
the orbit is n(t) = R(t) (cos phi, sin phi, 0) (conventions checked in `gw150914/validate_teob.py`).
Times in units of the total detector-frame mass M unless stated.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import EOBRun_module
import lal
import numpy as np
from bilby.gw import utils as gwutils
from bilby.gw.conversion import bilby_to_lalsimulation_spins
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from .base import EOBRun

# settings of the GW150914 PE run (meta_data in the result file)
REFERENCE_FREQUENCY = 15.0
WAVEFORM_MINIMUM_FREQUENCY = 15.0
SAMPLING_RATE = 2048.0  # gwsignal's conditioning: 0.5 / (2**13 * deltaF), deltaF = 1/8 Hz
MODES = [(2, 2), (2, 1), (3, 3), (4, 4)]


def mode_index(ell, m):
    """TEOBResumS's linear multipole index."""
    return ell * (ell - 1) // 2 + m - 2


def true_anomaly(mean_anomaly, e):
    """As gwsignal's eccentricity_utils: Kepler's equation by Newton iterations."""
    E = mean_anomaly
    for _ in range(100):
        E = E - (E - e * np.sin(E) - mean_anomaly) / (1 - e * np.cos(E))
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


def teob_parameters(sample):
    """The EOBRunPy input of the PE generator (gwsignal `convert_parameters_to_teob`)."""
    iota, s1x, s1y, s1z, s2x, s2y, s2z = bilby_to_lalsimulation_spins(
        theta_jn=sample["theta_jn"], phi_jl=sample["phi_jl"],
        tilt_1=sample["tilt_1"], tilt_2=sample["tilt_2"], phi_12=sample["phi_12"],
        a_1=sample["a_1"], a_2=sample["a_2"],
        mass_1=sample["mass_1"] * gwutils.solar_mass, mass_2=sample["mass_2"] * gwutils.solar_mass,
        reference_frequency=REFERENCE_FREQUENCY, phase=sample["phase"],
    )
    e = sample["eccentricity"]
    return {
        "M": sample["mass_1"] + sample["mass_2"],
        "q": sample["mass_1"] / sample["mass_2"],
        "chi1x": s1x, "chi1y": s1y, "chi1z": s1z,
        "chi2x": s2x, "chi2y": s2y, "chi2z": s2z,
        "ecc": e,
        "inclination": iota,
        "coalescence_angle": np.pi / 2 - sample["phase"],
        "srate_interp": SAMPLING_RATE,
        "initial_frequency": WAVEFORM_MINIMUM_FREQUENCY,
        "distance": sample["luminosity_distance"],
        "anomaly": true_anomaly(sample["mean_per_ano"], e),
        "arg_out": "yes",
        "interp_uniform_grid": "yes",
        "use_geometric_units": "no",
        "use_mode_lm": [mode_index(*lm) for lm in MODES],
    }


def _modes_from_dict(hlm):
    """Inertial modes from EOBRunPy's dictionary: keys 'k' (m > 0), '-k' (m < 0), 'l0'
    (m = 0) -> {(l, m): A exp(-i phase)} (TEOBResumS's convention)."""
    from_k = {mode_index(*lm): lm for lm in MODES}
    modes = {}
    for key, (A, phase) in hlm.items():
        h = np.asarray(A) * np.exp(-1j * np.asarray(phase))
        if key.startswith("-"):
            ell, m = from_k[int(key[1:])]
            modes[(ell, -m)] = h
        elif len(key) == 2 and key.endswith("0"):
            modes[(int(key[0]), 0)] = h
        else:
            modes[from_k[int(key)]] = h
    return modes


def _read_params(path):
    out = {}
    for line in open(path):
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


@dataclass
class Model:
    """What the export needs. Times in M unless stated; the dynamics time axis starts
    at 0 at the initial frequency. "Waveform time" [s] is relative to geocent_time, whose
    sample is the maximum of h_+ (see `pe_time_origin`)."""

    M: float  # total detector-frame mass [M_sun]
    m_1: float  # mass fractions
    m_2: float
    M_sec: float
    dyn: dict  # co-precessing EOB dynamics: t, r, phi, MOmega, ...
    spin: dict  # spin dynamics: t, S1, S2 [M^2], L_hat, alpha, beta, gamma (unwrapped), MOmega
    t_peak: float  # peak of the co-precessing A_22 (TEOB's t = 0); ringdown attached here
    t_origin: float  # sample of the maximum of the tapered h_+: placed at geocent_time in PE
    t_modes: np.ndarray  # [M] uniform grid of the modes and polarizations, dynamics time axis
    hI: dict  # inertial modes {(l, m): complex}, TEOB convention h = A exp(-i phi)
    hP: dict  # co-precessing modes {(l, m>0): complex}
    hpc: np.ndarray  # h_+ - i h_x [strain] as returned by TEOBResumS
    final_mass: float  # [M]
    final_spin: float
    chi1_start: np.ndarray  # I-frame spins at the start
    chi2_start: np.ndarray
    parameters: dict

    @property
    def t_attach(self):
        return self.t_peak

    def t_seconds(self, t_geom):
        """Model time [M] -> waveform time [s], relative to geocent_time."""
        return (np.asarray(t_geom) - self.t_origin) * self.M_sec

    def t_geometric(self, t_sec):
        return np.asarray(t_sec) / self.M_sec + self.t_origin

    def spin_state(self, t):
        """S1, S2 [M^2] and L_hat in the I-frame at times t [M] (clipped to the spin dynamics)."""
        s = self.spin
        t = np.clip(t, s["t"][0], s["t"][-1])
        S1 = CubicSpline(s["t"], s["S1"])(t)
        S2 = CubicSpline(s["t"], s["S2"])(t)
        L = CubicSpline(s["t"], s["L_hat"])(t)
        return S1, S2, L / np.linalg.norm(L, axis=-1, keepdims=True)

    @property
    def J_merger(self):
        """J at the peak as TEOB builds it for the post-merger Euler angles
        (prolong_euler_angles_TD): S1 + S2 + L_2PN(Omega) L_hat [M^2]."""
        s = self.spin
        i = np.searchsorted(s["t"], self.t_peak) - 1
        nu = self.m_1 * self.m_2
        v2 = s["MOmega"][i] ** (2 / 3)
        L2PN = nu / np.sqrt(v2) * (1 + v2 * (1.5 + nu / 6) + v2**2 * (3.375 - 2.375 * nu + nu**2 / 24))
        return s["S1"][i] + s["S2"][i] + L2PN * s["L_hat"][i]

    @property
    def Jfhat_attach(self):
        J = self.J_merger
        return J / np.linalg.norm(J)


def taper_start(h):
    """gwsignal's `taper_gwpy_timeseries(h, 'start')` (a port of XLALSimInspiralREAL8WaveTaper)."""
    from scipy.signal import find_peaks

    h = np.array(h, dtype=float)
    nonzero = np.flatnonzero(h)
    start, end = nonzero[0], nonzero[-1]
    mid = int((start + end) / 2)
    pks, _ = find_peaks(abs(h[start + 1:mid]))
    pks = pks[pks > 19]  # LALSIMULATION_RINGING_EXTENT
    n = mid - start if len(pks) < 2 else pks[1] + 1
    h[start] = 0.0
    i = np.arange(1, n - 1)
    z = (n - 1.0) / i + (n - 1.0) / (i - (n - 1.0))
    h[start + 1:start + n - 1] *= 1.0 / (np.exp(z) + 1.0)
    return h


def pe_time_origin(hpc):
    """Index of the sample placed at geocent_time in PE: gwsignal's
    `resize_gwpy_timeseries` (37da46cd) sets the time axis of the padded, start-tapered
    h_+ so that its maximum (signed, not |h|) is at t = 0, and bilby uses that epoch."""
    return int(np.argmax(taper_start(hpc.real)))


def run_model(sample):
    params = teob_parameters(sample)
    with tempfile.TemporaryDirectory() as tmp:
        t, hp, hc, hlm, dyn = EOBRun_module.EOBRunPy(dict(
            params, output_dynamics="yes", output_multipoles="yes",
            output_lm=params["use_mode_lm"], output_dir=tmp))
        spin_raw = np.loadtxt(Path(tmp) / "dynspin.txt")
        info = _read_params(Path(tmp) / "params.txt")
        hP = {}
        for ell, m in MODES:
            d = np.loadtxt(Path(tmp) / f"hlm_interp_l{ell}_m{m}.txt")
            hP[(ell, m)] = d[:, 1] * np.exp(-1j * d[:, 2])
            t_modes = d[:, 0]
    M = params["M"]
    M_sec = M * lal.MTSUN_SI
    m_1 = params["q"] / (1 + params["q"])
    spin = dict(
        t=spin_raw[:, 0], S1=spin_raw[:, 1:4], S2=spin_raw[:, 4:7], L_hat=spin_raw[:, 7:10],
        # the first row's alpha, gamma are reset to the second's (L = z there: undefined)
        alpha=np.unwrap(np.r_[spin_raw[1, 10], spin_raw[1:, 10]]),
        beta=spin_raw[:, 11],
        gamma=np.unwrap(np.r_[spin_raw[1, 12], spin_raw[1:, 12]]),
        MOmega=spin_raw[:, 13],
    )
    t_wave = np.asarray(t)
    return Model(
        M=M, m_1=m_1, m_2=1 - m_1, M_sec=M_sec,
        dyn={k: np.asarray(v) for k, v in dyn.items()}, spin=spin,
        t_peak=float(t_modes[0] - t_wave[0] / M_sec),
        t_origin=float(t_modes[pe_time_origin(np.asarray(hp) - 1j * np.asarray(hc))]),
        t_modes=t_modes, hI=_modes_from_dict(hlm), hP=hP,
        hpc=np.asarray(hp) - 1j * np.asarray(hc),
        final_mass=float(info["BH_final_mass"]), final_spin=float(info["BH_final_spin"]),
        chi1_start=np.array([params["chi1x"], params["chi1y"], params["chi1z"]]),
        chi2_start=np.array([params["chi2x"], params["chi2y"], params["chi2z"]]),
        parameters=params,
    )


def observer_phi_ref(sample):
    """TEOB's polarizations are -(nu M / D) sum_lm h_lm -2Y_lm(iota, phase): the observer
    is at azimuth `phase` in the I-frame, which is LAL's convention (azimuth pi/2 - phi_ref)
    with phi_ref = pi/2 - phase. The -1 is LAL's tetrad sign (checked in `gw150914/validate_teob.py`)."""
    return np.pi / 2 - sample["phase"]


def euler_rotation(alpha, beta, gamma, convention="zyz"):
    """Rotation from the co-precessing frame to the I-frame. "zyz": R = Rz(alpha) Ry(beta)
    Rz(gamma), with z_P -> L_hat = (sin b cos a, sin b sin a, cos b); "zyz-": the same with
    -gamma. The adopted one is selected in `gw150914/validate_teob.py`."""
    g = gamma if convention == "zyz" else -gamma
    return Rotation.from_euler("ZYZ", np.column_stack([alpha, beta, g])).as_matrix()


EULER_CONVENTION = "zyz-"  # gamma_dot = +alpha_dot cos(beta) in the spin dynamics: minimal rotation


def inertial_dynamics(model, t_geom=None, convention=None):
    """Orbit, spins and angular momenta in the I-frame, in units of M, up to the end of
    the dynamics."""
    d, s = model.dyn, model.spin
    if t_geom is None:
        t_geom = d["t"]
    t_geom = np.atleast_1d(t_geom)
    r = CubicSpline(d["t"], d["r"])(t_geom)
    phi = CubicSpline(d["t"], d["phi"])(t_geom)
    omega = CubicSpline(d["t"], d["MOmega"])(t_geom)
    ts = np.clip(t_geom, s["t"][0], s["t"][-1])
    angles = [CubicSpline(s["t"], s[k])(ts) for k in ("alpha", "beta", "gamma")]
    rot = euler_rotation(*angles, convention or EULER_CONVENTION)
    n_P = np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
    n = np.einsum("tij,tj->ti", rot, n_P)
    S1, S2, L_hat = model.spin_state(t_geom)
    return dict(
        t=t_geom, r=r, phi=phi, omega=omega, n=n,
        x1=model.m_2 * r[:, None] * n, x2=-model.m_1 * r[:, None] * n,
        ez_coprecessing=rot[:, :, 2],
        chi1=S1 / model.m_1**2, chi2=S2 / model.m_2**2,
        LN_hat=L_hat, S1=S1, S2=S2,
    )


def J_start(model):
    """J = L + S1 + S2 at the start [M^2], with the 2PN L of TEOB's spin dynamics."""
    s = model.spin
    nu = model.m_1 * model.m_2
    v2 = s["MOmega"][0] ** (2 / 3)
    L2PN = nu / np.sqrt(v2) * (1 + v2 * (1.5 + nu / 6) + v2**2 * (3.375 - 2.375 * nu + nu**2 / 24))
    return s["S1"][0] + s["S2"][0] + L2PN * s["L_hat"][0]


class TEOBResumSDali(EOBRun):
    """TEOBResumS-Dali run. The merger ("attachment") is the peak of the co-precessing
    A_22, where TEOB attaches its ringdown (the EOB dynamics run ~9 M further, to r = 1.5);
    the remnant has TEOB's final mass and spin, the spin along J at the peak as TEOB
    computes it for its post-merger Euler angles (`Model.J_merger`). The ringdown is fitted
    on the co-precessing (2,2) mode from the peak: the co-precessing frame then precesses
    about the final J with the QNM prescription, so this is the remnant's own (2,2) up to
    that slow rotation. Spins are given at 15 Hz; J(t -> -inf) neglects the eccentricity
    (quasi-circular PN equations)."""

    approximant = "TEOBResumS-Dali"

    def __init__(self, sample):
        self.model = m = run_model(sample)
        self.M, self.m_1, self.m_2, self.M_sec = m.M, m.m_1, m.m_2, m.M_sec
        self.final_mass, self.final_spin = m.final_mass, m.final_spin
        self.t_attach = self.t_peak = m.t_peak
        self.t_start, self.t_eval_end = m.dyn["t"][0], m.t_peak
        self.omega_ref = np.pi * WAVEFORM_MINIMUM_FREQUENCY * m.M_sec
        self.chi1_ref, self.chi2_ref = m.chi1_start, m.chi2_start
        self.inclination, self.phi_ref = m.parameters["inclination"], observer_phi_ref(sample)

    def inertial_dynamics(self, t_geom=None):
        return inertial_dynamics(self.model, t_geom)

    def t_geometric(self, t_sec):
        return self.model.t_geometric(t_sec)

    def t_seconds(self, t_geom):
        return self.model.t_seconds(t_geom)

    @property
    def Jfhat_attach(self):
        return self.model.Jfhat_attach

    def J_reference(self):
        return J_start(self.model)

    def ringdown_mode(self):
        return self.model.t_modes, self.model.hP[(2, 2)]

    def info(self):
        m = self.model
        return {"pe_time_origin_after_peak_ms": float(m.t_seconds(m.t_origin) - m.t_seconds(m.t_peak)) * 1e3}
