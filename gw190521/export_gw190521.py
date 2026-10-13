"""Per-frame scene data for GW190521 SEOBNRv4PHM posterior samples, in the same format
as `gw250114/export_scene.py` (see there for the conventions), so that the same
camera, Blender and panel scripts apply.

Differences from GW250114: the dynamics are SEOBNRv4PHM's (`eob_v4phm.py`); the spins
of the samples are those at f_min = 5 Hz (v4PHM ignores f_ref), so "J at the reference"
is J at the start of the dynamics; J(t -> -inf) is obtained by evolving those spins
backwards with pyseobnr's PN precession equations. The remnant spin is along v4PHM's
final J.

Usage: python export_gw190521.py --n-samples 200   (or no argument: max likelihood only)
"""

import argparse
import sys
from pathlib import Path

import bilby
import numpy as np

import eob_v4phm as ev

sys.path.insert(0, str(Path(__file__).parent.parent / "gw250114"))
import eob_dynamics as ed  # noqa: E402
from export_scene import REMNANT_DEFORMATION, areal_radius, export_set  # noqa: E402

OUTDIR = Path(__file__).parent / "products"
IFO = "L1"
REFERENCE_TIME = 1242442967.4
TIME_WINDOW = (-0.95, 0.30)  # detector time - REFERENCE_TIME [s]; merger at ~+0.05
FPS = 30
SLOWDOWN = 50  # GW190521 is ~4 times heavier than GW250114: its orbits are ~4 times slower
TRAIL_POINTS_PER_ORBIT = 96
PANEL_NOTE = "Coalescence phases re-drawn from the H1+L1 data\n(not usable in the release; approximate)\n"


def frame_times():
    n_frames = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) * SLOWDOWN * FPS)) + 1
    return np.linspace(*TIME_WINDOW, n_frames)


def export(sample):
    model = ev.run_model(sample)
    M_src_km = model.M / (1 + sample["redshift"]) * ev.KM_PER_MSUN

    rot = ed.rotation_I_to_sky(sample["iota"], sample["phase"], sample["psi"])
    sky = lambda v: v @ rot.T

    ifo = bilby.gw.detector.get_empty_interferometer(IFO)
    delay = ifo.time_delay_from_geocenter(sample["ra"], sample["dec"], sample["geocent_time"])

    t_det_rel = frame_times()
    t_wave = t_det_rel + REFERENCE_TIME - sample["geocent_time"] - delay
    t_geom = model.t_geometric(t_wave)
    t_dyn_end = model.dynamics[-1, 0]
    if t_geom[0] < model.dynamics[0, 0]:
        raise ValueError("time window starts before the dynamics")
    ta = model.t_attach
    merged = t_geom >= ta

    def evaluate(t):
        d = ev.inertial_dynamics(model, np.minimum(t, t_dyn_end))
        d.update({k: sky(d[k]) for k in ("x1", "x2", "chi1", "chi2", "LN_hat", "n")})
        for k in ("x1", "x2"):
            d[k] = d[k] * M_src_km
        d["phi"] = ev.orbital_phase(model, np.minimum(t, t_dyn_end))
        return d

    frames = evaluate(t_geom)
    at_attach = evaluate(np.atleast_1d(ta))
    omega_attach = model.dynamics[-1, 22]
    now_phase = (frames["phi"] + np.maximum(t_geom - ta, 0) * omega_attach) / (2 * np.pi)

    t_end = min(ta, t_geom[-1])
    t_dense = np.linspace(t_geom[0], t_end, 20000)
    phi_dense = ev.orbital_phase(model, t_dense)
    n_trail = int((phi_dense[-1] - phi_dense[0]) / (2 * np.pi) * TRAIL_POINTS_PER_ORBIT) + 2
    trail = evaluate(np.interp(np.linspace(phi_dense[0], phi_dense[-1], n_trail), phi_dense, t_dense))

    chi_f = abs(model.final_spin)
    e3 = rot @ model.Jfhat_attach
    n_a = at_attach["n"][0]
    e1 = n_a - (n_a @ e3) * e3
    e1 /= np.linalg.norm(e1)
    remnant_basis = np.column_stack([e1, np.cross(e3, e1), e3])
    fit = ev.fit_ringdown(model)
    h_of = lambda t: np.exp(-1j * np.outer(t - ta, fit["omegas"])) @ fit["amplitudes"]
    h = h_of(np.maximum(t_geom, ta))
    deformation = np.where(merged, REMNANT_DEFORMATION * np.conj(h / h_of(np.atleast_1d(ta))[0]), 0)
    # recoil, from the spins and orbit at -100 M from the peak; the remnant moves with it
    # from the attachment on
    k = evaluate(np.atleast_1d(model.t_peak + ed.KICK_EPOCH))
    kick, kick_err = ed.nrsur_kick(model.m_1, model.m_2, k["chi1"][0], k["chi2"][0], k["n"][0], k["LN_hat"][0])
    remnant_offset = np.maximum(t_geom - ta, 0)[:, None] * kick * M_src_km

    J_inf_I, _, _ = ev.J_hat_at_past_infinity(model)
    J_start_I = ev.inertial_dynamics(model, model.dynamics[:1, 0])["J"][0]
    J_start_I /= np.linalg.norm(J_start_I)

    chi1_mag, chi2_mag = np.linalg.norm(model.chi1_start), np.linalg.norm(model.chi2_start)
    out = dict(
        fps=FPS, slowdown=SLOWDOWN, ifo=IFO, reference_time=REFERENCE_TIME,
        t_det_rel=t_det_rel, t_wave=t_wave, merged=merged,
        x1=frames["x1"], x2=frames["x2"], chi1=frames["chi1"], chi2=frames["chi2"],
        LN_hat=frames["LN_hat"], now_phase=now_phase,
        trail1=trail["x1"], trail2=trail["x2"], trail_phase=trail["phi"] / (2 * np.pi),
        radius1=areal_radius(model.m_1, chi1_mag) * M_src_km,
        radius2=areal_radius(model.m_2, chi2_mag) * M_src_km,
        radius_f=areal_radius(model.final_mass, chi_f) * M_src_km,
        chi_f=chi_f * e3, remnant_basis=remnant_basis, deformation=deformation,
        kick_velocity=kick, remnant_offset=remnant_offset,
        J_hat_inf=rot @ J_inf_I, J_hat_ref=rot @ J_start_I, km_per_M=M_src_km,
    )
    info = {
        "sample": {k: sample[k] for k in [
            "mass_1", "mass_2", "mass_1_source", "mass_2_source", "redshift",
            "a_1", "a_2", "tilt_1", "tilt_2", "theta_jn", "iota", "psi", "phase",
            "ra", "dec", "geocent_time", "luminosity_distance", "log_likelihood"]},
        "delay_" + IFO: delay,
        "t_attach_det_rel": float(np.interp(ta, t_geom, t_det_rel)),
        "angle_J_inf_vs_J_fmin_deg": float(np.degrees(np.arccos(np.clip(J_inf_I @ J_start_I, -1, 1)))),
        "angle_J_inf_vs_line_of_sight_deg": float(np.degrees(np.arccos(out["J_hat_inf"][2]))),
        "final_mass_source": model.final_mass * model.M / (1 + sample["redshift"]),
        "final_spin": chi_f,
        "kick_km_s": float(np.linalg.norm(kick) * ed.lal.C_SI / 1e3),
        "kick_err_km_s": kick_err * ed.lal.C_SI / 1e3,
        "km_per_M_source": M_src_km,
        "ringdown_fit_residual": float(fit["residual"]),
        "A221_over_A220": [float(abs(fit["amplitudes"][1] / fit["amplitudes"][0])),
                           float(np.angle(fit["amplitudes"][1] / fit["amplitudes"][0]))],
        "M_omega_220": [float(fit["omegas"][0].real), float(fit["omegas"][0].imag)],
        "M_omega_221": [float(fit["omegas"][1].real), float(fit["omegas"][1].imag)],
    }
    return out, info


def export_gw190521(indices, name, processes=8):
    return export_set(
        indices, name, processes=processes, export_fn=export,
        load_posterior=ev.load_posterior, sample_as_dict=ev.sample_with_phase, outdir=OUTDIR,
        extra={"event": "GW190521", "approximant": "SEOBNRv4PHM",
               "ppd_file": f"ppd_SEOBNRv4PHM_{IFO}.npz", "panel_note": PANEL_NOTE},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--index", type=int, default=None)
    group.add_argument("--n-samples", type=int, default=None,
                       help="the first N of the random posterior samples used for the PPD")
    args = parser.parse_args()
    OUTDIR.mkdir(exist_ok=True)
    if args.n_samples:
        ppd = np.load(OUTDIR / f"ppd_SEOBNRv4PHM_{IFO}.npz")
        export_gw190521([int(i) for i in ppd["sample_indices"][: args.n_samples]],
                        f"cloud{args.n_samples}")
    else:
        index = args.index
        if index is None:
            index = int(np.argmax(ev.load_posterior()["log_likelihood"]))
        export_gw190521([index], f"sample{index}", processes=1)
