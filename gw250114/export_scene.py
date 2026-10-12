"""Export per-frame scene data for a set of posterior samples, to be read by `blender_scene.py`.

Timeline: frames are uniformly spaced in detector time at `IFO` (the detector
whose whitened strain is shown in the panel). The state of the binary shown at
detector time t_det is the one at waveform time t_det - geocent_time - delay_IFO,
i.e. the signal arriving at the detector at t_det was emitted by that state.
All samples share this timeline, so each one merges at its own detector time.

Space: sky frame (X west, Y north, Z towards Earth), lengths in source-frame km
(dynamics in units of the detector-frame mass M, times G M / c^2 / (1 + z)).

Bodies are drawn with their Kerr horizon areal radius sqrt(A / 4 pi) = sqrt(2 m r_+),
placed at the EOB (point-particle, centre-of-mass frame) positions. Before the
attachment time t_attach the two inspiralling BHs are shown; after it, the remnant
with pyseobnr's final mass and spin, the spin along the J direction at attachment
(which is how pyseobnr models it).

Ringdown: the remnant's surface is r = R_f [1 + Re(C(t) e^{-2 i phi}) sin^2 theta]
in a frame with z along the remnant spin and phi measured from the separation
vector at attachment, with
    C(t) = REMNANT_DEFORMATION * conj(h(t) / h(t_attach)),
h the 220 + 221 Kerr-QNM fit to the sample's own (2,2) mode in the final-J frame.
The l = m = 2 bar thus starts along the line joining the two BHs, rotates
with the pattern speed and decays with the damping times of the two QNMs, with
their fitted relative amplitude and phase. Its overall size is illustrative.

Grid: perpendicular to the mean J(t -> -inf) of the set, after reflecting the J's
pointing away from us through the sky plane (theta_JN is bimodal, and mirror images
through the sky plane look the same in an orthographic view along the line of sight).

Sky orientation: psi and psi + pi give identical detector responses, and bilby
samples psi in [0, pi). Each sample is shown with the representative that puts the
sky projection of its J(t -> -inf) on the same side as the (axial) mean of the set.
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import bilby
import numpy as np

import eob_dynamics as ed
from ppd import OUTDIR, REFERENCE_TIME

IFO = "L1"
TIME_WINDOW = (-0.29, 0.25)  # detector time - REFERENCE_TIME [s]; merger at ~0.217
FPS = 24
SLOWDOWN = 100  # video seconds per second of signal
TRAIL_POINTS_PER_ORBIT = 96
REMNANT_DEFORMATION = 0.3  # fractional l = m = 2 deformation at t_attach (illustrative)
SCENES = OUTDIR / "scenes"

# arrays holding sky-frame vectors along their last axis
VECTOR_KEYS = ("x1", "x2", "chi1", "chi2", "LN_hat", "trail1", "trail2",
               "chi_f", "J_hat_inf", "J_hat_ref")


def areal_radius(m, chi):
    """sqrt(A / 4 pi) for a Kerr BH, in units of m."""
    r_plus = m * (1 + np.sqrt(1 - chi**2))
    return np.sqrt(2 * m * r_plus)


def frame_times():
    n_frames = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) * SLOWDOWN * FPS)) + 1
    return np.linspace(*TIME_WINDOW, n_frames)


def export(sample):
    model, gen = ed.run_model(sample, with_modes=True)
    params = gen.parameters
    M_src_km = model.M / (1 + sample["redshift"]) * ed.KM_PER_MSUN  # km per unit of M

    rot = ed.rotation_I_to_sky(params["inclination"], params["phi_ref"], sample["psi"])
    N_I, _, _ = ed.observer_direction_I(params["inclination"], params["phi_ref"])
    assert np.allclose(rot @ N_I, [0, 0, 1])
    sky = lambda v: v @ rot.T

    ifo = bilby.gw.detector.get_empty_interferometer(IFO)
    delay = ifo.time_delay_from_geocenter(sample["ra"], sample["dec"], sample["geocent_time"])

    t_det_rel = frame_times()
    t_wave = t_det_rel + REFERENCE_TIME - sample["geocent_time"] - delay
    t_geom = ed.seconds_to_geometric(model)(t_wave)
    if t_geom[0] < model.dynamics[0, 0]:
        raise ValueError("time window starts before the dynamics")
    ta = model.t_attach
    merged = t_geom >= ta

    def evaluate(t):
        d = ed.inertial_dynamics(model, np.minimum(t, ta))
        d.update({k: sky(d[k]) for k in ("x1", "x2", "chi1", "chi2", "LN_hat", "n")})
        for k in ("x1", "x2"):
            d[k] = d[k] * M_src_km
        return d

    frames = evaluate(t_geom)
    at_attach = evaluate(np.atleast_1d(ta))

    # orbital phase [cycles], used to fade the trails behind the bodies; after the
    # attachment it keeps running at the orbital frequency of the attachment
    now_phase = (frames["phi"] + np.maximum(t_geom - ta, 0) * at_attach["omega"][0]) / (2 * np.pi)

    # trails, uniformly sampled in orbital phase up to the attachment (or the end of the window)
    t_end = min(ta, t_geom[-1])
    t_dense = np.linspace(t_geom[0], t_end, 20000)
    phi_dense = ed.inertial_dynamics(model, t_dense)["phi"]
    n_trail = int((phi_dense[-1] - phi_dense[0]) / (2 * np.pi) * TRAIL_POINTS_PER_ORBIT) + 2
    trail = evaluate(np.interp(np.linspace(phi_dense[0], phi_dense[-1], n_trail), phi_dense, t_dense))

    # remnant: frame with z along its spin and x along the separation at attachment
    chi_f = abs(model.final_spin)
    e3 = rot @ model.Jfhat_attach
    n_a = at_attach["n"][0]
    e1 = n_a - (n_a @ e3) * e3
    e1 /= np.linalg.norm(e1)
    remnant_basis = np.column_stack([e1, np.cross(e3, e1), e3])
    fit = ed.fit_ringdown(model)
    h_attach = ed.ringdown_h22(fit, np.atleast_1d(ta), ta)[0]
    h = ed.ringdown_h22(fit, np.maximum(t_geom, ta), ta)
    deformation = np.where(merged, REMNANT_DEFORMATION * np.conj(h / h_attach), 0)

    J_inf_I, _, _ = ed.J_hat_at_past_infinity(model)
    J_ref_I = ed.inertial_dynamics(model, np.atleast_1d(model.t_ref))["J"][0]
    J_ref_I /= np.linalg.norm(J_ref_I)

    chi1_mag = np.linalg.norm(model.chi1_v)
    chi2_mag = np.linalg.norm(model.chi2_v)
    out = dict(
        fps=FPS,
        slowdown=SLOWDOWN,
        ifo=IFO,
        reference_time=REFERENCE_TIME,
        t_det_rel=t_det_rel,
        t_wave=t_wave,
        merged=merged,
        x1=frames["x1"],
        x2=frames["x2"],
        chi1=frames["chi1"],
        chi2=frames["chi2"],
        LN_hat=frames["LN_hat"],
        now_phase=now_phase,
        trail1=trail["x1"],
        trail2=trail["x2"],
        trail_phase=trail["phi"] / (2 * np.pi),
        radius1=areal_radius(model.m_1, chi1_mag) * M_src_km,
        radius2=areal_radius(model.m_2, chi2_mag) * M_src_km,
        radius_f=areal_radius(model.final_mass, chi_f) * M_src_km,
        chi_f=chi_f * e3,
        remnant_basis=remnant_basis,
        deformation=deformation,
        J_hat_inf=rot @ J_inf_I,
        J_hat_ref=rot @ J_ref_I,
        km_per_M=M_src_km,
    )
    info = {
        "sample": {k: sample[k] for k in [
            "mass_1", "mass_2", "mass_1_source", "mass_2_source", "redshift",
            "a_1", "a_2", "tilt_1", "tilt_2", "theta_jn", "iota", "psi", "phase",
            "ra", "dec", "geocent_time", "luminosity_distance", "log_likelihood"]},
        "delay_" + IFO: delay,
        "t_attach_det_rel": float(np.interp(ta, t_geom, t_det_rel)),
        "angle_J_inf_vs_J_fref_deg": float(np.degrees(np.arccos(np.clip(J_inf_I @ J_ref_I, -1, 1)))),
        "angle_J_inf_vs_line_of_sight_deg": float(np.degrees(np.arccos(out["J_hat_inf"][2]))),
        "final_mass_source": model.final_mass * model.M / (1 + sample["redshift"]),
        "final_spin": chi_f,
        "km_per_M_source": M_src_km,
        "ringdown_fit_residual": float(fit["residual"]),
        "A221_over_A220": [float(abs(fit["amplitudes"][1] / fit["amplitudes"][0])),
                           float(np.angle(fit["amplitudes"][1] / fit["amplitudes"][0]))],
        "M_omega_220": [float(fit["omegas"][0].real), float(fit["omegas"][0].imag)],
        "M_omega_221": [float(fit["omegas"][1].real), float(fit["omegas"][1].imag)],
    }
    return out, info


def rotate_by_pi_about_line_of_sight(out, info):
    """psi -> psi + pi: an exact symmetry of the detector response."""
    flip = np.array([-1.0, -1.0, 1.0])
    for k in VECTOR_KEYS:
        out[k] = out[k] * flip
    out["remnant_basis"] = flip[:, None] * out["remnant_basis"]
    info["psi_shown"] = info["sample"]["psi"] + np.pi


def axial_mean_angle(vectors_xy):
    """Mean orientation (mod pi) of a set of 2D vectors, weighted by their length."""
    z = vectors_xy[:, 0] + 1j * vectors_xy[:, 1]
    return np.angle(np.sum(z**2 / np.maximum(np.abs(z), 1e-12))) / 2


def _export_index(args):
    export_fn, sample_as_dict, posterior, index = args
    return export_fn(sample_as_dict(posterior, index))


def export_set(indices, name, processes=8, export_fn=None, load_posterior=ed.load_posterior,
               sample_as_dict=ed.sample_as_dict, outdir=OUTDIR, extra=None):
    """Export the samples `indices` and write the manifest `outdir/<name>.json`.
    The defaults are GW250114's; another event passes its own export function
    (returning the same arrays), posterior access and output directory."""
    export_fn = export_fn or export
    scenes = outdir / "scenes"
    posterior = load_posterior()
    with Pool(processes) as pool:
        results = pool.map(_export_index, [(export_fn, sample_as_dict, posterior, i) for i in indices],
                           chunksize=1)

    J_xy = np.array([out["J_hat_inf"][:2] for out, _ in results])
    reference = axial_mean_angle(J_xy)
    axis = np.array([np.cos(reference), np.sin(reference)])
    # the axial mean is defined mod pi: orient it so that most samples keep bilby's psi
    if np.count_nonzero(J_xy @ axis < 0) > len(J_xy) / 2:
        axis = -axis

    scenes.mkdir(parents=True, exist_ok=True)
    labels = []
    for index, (out, info) in zip(indices, results):
        info["psi_shown"] = info["sample"]["psi"]
        if out["J_hat_inf"][:2] @ axis < 0:
            rotate_by_pi_about_line_of_sight(out, info)
        label = f"sample{index}"
        info["label"] = label
        np.savez(scenes / f"{label}.npz", label=label, **out)
        with open(scenes / f"{label}.json", "w") as f:
            json.dump(info, f, indent=2)
        labels.append(label)

    # theta_JN is bimodal (J towards / away from us); the two orientations are mirror
    # images through the sky plane, which project identically on the sky: the grid
    # normal is the mean of J reflected into the hemisphere facing the observer
    J = np.array([out["J_hat_inf"] for out, _ in results])
    J_mirrored = J * np.where(J[:, 2:] < 0, [1, 1, -1], 1)
    J_grid = J_mirrored.mean(axis=0)
    J_grid /= np.linalg.norm(J_grid)
    t_attach = np.array([info["t_attach_det_rel"] for _, info in results])
    manifest = {
        "name": name,
        "labels": labels,
        "indices": [int(i) for i in indices],
        "J_hat_grid": J_grid.tolist(),
        "max_angle_J_inf_mirrored_vs_grid_deg":
            float(np.degrees(np.arccos(np.clip(J_mirrored @ J_grid, -1, 1))).max()),
        "n_J_inf_away_from_observer": int(np.count_nonzero(J[:, 2] < 0)),
        "n_rotated_by_pi": int(sum(info["psi_shown"] != info["sample"]["psi"] for _, info in results)),
        "t_attach_det_rel_range": [float(t_attach.min()), float(t_attach.max())],
        "ringdown_fit_residual_max": float(max(info["ringdown_fit_residual"] for _, info in results)),
    }
    manifest.update(extra or {"event": "GW250114", "approximant": "SEOBNRv5PHM",
                              "ppd_file": f"ppd_SEOBNRv5PHM_{IFO}.npz"})
    with open(outdir / f"{name}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest | {"labels": f"{len(labels)} samples", "indices": "..."}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--index", type=int, default=None,
                       help="a single posterior sample (default: max likelihood)")
    group.add_argument("--n-samples", type=int, default=None,
                       help="the first N of the random posterior samples used for the PPD")
    args = parser.parse_args()
    if args.n_samples:
        ppd = np.load(OUTDIR / f"ppd_SEOBNRv5PHM_{IFO}.npz")
        indices = [int(i) for i in ppd["sample_indices"][: args.n_samples]]
        export_set(indices, f"cloud{args.n_samples}")
    else:
        index = args.index
        if index is None:
            index = int(np.argmax(ed.load_posterior()["log_likelihood"]))
        export_set([index], f"sample{index}", processes=1)
