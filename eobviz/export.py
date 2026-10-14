"""Export per-frame scene data for a set of posterior samples of an event, to be read by
`camera.py`, `blender_scene.py` and `panel.py`. Model-independent: everything
model-specific comes through `models.base.EOBRun`.

Timeline: frames are uniformly spaced in detector time at the event's detector (the one
whose whitened strain is shown in the panel). The state of the binary shown at
detector time t_det is the one at waveform time t_det - geocent_time - delay_IFO,
i.e. the signal arriving at the detector at t_det was emitted by that state.
All samples share this timeline, so each one merges at its own detector time.

Space: sky frame (X west, Y north, Z towards Earth), lengths in source-frame km
(dynamics in units of the detector-frame mass M, times G M / c^2 / (1 + z)).

Bodies are drawn with their Kerr horizon areal radius sqrt(A / 4 pi) = sqrt(2 m r_+),
placed at the EOB (point-particle, centre-of-mass frame) positions. Before the
attachment time t_attach the two inspiralling BHs are shown; after it, the remnant
with the model's final mass and spin, the spin along the model's remnant spin direction.

Ringdown: the remnant's surface is r = R_f [1 + Re(C(t) e^{-2 i phi}) sin^2 theta]
in a frame with z along the remnant spin and phi measured from the separation
vector at attachment, with
    C(t) = REMNANT_DEFORMATION * conj(h(t) / h(t_attach)),
h the 220 + 221 Kerr-QNM fit to the sample's own (2,2) mode in the remnant's frame.
The l = m = 2 bar thus starts along the line joining the two BHs, rotates
with the pattern speed and decays with the damping times of the two QNMs, with
their fitted relative amplitude and phase. Its overall size is illustrative.

Recoil: NRSur7dq4Remnant, from the spins and orbit at -100 M from the peak; the remnant
moves with it from the attachment on.

Grid: perpendicular to the mean J(t -> -inf) of the set, after reflecting the J's
pointing away from us through the sky plane (theta_JN is bimodal, and mirror images
through the sky plane look the same in an orthographic view along the line of sight).

Sky orientation: psi and psi + pi give identical detector responses, and bilby
samples psi in [0, pi). Each sample is shown with the representative that puts the
sky projection of its J(t -> -inf) on the same side as the (axial) mean of the set.
"""

import json
from multiprocessing import Pool

import bilby
import lal
import numpy as np

from . import physics
from .events import get_event
from .events.base import SAMPLE_INFO_KEYS

REMNANT_DEFORMATION = 0.3  # fractional l = m = 2 deformation at t_attach (illustrative)

# arrays holding sky-frame vectors along their last axis
VECTOR_KEYS = ("x1", "x2", "chi1", "chi2", "LN_hat", "trail1", "trail2",
               "chi_f", "J_hat_inf", "J_hat_ref", "kick_velocity", "remnant_offset")


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def export_sample(event, sample):
    """Scene arrays (`out`) and a summary (`info`) of one sample."""
    run = event.run_model(sample)
    M_src_km = run.M / (1 + sample["redshift"]) * physics.KM_PER_MSUN  # km per unit of M

    rot = run.rotation_I_to_sky(sample["psi"])
    N_I, _, _ = physics.observer_direction_I(run.inclination, run.phi_ref)
    assert np.allclose(rot @ N_I, [0, 0, 1])
    sky = lambda v: v @ rot.T

    ifo = bilby.gw.detector.get_empty_interferometer(event.ifo)
    delay = ifo.time_delay_from_geocenter(sample["ra"], sample["dec"], sample["geocent_time"])

    t_det_rel = event.frame_times()
    t_wave = t_det_rel + event.reference_time - sample["geocent_time"] - delay
    t_geom = run.t_geometric(t_wave)
    if t_geom[0] < run.t_start:
        raise ValueError("time window starts before the dynamics")
    ta = run.t_attach
    merged = t_geom >= ta

    def evaluate(t):
        d = run.inertial_dynamics(np.minimum(t, run.t_eval_end))
        d.update({k: sky(d[k]) for k in ("x1", "x2", "chi1", "chi2", "LN_hat", "n")})
        for k in ("x1", "x2"):
            d[k] = d[k] * M_src_km
        return d

    frames = evaluate(t_geom)
    at_attach = evaluate(np.atleast_1d(ta))

    # orbital phase [cycles], used to fade the trails behind the bodies; after the
    # attachment it keeps running at the orbital frequency of the attachment
    now_phase = (frames["phi"] + np.maximum(t_geom - ta, 0) * run.omega_attach) / (2 * np.pi)

    # trails, uniformly sampled in orbital phase up to the attachment (or the end of the window)
    t_end = min(ta, t_geom[-1])
    t_dense = np.linspace(t_geom[0], t_end, 20000)
    phi_dense = run.inertial_dynamics(t_dense)["phi"]
    n_trail = int((phi_dense[-1] - phi_dense[0]) / (2 * np.pi) * event.trail_points_per_orbit) + 2
    trail = evaluate(np.interp(np.linspace(phi_dense[0], phi_dense[-1], n_trail), phi_dense, t_dense))

    # remnant: frame with z along its spin and x along the separation at attachment
    chi_f = abs(run.final_spin)
    e3 = rot @ run.Jfhat_attach
    n_a = at_attach["n"][0]
    e1 = n_a - (n_a @ e3) * e3
    e1 /= np.linalg.norm(e1)
    remnant_basis = np.column_stack([e1, np.cross(e3, e1), e3])
    fit = run.fit_ringdown()
    h_attach = physics.ringdown_h22(fit, np.atleast_1d(ta), ta)[0]
    h = physics.ringdown_h22(fit, np.maximum(t_geom, ta), ta)
    deformation = np.where(merged, REMNANT_DEFORMATION * np.conj(h / h_attach), 0)
    k = evaluate(np.atleast_1d(run.t_peak + physics.KICK_EPOCH))
    kick, kick_err = physics.nrsur_kick(run.m_1, run.m_2, k["chi1"][0], k["chi2"][0], k["n"][0], k["LN_hat"][0])
    remnant_offset = np.maximum(t_geom - ta, 0)[:, None] * kick * M_src_km

    J_inf_I, _, _ = run.J_hat_at_past_infinity()
    J_ref_I = run.J_reference()
    J_ref_I = J_ref_I / np.linalg.norm(J_ref_I)

    out = dict(
        fps=event.fps,
        slowdown=event.slowdown,
        ifo=event.ifo,
        reference_time=event.reference_time,
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
        radius1=physics.areal_radius(run.m_1, np.linalg.norm(run.chi1_ref)) * M_src_km,
        radius2=physics.areal_radius(run.m_2, np.linalg.norm(run.chi2_ref)) * M_src_km,
        radius_f=physics.areal_radius(run.final_mass, chi_f) * M_src_km,
        chi_f=chi_f * e3,
        remnant_basis=remnant_basis,
        deformation=deformation,
        kick_velocity=kick,
        remnant_offset=remnant_offset,
        J_hat_inf=rot @ J_inf_I,
        J_hat_ref=rot @ J_ref_I,
        km_per_M=M_src_km,
    )
    c_km_s = lal.C_SI / 1e3
    info = {
        "sample": {k: sample[k] for k in (*SAMPLE_INFO_KEYS, *event.extra_sample_keys)},
        "delay_" + event.ifo: delay,
        "t_attach_det_rel": float(np.interp(ta, t_geom, t_det_rel)),
        "angle_J_inf_vs_J_ref_deg": angle_deg(J_inf_I, J_ref_I),
        "angle_J_inf_vs_line_of_sight_deg": float(np.degrees(np.arccos(out["J_hat_inf"][2]))),
        "final_mass_source": run.final_mass * run.M / (1 + sample["redshift"]),
        "final_spin": chi_f,
        "kick_km_s": float(np.linalg.norm(kick) * c_km_s),
        "kick_err_km_s": kick_err * c_km_s,
        "km_per_M_source": M_src_km,
        "ringdown_fit_residual": float(fit["residual"]),
        "A221_over_A220": [float(abs(fit["amplitudes"][1] / fit["amplitudes"][0])),
                           float(np.angle(fit["amplitudes"][1] / fit["amplitudes"][0]))],
        "M_omega_220": [float(fit["omegas"][0].real), float(fit["omegas"][0].imag)],
        "M_omega_221": [float(fit["omegas"][1].real), float(fit["omegas"][1].imag)],
        **run.info(),
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


_worker = {}


def _export_one(args):
    name, sample = args
    if _worker.get("name") != name:
        _worker.update(name=name, event=get_event(name))
    return export_sample(_worker["event"], sample)


def export_set(event, indices, name, processes=8):
    """Export the samples `indices` to `products/scenes/` and write the manifest
    `products/<name>.json`."""
    posterior = event.load_posterior()
    samples = [event.sample(posterior, i) for i in indices]
    del posterior
    with Pool(min(processes, len(samples))) as pool:
        results = pool.map(_export_one, [(event.name, s) for s in samples], chunksize=1)

    J_xy = np.array([out["J_hat_inf"][:2] for out, _ in results])
    reference = axial_mean_angle(J_xy)
    axis = np.array([np.cos(reference), np.sin(reference)])
    # the axial mean is defined mod pi: orient it so that most samples keep bilby's psi
    if np.count_nonzero(J_xy @ axis < 0) > len(J_xy) / 2:
        axis = -axis

    scenes = event.products / "scenes"
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
        "event": event.name,
        "approximant": event.approximant,
        "ppd_file": event.ppd_file().name,
        "panel_note": event.panel_note,
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
    with open(event.products / f"{name}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest | {"labels": f"{len(labels)} samples", "indices": "..."}, indent=2))
    return manifest


def ppd_indices(event, n_samples):
    """The first n of the random posterior samples of the PPD (so that the panel has
    their whitened waveforms)."""
    with np.load(event.ppd_file()) as ppd:
        return [int(i) for i in ppd["sample_indices"][:n_samples]]
