"""Camera path for a set of exported samples (`export_scene.py`), read by `blender_scene.py`.

Shots, in detector time:
1. from Earth (on the line of sight, far away, i.e. nearly orthographic), wide,
   slowly zooming in as the orbits shrink;
2. a close-up that follows one cluster of bodies (the bodies of the majority theta_JN
   family that are near the same body of a reference sample), from above its orbital
   plane, co-rotating with it;
3. back to the line of sight, framing the merger and the ringdown.

A shot is a target point, a viewing direction (target -> camera), the width of the view
at the target and the distance of the camera. Shots are blended with smoothstep weights:
the target linearly, the direction by slerp, width and distance logarithmically.

The direction to Earth is +Z; `earth_marker` (0..1) says how far the camera is from it,
used to show a marker on the line of sight when the camera is not on it.

Usage: python camera.py --name cloud200
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ppd import OUTDIR

ASPECT = 16 / 9
FAR_KM = 40000.0  # camera distance on the line of sight (nearly orthographic)
MARGIN_KM = 250.0  # around the close-up cluster
ARROW_KM = 200.0  # spin arrows protrude by up to this
WIDE_FACTOR = 1.1  # initial width relative to the orbit

# shot schedule, in seconds of video before the (median) merger
CLOSE_UP_IN = (39.7, 33.7)
CLOSE_UP_OUT = (15.7, 8.7)


def smoothstep(x, x0, x1):
    u = np.clip((x - x0) / (x1 - x0), 0, 1)
    return u * u * (3 - 2 * u)


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def slerp(a, b, w):
    a, b = normalize(a), normalize(b)
    cos = np.clip(np.sum(a * b, axis=-1, keepdims=True), -1, 1)
    omega = np.arccos(cos)
    small = omega < 1e-6
    so = np.where(small, 1, np.sin(omega))
    w = w[:, None]
    out = np.where(small, (1 - w) * a + w * b,
                   (np.sin((1 - w) * omega) * a + np.sin(w * omega) * b) / so)
    return normalize(out)


def load(name, products=OUTDIR):
    manifest = json.load(open(products / f"{name}.json"))
    scenes = [np.load(products / "scenes" / f"{label}.npz") for label in manifest["labels"]]
    infos = [json.load(open(products / "scenes" / f"{label}.json")) for label in manifest["labels"]]
    return manifest, scenes, infos


def view_width_for_orbit(scenes):
    """Width of the on-axis view that fits the bodies (horizons and spin arrows) at each
    frame; orbit ellipses must fit the frame height too."""
    r = np.array([
        np.where(d["merged"], float(d["radius_f"]),
                 np.maximum(np.linalg.norm(d["x1"], axis=-1) + float(d["radius1"]),
                            np.linalg.norm(d["x2"], axis=-1) + float(d["radius2"])))
        for d in scenes]) + ARROW_KM  # (N, F)
    R = np.percentile(r, 95, axis=0)
    R = np.maximum.accumulate(R[::-1])[::-1]  # never zoom out again
    return 2 * 1.1 * R * ASPECT


def cluster_track(scenes, infos):
    """Centroid of one cluster of bodies and the mean orbital-plane normal of its family."""
    J_z = np.array([d["J_hat_inf"][2] for d in scenes])
    family = J_z > 0 if np.count_nonzero(J_z > 0) >= len(J_z) / 2 else J_z < 0
    members = np.flatnonzero(family)
    ref = members[np.argmax([infos[i]["sample"]["log_likelihood"] for i in members])]
    x_ref = scenes[ref]["x1"]
    points = []
    for i in members:
        d = scenes[i]
        closer1 = np.linalg.norm(d["x1"] - x_ref, axis=-1) < np.linalg.norm(d["x2"] - x_ref, axis=-1)
        points.append(np.where(closer1[:, None], d["x1"], d["x2"]))
    points = np.array(points)  # (members, F, 3)
    merged = np.array([scenes[i]["merged"] for i in members])
    centroid = np.median(points, axis=0)
    spread = np.percentile(np.linalg.norm(points - centroid, axis=-1), 80, axis=0)
    radius = max(float(scenes[i][f"radius{k}"]) for i in members for k in (1, 2))
    # after the first merger the clusters are meaningless; hold the last values
    f_first_merge = int(np.argmax(merged.any(axis=0))) if merged.any() else points.shape[1]
    centroid[f_first_merge:] = centroid[f_first_merge - 1]
    spread[f_first_merge:] = spread[f_first_merge - 1]
    centroid = gaussian_filter1d(centroid, 3, axis=0, mode="nearest")
    spread = gaussian_filter1d(spread, 10, mode="nearest")
    L = normalize(np.mean([scenes[i]["LN_hat"] for i in members], axis=0))
    L[f_first_merge:] = L[f_first_merge - 1]
    L = normalize(gaussian_filter1d(L, 3, axis=0, mode="nearest"))
    return centroid, L, spread, radius, len(members), int(ref)


def camera_path(name, products=OUTDIR):
    manifest, scenes, infos = load(name, products)
    t = scenes[0]["t_det_rel"]
    n = len(t)
    t_merge = np.median([info["t_attach_det_rel"] for info in infos])
    slowdown = float(scenes[0]["slowdown"])
    close_up_in = [t_merge - x / slowdown for x in CLOSE_UP_IN]
    close_up_out = [t_merge - x / slowdown for x in CLOSE_UP_OUT]

    # shot 1/3: on the line of sight
    W_fit = view_width_for_orbit(scenes)
    zoom = smoothstep(t, t[0], close_up_in[0])
    W_axis = W_fit * np.exp((1 - zoom) * np.log(WIDE_FACTOR))
    axis = dict(target=np.zeros((n, 3)), direction=np.tile([0, 0, 1.0], (n, 1)),
                up=np.tile([0, 1.0, 0], (n, 1)), width=W_axis, distance=np.full(n, FAR_KM))

    # shot 2: co-rotating close-up of a cluster, from outside, above its orbital plane
    # and a little behind it
    c, L, spread, radius, n_family, ref = cluster_track(scenes, infos)
    radial = normalize(c - np.sum(c * L, axis=-1, keepdims=True) * L)
    motion = np.cross(L, radial)
    direction = normalize(0.45 * radial + 0.85 * L - 0.3 * motion)
    rc = np.linalg.norm(c, axis=-1)
    # frame the cluster and the centre of mass; wider if the cluster is diffuse
    width = np.maximum(1.1 * rc, 2.5 * spread) + 2 * (radius + MARGIN_KM)
    close = dict(target=0.85 * c, direction=direction, up=L, width=width, distance=1.5 * width)

    w = smoothstep(t, *close_up_in) * (1 - smoothstep(t, *close_up_out))
    target = (1 - w[:, None]) * axis["target"] + w[:, None] * close["target"]
    direction = slerp(axis["direction"], close["direction"], w)
    up = slerp(axis["up"], close["up"], w)
    up = normalize(up - np.sum(up * direction, axis=-1, keepdims=True) * direction)
    width = np.exp((1 - w) * np.log(axis["width"]) + w * np.log(close["width"]))
    distance = np.exp((1 - w) * np.log(axis["distance"]) + w * np.log(close["distance"]))
    fov = 2 * np.arctan(width / (2 * distance))

    off_axis = np.degrees(np.arccos(np.clip(direction[:, 2], -1, 1)))
    out = dict(
        t_det_rel=t,
        location=target + distance[:, None] * direction,
        target=target,
        direction=direction,
        up=up,
        width=width,
        distance=distance,
        fov=fov,
        off_axis_deg=off_axis,
        earth_marker=smoothstep(off_axis, 4, 15),
        close_up_weight=w,
        cluster_centroid=c,
    )
    np.savez(products / f"{name}_camera.npz", **out)
    print(f"{name}: {n} frames, merger at {t_merge:.4f} s; close-up on a cluster of the "
          f"{n_family}-sample family (reference sample {manifest['labels'][ref]}); "
          f"max off-axis angle {off_axis.max():.0f} deg")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--products", type=Path, default=OUTDIR, help="event products directory")
    args = parser.parse_args()
    camera_path(args.name, args.products)
