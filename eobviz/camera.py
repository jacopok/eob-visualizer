"""Camera path for a set of exported samples (`export.py`), read by `blender_scene.py`.

Shots, in detector time:
1. from Earth (on the line of sight, far away, i.e. nearly orthographic), wide,
   slowly zooming in as the orbits shrink;
2. a close-up that follows one cluster of bodies (the bodies of the majority theta_JN
   family that are near the same body of a reference sample), from above its orbital
   plane, co-rotating with it;
   or (`--shot orbit`) a slow orbit: the camera descends from the line of sight to
   ORBIT_ELEVATION_DEG above the grid plane, circles once around the grid normal (in the
   sense of most orbits), framing the whole set, and climbs back;
3. back to the line of sight, framing the merger and the ringdown.

With a storyboard (`--story degeneracies`, see `story.py`) the video frames are no longer
the data frames: the evolution pauses early on while the camera turns to a side view
(blended over the shots above, in video time), the close-up follows a cluster of the
canonical set and the grid is perpendicular to that set's mean J(t -> -inf). The output
then also holds the map to the data frames, the opacities of the sets of samples and the
captions. `--fps` sets the frame rate of the video (default: that of the data).

A shot is a target point, a viewing direction (target -> camera), the width of the view
at the target and the distance of the camera. Shots are blended with smoothstep weights:
the target linearly, the direction by slerp, width and distance logarithmically.

The direction to Earth is +Z; `earth_marker` (0..1) says how far the camera is from it,
used to show a marker on the line of sight when the camera is not on it.

Usage: python -m eobviz camera EVENT --name cloud200 [--shot orbit] [--story degeneracies] [--fps 10]
"""

import json

import numpy as np
from scipy.ndimage import gaussian_filter1d

ASPECT = 1920 / 810  # 3D view above the 1920x270 panel
FAR_KM = 40000.0  # camera distance on the line of sight (nearly orthographic)
MARGIN_KM = 250.0  # around the close-up cluster
ARROW_KM = 300.0  # spin arrows protrude by up to this
WIDE_FACTOR = 1.1  # initial width relative to the orbit

# shot schedule, in seconds of video before the (median) merger
CLOSE_UP_IN = (39.7, 33.7)
CLOSE_UP_OUT = (15.7, 8.7)
ORBIT_ELEVATION_DEG = 30.0  # above the grid plane, on the observer's side
ORBIT_DISTANCE = 2.0  # camera distance in units of the view width (perspective)


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


def load_scene(path):
    """All the arrays of a scene file, read once (an open NpzFile re-reads an array from
    disk at every access)."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def load(name, products):
    manifest = json.load(open(products / f"{name}.json"))
    scenes = [load_scene(products / "scenes" / f"{label}.npz") for label in manifest["labels"]]
    infos = [json.load(open(products / "scenes" / f"{label}.json")) for label in manifest["labels"]]
    return manifest, scenes, infos


def view_width_for_orbit(scenes):
    """Width of the on-axis view that fits the bodies (horizons and spin arrows) at each
    frame; orbit ellipses must fit the frame height too."""
    r = np.array([
        np.where(d["merged"], float(d["radius_f"]) + np.linalg.norm(d["remnant_offset"], axis=-1),
                 np.maximum(np.linalg.norm(d["x1"], axis=-1) + float(d["radius1"]),
                            np.linalg.norm(d["x2"], axis=-1) + float(d["radius2"])))
        for d in scenes]) + ARROW_KM  # (N, F)
    R = np.percentile(r, 95, axis=0)
    R = np.maximum.accumulate(R[::-1])[::-1]  # never zoom out again
    return 2 * 1.1 * R * ASPECT


def cluster_track(scenes, infos, fps=24, family=None):
    """Centroid of one cluster of bodies and the mean orbital-plane normal of its family
    (default: the majority theta_JN family)."""
    J_z = np.array([d["J_hat_inf"][2] for d in scenes])
    if family is None:
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
    centroid = gaussian_filter1d(centroid, 3 * fps / 24, axis=0, mode="nearest")
    spread = gaussian_filter1d(spread, 10 * fps / 24, mode="nearest")
    L = normalize(np.mean([scenes[i]["LN_hat"] for i in members], axis=0))
    L[f_first_merge:] = L[f_first_merge - 1]
    L = normalize(gaussian_filter1d(L, 3 * fps / 24, axis=0, mode="nearest"))
    return centroid, L, spread, radius, len(members), int(ref)


def orbit_shot(t, t_start, t_end, J_grid, scenes, width):
    """One turn around the grid normal at ORBIT_ELEVATION_DEG, starting (and ending) at
    the azimuth of the line of sight, targeting the centre of mass."""
    n = len(t)
    a = normalize(np.asarray(J_grid, float))
    z = np.array([0, 0, 1.0])
    e1 = normalize(z - (z @ a) * a)
    e2 = np.cross(a, e1)
    sense = np.sign(np.sum([np.sign(d["LN_hat"][0] @ a) for d in scenes])) or 1.0
    psi = sense * 2 * np.pi * np.clip((t - t_start) / (t_end - t_start), 0, 1)
    el = np.radians(ORBIT_ELEVATION_DEG)
    direction = (np.cos(el) * (np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2)
                 + np.sin(el) * a)
    return dict(target=np.zeros((n, 3)), direction=direction, up=np.tile(a, (n, 1)),
                width=width, distance=ORBIT_DISTANCE * width)


def blend(a, b, w):
    """Shot a, moving to shot b with weight w (per frame)."""
    direction = slerp(a["direction"], b["direction"], w)
    up = slerp(a["up"], b["up"], w)
    return dict(
        target=(1 - w[:, None]) * a["target"] + w[:, None] * b["target"],
        direction=direction,
        up=normalize(up - np.sum(up * direction, axis=-1, keepdims=True) * direction),
        width=np.exp((1 - w) * np.log(a["width"]) + w * np.log(b["width"])),
        distance=np.exp((1 - w) * np.log(a["distance"]) + w * np.log(b["distance"])),
    )


def camera_path(name, products, shot="close-up", story=None, fps=None):
    manifest, scenes, infos = load(name, products)
    t = scenes[0]["t_det_rel"]
    n = len(t)
    data_fps = float(scenes[0]["fps"])
    family = None
    if story is not None:
        from . import story as storyboard

        board = storyboard.degeneracies(scenes, infos, data_fps, fps or data_fps)
        family = np.isin(board["sets"], (storyboard.CANONICAL, storyboard.RELABELED))
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
    c, L, spread, radius, n_family, ref = cluster_track(scenes, infos, data_fps, family)
    radial = normalize(c - np.sum(c * L, axis=-1, keepdims=True) * L)
    motion = np.cross(L, radial)
    direction = normalize(0.45 * radial + 0.85 * L - 0.3 * motion)
    rc = np.linalg.norm(c, axis=-1)
    # frame the cluster and the centre of mass; wider if the cluster is diffuse
    width = np.maximum(1.1 * rc, 2.5 * spread) + 2 * (radius + MARGIN_KM)
    close = dict(target=0.85 * c, direction=direction, up=L, width=width, distance=1.5 * width)
    if shot == "orbit":
        close = orbit_shot(t, close_up_in[0], close_up_out[1], manifest["J_hat_grid"], scenes, W_fit)

    w = smoothstep(t, *close_up_in) * (1 - smoothstep(t, *close_up_out))
    path = blend(axis, close, w)
    extra = {}
    data_frame = np.arange(n)
    if story is not None:
        # in video frames from here on
        data_frame = board["data_frame"]
        path = {k: v[data_frame] for k, v in path.items()}
        w, c = w[data_frame], c[data_frame]
        d, up, n_distinct = storyboard.side_view(scenes, board["sets"], board["pause_frame"])
        m = len(data_frame)
        width = np.full(m, W_fit[board["pause_frame"]])
        side = dict(target=np.zeros((m, 3)), direction=np.tile(d, (m, 1)), up=np.tile(up, (m, 1)),
                    width=width, distance=ORBIT_DISTANCE * width)
        path = blend(path, side, board["side_weight"])
        right = np.cross(up, d)  # of the side view, in which the sets move apart
        J = np.array([s["J_hat_inf"] for s in scenes])[family].mean(axis=0)
        extra = dict(
            data_frame=data_frame,
            side_weight=board["side_weight"],
            set_opacity=board["set_opacity"],
            set_offset=board["set_shift"][:, :, None] * width[0] * right,
            swap=board["swap"],
            sets=board["sets"],
            set_labels=np.array(manifest["labels"]),
            grid_normal=J / np.linalg.norm(J),
            caption_text=np.array([x["text"] for x in board["captions"]]),
            caption_row=np.array([x["row"] for x in board["captions"]]),
            caption_in=np.array([x["t_in"] for x in board["captions"]]),
            caption_out=np.array([x["t_out"] for x in board["captions"]]),
            caption_x=np.array([x["x"] for x in board["captions"]]),
        )
        print(f"storyboard: {m} video frames at {fps or data_fps:g} fps, paused at data frame "
              f"{board['pause_frame']}; side view {np.round(d, 2)} ({n_distinct} distinct clusters); "
              f"sets {np.bincount(board['sets'], minlength=storyboard.N_SETS).tolist()} "
              f"(canonical, theta, psi, psi+theta, relabeled)")
    target, direction, up = path["target"], path["direction"], path["up"]
    width, distance = path["width"], path["distance"]
    fov = 2 * np.arctan(width / (2 * distance))

    off_axis = np.degrees(np.arccos(np.clip(direction[:, 2], -1, 1)))
    out = dict(
        t_det_rel=t[data_frame],
        fps=(fps or data_fps) if story else data_fps,
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
        **extra,
    )
    np.savez(products / f"{name}_camera.npz", **out)
    what = ("orbit around the centre of mass" if shot == "orbit" else
            f"close-up on a cluster of the {n_family}-sample family (reference sample {manifest['labels'][ref]})")
    print(f"{name}: {len(data_frame)} frames, merger at {t_merge:.4f} s; {what}; "
          f"max off-axis angle {off_axis.max():.0f} deg")
    return out

