"""Storyboard "degeneracies" of a cloud of samples, read by `camera.py`: the video stops
early in the inspiral, turns to show the copies of the orbits allowed by the two
degeneracies of the sky orientation, restricts the cloud to a canonical domain and
relabels its BHs by orbit clustering, then goes on to the merger.

1. The cloud evolves, seen from Earth: the theta_JN copies are mirror images through
   the sky plane, the psi copies rotated by pi about the line of sight, and they all
   project onto nearly the same ellipses.
2. The evolution slows to a stop while the camera turns to a side view in which the
   copies are apart (`side_view`).
3. psi: F_+ + i F_x is proportional to e^{-2 i psi}, so psi -> psi + pi leaves the
   signal unchanged. The samples with 0 < psi < pi move to the left of the view, those
   with pi < psi < 2 pi to the right; the latter fade out and the former come back.
4. theta_JN: theta_JN -> pi - theta_JN flips the sign of h_x, which the two nearly
   co-aligned LIGO detectors barely constrain. Same, keeping 0 < theta_JN < pi/2.
5. Labels: m_1 >= m_2 sample by sample labels some orbits against the majority
   (q ~ 1, `labels.synchronize_signs` on the canonical samples). Those move to the
   right, swap colours, and come back.
6. The camera goes back to the line of sight and the evolution resumes with the
   canonical set only; the other copies fade back in after the merger.

Samples are split in sets: CANONICAL, THETA (pi/2 < theta_JN), PSI (pi <= psi),
PSI_THETA (both), RELABELED (canonical, BHs swapped). Times are in seconds of video.
"""

import numpy as np

CANONICAL, THETA, PSI, PSI_THETA, RELABELED = range(5)
N_SETS = 5

RAMP = 0.7  # [s] of the transitions of the opacities
MOVE = 1.2  # [s] for the sets to move apart or back
SPLIT = 0.22  # offset of the two sides of a fold, in units of the view width (left -, right +)
GHOSTS_BACK = 0.5  # [s] after the (median) merger at which the folded copies fade back in
SLOW = 2.0  # [s] for the evolution to stop or to resume
TURN = 3.5  # [s] of the camera turn to and from the side view
PAUSE_AT = 4.0  # [s] of video at which the evolution starts slowing down
SECTION = 11.0  # [s] for each degeneracy
LABELS = 9.0  # [s] for the relabeling
SWAP = 1.0  # [s] of the colour swap of the relabeled BHs
ANALYSIS_WINDOW = (0.10, 0.25)  # detector time - reference time [s] of the orbit clustering

PSI_LOW, PSI_HIGH = (CANONICAL, RELABELED, THETA), (PSI, PSI_THETA)
THETA_LOW, THETA_HIGH = (CANONICAL, RELABELED), (THETA,)


def smoothstep(x, x0, x1):
    u = np.clip((x - x0) / (x1 - x0), 0, 1)
    return u * u * (3 - 2 * u)


def sample_sets(scenes, infos):
    """The set of each sample, and the agreement scores of the orbit clustering."""
    psi_high = np.array([info["psi_shown"] >= np.pi for info in infos])
    theta_high = np.array([info["sample"]["theta_jn"] > np.pi / 2 for info in infos])
    sets = np.where(psi_high, np.where(theta_high, PSI_THETA, PSI), np.where(theta_high, THETA, CANONICAL))
    canonical = np.flatnonzero(sets == CANONICAL)
    t = scenes[0]["t_det_rel"]
    weight = ((t >= ANALYSIS_WINDOW[0]) & (t <= ANALYSIS_WINDOW[1])).astype(float)
    u = []
    for i in canonical:
        d = scenes[i]
        n = d["x1"] - d["x2"]
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        u.append(np.where(d["merged"][:, None], np.nan, n))
    from .labels import synchronize_signs

    s, score = synchronize_signs(np.array(u), weight)
    sets[canonical[s < 0]] = RELABELED
    return sets, score


def body_cluster(points):
    """For the bodies `points` (n, 3) of an orbit cloud around the origin, which form two
    opposite clusters: the centre of one of them."""
    axis = np.linalg.eigh(points.T @ points)[1][:, -1]
    return axis * np.mean(np.abs(points @ axis))


def side_view(scenes, sets, frame):
    """A viewing direction (and its up vector) in which the clusters of bodies of the four
    sky-orientation copies are apart at `frame`, with the line of sight up."""
    groups = [(CANONICAL, RELABELED), (THETA,), (PSI,), (PSI_THETA,)]
    centres = []
    for group in groups:
        members = np.flatnonzero(np.isin(sets, group))
        c = body_cluster(np.concatenate([[scenes[i]["x1"][frame], scenes[i]["x2"][frame]] for i in members]))
        centres += [c, -c]
    centres = np.array(centres)
    size = np.median(np.linalg.norm(centres, axis=-1))
    # clusters that coincide in space cannot be separated by any view
    distinct = []
    for c in centres:
        if all(np.linalg.norm(c - x) > 0.3 * size for x in distinct):
            distinct.append(c)
    distinct = np.array(distinct)
    best, best_score = None, -1.0
    z = np.array([0, 0, 1.0])
    for theta in np.radians(np.arange(20, 91, 2.5)):  # from the line of sight
        for phi in np.radians(np.arange(0, 360, 2.5)):
            d = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
            p = distinct - np.outer(distinct @ d, d)
            sep = min(np.linalg.norm(a - b) for k, a in enumerate(p) for b in p[k + 1:])
            score = sep * (1 + 0.1 * np.cos(theta))  # a little towards Earth when it costs nothing
            if score > best_score:
                best, best_score = d, score
    up = z - (z @ best) * best
    return best, up / np.linalg.norm(up), len(distinct)


class Track:
    """A per-frame value that moves smoothly (over RAMP) between the values it is set to."""

    def __init__(self, v, value):
        self.v = v
        self.value = np.full(len(v), float(value))

    def set(self, at, value, ramp=RAMP):
        w = smoothstep(self.v, at, at + ramp)
        self.value = (1 - w) * self.value + w * value


def degeneracies(scenes, infos, data_fps, fps):
    """The storyboard: per video frame, the data frame shown, the opacity and the
    sideways offset (in units of the view width) of each set, the colour swap of
    RELABELED and the weight of the side view; the captions; the sets of the samples."""
    sets, score = sample_sets(scenes, infos)
    t_data = scenes[0]["t_det_rel"]
    n_data = len(t_data)

    resume = PAUSE_AT + SLOW / 2 + TURN + 0.5 + 2 * SECTION + LABELS
    data_duration = (n_data - 1) / data_fps
    # data time u(v): the rate du/dv goes smoothly to 0 and back to 1
    fine = np.arange(0, data_duration + resume + 2 * SLOW, 1 / (20 * fps))
    rate = 1 - smoothstep(fine, PAUSE_AT, PAUSE_AT + SLOW) + smoothstep(fine, resume, resume + SLOW)
    u = np.concatenate([[0], np.cumsum(rate[1:] * np.diff(fine))])
    v = np.arange(int(np.interp(data_duration, u, fine) * fps) + 1) / fps
    data_frame = np.clip(np.round(np.interp(v, fine, u) * data_fps).astype(int), 0, n_data - 1)
    pause_frame = int(data_frame[np.searchsorted(v, PAUSE_AT + SLOW)])
    t_merge = np.median([info["t_attach_det_rel"] for info in infos])
    v_merge = v[np.searchsorted(t_data[data_frame], t_merge)]

    side = Track(v, 0.0)
    side.set(PAUSE_AT, 1.0, TURN)
    side.set(resume - 1.0, 0.0, TURN)
    swap = Track(v, 0.0)
    opacity = [Track(v, 1.0) for _ in range(N_SETS)]
    shift = [Track(v, 0.0) for _ in range(N_SETS)]
    captions = []

    def show(sets_, value, at, ramp=RAMP):
        for s in sets_:
            opacity[s].set(at, value, ramp)

    def move(sets_, value, at):
        for s in sets_:
            shift[s].set(at, value, MOVE)

    def caption(text, row, t_in, t_out, x=None):
        """Row 0-2: top left; row 3: under the sets, centred on x (fraction of the width);
        row 4: as row 2, in green."""
        captions.append(dict(text=text, row=row, t_in=float(t_in), t_out=float(t_out),
                             x=-1.0 if x is None else float(x)))

    def fold(t, keep, drop, keep_label, drop_label, keep_caption):
        """Move the two sides apart, label them, fade out `drop`, bring `keep` back."""
        move(keep, -SPLIT, t + 2)
        move(drop, SPLIT, t + 2)
        caption(keep_label, 3, t + 2.6, t + 7.0, 0.5 - SPLIT)
        caption(drop_label, 3, t + 2.6, t + 7.0, 0.5 + SPLIT)
        show(drop, 0.0, t + 6.8)
        move(keep + drop, 0.0, t + 7.8)  # (drop, unseen: in place for after the merger)
        caption(keep_caption, 2, t + 8.0, t + SECTION - 0.5)

    t0 = PAUSE_AT + SLOW / 2 + TURN + 0.5
    caption(r"$h = F_+(\psi)\,h_+ + F_\times(\psi)\,h_\times, \qquad F_+ + i\,F_\times \propto e^{-2 i \psi}$",
            0, t0, t0 + SECTION - 0.5)
    caption(r"$\psi \to \psi + \pi$: exactly the same signal", 1, t0 + 0.8, t0 + SECTION - 0.5)
    fold(t0, PSI_LOW, PSI_HIGH, r"$0 < \psi < \pi$", r"$\pi < \psi < 2\pi$", r"keep $0 < \psi < \pi$")

    t1 = t0 + SECTION
    caption(r"$h_+ \propto \dfrac{1 + \cos^2\theta_{JN}}{2}, \qquad h_\times \propto \cos\theta_{JN}$",
            0, t1, t1 + SECTION - 0.5)
    caption(r"$\theta_{JN} \to \pi - \theta_{JN}$: $h_\times \to -h_\times$, "
            r"barely constrained by the nearly co-aligned LIGO detectors", 1, t1 + 0.8, t1 + SECTION - 0.5)
    fold(t1, THETA_LOW, THETA_HIGH, r"$0 < \theta_{JN} < \pi/2$", r"$\pi/2 < \theta_{JN} < \pi$",
         r"keep $0 < \theta_{JN} < \pi/2$")

    t2 = t1 + SECTION
    caption(r"BH 1 is the heavier one, sample by sample: $m_1 \geq m_2$", 0, t2, t2 + LABELS - 0.5)
    caption(r"$m_1 \simeq m_2$: similar orbits can get opposite labels", 1, t2 + 0.8, t2 + LABELS - 0.5)
    move((CANONICAL,), -SPLIT, t2 + 2)
    move((RELABELED,), SPLIT, t2 + 2)
    caption("labelled like the majority", 3, t2 + 2.6, t2 + 6.0, 0.5 - SPLIT)
    caption("labelled against the majority", 3, t2 + 2.6, t2 + 6.0, 0.5 + SPLIT)
    swap.set(t2 + 4.0, 1.0, SWAP)
    move((CANONICAL, RELABELED), 0.0, t2 + 5.8)
    caption("BH 1: the body whose orbit matches BH 1 in the other samples", 2, t2 + 6.2, t2 + LABELS - 0.5)
    assert abs(t2 + LABELS - resume) < 1e-9

    # the folded copies, after the merger
    show((THETA, PSI, PSI_THETA), 1.0, v_merge + GHOSTS_BACK, 1.5)
    caption("all the copies", 4, v_merge + GHOSTS_BACK, v_merge + GHOSTS_BACK + 4)

    return dict(
        data_frame=data_frame,
        pause_frame=pause_frame,
        side_weight=side.value,
        set_opacity=np.stack([o.value for o in opacity], axis=1),
        set_shift=np.stack([x.value for x in shift], axis=1),
        swap=swap.value,
        captions=captions,
        sets=sets,
        n_relabeled=int(np.count_nonzero(sets == RELABELED)),
        clustering_score=score,
    )
