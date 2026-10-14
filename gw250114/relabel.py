"""Relabel the component BHs of the posterior samples by clustering their orbits,
in the spirit of arXiv:2409.07519 but with a trajectory-based distance.

Swapping the labels of the two BHs describes the same binary, so the likelihood
is invariant; the m_1 >= m_2 convention picks one representative. Here instead
"BH 1" is the body whose orbit agrees with that of BH 1 in the other samples.

For each sample, u(t) is the direction of BH 1 from the centre of mass (n = x_1 - x_2
normalized) in the sky frame, on the common detector-time axis of the renders
(`eobviz/export.py`). Relabeling flips u -> -u. With
    C_ij = sum_t w(t) u_i(t) . u_j(t),
the signs s_i maximize sum_ij s_i s_j C_ij = sum_t w |sum_i s_i u_i|^2
(Z2 synchronization): spectral initialization (leading eigenvector of C),
then Lloyd iterations s_i = sign(sum_t w u_i . mu), mu = sum_j s_j u_j.
w(t) is uniform within ANALYSIS_WINDOW and zero after each sample's attachment.

psi -> psi + pi is an exact symmetry of the detector response which maps
(u_x, u_y) -> -(u_x, u_y), close to a relabeling for face-on orbits: it is fixed
first with the same rule as the renders (sky projection of J(t -> -inf) on the
side of the set's axial mean).

The per-sample u(t) are cached in products/relabel_cache.
"""

import argparse
import json
from multiprocessing import Pool

import sys
from pathlib import Path

import bilby
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eobviz.events import get_event  # noqa: E402
from eobviz.export import axial_mean_angle  # noqa: E402
from eobviz.models.seobnrv5 import SEOBNRv5PHM  # noqa: E402

EVENT = get_event("GW250114")
IFO, TIME_WINDOW, REFERENCE_TIME, OUTDIR = EVENT.ifo, EVENT.time_window, EVENT.reference_time, EVENT.products
CACHE = OUTDIR / "relabel_cache"
SAMPLE_RATE = 4096  # [Hz] of the cached u(t)
ANALYSIS_WINDOW = (0.10, 0.25)  # detector time - REFERENCE_TIME [s]; merger at ~0.217

# parameters exchanged by a relabeling (the azimuthal angles are not handled)
SWAP_PAIRS = [("mass_1", "mass_2"), ("mass_1_source", "mass_2_source"), ("a_1", "a_2"),
              ("tilt_1", "tilt_2"), ("cos_tilt_1", "cos_tilt_2"),
              ("chi_1_in_plane", "chi_2_in_plane")]


def time_grid():
    return np.arange(TIME_WINDOW[0], TIME_WINDOW[1], 1 / SAMPLE_RATE)


def compute_u(sample):
    """u(t) on `time_grid()` in the sky frame (NaN after the attachment),
    and J(t -> -inf) in the sky frame."""
    run = SEOBNRv5PHM(sample, with_modes=False)
    rot = run.rotation_I_to_sky(sample["psi"])

    ifo = bilby.gw.detector.get_empty_interferometer(IFO)
    delay = ifo.time_delay_from_geocenter(sample["ra"], sample["dec"], sample["geocent_time"])
    t_wave = time_grid() + REFERENCE_TIME - sample["geocent_time"] - delay
    t_geom = run.t_geometric(t_wave)
    if t_geom[0] < run.t_start:
        raise ValueError("time window starts before the dynamics")

    ta = run.t_attach
    u = run.inertial_dynamics(np.minimum(t_geom, ta))["n"] @ rot.T
    u[t_geom > ta] = np.nan
    J_inf, _, _ = run.J_hat_at_past_infinity()
    return u, rot @ J_inf


def _cache_one(index):
    path = CACHE / f"sample{index}.npz"
    if path.exists():
        return
    sample = EVENT.sample(EVENT.load_posterior(), index)
    try:
        u, J_hat_inf = compute_u(sample)
    except Exception as e:  # keep going, report at the end
        print(f"sample {index}: {e!r}", flush=True)
        return
    np.savez(path, u=u.astype(np.float32), J_hat_inf=J_hat_inf)


def cache(indices, processes):
    CACHE.mkdir(parents=True, exist_ok=True)
    with Pool(processes) as pool:
        for k, _ in enumerate(pool.imap_unordered(_cache_one, indices, chunksize=4)):
            if k % 100 == 0:
                print(f"{k}/{len(indices)}", flush=True)


def load_cached():
    indices, us, Js = [], [], []
    for path in sorted(CACHE.glob("sample*.npz")):
        with np.load(path) as f:
            us.append(f["u"])
            Js.append(f["J_hat_inf"])
        indices.append(int(path.stem.removeprefix("sample")))
    return np.array(indices), np.array(us, dtype=float), np.array(Js)


def fix_psi(u, J):
    """psi -> psi + pi where needed, with the convention of `eobviz.export.export_set`."""
    axis_angle = axial_mean_angle(J[:, :2])
    axis = np.array([np.cos(axis_angle), np.sin(axis_angle)])
    if np.count_nonzero(J[:, :2] @ axis < 0) > len(J) / 2:
        axis = -axis
    flip = J[:, :2] @ axis < 0
    u = u.copy()
    u[flip, :, :2] *= -1
    return u, flip


def synchronize_signs(u, weight, max_iter=100):
    """Signs s maximizing sum_ij s_i s_j C_ij, and the per-sample agreement
    score_i = sum_t w s_i u_i . mu / sqrt(sum_t w |u_i|^2 sum_t w |mu|^2) in [-1, 1]."""
    from scipy.sparse.linalg import svds

    v = np.nan_to_num(u) * np.sqrt(weight).astype(u.dtype)[None, :, None]
    V = v.reshape(len(v), -1).astype(np.float32)
    # leading eigenvector of C = V V^T, without forming C
    left, _, _ = svds(V, k=1)
    s = np.where(left[:, 0] >= 0, 1, -1)
    for _ in range(max_iter):
        mu = s @ V
        new = np.where(V @ mu >= 0, 1, -1)
        if np.array_equal(new, s):
            break
        s = new
    mu = s @ V
    score = s * (V @ mu) / np.sqrt(np.sum(V**2, axis=1) * (mu @ mu))
    return s, score


def relabeled(posterior, indices, s):
    """Posterior samples at `indices`, with the BHs swapped where s = -1."""
    out = posterior[indices].copy()
    swap = s < 0
    names = posterior.dtype.names
    for a, b in SWAP_PAIRS:
        if a not in names or b not in names:
            continue
        out[a][swap], out[b][swap] = posterior[b][indices][swap], posterior[a][indices][swap]
    out["mass_ratio"][swap] = 1 / posterior["mass_ratio"][indices][swap]
    return out


def corner_plot(original, new, path):
    import corner
    import matplotlib.pyplot as plt

    keys = ["mass_1_source", "mass_2_source", "a_1", "a_2", "cos_tilt_1", "cos_tilt_2"]
    labels = [r"$m_1^{\rm src}$", r"$m_2^{\rm src}$", r"$a_1$", r"$a_2$",
              r"$\cos t_1$", r"$\cos t_2$"]
    stack = lambda p: np.column_stack([p[k] for k in keys])
    ranges = [(min(original[k].min(), new[k].min()), max(original[k].max(), new[k].max()))
              for k in keys]
    kwargs = dict(labels=labels, range=ranges, bins=30, smooth=1.0, plot_datapoints=False,
                  plot_density=False, levels=(0.5, 0.9))
    fig = corner.corner(stack(original), color="C0", hist_kwargs=dict(density=True), **kwargs)
    corner.corner(stack(new), fig=fig, color="C3", hist_kwargs=dict(density=True), **kwargs)
    fig.legend(handles=[plt.Line2D([], [], color="C0", label=r"$m_1 \geq m_2$"),
                        plt.Line2D([], [], color="C3", label="orbit clustering")],
               loc="upper right", fontsize=14)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def phase_plot(u, s, t, times, path, unit="s"):
    """Azimuth on the sky of BH 1 at `times` (on the grid `t` of u), before and after relabeling."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(times), figsize=(3 * len(times), 6),
                             subplot_kw=dict(projection="polar"))
    for col, tk in enumerate(times):
        k = np.argmin(np.abs(t - tk))
        for row, (sign, title) in enumerate([(np.ones_like(s), r"$m_1 \geq m_2$"),
                                             (s, "orbit clustering")]):
            v = sign[:, None] * u[:, k, :2]
            ok = np.all(np.isfinite(v), axis=1)
            axes[row, col].hist(np.arctan2(v[ok, 1], v[ok, 0]), bins=72, color=f"C{3 * row}")
            axes[row, col].set_yticks([])
            axes[row, col].set_title(f"{title}\n$t = {tk:.3f}$ {unit}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def analyze():
    posterior = EVENT.load_posterior()
    indices, u, J = load_cached()
    u, psi_flip = fix_psi(u, J)
    t = time_grid()
    weight = ((t >= ANALYSIS_WINDOW[0]) & (t <= ANALYSIS_WINDOW[1])).astype(float)

    s, score = synchronize_signs(u, weight)
    # the overall sign is free: keep most samples' original labels
    if np.count_nonzero(s < 0) > len(s) / 2:
        s, score = -s, score
    swapped = s < 0
    print(f"{len(s)} samples, {np.count_nonzero(swapped)} relabeled "
          f"({np.mean(swapped):.1%}); psi flipped for {np.count_nonzero(psi_flip)}")
    print("agreement score quantiles (5, 50, 95%):", np.percentile(score, [5, 50, 95]).round(3))

    q = posterior["mass_ratio"][indices]
    for lo, hi in [(0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        sel = (q > lo) & (q <= hi)
        if sel.any():
            print(f"  q in ({lo}, {hi}]: {np.count_nonzero(sel):5d} samples, "
                  f"{np.mean(swapped[sel]):.1%} relabeled")

    new = relabeled(posterior, indices, s)
    np.savez(OUTDIR / "relabel.npz", indices=indices, sign=s, score=score, psi_flip=psi_flip)
    corner_plot(posterior[indices], new, OUTDIR / "relabel_corner.png")
    phase_plot(u, s, t, np.linspace(ANALYSIS_WINDOW[0], 0.21, 5), OUTDIR / "relabel_phase.png")

    cloud = json.loads((OUTDIR / "cloud200.json").read_text())["labels"]
    in_cloud = np.isin(indices, [int(label.removeprefix("sample")) for label in cloud])
    print(f"render cloud200: {np.count_nonzero(in_cloud)} cached, "
          f"{np.count_nonzero(swapped & in_cloud)} relabeled:",
          sorted(indices[swapped & in_cloud].tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=2000, help="random samples to cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if not args.analyze_only:
        n_post = len(EVENT.load_posterior())
        rng = np.random.default_rng(args.seed)
        cloud = json.loads((OUTDIR / "cloud200.json").read_text())["labels"]
        indices = {int(label.removeprefix("sample")) for label in cloud}
        indices |= set(rng.choice(n_post, size=args.n, replace=False).tolist())
        cache(sorted(indices), args.processes)
    analyze()


if __name__ == "__main__":
    main()
