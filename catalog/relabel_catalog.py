"""Orbit-based relabeling of the component BHs (`gw250114/relabel.py`) for every event of
the catalog with a supported EOB run (see `catalog.py inventory`).

Differences from the GW250114 analysis, so that one setting serves all events:
- time axis: geocentric, tau = t_geocent - t_c with t_c the median geocent_time of the run
  (the waveform's t = 0, i.e. the peak), instead of detector time at L1;
- grid and window in units of the median detector-frame total mass M of the run:
  WINDOW_M before the peak, sampled every STEP_M. For GW250114 (M ~ 68 M_sun) the
  window is ~0.12 s, close to the one used there. It should not start much earlier:
  sub-populations that agree in phase at merger dephase going back in time, and
  they would be split by the relabeling;
- waveform settings read from each run's config.

Supported: SEOBNRv5PHM runs of bilby with gwsignal (GWTC-4.1), whose waveform call is
the one in `eob_dynamics` (validated for GW250114 in `validate_dynamics.py`).
Not supported yet, each needs its waveform call and conventions (phase, time,
polarization angle) reproduced and validated, as done for GW190521 in
`gw190521/validate_v4phm.py`:
- RIFT SEOBNRv5PHM (GWTC-5.0): Cartesian spins at f_ref = f_min, l_max = 4;
- SEOBNRv4PHM (GWTC-2.1, GWTC-3), RIFT or LALInference: `gw190521/eob_v4phm.py`
  (LALInference) is the starting point.

Outputs: data/u/<event>/<grid>/sample<i>.npz (cache), products/<event>/<window>/
{relabel.npz, corner.png, phase.png, score_q.png}, products/summary.csv.

Usage:
    python relabel_catalog.py run [--events GW231231_154016 ...] [--n 1000] [--processes 8]
                                  [--grid -400 50] [--step 0.5]
    python relabel_catalog.py windows --events GW231231_154016 --windows -350:0 -1000:0 ...
                                      [--grid ... --step ...]   (on cached u, no recomputation)
    python relabel_catalog.py summary
"""

import argparse
import csv
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

import catalog

sys.path.insert(0, str(Path(__file__).parent.parent / "gw250114"))
import eob_dynamics as ed  # noqa: E402
import relabel  # noqa: E402

PRODUCTS = Path(__file__).parent / "products"
U_CACHE = catalog.DATA / "u"
WINDOW_M = (-350.0, 0.0)  # tau / M of the analysis window
GRID_M = (-400.0, 50.0)  # tau / M of the cached u(t)
STEP_M = 0.5
SECONDS_PER_MSUN = ed.lal.MTSUN_SI


class Run:
    """One EOB run of one event: its samples, settings and common time grid."""

    def __init__(self, row, grid=GRID_M, step=STEP_M):
        self.event, self.label = row["event"], row["label"]
        self.settings = row["settings"]
        with h5py.File(catalog.SLIM / f"{self.event}.h5", "r") as f:
            self.posterior = f[self.label]["posterior_samples"][()]
        self.t_c = float(np.median(self.posterior["geocent_time"]))
        self.M_sec = float(np.median(self.posterior["total_mass"])) * SECONDS_PER_MSUN
        self.tau_M = np.arange(grid[0], grid[1], step)
        self.cache = U_CACHE / self.event / f"grid{grid[0]:g}_{grid[1]:g}_{step:g}"
        self.products = PRODUCTS / self.event

    def compute_u(self, sample):
        """u(tau) in the sky frame (NaN after the attachment), and J(t -> -inf)."""
        model, gen = ed.run_model(sample, settings=self.settings)
        params = gen.parameters
        rot = ed.rotation_I_to_sky(params["inclination"], params["phi_ref"], sample["psi"])
        t_wave = self.t_c + self.tau_M * self.M_sec - sample["geocent_time"]
        t_geom = ed.seconds_to_geometric(model)(t_wave)
        if t_geom[0] < model.dynamics[0, 0]:
            raise ValueError("time grid starts before the dynamics")
        ta = model.t_attach
        u = ed.inertial_dynamics(model, np.minimum(t_geom, ta))["n"] @ rot.T
        u[t_geom > ta] = np.nan
        J_inf, _, _ = ed.J_hat_at_past_infinity(model)
        return u, rot @ J_inf


_RUN = None


def _init_worker(run):
    global _RUN
    _RUN = run


def _cache_one(index):
    run = _RUN
    path = run.cache / f"sample{index}.npz"
    if path.exists():
        return None
    sample = {name: float(run.posterior[name][index]) for name in run.posterior.dtype.names}
    try:
        u, J_hat_inf = run.compute_u(sample)
    except Exception as e:
        return f"{run.event} sample {index}: {e!r}"
    np.savez(path, u=u.astype(np.float32), J_hat_inf=J_hat_inf)
    return None


def load_u(run):
    indices, us, Js = [], [], []
    for path in sorted(run.cache.glob("sample*.npz")):
        with np.load(path) as f:
            us.append(f["u"])
            Js.append(f["J_hat_inf"])
        indices.append(int(path.stem.removeprefix("sample")))
    return np.array(indices), np.array(us, dtype=np.float32), np.array(Js)


def null_score(u, seed=0, n_max=4000):
    """Median agreement score after rotating each sample's u by a random angle about the
    line of sight: what the clustering scores when the orbits' orientation on the sky
    carries no information. Compare with the median score of the data."""
    rng = np.random.default_rng(seed)
    if len(u) > n_max:
        u = u[rng.choice(len(u), n_max, replace=False)]
    angle = rng.uniform(0, 2 * np.pi, len(u))[:, None]
    v = u.copy()
    v[..., 0] = np.cos(angle) * u[..., 0] - np.sin(angle) * u[..., 1]
    v[..., 1] = np.sin(angle) * u[..., 0] + np.cos(angle) * u[..., 1]
    return float(np.median(relabel.synchronize_signs(v, np.ones(v.shape[1]))[1]))


def window_tag(window):
    return f"window{window[0]:g}_{window[1]:g}"


def score_plot(q, score, s, path):
    """Agreement score against the mass ratio, for kept and relabeled samples."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for keep, color, label in [(s > 0, "C0", "kept"), (s < 0, "C3", "relabeled")]:
        ax.scatter(q[keep], score[keep], s=2, alpha=0.3, color=color, label=label, rasterized=True)
    ax.set_xlabel("mass ratio $m_2 / m_1$ (PE labels)")
    ax.set_ylabel("agreement score")
    ax.legend(markerscale=5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def analyze(run, window=WINDOW_M, loaded=None):
    """Relabel with the analysis window `window` (tau / M); outputs in products/<event>/<window>/.
    loaded: the output of `load_u(run)` and `relabel.fix_psi`, to reuse across windows."""
    indices, u, psi_flip = loaded or (lambda i, u, J: (i, *relabel.fix_psi(u, J)))(*load_u(run))
    inside = (run.tau_M >= window[0]) & (run.tau_M <= window[1])
    s, score = relabel.synchronize_signs(u[:, inside], np.ones(np.count_nonzero(inside)))
    if np.count_nonzero(s < 0) > len(s) / 2:
        s = -s
    out = run.products / window_tag(window)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "relabel.npz", indices=indices, sign=s, score=score, psi_flip=psi_flip)
    relabel.corner_plot(run.posterior[indices], relabel.relabeled(run.posterior, indices, s),
                        out / "corner.png")
    relabel.phase_plot(u, s, run.tau_M, np.linspace(*window, 5), out / "phase.png", unit="M")
    q = run.posterior["mass_ratio"][indices]
    score_plot(q, score, s, out / "score_q.png")
    return dict(event=run.event, label=run.label, window=window_tag(window), n=len(s),
                relabeled=float(np.mean(s < 0)), psi_flipped=float(np.mean(psi_flip)),
                score_5=float(np.percentile(score, 5)), score_50=float(np.median(score)),
                score_50_null=null_score(u[:, inside]),
                ambiguous=float(np.mean(score < 0.3)), q_median=float(np.median(q)),
                total_mass=run.M_sec / SECONDS_PER_MSUN), s


def compare_windows(row, windows, grid, step):
    """Relabel one event with several windows on the same cached u(t), and compare the labels."""
    run = Run(row, grid, step)
    indices, u, J = load_u(run)
    u, psi_flip = relabel.fix_psi(u, J)
    rows, signs = [], []
    for window in windows:
        summary, s = analyze(run, window, (indices, u, psi_flip))
        rows.append(summary)
        signs.append(s)
        print("  " + ", ".join(f"{k} = {v:.3g}" if isinstance(v, float) else f"{k} = {v}"
                               for k, v in summary.items()), flush=True)
    # fraction of samples with the same label, up to the overall (free) sign
    agreement = np.array([[(1 + abs(np.mean(a * b))) / 2 for b in signs] for a in signs])
    print("label agreement between windows:")
    print("\n".join(f"  {window_tag(w):>22}: " + " ".join(f"{x:.3f}" for x in a)
                    for w, a in zip(windows, agreement)))
    np.savez(run.products / "windows.npz", windows=np.array(windows), agreement=agreement,
             indices=indices, signs=np.array(signs))
    write_summary(rows)


def run_events(rows, n, processes, seed, grid=GRID_M, step=STEP_M):
    PRODUCTS.mkdir(parents=True, exist_ok=True)
    summary = []
    for row in rows:
        run = Run(row, grid, step)
        run.cache.mkdir(parents=True, exist_ok=True)
        if n:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(run.posterior), size=min(n, len(run.posterior)), replace=False)
        else:
            indices = np.arange(len(run.posterior))
        print(f"{run.event} {run.label}: {len(indices)} samples, M = "
              f"{run.M_sec / SECONDS_PER_MSUN:.1f} M_sun", flush=True)
        with Pool(processes, initializer=_init_worker, initargs=(run,)) as pool:
            for error in pool.imap_unordered(_cache_one, indices.tolist(), chunksize=4):
                if error:
                    print("  " + error, flush=True)
        summary.append(analyze(run)[0])
        print("  " + ", ".join(f"{k} = {v:.3g}" for k, v in summary[-1].items()
                               if isinstance(v, float)), flush=True)
    write_summary(summary)


def write_summary(new_rows):
    """Merge into products/summary.csv (one row per event and run)."""
    path = PRODUCTS / "summary.csv"
    rows = {}
    if path.exists():
        with open(path) as f:
            rows = {(r["event"], r["label"], r["window"]): r for r in csv.DictReader(f)}
    rows.update({(r["event"], r["label"], r["window"]): r for r in new_rows})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(rows.values()))))
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: r["event"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--events", nargs="*")
    p.add_argument("--n", type=int, default=1000, help="samples per event (0: all)")
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("windows", help="compare analysis windows on cached u(t)")
    p.add_argument("--events", nargs="*")
    p.add_argument("--windows", nargs="+", required=True, help="start:end in M, e.g. -350:0")
    # let "-350:0" be read as a value, not as an option
    p._negative_number_matcher = re.compile(r"^-?[\d.]+:-?[\d.]+$|^-\d+$|^-\d*\.\d+$")
    for p in sub.choices.values():
        p.add_argument("--grid", type=float, nargs=2, default=GRID_M, help="cached tau range [M]")
        p.add_argument("--step", type=float, default=STEP_M, help="cached tau step [M]")
    sub.add_parser("summary")
    args = parser.parse_args()

    if args.command == "run":
        rows = [r for r in catalog.load_inventory() if r["supported"]]
        if args.events:
            rows = [r for r in rows if r["event"] in args.events]
        run_events(rows, args.n, args.processes, args.seed, args.grid, args.step)
    elif args.command == "windows":
        rows = [r for r in catalog.load_inventory() if r["supported"]]
        if args.events:
            rows = [r for r in rows if r["event"] in args.events]
        windows = [tuple(float(x) for x in w.split(":")) for w in args.windows]
        for row in rows:
            print(row["event"], row["label"], flush=True)
            compare_windows(row, windows, args.grid, args.step)
    elif args.command == "summary":
        with open(PRODUCTS / "summary.csv") as f:
            print(f.read())


if __name__ == "__main__":
    main()
