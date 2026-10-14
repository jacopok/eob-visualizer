"""Whitened time-domain posterior predictive distribution (PPD) of an event in one detector,
following `waveform_reconstruction.py` of the GW250114 paper: whitened data, whitened
detector responses of random posterior samples, 90% symmetric interval. Data loading,
whitening and waveforms are the event's (`events`).

The whitened waveforms of all the drawn samples (and of the max-likelihood one, last)
are stored: the panel of a render shows the samples of the scene.
"""

from multiprocessing import Pool

import numpy as np

from .events import get_event

CONFIDENCE_LEVEL = 0.9
_worker = {}


def _init_worker(event_name, ifo_name, approximant):
    event = get_event(event_name)
    _worker.update(event=event, ifo=event.load_interferometer(ifo_name), approximant=approximant)


def _whitened_waveform(sample):
    return _worker["event"].whitened_waveform(_worker["ifo"], sample, _worker["approximant"])


def compute_ppd(event, ifo_name=None, n_samples=500, approximant=None, seed=0, processes=8):
    """Whitened waveforms for `n_samples` random posterior samples (without replacement,
    as in the paper script) plus the max-likelihood sample."""
    ifo_name = ifo_name or event.ifo
    approximant = approximant or event.approximant
    posterior = event.load_posterior(approximant)
    rng = np.random.default_rng(seed)
    indices = list(rng.choice(len(posterior), n_samples, replace=False))
    indices.append(event.max_likelihood_index(posterior))
    samples = [event.sample(posterior, i) for i in indices]

    with Pool(processes, initializer=_init_worker, initargs=(event.name, ifo_name, approximant)) as pool:
        waveforms = np.array(pool.map(_whitened_waveform, samples, chunksize=1))

    ifo = event.load_interferometer(ifo_name)
    delta = (1 + CONFIDENCE_LEVEL) / 2
    ppd = waveforms[:n_samples]
    return dict(
        times=ifo.time_array - event.reference_time,
        data=event.whitened_data(ifo),
        lower=np.percentile(ppd, (1 - delta) * 100, axis=0),
        upper=np.percentile(ppd, delta * 100, axis=0),
        median=np.median(ppd, axis=0),
        sample_indices=np.array(indices),
        waveforms=waveforms,
        reference_time=event.reference_time,
        ifo=ifo_name,
        approximant=approximant,
    )


def plot_ppd(ppd, path, time_range, highlight=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.5))
    t = ppd["times"]
    sel = (t >= time_range[0]) & (t <= time_range[1])
    ax.plot(t[sel], ppd["data"][sel], c="grey", label=f"{ppd['ifo']} data")
    ax.fill_between(
        t[sel], ppd["lower"][sel], ppd["upper"][sel], color="C0", zorder=4,
        label=f"{ppd['approximant']} reconstruction (90%)",
    )
    if highlight is not None:
        ax.plot(t[sel], highlight[sel], c="k", lw=1, zorder=5, label="max-likelihood sample")
    ax.set_xlim(*time_range)
    ax.set_ylim(-15, 15)
    ax.set_xlabel(f"Time - {ppd['reference_time']:.1f} [s]")
    ax.set_ylabel(r"Whitened strain [$\sigma$]")
    ax.legend(loc="lower left", fontsize="x-small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_ppd(event, ifo_name=None, n_samples=500, approximant=None, processes=8):
    ppd = compute_ppd(event, ifo_name, n_samples, approximant, processes=processes)
    path = event.ppd_file(ppd["ifo"], ppd["approximant"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **ppd)
    plot_ppd(ppd, path.with_suffix(".png"), event.ppd_time_range, highlight=ppd["waveforms"][-1])
    return path
