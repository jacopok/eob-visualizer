"""Whitened time-domain posterior predictive distribution of the GW250114 signal
in one detector, following `waveform_reconstruction.py` of the GW250114 paper
(S250114ax/source/scripts): same data dump, same whitening, same
window correction sqrt(beta), same 90% symmetric interval.

The waveforms are computed with SEOBNRv5PHM using the settings of its PE run;
`--approximant NRSur7dq4` reproduces the paper figure instead.
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PAPER_DATA = Path("/home/jacopo/Documents/S250114ax/s250114ax/source/data")
DATA_DUMP = (
    PAPER_DATA
    / "bilby-NRSur7dq4_prod_data0_1420878141-222656_generation_data_dump.pickle"
)
NRSUR_RESULT = (
    PAPER_DATA / "bilby-NRSur7dq4_prod_data0_1420878141-222656_analysis_H1L1_merge_result.hdf5"
)
OUTDIR = Path(__file__).parent / "products"

# as in waveform_reconstruction.py
REFERENCE_TIME = 1420878141.0
TIME_RANGE = (0.06, 0.24)
CONFIDENCE_LEVEL = 0.9
ALPHA = 1 / 4
BETA = 1 - 5 * ALPHA / 8


def whitened_time_series(frequency_mask, whitened_frequency_series):
    """Copied from waveform_reconstruction.py (itself from bilby)."""
    frequency_window_factor = np.sum(frequency_mask) / len(frequency_mask)
    return (
        np.fft.irfft(whitened_frequency_series)
        * np.sqrt(np.sum(frequency_mask))
        / frequency_window_factor
    )


def load_interferometer(name):
    from bilby_pipe.utils import DataDump

    data = DataDump.from_pickle(DATA_DUMP)
    return next(ifo for ifo in data.interferometers if ifo.name == name)


def whitened_data(ifo):
    return whitened_time_series(
        ifo.frequency_mask, ifo.whitened_frequency_domain_strain * np.sqrt(BETA)
    )


def whiten_response(ifo, fd_response):
    whitened_fd = fd_response / (
        ifo.amplitude_spectral_density_array * np.sqrt(ifo.duration / 4)
    )
    return whitened_time_series(ifo.frequency_mask, whitened_fd) * np.sqrt(BETA)


def make_generator(approximant, ifo):
    import bilby

    if approximant == "SEOBNRv5PHM":
        import eob_dynamics as ed

        return bilby.gw.waveform_generator.WaveformGenerator(
            duration=ifo.duration,
            sampling_frequency=ifo.sampling_frequency,
            start_time=ifo.start_time,
            frequency_domain_source_model=bilby.gw.source.gwsignal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(
                waveform_approximant="SEOBNRv5PHM",
                reference_frequency=ed.REFERENCE_FREQUENCY,
                minimum_frequency=ed.WAVEFORM_MINIMUM_FREQUENCY,
                maximum_frequency=ed.MAXIMUM_FREQUENCY,
                lmax_nyquist=ed.LMAX_NYQUIST,
                catch_waveform_errors=True,
            ),
        )
    elif approximant == "NRSur7dq4":
        from bilby.gw.result import CompactBinaryCoalescenceResult

        os.environ["LAL_DATA_PATH"] = PAPER_DATA.as_posix()
        res = CompactBinaryCoalescenceResult.from_hdf5(filename=NRSUR_RESULT)
        return res.waveform_generator_class(
            duration=res.duration,
            sampling_frequency=res.sampling_frequency,
            start_time=res.start_time,
            frequency_domain_source_model=res.frequency_domain_source_model,
            time_domain_source_model=res.time_domain_source_model,
            parameter_conversion=res.parameter_conversion,
            waveform_arguments=res.waveform_arguments,
        )
    raise ValueError(approximant)


def load_samples(approximant):
    if approximant == "SEOBNRv5PHM":
        import eob_dynamics as ed

        posterior = ed.load_posterior()
        return [ed.sample_as_dict(posterior, i) for i in range(len(posterior))]
    elif approximant == "NRSur7dq4":
        from bilby.gw.result import CompactBinaryCoalescenceResult

        res = CompactBinaryCoalescenceResult.from_hdf5(filename=NRSUR_RESULT)
        return [dict(row) for _, row in res.posterior.iterrows()]
    raise ValueError(approximant)


_worker = {}


def _init_worker(approximant, ifo_name):
    _worker["ifo"] = load_interferometer(ifo_name)
    _worker["wfg"] = make_generator(approximant, _worker["ifo"])


def _whitened_waveform(params):
    ifo, wfg = _worker["ifo"], _worker["wfg"]
    pols = wfg.frequency_domain_strain(dict(params))
    return whiten_response(ifo, ifo.get_detector_response(pols, params))


def compute_ppd(approximant, ifo_name, n_samples, extra_indices=(), seed=0, processes=8):
    """Whitened waveforms for `n_samples` random posterior samples
    (without replacement, as in the paper script) plus the samples at `extra_indices`."""
    samples = load_samples(approximant)
    rng = np.random.default_rng(seed)
    indices = list(rng.choice(len(samples), n_samples, replace=False)) + list(extra_indices)

    with Pool(processes, initializer=_init_worker, initargs=(approximant, ifo_name)) as pool:
        waveforms = pool.map(_whitened_waveform, [samples[i] for i in indices], chunksize=1)

    ifo = load_interferometer(ifo_name)
    times = ifo.time_array - REFERENCE_TIME
    data = whitened_data(ifo)
    waveforms = np.array(waveforms)
    delta = (1 + CONFIDENCE_LEVEL) / 2
    ppd = waveforms[:n_samples]
    return dict(
        times=times,
        data=data,
        lower=np.percentile(ppd, (1 - delta) * 100, axis=0),
        upper=np.percentile(ppd, delta * 100, axis=0),
        median=np.median(ppd, axis=0),
        sample_indices=np.array(indices),
        waveforms=waveforms,
        reference_time=REFERENCE_TIME,
        ifo=ifo_name,
        approximant=approximant,
    )


def plot_ppd(ppd, path, time_range=TIME_RANGE, highlight=None):
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
        ax.plot(t[sel], highlight[sel], c="k", lw=1, zorder=5, label="shown sample")
    ax.set_xlim(*time_range)
    ax.set_ylim(-15, 15)
    ax.set_xlabel(f"Time - {ppd['reference_time']:.1f} [s]")
    ax.set_ylabel(r"Whitened strain [$\sigma$]")
    ax.legend(loc="lower left", fontsize="x-small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approximant", default="SEOBNRv5PHM", choices=["SEOBNRv5PHM", "NRSur7dq4"])
    parser.add_argument("--ifo", default="L1")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--include-maxl", action="store_true",
                        help="also compute the whitened waveform of the max-likelihood sample")
    args = parser.parse_args()

    extra = []
    if args.include_maxl:
        samples = load_samples(args.approximant)
        extra = [int(np.argmax([s["log_likelihood"] for s in samples]))]

    OUTDIR.mkdir(exist_ok=True)
    ppd = compute_ppd(args.approximant, args.ifo, args.n_samples, extra)
    stem = f"ppd_{args.approximant}_{args.ifo}"
    np.savez(OUTDIR / f"{stem}.npz", **ppd)
    plot_ppd(ppd, OUTDIR / f"{stem}.png",
             highlight=ppd["waveforms"][-1] if extra else None)
