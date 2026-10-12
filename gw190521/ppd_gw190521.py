"""Whitened time-domain posterior predictive distribution of GW190521 in one detector,
with the method of the GW250114 paper script (`gw250114/ppd.py`), adapted to the
GWTC-2.1 SEOBNRv4PHM (LALInference) run:

- data: GWOSC open strain (data/, fetched with gwpy), resampled to the run's 512 Hz,
  in a 4 s segment (the run's seglen) centred on the trigger;
- PSD: the one stored in the release for this run; band 20-224 Hz (flow, fhigh);
- Tukey window with 0.4 s roll-off, and the corresponding sqrt(beta) correction;
- waveforms: SEOBNRv4PHM exactly as in PE (`eob_v4phm.run_model`), projected with bilby.
  The release has no calibration parameters, so no calibration model is applied.

Usage: python ppd_gw190521.py [--ifo L1] [--n-samples 500]
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import bilby
import h5py
import numpy as np

import eob_v4phm as ev

sys.path.insert(0, str(Path(__file__).parent.parent / "gw250114"))
from ppd import CONFIDENCE_LEVEL, plot_ppd, whitened_time_series  # noqa: E402

DATA = Path(__file__).parent / "data"
OUTDIR = Path(__file__).parent / "products"
REFERENCE_TIME = 1242442967.4
SEGMENT = (REFERENCE_TIME - 2, REFERENCE_TIME + 2)
F_LOW, F_HIGH = 20.0, 224.0
ROLL_OFF = 0.4
ALPHA = 2 * ROLL_OFF / (SEGMENT[1] - SEGMENT[0])
BETA = 1 - 5 * ALPHA / 8  # mean square of the Tukey window
TIME_RANGE = (-0.15, 0.1)


def load_interferometer(name):
    from gwpy.timeseries import TimeSeries

    ts = TimeSeries.read(DATA / f"{name}_GWOSC_4KHZ_64s.hdf5")
    ts = ts.resample(ev.SAMPLING_RATE).crop(*SEGMENT)
    ifo = bilby.gw.detector.get_empty_interferometer(name)
    ifo.strain_data.roll_off = ROLL_OFF
    ifo.set_strain_data_from_gwpy_timeseries(ts)
    ifo.strain_data.roll_off = ROLL_OFF
    with h5py.File(ev.POSTERIOR_FILE, "r") as f:
        psd = f[ev.POSTERIOR_LABEL]["psds"][name][()]
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd[:, 0], psd_array=psd[:, 1])
    ifo.minimum_frequency, ifo.maximum_frequency = F_LOW, F_HIGH
    return ifo


def whiten(ifo, fd):
    mask = ifo.frequency_mask
    white = np.zeros_like(fd)
    white[mask] = fd[mask] / (ifo.amplitude_spectral_density_array[mask] * np.sqrt(ifo.duration / 4))
    return whitened_time_series(mask, white) * np.sqrt(BETA)


def whitened_data(ifo):
    return whiten(ifo, ifo.frequency_domain_strain)


def frequency_domain_polarizations(model, ifo):
    """FFT of the PE time-domain polarizations on the segment's frequency grid,
    with the waveform's t = 0 as time origin."""
    n_seg = len(ifo.time_array)
    h = model.hpc[-n_seg:]  # (all GW190521 samples are shorter than the segment)
    epoch = model.epoch + (len(model.hpc) - len(h)) / ev.SAMPLING_RATE
    dt = 1 / ev.SAMPLING_RATE
    phase = np.exp(-2j * np.pi * ifo.frequency_array * epoch)
    return {
        "plus": np.fft.rfft(h.real, n_seg) * dt * phase,
        "cross": np.fft.rfft(-h.imag, n_seg) * dt * phase,
    }


_worker = {}


def _init_worker(ifo_name):
    _worker["ifo"] = load_interferometer(ifo_name)


def whitened_waveform(sample, ifo=None):
    ifo = ifo or _worker["ifo"]
    model = ev.run_model(sample)
    response = ifo.get_detector_response(frequency_domain_polarizations(model, ifo), sample)
    return whiten(ifo, response)


def compute_ppd(ifo_name, n_samples, extra_indices=(), seed=0, processes=8):
    posterior = ev.load_posterior()
    rng = np.random.default_rng(seed)
    indices = list(rng.choice(len(posterior), n_samples, replace=False)) + list(extra_indices)
    samples = [ev.sample_with_phase(posterior, i) for i in indices]  # see reconstruct_phase.py
    with Pool(processes, initializer=_init_worker, initargs=(ifo_name,)) as pool:
        waveforms = np.array(pool.map(whitened_waveform, samples, chunksize=1))
    ifo = load_interferometer(ifo_name)
    delta = (1 + CONFIDENCE_LEVEL) / 2
    ppd = waveforms[:n_samples]
    return dict(
        times=ifo.time_array - REFERENCE_TIME,
        data=whitened_data(ifo),
        lower=np.percentile(ppd, (1 - delta) * 100, axis=0),
        upper=np.percentile(ppd, delta * 100, axis=0),
        median=np.median(ppd, axis=0),
        sample_indices=np.array(indices),
        waveforms=waveforms,
        reference_time=REFERENCE_TIME,
        ifo=ifo_name,
        approximant="SEOBNRv4PHM",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifo", default="L1")
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()
    i_maxl = int(np.argmax(ev.load_posterior()["log_likelihood"]))
    OUTDIR.mkdir(exist_ok=True)
    ppd = compute_ppd(args.ifo, args.n_samples, [i_maxl])
    stem = f"ppd_SEOBNRv4PHM_{args.ifo}"
    np.savez(OUTDIR / f"{stem}.npz", **ppd)
    plot_ppd(ppd, OUTDIR / f"{stem}.png", time_range=TIME_RANGE, highlight=ppd["waveforms"][-1])
