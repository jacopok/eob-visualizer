"""Reconstruct the coalescence phase of the GW190521 SEOBNRv4PHM samples.

In the GWTC-2.1 release of this (RIFT) run every sample has the same geocent_time and
the stored phases are uniformly distributed: time and phase were marginalized over, and
the stored values do not go with the other parameters (the whitened waveforms built with
them are out of phase with the data by up to 2 rad, sample by sample, while with a
suitable phase every sample matches the data to within 0.6 ms at the stored time).

As an approximation, for each sample we draw the phase from its conditional posterior
p(phase | other parameters, data) ∝ exp(log L) with log L the H1 + L1 Gaussian
likelihood (release PSDs, 20-224 Hz, no calibration model) at the stored time, on a
grid. The phase only sets the azimuth of the observer, so the model is run once per
sample. Virgo is left out (its contribution to this event is small).

Usage: python reconstruct_phase.py   (in ../gw250114/.venv)
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eobviz.events import get_event  # noqa: E402
from eobviz.models import seobnrv4 as ev  # noqa: E402

EVENT = get_event("GW190521")

PHASE_GRID = np.linspace(0, 2 * np.pi, 180, endpoint=False)
IFOS = ("H1", "L1")
_worker = {}


def _init():
    _worker["ifos"] = [EVENT.load_interferometer(name) for name in IFOS]


def log_likelihood_vs_phase(sample):
    model = ev.run_model(sample)
    logl = np.zeros(len(PHASE_GRID))
    for k, phase in enumerate(PHASE_GRID):
        model.hpc = ev.polarizations(model, sample["iota"], phase)
        s = dict(sample, phase=phase)
        for ifo in _worker["ifos"]:
            mask = ifo.frequency_mask
            h = ifo.get_detector_response(EVENT.frequency_domain_polarizations(model, ifo), s)[mask]
            d = ifo.frequency_domain_strain[mask]
            psd = ifo.power_spectral_density_array[mask]
            inner = lambda a, b: 4 / ifo.duration * np.real(np.sum(np.conj(a) * b / psd))
            logl[k] += inner(d, h) - inner(h, h) / 2
    return logl


if __name__ == "__main__":
    posterior = EVENT.load_posterior()
    indices = np.arange(len(posterior))
    with Pool(8, initializer=_init) as pool:
        logl = np.array(pool.map(log_likelihood_vs_phase,
                                 [EVENT.release_sample(posterior, i) for i in indices], chunksize=4))
    rng = np.random.default_rng(1)
    weights = np.exp(logl - logl.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    drawn = np.array([rng.choice(PHASE_GRID, p=w) for w in weights])
    drawn += rng.uniform(0, PHASE_GRID[1], len(drawn))  # within the grid cell
    EVENT.phases_file.parent.mkdir(exist_ok=True)
    np.savez(EVENT.phases_file, indices=indices, phase_drawn=drawn,
             phase_max=PHASE_GRID[np.argmax(logl, axis=1)], phase_grid=PHASE_GRID, log_likelihood=logl,
             phase_release=posterior["phase"])
    spread = np.degrees(np.sqrt(np.sum(weights * np.angle(np.exp(1j * (PHASE_GRID[None] - drawn[:, None]))) ** 2, axis=1)))
    print(f"{len(indices)} samples; conditional phase posterior width (rms): median {np.median(spread):.0f} deg; "
          f"max log L over phase: median {np.median(logl.max(axis=1)):.1f} (release log_likelihood median "
          f"{np.median(posterior['log_likelihood']):.1f})")
