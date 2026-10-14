"""GW190521: GWTC-2.1 SEOBNRv4PHM (LALInference) run, `C01:SEOBNRv4PHM` in the release.

PPD with the method of the GW250114 paper script, adapted to this run:
- data: GWOSC open strain (gw190521/data/, fetched with gwpy), resampled to the run's
  512 Hz, in a 4 s segment (the run's seglen) centred on the trigger;
- PSD: the one stored in the release for this run; band 20-224 Hz (flow, fhigh);
- Tukey window with 0.4 s roll-off, and the corresponding sqrt(beta) correction;
- waveforms: SEOBNRv4PHM exactly as in PE, projected with bilby. The release has no
  calibration parameters, so no calibration model is applied.

Coalescence phases: those of the release are not usable (time and phase were
marginalized over); each sample's phase is replaced by one drawn from its conditional
posterior (`gw190521/reconstruct_phase.py`, see gw190521/README.md).
"""

from functools import cached_property

import h5py
import numpy as np

from .. import REPO
from .base import Event, tukey_beta

SAMPLING_RATE = 512.0
SEGMENT_DURATION = 4.0
ROLL_OFF = 0.4


class GW190521(Event):
    name = "GW190521"
    approximant = "SEOBNRv4PHM"
    model_settings = dict(f_min=5.0, sampling_rate=SAMPLING_RATE)  # fmin-template, srate
    ifo = "L1"
    reference_time = 1242442967.4
    time_window = (-0.95, 0.30)  # merger at ~+0.05
    slowdown = 50  # ~4 times heavier than GW250114: its orbits are ~4 times slower
    panel_note = "Coalescence phases re-drawn from the H1+L1 data\n(not usable in the release; approximate)\n"
    ppd_time_range = (-0.15, 0.1)
    ppd_beta = tukey_beta(ROLL_OFF, SEGMENT_DURATION)
    band = (20.0, 224.0)
    posterior_file = REPO / "IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_nocosmo.h5"
    posterior_label = "C01:SEOBNRv4PHM"

    @property
    def phases_file(self):
        return self.products / "phase_reconstruction.npz"

    def load_posterior(self, approximant=None):
        with h5py.File(self.posterior_file, "r") as f:
            return f[self.posterior_label]["posterior_samples"][()]

    @cached_property
    def reconstructed_phases(self):
        if not self.phases_file.exists():
            raise FileNotFoundError("run gw190521/reconstruct_phase.py first")
        d = np.load(self.phases_file)
        return dict(zip(d["indices"].tolist(), d["phase_drawn"]))

    def release_sample(self, posterior, index):
        """The sample as in the release (with its unusable phase)."""
        return super().sample(posterior, index)

    def sample(self, posterior, index):
        """The sample with its coalescence phase replaced by the reconstructed one."""
        s = self.release_sample(posterior, index)
        s["phase_release"] = s["phase"]
        s["phase"] = float(self.reconstructed_phases[index])
        return s

    def load_interferometer(self, name):
        import bilby
        from gwpy.timeseries import TimeSeries

        segment = (self.reference_time - SEGMENT_DURATION / 2, self.reference_time + SEGMENT_DURATION / 2)
        ts = TimeSeries.read(self.directory / "data" / f"{name}_GWOSC_4KHZ_64s.hdf5")
        ts = ts.resample(SAMPLING_RATE).crop(*segment)
        ifo = bilby.gw.detector.get_empty_interferometer(name)
        ifo.strain_data.roll_off = ROLL_OFF
        ifo.set_strain_data_from_gwpy_timeseries(ts)
        ifo.strain_data.roll_off = ROLL_OFF
        with h5py.File(self.posterior_file, "r") as f:
            psd = f[self.posterior_label]["psds"][name][()]
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd[:, 0], psd_array=psd[:, 1])
        ifo.minimum_frequency, ifo.maximum_frequency = self.band
        return ifo

    @staticmethod
    def frequency_domain_polarizations(model, ifo):
        """FFT of the PE time-domain polarizations (of a `seobnrv4.Model`) on the segment's
        frequency grid, with the waveform's t = 0 as time origin."""
        n_seg = len(ifo.time_array)
        h = model.hpc[-n_seg:]  # (all GW190521 samples are shorter than the segment)
        dt = 1 / SAMPLING_RATE
        epoch = model.epoch + (len(model.hpc) - len(h)) * dt
        phase = np.exp(-2j * np.pi * ifo.frequency_array * epoch)
        return {
            "plus": np.fft.rfft(h.real, n_seg) * dt * phase,
            "cross": np.fft.rfft(-h.imag, n_seg) * dt * phase,
        }

    def whitened_waveform(self, ifo, sample, approximant=None):
        model = self.run_model(sample).model
        return self.whiten(ifo, ifo.get_detector_response(self.frequency_domain_polarizations(model, ifo), sample))
