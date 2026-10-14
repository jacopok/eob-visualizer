"""What the pipeline needs to know about an event: its PE run (posterior, waveform model
and settings), the timeline of the render, and the data shown in the panel."""

import numpy as np

from . import event_directory

# posterior parameters copied to each scene's info file
SAMPLE_INFO_KEYS = (
    "mass_1", "mass_2", "mass_1_source", "mass_2_source", "redshift",
    "a_1", "a_2", "tilt_1", "tilt_2", "theta_jn", "iota", "psi", "phase",
    "ra", "dec", "geocent_time", "luminosity_distance", "log_likelihood")


def whitened_time_series(frequency_mask, whitened_frequency_series):
    """From `waveform_reconstruction.py` of the GW250114 paper (itself from bilby)."""
    frequency_window_factor = np.sum(frequency_mask) / len(frequency_mask)
    return (
        np.fft.irfft(whitened_frequency_series)
        * np.sqrt(np.sum(frequency_mask))
        / frequency_window_factor
    )


def tukey_beta(roll_off, duration):
    """Mean square of a Tukey window with the given roll-off (bilby's convention)."""
    return 1 - 5 * (2 * roll_off / duration) / 8


class Event:
    """Subclasses set the class attributes below and implement `load_posterior` and, for
    the PPD, `load_interferometer` and `whitened_waveform`."""

    name = None
    approximant = None  # waveform model of the PE run, a key of `models.MODELS`
    model_settings = {}  # keyword arguments of the model class
    ifo = None  # detector of the panel; frames are uniform in its time
    reference_time = None  # GPS origin of the time axes
    time_window = None  # (start, end) of the render, detector time - reference_time [s]
    fps = 30
    slowdown = None  # video seconds per second of signal
    trail_points_per_orbit = 96
    extra_sample_keys = ()  # beyond SAMPLE_INFO_KEYS
    panel_note = None
    # PPD: plotted time range, and the whitening's sqrt(beta) correction
    ppd_time_range = None
    ppd_beta = None

    @property
    def directory(self):
        return event_directory(self.name)

    @property
    def products(self):
        return self.directory / "products"

    @property
    def renders(self):
        return self.directory / "renders"

    # posterior and model

    def load_posterior(self, approximant=None):
        """Posterior samples (a structured array, or a DataFrame) of the PE run with
        `approximant` (default: the event's)."""
        raise NotImplementedError

    def sample(self, posterior, index):
        """One posterior sample as a dict of floats."""
        if hasattr(posterior, "iloc"):
            return dict(posterior.iloc[index])
        return {name: float(posterior[name][index]) for name in posterior.dtype.names}

    def max_likelihood_index(self, posterior=None):
        posterior = self.load_posterior() if posterior is None else posterior
        return int(np.argmax(np.asarray(posterior["log_likelihood"])))

    def run_model(self, sample):
        """`models.base.EOBRun` of the sample, called as in PE."""
        from ..models import get_model

        return get_model(self.approximant)(sample, **self.model_settings)

    def frame_times(self):
        """Detector time - reference_time [s] of the video frames."""
        n_frames = int(round((self.time_window[1] - self.time_window[0]) * self.slowdown * self.fps)) + 1
        return np.linspace(*self.time_window, n_frames)

    # data and posterior predictive distribution (PPD)

    def ppd_file(self, ifo=None, approximant=None):
        return self.products / f"ppd_{approximant or self.approximant}_{ifo or self.ifo}.npz"

    def load_interferometer(self, name):
        """bilby Interferometer holding the data, the PSD, the band and calibration of PE."""
        raise NotImplementedError

    def whitened_waveform(self, ifo, sample, approximant=None):
        """Whitened time-domain response of `ifo` to the sample, on `ifo.time_array`."""
        raise NotImplementedError

    def whiten(self, ifo, fd):
        """Whitened time series of a frequency series, zero outside the analysis band."""
        mask = ifo.frequency_mask
        white = np.zeros_like(fd)
        white[mask] = fd[mask] / (ifo.amplitude_spectral_density_array[mask] * np.sqrt(ifo.duration / 4))
        return whitened_time_series(mask, white) * np.sqrt(self.ppd_beta)

    def whitened_data(self, ifo):
        return self.whiten(ifo, ifo.frequency_domain_strain)
