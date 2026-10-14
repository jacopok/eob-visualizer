"""GW150914: TEOBResumS-Dali (eccentric, precessing) bilby run of Gamba et al.,
CQG 10.1088/1361-6382/adfe50 (f_min = f_ref = 15 Hz, modes (2,2), (2,1), (3,3), (4,4)).

PPD with the method of the GW250114 paper script, for this run (settings from its meta_data):
- data: GWOSC 4 kHz strain (GWTC-1 R1 frames, as fetched by bilby_pipe for channel GWOSC),
  8 s segment ending 2 s after the trigger time, Tukey roll-off 0.4 s, and the
  corresponding sqrt(beta) correction;
- PSDs: the GWTC-1 ones used in PE; band 20-896 Hz;
- waveforms: TEOBResumS-Dali as in PE, through the same frequency-domain path as
  gwsignal's conditioning (start taper, zero padding to 8 s, FFT with TEOB's time axis as
  epoch), projected with bilby with the sample's spline calibration.
"""

from pathlib import Path

import h5py
import numpy as np

from .base import Event, tukey_beta

DATA = Path("/home/jacopo/Documents/gw150914_decade/dat/GW150914")
TRIGGER_TIME = 1126259462.391
DURATION, POST_TRIGGER = 8.0, 2.0
START_TIME = TRIGGER_TIME + POST_TRIGGER - DURATION
ROLL_OFF = 0.4
F_LOW, F_HIGH = 20.0, 896.0
CALIBRATION_NODES = 10


class GW150914(Event):
    name = "GW150914"
    approximant = "TEOBResumS-Dali"
    ifo = "H1"  # the louder detector (SNR ~20 vs ~15)
    reference_time = 1126259462.4
    time_window = (-0.48, 0.175)  # merger at ~+0.02
    slowdown = 100  # GW150914's mass is close to GW250114's
    extra_sample_keys = ("eccentricity", "mean_per_ano")
    panel_note = "Eccentric, precessing TEOBResumS-Dali posterior\n"
    ppd_time_range = (-0.15, 0.05)
    ppd_beta = tukey_beta(ROLL_OFF, DURATION)
    posterior_file = Path(
        "/home/jacopo/Documents/gw150914_decade/dat/post_bilby/15_hz_post/"
        "gw150914_teob_prc_ecc_allmodes_15_data0_1126259462-391_analysis_H1L1_result.hdf5"
    )

    def load_posterior(self, approximant=None):
        """Posterior samples as a structured array (real-valued columns only)."""
        with h5py.File(self.posterior_file, "r") as f:
            g = f["posterior"]
            names = [k for k in g if g[k].dtype.kind == "f"]
            columns = [g[k][()] for k in names]
        out = np.empty(len(columns[0]), dtype=[(k, float) for k in names])
        for k, c in zip(names, columns):
            out[k] = c
        return out

    def load_interferometer(self, name):
        import bilby
        from gwpy.timeseries import TimeSeries

        ts = TimeSeries.read(DATA / f"{name[0]}-{name}_GWOSC_4KHZ_R1-1126259447-32.gwf",
                             f"{name}:GWOSC-4KHZ_R1_STRAIN", format="gwf.lalframe")
        ts = ts.crop(START_TIME, START_TIME + DURATION)
        ifo = bilby.gw.detector.get_empty_interferometer(name)
        ifo.strain_data.roll_off = ROLL_OFF
        ifo.set_strain_data_from_gwpy_timeseries(ts)
        ifo.strain_data.frequency_domain_strain  # applies the window, sets the window factor
        psd = np.loadtxt(DATA / f"GWTC1_GW150914_PSDs_{name}.dat")
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd[:, 0], psd_array=psd[:, 1])
        ifo.minimum_frequency, ifo.maximum_frequency = F_LOW, F_HIGH
        ifo.calibration_model = bilby.gw.calibration.CubicSpline(
            prefix=f"recalib_{name}_", minimum_frequency=F_LOW, maximum_frequency=F_HIGH,
            n_points=CALIBRATION_NODES)
        return ifo

    def whitened_waveform(self, ifo, sample, approximant=None):
        model = self.run_model(sample).model
        pols = frequency_domain_polarizations(model, ifo.frequency_array)
        return self.whiten(ifo, ifo.get_detector_response(pols, sample))


def frequency_domain_polarizations(model, frequency_array, hpc=None):
    """The PE frequency-domain path for a `teob_dali.Model`: TD polarizations (TEOB's by
    default) tapered at the start, placed at the end of an 8 s buffer, FFT'd, with the
    maximum of h_+ at geocent_time (`teob_dali.pe_time_origin`; bilby shifts by
    1/df + epoch)."""
    from ..models import teob_dali as td

    hpc = model.hpc if hpc is None else hpc
    dt = 1 / td.SAMPLING_RATE
    n = int(round(DURATION / dt))
    hp, hc = td.taper_start(hpc.real), td.taper_start(-hpc.imag)
    if len(hp) > n:
        raise ValueError("waveform longer than the segment")
    epoch = -(n - len(hp) + td.pe_time_origin(hpc)) * dt  # time of the first buffer sample
    freqs = np.fft.rfftfreq(n, dt)
    shift = np.exp(-2j * np.pi * freqs * epoch)
    out = {}
    for key, h in (("plus", hp), ("cross", hc)):
        buf = np.zeros(n)
        buf[n - len(h):] = h
        fd = np.fft.rfft(buf) * dt * shift
        full = np.zeros(len(frequency_array), dtype=complex)
        m = min(len(full), len(fd))
        full[:m] = fd[:m]
        full *= (frequency_array >= td.WAVEFORM_MINIMUM_FREQUENCY) & (frequency_array <= F_HIGH)
        out[key] = full
    return out


def polarizations_mirrored(model, sample):
    """Alternative tested in `gw150914/validate_teob.py`: only the m >= 0 inertial modes,
    with the m < 0 ones from h_{l,-m} = (-1)^l conj(h_lm) (exact only without precession)."""
    import lal

    iota, phase = model.parameters["inclination"], sample["phase"]
    nuMD = model.m_1 * model.m_2 * model.M * lal.MRSUN_SI / (sample["luminosity_distance"] * 1e6 * lal.PC_SI)
    h = 0
    for (ell, m), hlm in model.hI.items():
        if m < 0:
            continue
        h = h + hlm * lal.SpinWeightedSphericalHarmonic(iota, phase, -2, ell, m)
        if m > 0:
            h = h + (-1) ** ell * np.conj(hlm) * lal.SpinWeightedSphericalHarmonic(iota, phase, -2, ell, -m)
    return -nuMD * h
