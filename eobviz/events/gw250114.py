"""GW250114: SEOBNRv5PHM bilby run (reweighted), as in the GW250114 paper.

PPD as `waveform_reconstruction.py` of the paper (S250114ax/source/scripts): same data
dump, same whitening, same window correction sqrt(beta), same 90% symmetric interval.
The waveforms are computed with SEOBNRv5PHM using the settings of its PE run;
`approximant="NRSur7dq4"` reproduces the paper figure instead. The data dump of the
NRSur7dq4 run is used: its PSDs are identical to the SEOBNRv5PHM run's.
"""

import os
from pathlib import Path

import h5py
import numpy as np

from .. import REPO
from .base import Event, whitened_time_series

PAPER_DATA = Path("/home/jacopo/Documents/S250114ax/s250114ax/source/data")
DATA_DUMP = PAPER_DATA / "bilby-NRSur7dq4_prod_data0_1420878141-222656_generation_data_dump.pickle"
NRSUR_RESULT = PAPER_DATA / "bilby-NRSur7dq4_prod_data0_1420878141-222656_analysis_H1L1_merge_result.hdf5"


class GW250114(Event):
    name = "GW250114"
    approximant = "SEOBNRv5PHM"
    ifo = "L1"
    reference_time = 1420878141.0  # as in waveform_reconstruction.py
    time_window = (-0.29, 0.37)  # merger at ~0.217
    slowdown = 100
    ppd_time_range = (0.06, 0.24)
    ppd_beta = 1 - 5 * (1 / 4) / 8  # as in waveform_reconstruction.py
    posterior_file = REPO / "GW250114_posteriors" / "SEOB_posterior_samples.h5"
    posterior_label = "bilby-SEOBNRv5PHM-reweighted"

    def __init__(self):
        self._generators = {}

    def load_posterior(self, approximant=None):
        if approximant in (None, self.approximant):
            with h5py.File(self.posterior_file, "r") as f:
                return f[self.posterior_label]["posterior_samples"][()]
        if approximant == "NRSur7dq4":
            return self._nrsur_result().posterior
        raise ValueError(approximant)

    @staticmethod
    def _nrsur_result():
        from bilby.gw.result import CompactBinaryCoalescenceResult

        os.environ["LAL_DATA_PATH"] = PAPER_DATA.as_posix()
        return CompactBinaryCoalescenceResult.from_hdf5(filename=NRSUR_RESULT)

    def load_interferometer(self, name):
        from bilby_pipe.utils import DataDump

        data = DataDump.from_pickle(DATA_DUMP)
        return next(ifo for ifo in data.interferometers if ifo.name == name)

    def whitened_data(self, ifo):
        return whitened_time_series(
            ifo.frequency_mask, ifo.whitened_frequency_domain_strain * np.sqrt(self.ppd_beta))

    def whiten(self, ifo, fd):
        """As the paper script: no band mask before the inverse FFT."""
        whitened_fd = fd / (ifo.amplitude_spectral_density_array * np.sqrt(ifo.duration / 4))
        return whitened_time_series(ifo.frequency_mask, whitened_fd) * np.sqrt(self.ppd_beta)

    def waveform_generator(self, ifo, approximant=None):
        import bilby

        from ..models import seobnrv5

        approximant = approximant or self.approximant
        if approximant == "SEOBNRv5PHM":
            return bilby.gw.waveform_generator.WaveformGenerator(
                duration=ifo.duration,
                sampling_frequency=ifo.sampling_frequency,
                start_time=ifo.start_time,
                frequency_domain_source_model=bilby.gw.source.gwsignal_binary_black_hole,
                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                waveform_arguments=dict(
                    waveform_approximant="SEOBNRv5PHM",
                    reference_frequency=seobnrv5.REFERENCE_FREQUENCY,
                    minimum_frequency=seobnrv5.WAVEFORM_MINIMUM_FREQUENCY,
                    maximum_frequency=seobnrv5.MAXIMUM_FREQUENCY,
                    lmax_nyquist=seobnrv5.LMAX_NYQUIST,
                    catch_waveform_errors=True,
                ),
            )
        if approximant == "NRSur7dq4":
            res = self._nrsur_result()
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

    def whitened_waveform(self, ifo, sample, approximant=None):
        key = (ifo.name, approximant)
        if key not in self._generators:
            self._generators[key] = self.waveform_generator(ifo, approximant)
        pols = self._generators[key].frequency_domain_strain(dict(sample))
        return self.whiten(ifo, ifo.get_detector_response(pols, sample))
