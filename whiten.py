from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.psd import welch, interpolate
from pycbc.catalog import Merger
from pycbc.types.timeseries import TimeSeries

def whiten(data, delta_t):
    
    h1 = Merger("GW150914").strain('L1')
    h1 = highpass_fir(h1, 15, 8).resample(delta_t)

    # Calculate the noise spectrum
    psd = interpolate(welch(h1), 1.0 / h1.duration)

    # whiten
    timeseries_pycbc = TimeSeries(data, delta_t)
    frequencyseries = timeseries_pycbc.to_frequencyseries()
    print(frequencyseries.delta_f)
    print(psd.delta_f)
    white_strain = (frequencyseries / psd ** 0.5).to_timeseries()

    return white_strain.data

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from maximum_likelihood_scenario import constrained_prior
    
    