from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.psd import welch, interpolate
from pycbc.catalog import Merger
from pycbc.types.timeseries import Timeseries

def whiten(data, delta_t):
    
    h1 = Merger("GW190521").strain('L1')
    h1 = highpass_fir(h1, 15, 8)

    # Calculate the noise spectrum
    psd = interpolate(welch(h1), 1.0 / h1.duration)

    # whiten
    timeseries_pycbc = Timeseries(data, delta_t)
    white_strain = (timeseries_pycbc.to_frequencyseries() / psd ** 0.5).to_timeseries()

    return white_strain.data

if __name__ == '__main__':
    from coordinate_conversion import get_time_and_coordinates
    from maximum_likelihood_scenario import get_t_hp
    import matplotlib.pyplot as plt
    import numpy as np

    t, hp = get_t_hp()

    plt.plot(t, hp)
    plt.plot(t, whiten(hp, t[1]-t[0]))
    plt.show()