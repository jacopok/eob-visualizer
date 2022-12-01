from coordinate_conversion import get_time_and_coordinates
from maximum_likelihood_scenario import get_t_hp
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def spectrogram(t, hp):
    srate = 1 / (t[1] - t[0])
    f, t, Sxx = signal.spectrogram(hp, srate, nperseg=64)
    plt.pcolormesh(t, f, Sxx, shading="gouraud")
    plt.ylabel("Frequency [1/M]")
    plt.xlabel("Time [M]")

if __name__ == '__main__':
    t, hp = get_t_hp()

    spectrogram(t, hp)
    plt.show()
