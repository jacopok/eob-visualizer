from coordinate_conversion import get_time_and_coordinates
from maximum_likelihood_scenario import constrained_prior
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

if __name__ == '__main__':

    constrained_prior.compute()
    plt.plot(constrained_prior.t, constrained_prior.hp)
    plt.plot(constrained_prior.t, constrained_prior.hp_highpass)
    # plt.plot(t, interp1d(t2, hp, kind='cubic')(t))
    plt.show()