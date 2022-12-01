from coordinate_conversion import get_time_and_coordinates
from maximum_likelihood_scenario import get_t_hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

if __name__ == '__main__':
    t, r_1, r_2 = get_time_and_coordinates()
    t2, hp = get_t_hp()

    plt.plot(t2, hp)
    plt.plot(t, interp1d(t2, hp, kind='cubic')(t))
    plt.show()