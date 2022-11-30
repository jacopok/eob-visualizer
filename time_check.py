from coordinate_conversion import get_time_and_coordinates
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    t, r_1, r_2 = get_time_and_coordinates()

    # plt.hist(np.log10(np.ediff1d(t)), bins=100)
    plt.plot(t)
    plt.show()