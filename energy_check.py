from coordinate_conversion import get_dynamics
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    dyn = get_dynamics()

    plt.plot(dyn['t'], dyn['E'])
    plt.show()