from maximum_likelihood_scenario import Event
from plots import plot_event

if __name__ == '__main__':
    plot_event(
        Event(
            mass_1=90., 
            mass_2 = 66., 
            p_0_phi = None, 
            E0_over_M = None, 
            chi_p = .72,
            chi_eff = -0.5,
            redshift = 0.75,
            title='Quasicircular', 
            hyperbolic=False,
            precessing=True,
        ),
        zoom_levels=(12, 12)
    )