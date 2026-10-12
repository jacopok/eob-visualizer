from maximum_likelihood_scenario import Event
from plots import plot_event

if __name__ == '__main__':
    plot_event(
        Event(
            mass_1=81., 
            mass_2 = 52., 
            p_0_phi = 4.25, 
            E0_over_M = 1.001, 
            redshift = 0.918, 
            title='ManyPasses'
        ),
        zoom_levels=(250, 80)
    )
