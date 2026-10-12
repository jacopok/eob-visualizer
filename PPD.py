# from coordinate_conversion import get_time_and_coordinates
# from maximum_likelihood_scenario import Event
# import pandas as pd
import h5py

if __name__ == '__main__':

    fname = 'IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_nocosmo.h5'
    f = h5py.File(fname, 'r')
    data = f['C01:SEOBNRv4PHM']['posterior_samples']
    # available parameters:
    # [('mass_1', '<f8'), ('mass_2', '<f8'), ('spin_1x', '<f8'), ('spin_1y', '<f8'), ('spin_1z', '<f8'), ('spin_2x', '<f8'), ('spin_2y', '<f8'), ('spin_2z', '<f8'), ('chirp_mass', '<f8'), ('symmetric_mass_ratio', '<f8'), ('ra', '<f8'), ('dec', '<f8'), ('geocent_time', '<f8'), ('phase', '<f8'), ('iota', '<f8'), ('psi', '<f8'), ('luminosity_distance', '<f8'), ('Npts', '<f8'), ('log_likelihood', '<f8'), ('p', '<f8'), ('ps', '<f8'), ('neff', '<f8'), ('total_mass', '<f8'), ('mass_ratio', '<f8'), ('chi_eff', '<f8'), ('chi_p', '<f8'), ('tilt_1', '<f8'), ('tilt_2', '<f8'), ('phi_12', '<f8'), ('theta_jn', '<f8'), ('phi_jl', '<f8'), ('redshift', '<f8'), ('comoving_distance', '<f8'), ('inverted_mass_ratio', '<f8'), ('mass_1_source', '<f8'), ('mass_2_source', '<f8'), ('total_mass_source', '<f8'), ('chirp_mass_source', '<f8'), ('a_1', '<f8'), ('a_2', '<f8'), ('phi_1', '<f8'), ('phi_2', '<f8'), ('tilt_1_infinity_only_prec_avg', '<f8'), ('tilt_2_infinity_only_prec_avg', '<f8'), ('chi_p_2spin', '<f8'), ('spin_1z_infinity_only_prec_avg', '<f8'), ('spin_2z_infinity_only_prec_avg', '<f8'), ('chi_eff_infinity_only_prec_avg', '<f8'), ('chi_p_infinity_only_prec_avg', '<f8'), ('beta', '<f8'), ('psi_J', '<f8'), ('final_spin', '<f8'), ('peak_luminosity', '<f8'), ('final_mass', '<f8'), ('cos_tilt_1_infinity_only_prec_avg', '<f8'), ('cos_tilt_2_infinity_only_prec_avg', '<f8'), ('cos_tilt_1', '<f8'), ('cos_tilt_2', '<f8'), ('final_mass_source', '<f8'), ('radiated_energy', '<f8'), ('cos_theta_jn', '<f8'), ('viewing_angle', '<f8'), ('cos_iota', '<f8')]

    
    evt = Event(mass_1=81., mass_2 = 52., p_0_phi = 4.24, E0_over_M = 1.014, redshift = 0.918)

    # t, r1, r2 = get_time_and_coordinates(evt)