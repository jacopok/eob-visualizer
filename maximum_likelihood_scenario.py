from EOBRun_module import EOBRunPy
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo

class Event:
    
    def __init__(self, mass_1, mass_2, redshift, p_0_phi, E0_over_M, title):
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.redshift = redshift
        self.p_0_phi = p_0_phi
        self.E0_over_M = E0_over_M
        self.title = title
        
    @property
    def mass_ratio(self):
        return self.mass_1 / self.mass_2

    @property
    def total_mass(self):
        return self.mass_1 + self.mass_2

    @property
    def red_mass_ratio(self):
        return self.mass_ratio / (1 + self.mass_ratio)**2
    
    @property
    def par(self):
        return {
            'M'                  : self.total_mass,
            'q'                  : self.mass_ratio,
            'distance'           : cosmo.luminosity_distance(self.redshift).value,
            'inclination'        : 0,
            'chi1'               : 0.,
            'chi2'               : 0.,
            'Lambda1'            : 0.,
            'Lambda2'            : 0.,
            'dt'                 : 0.5,
            'dt_interp'          : 0.5,
            'domain'             : 0,                 #Set 1 for FD. Default = 0
            'arg_out'            : 1,                 #Output hlm/hflm. Default = 0
            'nqc'                : 2,
            'nqc_coefs_hlm'      : 0,
            'nqc_coefs_flx'      : 0,
            'use_mode_lm'        : [1],               #List of modes to use/output through EOBRunPy
            'output_lm'          : [1],               #List of modes to print on file
            'output_dynamics'    : 0,                 #output of the dynamics
            'ode_tstep_opt'      : 1,                 #fixing uniform or adaptive. Default = 1 
            'srate_interp'       : 3000000.,          #srate at which to interpolate. Default = 4096.
            'use_geometric_units': 1,                 #output quantities in geometric units. Default = 1
            # 'r0':r,
            'interp_uniform_grid': 1,                 #interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
            'ecc'                : 0.18,              #Eccentricity. Default = 0.
            'j_hyp'              : self.p_0_phi,                 #J_hyp. Default = 0.
            'r_hyp'              : 1500.,                 #r_hyp. Default = 0.
            'H_hyp'              : self.E0_over_M,                #H_hyp. Default = 0.
            'ode_tmax'           : 20e4,
            'output_hpc'         : 0,                 #output waveform. Default = 1.
        }
    
    def compute(self):
        self.t, self.hp, _, _, self.dyn = EOBRunPy(self.par)

# median values for constrained energy prior

constrained_prior = Event(mass_1=81., mass_2 = 52., p_0_phi = 4.24, E0_over_M = 1.014, redshift = 0.918, title='ConstrainedPrior') 
unconstrained_prior = Event(mass_1=85., mass_2 = 59., p_0_phi = 4.18, E0_over_M = 1.014, redshift = 0.75,title='UnconstrainedPrior')

if __name__ == '__main__':

    evt = Event(mass_1=81., mass_2 = 52., p_0_phi = 4.24, E0_over_M = 1.014, redshift = 0.918)

    evt.compute()
    dyn = evt.dyn
    # plt.loglog(dyn['t'], dyn['r'])
    # plt.show()
    print(dyn.keys())
