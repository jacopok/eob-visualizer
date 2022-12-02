from EOBRun_module import EOBRunPy
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from scipy import signal
# from whiten import whiten
import numpy as np

class Event:
    
    def __init__(
        self, 
        mass_1, 
        mass_2, 
        redshift, 
        p_0_phi, 
        E0_over_M, 
        title='',
        chi_p=0.,
        chi_eff=0.,
        hyperbolic=True, 
        precessing=False):
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.redshift = redshift
        self.p_0_phi = p_0_phi
        self.E0_over_M = E0_over_M
        self.title = title
        self.chi_p = chi_p
        self.chi_eff = chi_eff
        self.hyperbolic = hyperbolic
        self.precessing = precessing
        
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
        par = {
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
            'ecc'                : 0.,              #Eccentricity. Default = 0.
            'ode_tmax'           : 20e4,
            'output_hpc'         : 0,                 #output waveform. Default = 1.
        }
        if self.hyperbolic:
            par |= {
            'j_hyp'              : self.p_0_phi,                 #J_hyp. Default = 0.
            'r_hyp'              : 1500.,                 #r_hyp. Default = 0.
            'H_hyp'              : self.E0_over_M,                #H_hyp. Default = 0.
            }
        if self.precessing:
            A = 2 + 3 * self.mass_ratio / 2
            par |= {
                'chi1z': self.chi_eff,
                'chi2z': self.chi_eff,
                'chi1y': 0.,
                'chi2y': 0.,
                'chi1x': self.chi_p / A,
                'chi2x': 0.,
            }
            
        return par
    
    def compute(self):
        self.t, self.hp, _, _, self.dyn = EOBRunPy(self.par)
        self.compute_highpass()

    @property
    def M_in_seconds(self):
        return 4.92549095e-06 * self.total_mass * (1+self.redshift)

    def compute_highpass(self):
        srate = 1/(self.t[1] - self.t[0]) / self.M_in_seconds
        
        delta_t = 2**-int(np.log2(srate))
        
        sos = signal.butter(6, 50, 'hp', fs=srate, output='sos')
        self.hp_highpass = signal.sosfilt(sos, self.hp)
        # self.hp_highpass = whiten(self.hp, delta_t)

# maximum likelihood scenario, unconstrained prior
# 254.42661600355004 1.0303276645477257 4.240438656074215 1.0142366764445196 4057.9491147309354
# M, q, J_hyp, H_hyp, DL

# 

constrained_prior = Event(mass_1=81., mass_2 = 52., p_0_phi = 4.24, E0_over_M = 1.014, redshift = 0.918, title='ConstrainedPrior')
unconstrained_prior = Event(mass_1=77.92626388591152, mass_2 = 75.632506596936, p_0_phi = 4.240438656074215, E0_over_M = 1.0142366764445196, redshift = 0.65686802,title='UnconstrainedPrior')

if __name__ == '__main__':

    evt = Event(mass_1=81., mass_2 = 52., p_0_phi = 4.24, E0_over_M = 1.014, redshift = 0.918)

    evt.compute()
    dyn = evt.dyn
    # plt.loglog(dyn['t'], dyn['r'])
    # plt.show()
    print(dyn.keys())
