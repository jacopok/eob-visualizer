from EOBRun_module import EOBRunPy
import matplotlib.pyplot as plt

M1 = 81.
M2 = 52.
Deff = 6100.
iota = 0.
p_0_phi = 4.24
E0_over_M = 1.014

def get_dynamics():
    # median values for constrained energy prior

    t, hp, hc, hlm, dyn = EOBRunPy({
        'M'                  : M1+M2,
        'q'                  : M1/M2,
        'distance'           : Deff,
        'inclination'        : iota,
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
        'use_geometric_units': 0,                 #output quantities in geometric units. Default = 1
        # 'r0':r,
        'interp_uniform_grid': 1,                 #interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)
        'ecc'                : 0.18,              #Eccentricity. Default = 0.
        'j_hyp'              : p_0_phi,                 #J_hyp. Default = 0.
        'r_hyp'              : 1500.,                 #r_hyp. Default = 0.
        'H_hyp'              : E0_over_M,                #H_hyp. Default = 0.
        'ode_tmax'           : 20e4,
        'output_hpc'         : 0,                 #output waveform. Default = 1.
    })
    return dyn

if __name__ == '__main__':

    dyn = get_dynamics()
    plt.loglog(dyn['t'], dyn['r'])
    plt.show()