#!/usr/bin/python

"""
Script to generate parameter space 2d plots (E,j,N)

 * Works in parallel with multiprocessing
 * Generates a .npz files and the plots

NOTES

 * The encounters are computed as number of peaks in the orbital frequency 
   A minimum 'height' is necessary to define a 'peak'. 
 * Initial data parameters can be sampled linearly or randomly

SB 08/2020
"""

# Standard python imports
import glob, itertools, matplotlib, matplotlib.pyplot as plt, multiprocessing as mp, numpy as np, os, sys, time
from functools          import partial
from matplotlib         import rc, colors
from scipy.interpolate  import LinearNDInterpolator

try:
    from scipy.signal import find_peaks
    FIND_PEAKS = 1
except:
    print("update scipy?") 
    FIND_PEAKS = 0
matplotlib.use('Agg') # avoid DISPLAY error
rc('text', usetex=True)

# EOB imports
# Set your path to the EOB code
#sys.path.insert(1, '/home/sebastiano.bernuzzi/teobresums/Python')
# Check that the script is sourcing the local installation.
# print(EOBRun_module.__file__)
import EOBRun_module

#
# Set global parameters
#

number_of_processes = 4             # parallel processes
test                = 0             # dummy test for pool/map logic
DEBUG               = 0             # print parameters values

# Initial conditions parameters ranges 
jmin, jmax          = 1.15, 1.55    # Bounds for j in units of pphi_LSO
Emax                = []            # if not empty, force to draw the given maximum energy
jsample             = 100           # no samples in j 
Esample             = 100           # no samples in E
j_sample_randomly   = 1        
E_sample_randomly   = 1        

# Hard-coded dynamics-related parameters
peak_height         = 0.05 #0.035   # height to define a peak
eob_initial_radius  = 1500.#10000.  # r0

# Binary parameters ranges to consider
range_q    = np.array([1,2,4,8])

range_chi1 = np.linspace(-0.5,0.5,5)
range_chi2 = np.linspace(-0.5,0.5,5)

# FIXME: need a different way to fix the seed in the multi-threading case.
np.random.seed(666)
runtag = 'spin_HMs_no32_NQC_eccentric_b41a5d1_2022-10-28'

# ##############################################################
# DO NOT CHANGE BELOW HERE UNLESS YOU KNOW WHAT YOU ARE DOING
# ##############################################################

#
# Utilities 
#

def modes_to_k(modes):

    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]

def drawn(a,b,N):
    """
    Draw randomly N samples in [a,b), return array
    """
    return (b - a) * np.random.sample(N) + a

def combine_parameters(pars):
    """
    Given a dictionary of parameters and their values in lists, 
    Generate all the combinations (return the key list for the ordering)
    """
    x, k = [], []
    for key in pars.keys(): 
        x.append(pars[key])
        k.append(key)
    x = list(itertools.product(*x))
    return x, k

def D4(f, dx):
    """
    4th order drvt 
    """
    n = len(f)
    df = np.zeros_like(f)
    f = np.array(f)
    oo12dx  = 1./(12*dx);
    i = np.arange(2, n-2)
    df[i] = (8.*(f[i+1]-f[i-1]) - f[i+2] + f[i-2])*oo12dx
    i = 0
    df[i] = (-25.*f[i] + 48.*f[i+1] - 36.*f[i+2] + 16.*f[i+3] - 3.*f[i+4])*oo12dx
    i = 1
    df[i] = (-3.*f[i-1] - 10.*f[i] + 18.*f[i+1] - 6.*f[i+2] + f[i+3])*oo12dx
    i = n-2
    df[i] = - (-3.*f[i+1] - 10.*f[i] + 18.*f[i-1] - 6.*f[i-2] + f[i-3])*oo12dx
    i = n-1
    df[i] = - (-25.*f[i] + 48.*f[i-1] - 36.*f[i-2] + 16.*f[i-3] - 3.*f[i-4])*oo12dx
    return df

def output_fname(q,chi1,chi2):
    """
    Output filename labelling mass ratio and spin
    """
    return "span_dyn_data_{}/q{:03d}_A{:+04d}_B{:+04d}".format(runtag,int(q),int(100*chi1),int(100*chi2))    

def build_colormap(old_cmp_name, val, clr):
    """
    Outputs a colormap similar to old_cmp_name, but with a new specified RGBA color
    associated to val.
    """
    from matplotlib.colors import ListedColormap

    old_cmp   = matplotlib.cm.get_cmap(old_cmp_name)
    newcolors = old_cmp(np.linspace(0, 1, 512)) # used for q=128
    #newcolors = old_cmp(np.linspace(0, 1, 256)) # default
    val_idx   = int(len(newcolors)*val) +4 #FIXME
    newcolors[val_idx-2:val_idx+2, :] = clr
    newcmp = ListedColormap(newcolors)

    return newcmp

def scatterplot_EjN(E,j,N,fname=None, qval=0, chi1val=0.0, chi2val=0.0, EM=None,JM=None):
    
    """
    Scatter plot (E,j,N)
    """
    fig, ax = plt.subplots()
    map_len = max(N)

    # I have no idea why I need to change the behaviour for this range
    # possibly because the maximum number of encounters shift the map in such a way that the magenta disappears
    if(qval>=8 and qval<=32):
        cmap = plt.cm.jet  # define the colormap
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # force the second color entry (scaled to the map length) to be magenta
        cmaplist[1*int(cmap.N/map_len)] = (1,0,1,1)
        cmp = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist,  cmap.N)
    else:
        p2  = (2.-min(N))/map_len
        cmp = build_colormap('jet', p2, [1,0,1,1])

    sp = ax.scatter(j, E, s=2.1, c=N, marker = 'o', cmap=cmp)
    ax.grid(alpha=0.5)
    ax.set_xlabel(r'$p_{\varphi}^0$',fontsize=17)
    ax.set_ylabel(r'$E_0/M$',fontsize=17)

    if qval == 128:
        ax.set_ylim(min(E)-0.0002, max(E) + 0.0002) 
        cbar = fig.colorbar(sp, ticks=range(1, map_len)[::11])  # when there are too many points (q=128)
    elif qval==64:
        ax.set_ylim(min(E)-0.0004, max(E) + 0.0004)
        cbar = fig.colorbar(sp, ticks=range(1, map_len)[::7])  # when there are too many points (q=64)
    elif qval==32:
        ax.set_ylim(min(E)-0.0008, max(E) + 0.0008)
        cbar = fig.colorbar(sp, ticks=range(1, map_len)[::4])  # when there are too many points (q>=32)    
    else:
        if qval==16:
            ax.set_ylim(min(E)-0.0015, max(E) + 0.0015)
        cbar = fig.colorbar(sp, ticks=range(1, map_len))        # default

    cbar.set_label('$N$ (number of $\Omega$ peaks)',fontsize=17)#, rotation=270)
    if(qval and chi1val and chi2val):
        ax.text(0.45, 0.8, r'$q =$ %s, $\chi_1 =$ %s, $\chi_2 =$ %s' %(qval, chi1val, chi2val), fontsize=17, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

    ax.text(0.75, 0.3, r'Scattering', color='white', fontsize=17, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax.text(0.25, 0.6, r'Capture: $E_0>E_0^{\rm max}$', color='black', fontsize=17, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    plt.tight_layout()

    if((JM is not None) and (EM is not None)):
        ax.scatter(JM, EM, marker='*',s=100, color='orange', label=r'max $\mathcal{L}$')
        ax.axvline(JM, color='orange', linestyle='--')
        ax.axhline(EM, color='orange', linestyle='--')
    if fname:
        fig.savefig(fname+"_scatter_magenta.pdf",dpi=100)
        fig.savefig(fname+"_scatter_magenta.png")        
        plt.close(fig)
    else:
        plt.show()        

def plot_Nfraction(C,fname=None):
    """
    Plot N(symmetric mass ratio) and C(spin) 
    Use dictionary data from combine_encounters()
    """
    Nmax = C['Nmax']
    q = C['q']
    s = C['s']
    nu = q/(1.+q)**2

    # N=0 --> N=1     
    C[1] = C[1] + C[0]
    
    plt.style.use('seaborn-colorblind')   #more styles here https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
    plt.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots()
    ax2 = ax.twiny()
    for n in range(2, Nmax):
        if n == 2:
            ax.plot(nu,C[n],'k',linewidth=1.5, marker='o', label=r'$N=$ %s'%(n))
        else:
            ax.plot(nu,C[n], linewidth=1.5, marker='o', label=r'$N=$ %s'%(n))
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    tks = [round(jmax*EOBRun_module.pph_lso_orbital_py(i),2) for i in ax.get_xticks()]
    ax2.set_xticklabels(tks)
    ax2.set_xlabel(r'${\rm max}(p_\varphi^0)$')

    ax.grid()
    ax.set_xlabel(r'$\nu$')
    ax.set_ylabel(r'$Y_N$ Fraction of mergers with $N$ encounters')
    ax.set_yscale('log')
    ax.legend()
    if fname:
        fig.savefig(fname+"_q.pdf",dpi=100)
        fig.savefig(fname+"_q.png")
        plt.close(fig)
    else:
        plt.show()        

    fig, ax = plt.subplots()
    for n in range(Nmax):
        ax.plot(s,C[n],label="$n={}$".format(n))
    ax.grid()
    ax.set_xlabel('$S$ Symmetric spin parameter')
    ax.set_ylabel('$Y_n$ Fraction of mergers with $n$ encounters')
    ax.set_yscale('log')
    ax.legend()
    if fname:
        fig.savefig(fname+"_spin.pdf",dpi=100)
        fig.savefig(fname+"_spin.png")
        plt.close(fig)
    else:
        plt.show()        

#
# EOB 
#

def symm_spin_par(chi1, chi2, q, M=1):
    """
    Symmetric spin parameter
    hat{S} = (S_1 + S_2)/M^2 = chi1 X_1^2 + chi2 X_2^2
    """
    m2 = M/(q+1.)
    m1 = q*m2
    X1 = m1/M
    X2 = m2/M
    return chi1 * X1**2 + chi2 * X2**2

def Hamiltonian(r, pph, nu):
    """
    Circularized EOB Hamiltonian: nonspinning
    """
    A, dA, d2A  = EOBRun_module.eob_metric_A5PNlog_py(r, nu)
    Heff0       = np.sqrt(A*(1.+(pph/r)**2))
    E0          = np.sqrt(1. + 2.*nu*(Heff0-1.))

    return E0

def SpinHamiltonian(r, pph, q, chi1, chi2):
    """
    Circularized EOB Hamiltonian: spinning
    """
    hatH = EOBRun_module.eob_ham_s_py(r, q, pph, 0., chi1, chi2)
    nu   = q/(1+q)**2
    E0   = nu*hatH[0]

    return E0

def EnergyLimitsSpin(rmx, q, pph_hyp, chi1, chi2, N=100000):
    """
    Determine min/max energy and inflection point:spinning
    """

    # r_min is the smallest radius at which the potenatial can peak.
    # For |chi|<0.5 and pphi < 1.55 * pphi_LSO, r_min > 1.5 (equal mass case, where r_min is smaller), 
    # so we set 1.3 to be conservative and ignore nans.

    rmin = 1.3

    x    = np.linspace(rmin,rmx, N)
    dx   = x[1]-x[0]
    E0   = [SpinHamiltonian(xi, pph_hyp, q, chi1, chi2) for xi in x]
    Emin = SpinHamiltonian(rmx, pph_hyp, q, chi1, chi2)
    # Determine the max energy allowed. For large q, A will go below zero, so ignore those values by removing nans.
    Emx  = np.nanmax(E0)     

    # Determine the inflection point of the potential energy E0.
    try:
        dE0   = D4(E0,  dx) 
        d2E0  = D4(dE0, dx)
        jflex = np.where(d2E0 >= 0)[0][0] #FIXME: this needs fixing
        Einfl = E0[jflex]
    except:
        Einfl = 0.0
        jflex = 0.0 

    return Emin, Emx, Einfl, jflex


def EnergyLimits(rmx, nu, pph_hyp, N=100000):
    """
    Determine min/max energy and inflection point: nonspinning
    """
    # This is a minimum radius (valid across the whole [q,pphi] parameter space)
    # to set the range in which the peak of the potential should be searched.
    # The minimum is achieved for comparable masses and large pphi.
    # In the non-spinning case, r_min never goes below 1.8, so we set 1.5 to be conservative
    # and by ignoring nans, we discard values for which A becomes negative.
    rmin = 1.5
    x    = np.linspace(rmin,rmx, N)
    dx   = x[1]-x[0]
    E0   = [Hamiltonian(xi, pph_hyp, nu) for xi in x]
    Emin = Hamiltonian(rmx, pph_hyp, nu)
    # Determine the max energy allowed. For large q, A will go below zero, so ignore those values by removing nans.
    Emx  = np.nanmax(E0)

    # Determine the inflection point of the potential energy E0.
    try:
        dE0   = D4(E0,  dx) 
        d2E0  = D4(dE0, dx) 
        jflex = np.where(d2E0 >= 0)[0][0] #FIXME: this needs fixing
        Einfl = E0[jflex]
    except:
        Einfl = 0.0
        jflex = 0.0 

    return Emin, Emx, Einfl, jflex

def run_eob(data,keys,parameters):
    """
    Run EOB
    Add the result to the tuple
    """
    # Set current initial data parameters
    parameters['j_hyp'] = data[0] # the index must match the way data are packed!
    parameters['H_hyp'] = data[1] # the index must match the way data are packed!
    try:
        # Run the wf generator
        t, hp, hc, hlm, dyn = EOBRun_module.EOBRunPy(parameters)
    except Exception:
        print('Waveform generation failed: {} = {}',format(keys,data))
        N = None
        # IF HERE, RUN FAILED
        #TODO: understand how to catch this exception within multiprocess pool/map
    # Compute no peaks
    Momg_o = dyn['MOmega_orb']
    if FIND_PEAKS:
        pks, _  = find_peaks(Momg_o, height=parameters['peak_height']) 
        N = len(pks)
    else:
        N = None
        print("you really need scipy/find_peaks")
        sys.exit()

    if (np.all((Momg_o == 0))):
        # Here EOB code actually failed
        # but with a workaround in the EOB code we do not raise the failure,
        # rather it returned all zeros
        N = None
    if(DEBUG):
        print('E: {}\tJ: {}\tN: {}'.format(parameters['H_hyp'], parameters['j_hyp'], N))

    # Add the data to the tuple
    data = data + (N,)
    return data 

#
# Main routines
#

def span_dynamics(run,data,keys,parameters=None,fname=None,nproc=1,plotme=True):
    """
    Calculate the dynamics spanning all the combination of the initial
    data listed in inipars
    """    
    # Run a pool of nproc 
    pool = mp.Pool(processes=nproc)
    task = partial(run, keys=keys, parameters=parameters)
    results = pool.map(task, data)
    pool.close() 
    pool.join()

    if fname:
        # Unpack data
        data = list(map(list,zip(*results)))
        # if(DEBUG):
        #     print('\nJ0 \t E0 \t  N\n')
        #     for i in range(len(data[0])):
        #         print('{:.6f} {:.6f} {}'.format(data[0][i], data[1][i], data[2][i]))

        for idx,k in enumerate(keys):
            if k=='H_hyp': E = data[idx]
            if k=='j_hyp': j = data[idx]
            if k=='N': N = data[idx]        
        # Save in binary file
        np.savez(fname, E=E,j=j,N=N, q=parameters['q'],chi1=parameters['chi1'],chi2=parameters['chi2'])
        # Count encounters
        nn, counts = np.unique(N, return_counts=True)
        np.savez(fname+"_histo",nn=nn,counts=counts, q=parameters['q'],chi1=parameters['chi1'],chi2=parameters['chi2'])        
        if plotme:
            # Plot
            fig, ax = plt.subplots()
            sp = ax.scatter(j,E,s=20,c=N, marker = 'o')
            ax.grid()
            ax.set_xlabel('$j$, Initial angular momentum')
            ax.set_ylabel('$E$, Initial energy')
            cbar = fig.colorbar(sp)
            cbar.set_label('$N$, number of encounters')
            fig.savefig(fname+".pdf",dpi=100)  
            fig.savefig(fname+".png")            
            plt.close(fig)

    return

def count_Nfraction(fnames,Nmax=6,
                    output_fname=None, output_fig=None):
    """
    Combine histograms for encounters from several runs 
    Postprocess data in all files 'folder/fname'
    """
    size = len(fnames)
    # Create dictionary for encounters
    C = {}
    for i in range(Nmax):   #FIXME: rough fix for bug
       C[i] = np.zeros(size)
    
    C['q'] = np.empty(size) # mass ratio
    C['s'] = np.empty(size) # symm spin
    C['Nmax'] = Nmax
    for (idx,f) in enumerate(fnames):
        with np.load(f) as data:
            nn = data['nn']
            cn = data['counts']
            C['q'][idx] = data['q']
            C['s'][idx] = symm_spin_par(data['chi1'],data['chi2'],data['q'])# M=1
        totc = np.sum(cn)
        for i in range(Nmax):
            if i in nn:
                C[i][idx] = float(cn[np.where(nn == i)])/totc #fraction of total counts
    if output_fname:        
        np.savez(output_fname,C=C,Nmax=Nmax)
    if output_fig:
        plot_Nfraction(C,fname=output_fig)
    return

def main():
    """
    SPAN EOB DYNAMICS
    """    

    if not os.path.exists('span_dyn_data_{}'.format(runtag)):
        os.makedirs('span_dyn_data_{}'.format(runtag))

    # All modes, except (3,2) for which we have seen pathological wfs
    modes   = [[2,2], [2,1], [4,3], [3,3], [4,4], [5,5]]
    k_modes = modes_to_k(modes)

    # Build pphi interpolator for spinning case
    q_vec_interp, chi1_vec_interp, chi2_vec_interp, pphi_LSO_vec_interp = np.genfromtxt('pphi_LSO_spinning/Data/q_chi1_chi2_pphi_LSO.txt', unpack=True)    
    coords_interp = np.column_stack((q_vec_interp, chi1_vec_interp, chi2_vec_interp))
    interp_pphi   = LinearNDInterpolator(coords_interp, pphi_LSO_vec_interp)       

    # EOB parameters
    # -------------------
    r0  = eob_initial_radius         
    eobpars = {

               # Source parameters that won't be varied
               'M'                  : 1.,
               'LambdaAl2'            : 0.,
               'LambdaBl2'            : 0.,
               'ecc'                : 0.0,               #Eccentricity. Default = 0.
               'r_hyp'              : r0,

                # NQC: default they are on. Uncomment below to disabilitate
                # 'nqc'                : 2,                 # NQC=2 --> Set manually
                # 'nqc_coefs_hlm'      : 0,                 # No NQC in wave 
                # 'nqc_coefs_flx'      : 0,                 # No NQC in flux

                # Ode-related parameters
               'dt'                 : 0.5,
               'dt_interp'          : 0.5,
               'ode_tmax'           : 20e5,
               'ode_tstep_opt'      : "yes",                 #fixing uniform or adaptive. Default = 1 
               'srate_interp'       : 4096.,             #srate at which to interpolate. Default = 4096.
               'interp_uniform_grid': "no",                 #interpolate mode by mode on a uniform grid. Default = 0 (no interpolation)

                # I/O parameters
               'arg_out'            : "yes",                 #this gives the output
               'use_geometric_units': "no",                 #output quantities in geometric units. Default = 1
               'domain'             : 0,                 #Set 1 for FD. Default = 0
               'use_mode_lm'        : k_modes,           #List of modes to use in h+,hx/output through EOBRunPy
               'output_hpc'         : "no"
    }
    # add some extra parameters for this specific application
    eobpars['peak_height'] = peak_height
    # Loop on BBH pars 
    # -------------------
    for chi2 in range_chi2:
        for chi1 in range_chi1:
            if chi1 != chi2:
                continue
            for q in range_q:
                if os.path.exists(output_fname(q,chi1,chi2) + '.npz'):
                    print(f'Skipping {q}, {chi1}, {chi2}')
                    continue
                else:
                    print(f'path {output_fname(q,chi1,chi2)} does not exist, running')
                if(np.abs(chi1)>0.5 or np.abs(chi2)>0.5):
                    raise ValueError("r_min and pphi_LSO were not tested for spins greater than 0.5, aborting.")

                print('\nq: {}, spin1: {}, spin2: {}\n'.format(q, chi1, chi2))

                # Set EOB pars for this BBH
                eobpars['q'] = q
                eobpars['chi1'] = chi1
                eobpars['chi2'] = chi2            
                # Sample j in [jmin,jmax]
                if j_sample_randomly:
                    range_j = drawn(jmin,jmax,jsample) 
                else:
                    range_j = np.linspace(jmin,jmax,jsample)

                # Sample E between Emin and Emax and generate the data ...
                nu = q/(1.+q)**2

                data = [] # this is a stupid way, there is a better one
                for x in range_j:

                    if (np.abs(chi1)>1e-3) or (np.abs(chi2) > 1e-3):
                        pphi_lso = interp_pphi(q, chi1, chi2)
                    else:
                        pphi_lso = EOBRun_module.pph_lso_orbital_py(nu)

                    J = x * pphi_lso

                    if (np.abs(chi1)>1e-3) or (np.abs(chi2) > 1e-3):
                        Emn, Emx, Einfl, jflex = EnergyLimitsSpin(r0,q,J,chi1,chi2)
                    else:
                        Emn, Emx, Einfl, jflex = EnergyLimits(r0,nu,J)

                    if(DEBUG):
                        print('J0: {:.6f}, E_min: {:.6f}, E_infl: {:.6f}, E_max: {:.6f}, pphi_lso: {:.6f}'.format(J,Emn,Einfl,Emx,pphi_lso))

                    if Emax:
                        # Force to draw to this Emax
                        Emn = Emax            
                        print('Force Emin=Emax={}'.format(Emax))
                    if E_sample_randomly:
                        Elims = drawn(Emn, Emx, Esample)
                    else:
                        Elims = np.linspace(Emn, Emx, Esample)
                    # The first point is excluded because is below the minimum value of the potential and the orbit is not allowed.
                    for e in Elims[1:-2]:
                        data.append((J,e))
                        if(DEBUG):
                            print('{}     {} {} {}'.format(data[-1],Emn,Emx,J))

                # ... now data is a list containing all the possible
                # combination of initial data parameters as tuples (J,E)
                
                # Keys for the data
                keys = ['j_hyp', 'H_hyp', 'N']            
                # Do it
                span_dynamics(run_eob,data,keys,parameters=eobpars,
                              fname=output_fname(q,chi1,chi2),
                              nproc=number_of_processes)  

                with np.load(output_fname(q,chi1,chi2)+'.npz') as data_plot:
                    E,j,N = data_plot['E'], data_plot['j'], data_plot['N']
                    # set N = 0 to N = 1
                    N = [n if n !=0 else 1 for n in N]
                    # plot
                    scatterplot_EjN(E,j,N,fname=output_fname(q,chi1,chi2), qval=q, chi1val=chi1, chi2val=chi2)

    # Count encounters
    # -------------------
    fnin = glob.glob('span_dyn_data_{}/*_histo.npz'.format(runtag))
    fnout = "span_dyn_data_{}/summary_fraction_encounters".format(runtag)
    count_Nfraction(fnin, 
                    Nmax=6,
                    output_fname=fnout, 
                    output_fig=fnout)
    return

#
# Test run
#
    
def run_test(data,keys,parameters):
    """
    Run a test
    Add the result to the tuple
    """
    res = np.rint(np.abs(data[1]-data[0])) 
    data = data + (res,)
    print("test run {} {}".format(keys,data))
    return data 

def make_test():
    """
    Dummy test
    """
    # Randomly sample initial data parameters
    # The following simply span a square
    # To start with more complicated distributions see e.g.
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.triangular.html
    E = drawn(1,2, Esample)
    j = drawn(3,4, jsample)
    # Pack them into a dictionary
    inipars = {'H_hyp': E, 'j_hyp': j}                
    # Generate combinations
    data, keys = combine_parameters(inipars)
    ##print(data) #DEBUG    
    # Extend keys
    keys.append('N')    
    # Dummy
    eobpars = {}
    eobpars['q'] = 42
    eobpars['chi1'] = 42
    eobpars['chi2'] = 42
    # Do it
    span_dynamics(run_test,data,keys,eobpars,
                  fname="test",
                  nproc=number_of_processes)
    return

if __name__ == "__main__": 

    N_tot = len(range_q)*len(range_chi1)*len(range_chi2)*jsample*Esample
    print('Total number of iterations: {}'.format(N_tot))

    if test: make_test()
    else: main()
