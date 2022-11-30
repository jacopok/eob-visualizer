from coordinate_conversion import qp_from_dyn, get_dynamics, M1, M2, eob_to_adm
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    dyn = get_dynamics()
    mass_ratio = M1 / M2 
    q, p = qp_from_dyn(dyn, mass_ratio)
    
    q_adm, p_adm = eob_to_adm(q, p, mass_ratio / (1 + mass_ratio)**2)
    
    t = dyn['t']
    plt.plot(np.sqrt(np.einsum('ij,ij->j', q_adm, q_adm)), label='r_adm computed')
    plt.plot(np.sqrt(np.einsum('ij,ij->j', q, q)), label='r_eob computed')
    plt.plot(dyn['r'], label='r_eob from dyn', ls=':')
    plt.yscale('log')
    plt.legend()
    plt.show()
    # plt.savefig('radii_check.png')