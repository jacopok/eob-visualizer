import numpy as np
import EOBRun_module
from maximum_likelihood_scenario import Event


def eob_to_adm(q_vec, p_vec, nu):
    """EOB to ADM functions
    stolen from TEOBResumS, specifically:
    https://bitbucket.org/eob_ihes/teobresums/src/b41a5d126166d899589ae9060ea2dc4f656d3b10/Python/Utils/hyp_priors_to_ADM.py#lines-90
    
    q and p should be of shape (2, N)
    
    """

    q2      = np.einsum('ij,ij->j', q_vec, q_vec);
    q       = np.sqrt(q2);
    q3      = q*q2;
    q4      = q*q3;
    p2      = np.einsum('ij,ij->j', p_vec, p_vec);
    p       = np.sqrt(p2);
    p3      = p*p2;
    p4      = p*p3; 
    qdotp   = np.einsum('ij,ij->j', q_vec, p_vec);
    qdotp2  = qdotp*qdotp;
    nu2     = nu*nu;

    cQ_1PN_q = nu*p2/2 - (1 + nu/2)/q;
    cQ_1PN_p = nu*qdotp; 
    cQ_2PN_q = -nu/8*(1 + nu)*p4 + 3/4*nu*(nu/2 - 1)*p2/q - nu*(2 + 5/8*nu)*qdotp2/q3 + (-nu2 + 7*nu - 1)/4/q2;
    cQ_2PN_p = qdotp*(nu*(nu - 1)/2*p2 + nu/2*(-5 + nu/2)/q);
    # coefficients for momenta
    cP_1PN_q = -(1 + nu/2)*qdotp/q3;
    cP_1PN_p = -nu/2*p2 + (1 + nu/2)/q;
    cP_2PN_q = qdotp/q3*(3/4*nu*(nu/2 - 1)*p2 + 3/8*nu2*qdotp2/q2 + (-3/2 + 5/2*nu - 3/4*nu2)/q);
    cP_2PN_p = nu*(1 + 3*nu)/8*p4 - nu/4*(1 + 7/2*nu)*p2/q + nu*(1 + nu/8)*qdotp2/q3 + (5/4 - 3/4*nu + nu2/2)/q2;

    # Put all together 
    Q_vec = q_vec + cQ_1PN_q*q_vec + cQ_1PN_p*p_vec + cQ_2PN_q*q_vec + cQ_2PN_p*p_vec; 
    P_vec = p_vec + cP_1PN_q*q_vec + cP_1PN_p*p_vec + cP_2PN_q*q_vec + cP_2PN_p*p_vec;

    return Q_vec, P_vec

def adm_q_to_single_body(q_vec, mass_ratio):
    
    # mass ratio > 1
    # mass_ratio = (- 2 * nu + 1 + np.sqrt(1 - 4 * nu) ) / (2 * nu)
    
    r_1 = q_vec / (mass_ratio + 1)
    r_2 = - q_vec / (1 / mass_ratio + 1)
    
    return r_1, r_2

@np.vectorize
def compute_radial_momentum(r, p_r_star, mass_ratio):
    A, B = EOBRun_module.eob_metricAB_py(r, mass_ratio);
    return p_r_star * np.sqrt(B/A);


def qp_from_dyn(dyn, mass_ratio):

    # initial coordinate and momenta
    time = dyn['t']
    r_eob = dyn['r']
    p_r_star_eob = dyn['Prstar']
    phi_eob = dyn['phi']
    p_phi_eob = dyn['Pphi']
    
    p_r_eob = compute_radial_momentum(r_eob, p_r_star_eob, mass_ratio)
    
    x_eob = r_eob * np.cos(phi_eob)
    y_eob = r_eob * np.sin(phi_eob)
    
    # covariant transformation laws
    # p_x = ∂φ/∂x p_φ + ∂r/∂x p_r
    # p_y = ∂φ/∂y p_φ + ∂r/∂y p_r
    p_x_eob = - y_eob / r_eob**2 * p_phi_eob + x_eob / r_eob * p_r_eob
    p_y_eob = x_eob / r_eob**2 * p_phi_eob + y_eob / r_eob * p_r_eob
    
    # contravariant transformation laws
    # p_x = ∂x/∂φ p_φ + ∂x/∂r p_r
    # p_y = ∂y/∂φ p_φ + ∂y/∂r p_r
    # p_x_eob = - r_eob * np.sin(phi_eob) * p_phi_eob + np.cos(phi_eob) * p_r_eob
    # p_y_eob = r_eob * np.cos(phi_eob) * p_phi_eob + np.sin(phi_eob) * p_r_eob
    
    # psq_1 = p_x_eob ** 2 + p_y_eob ** 2
    # psq_2 = p_r_eob ** 2 + p_phi_eob **2 / r_eob**2 
    
    # import matplotlib.pyplot as plt
    # plt.plot(time, psq_1, label='mine')
    # plt.plot(time, psq_2, label='computed')
    # plt.legend()
    # plt.show()
    
    q_eob = np.vstack((x_eob, y_eob))
    p_eob = np.vstack((p_x_eob, p_y_eob))
    
    return q_eob, p_eob

def get_time_and_coordinates(event: Event):
    
    q_eob, p_eob = qp_from_dyn(event.dyn, event.mass_ratio)
    q_adm, p_adm = eob_to_adm(q_eob, p_eob, event.red_mass_ratio)
    r_1, r_2 = adm_q_to_single_body(q_adm, event.mass_ratio)
    
    t = event.dyn['t']

    return t, r_1, r_2
