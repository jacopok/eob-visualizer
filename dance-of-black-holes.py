import numpy as np

def polarizations_simple(r, phi, rdot, phidot, M=1, R=1, eta=0.25, iota=0.):
    
    hplus = eta * M / R * (
        (1+np.cos(iota)**2) * (
            (rdot**2 - r**2 *phidot**2 - M/r) *np.cos(2*phi) 
        - 2 * rdot * r * phidot * np.sin(2*phi)
        ) 
    - np.sin(iota)**2 * (
        rdot**2 + r**2 * phidot**2 - M/r 
    )
    )
    hcross = 2 * eta * M / R * np.cos(iota) * (
        2 * r * rdot * phidot * np.cos(2*phi)
        + (rdot**2 - r**2 * phidot**2 - M/r) * np.sin(2*phi)
    )
    
    return hplus, hcross

