"""
Transient condunction in plate fuel elements.
"""

import numpy as np 

def A(x, fk, fc_p, frho, T):
    """Generate conduction problem system matrix given constants and temperature.

    Args:
        fk (_type_): function k(x, )
        fc_p (_type_): _description_
        frho (_type_): _description_
        T (_type_): _description_
    """

    assert len(x) == len(T)
    n = len(x)

    Δx = x[1] - x[0]

    # Generate discrete values of constants
    k = fk(x, T)
    c_p = fc_p(x, T)
    rho = frho(x)

    # Initialize system matrix
    A = np.zeros((n, n))

    # Left boundary condition
    A[0, 0] = -1
    A[0, 1] = 1

    # Right boundary condition
    A[-1, -1] = 1

    a = (k[i+1] - k[i-1])/(4*Δx*rho[i]*c_p[i])
    b = k[i] / (Δx**2*rho[i]*c_p[i])

    for i in range(1, n-1):
        A[i, i-1] = -a+b
        A[i, i]   = -2*b
        A[i, i+1] = a 

    return A 

def k_system(x, T):



def k_fuel(T_K, B=0.0) : 
    """ Calculate fuel conductivity (W/m-K) for temperature T_K (K)
    
    From J.D. Hales et al. (2013) "Bison Theory" (NFIR model)
    """
    from numpy import exp, tanh
    # kelvin to celsius
    T = T_K - 273.15;
    # thermal recovery function
    rf = 0.5*(1.0+tanh((T-900.0)/150.0))
    # phonon contribution at start of thermal recovery [Hales eq. 8.14]
    kps = 1.0/(0.09592+0.00614*B-0.000014*B**2+(0.00025-0.00000181*B)*T)
    # phonon contribution at the end of thermal recovery [Hales eq. 8.15]
    kpend = 1.0/(0.09592+0.0026*B+(0.00025-0.00000027*B)*T)
    # unirradiated material at 95% th. density [Hales eq. 8.17]
    kel = 0.0132*exp((0.00188)*T)
    k = (1.0-rf)*kps + rf*kpend + kel
    return k

def k_cladding(T) :
    """ Returns thermal conductivity (W/m) for Zr-4 at temperature T (K)
    
    Reference: INL/EXT-14-32591, Revision 1
    """
    return 7.511 + 2.088e-2*T - 1.45e-5*T**2 + 7.668e-9*T**3

def k(x, T):

    k


