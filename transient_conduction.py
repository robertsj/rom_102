"""
Transient condunction in plate fuel elements.

Provides tools for solving

  rho*c_p*T_t = (d/dx)*[k(x,T)*dT/dx] + q'''

s.t.

  T'(0, t) = 0 and   T(w, t) = 0 .

where w is the wall width.    

Uses finite-difference method:

  T_t = (1/[rho*c_p]) * [(ΔkΔT + kΔ^2 T) + q''')

where Δ and Δ^2 indicate first and second central differences, i.e.,

  Δy[i]   = (y[i+1]-y[i-1])/(2Δ)
  Δ^2y[i] = (y[i+1]-2*y[i]+y[i-1])/(2Δ^2)

Solution uses odeint in time.

The user provides:

  - function k(x, T)   # array x, array T  len(x) == len(T)
  - function c_p(x, T) # array x, array T  len(x) == len(T)
  - function rho(x)    # array x
  - function q(x, t)   # array x, float t
  - function T0(x)     # array x, (initial condition)
  - float w 
  - float t_final
  - int n_x (or d_x)
  - int n_t (or d_t)
  - float d_x (adjusts down in size to produce integer n_x)
  - float d_t (adjusts down in size to produce integer n_t)
  - array t (times when solution is wanted)

Uses backward Euler with fixed step sizes in x and t.


"""

import numpy as np 
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

def rhs(t, T, x, k, c_p, rho, q):
    """ A@T + q'''/(rho*c_p)
    """
    A = get_A(x, T, k, c_p, rho)

    qq = q(x, t)
    qq[-1] = 0.0 # does not get added to bc
    qq[0] = 0.0
    return A@T + qq/(rho(x)*c_p(x, T)) 

def get_source(T, t, x, k, c_p, rho, q):
    qq = q(x, t)
    qq[-1] = 0.0 # does not get added to bc
    qq[0] = 0.0 
    return qq 

def solve_euler(k, c_p, rho, q, T0, w, nx, times):
    x = np.linspace(0, w, nx)
    ic = T0(x)

    A = get_A(x, x**0, k, c_p, rho)

    qq = q(x, 1)
    qq[-1] = 0.0 # does not get added to bc
    qq[0] = 0.0

    # dydt = A*y + qq
    #  y1-y0 = dt*A*y1 + dt*qq
    #  (I-dt*A)*y1 = y0 + dt*qq
    I = np.eye(len(x))

    dt = 0.5
    nt = 101
    sol = np.zeros((len(x), nt))
    sol[:, 0] = ic 

    for i in range(1, 101):
        y0 = sol[:, i-1]
        y1 = np.linalg.solve((I-dt*A), y0+dt*qq)
        sol[:, i] = y1 
    return sol, x



def solve(k, c_p, rho, q, T0, w, nx, times):
    x = np.linspace(0, w, nx)
    ic = T0(x)
    dx = x[1]-x[0]    
    sol = solve_ivp(rhs, t_span=(0, times[-1]), y0=ic, t_eval=times, args=(x, k, c_p, rho, q), method="BDF")
    sol=sol.y
    return sol, x


def get_A(x, T, fk, fc_p, frho):
    """Generate conduction problem system matrix given constants and temperature.

    Args:
        T (_type_): _description_
        fk (_type_): function k(x, )
        fc_p (_type_): _description_
        frho (_type_): _description_
        T (_type_): _description_
    """
 
    assert len(x) == len(T)
    n = len(x)
    Δx = x[1] - x[0]
    # Initialize system matrix
    A = np.zeros((n, n))
    # Left boundary condition
    A[0, 0] = -1/Δx
    A[0, 1] = 1/Δx
    # Right boundary condition
    A[-1, -1] = 1
    # Internal cells
    k = fk(x, T)
    c_p = fc_p(x, T)
    rho = frho(x)
    for i in range(1, n-1):
        a = (k[i+1] - k[i-1])/(4*Δx**2*rho[i]*c_p[i])
        b = k[i] / (Δx**2*rho[i]*c_p[i])
        A[i, i-1] = -a+b
        A[i, i]   = -2*b
        A[i, i+1] = a+b
    return A 

 
def get_pod_basis(snapshots, rank):

    U, S, V = np.linalg.svd(snapshots)

    return U[:, 0:rank], S[:, 0:rank]

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


if __name__ == "__main__":

    nx = 77

    def ic(x):
        y = x**0; y[x>0.5] = 0.0
        return y

    k = lambda x, T: 1.0*x**0  
    c_p = lambda x, T: 1*x**0
    rho = lambda x: 1*x**0
    q = lambda x, t: 1*x**0

    def q(x, t):
        v = x**0 
        if t > 5:
            v[x<.5] = 0
        return v

    times = np.linspace(0, 50, 11)

    xx = np.linspace(0, 1, nx)
    A = get_A(xx, ic(xx), k, c_p, rho)
    qq = get_source(ic(xx), 8, xx, k, c_p, rho, q)
    T_ss = np.linalg.solve(-A, qq)
    plt.plot(xx, T_ss, 'g--x', label="steady")

    sol, x = solve(k, c_p, rho, q, ic, 1.0, 101, times)

    for i in range(len(times)):
        plt.plot(x, sol[:, i], color=plt.cm.copper((i/len(times)))); 
    plt.xlabel("x"); plt.ylabel("T(x)");

    plt.legend()
    plt.show()


