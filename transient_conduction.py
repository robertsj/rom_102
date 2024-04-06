"""
Transient conduction in 1-D cartesian geometry.

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

Solution uses solve_ivp in time.
"""

import numpy as np 
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from time import time 


def get_source(T, t, x, k, c_p, rho, q):
    qq = q(x, t)
    qq[-1] = 0.0 # does not get added to bc
    qq[0]  = 0.0 
    return  qq/(rho(x)*c_p(x, T))  

def get_A(x, T, fk, fc_p, frho):
    """Generate conduction problem system matrix given constants and temperature.

    Args:
        x (array): points
        T (array): temperature at points
        fk (function): conductivity as function of x and T
        fc_p (function): specific heat as function of x and T
        frho (function): density as function of x 
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

def rhs(t, T, x, k, c_p, rho, q):
    """ A@T + q'''/(rho*c_p)
    """
    A = get_A(x, T, k, c_p, rho)
    s = get_source(T, t, x, k, c_p, rho, q)
    return A@T + s

def solve_steady_state(k, c_p, rho, q, ic, w, nx, verbose = False):
    """Solve the full-order model for steady state, i.e., dT/dt = 0.

    This assumes a constant source at t = 0.
    """
    x = np.linspace(0, w, nx)
    T_ss_0 = ic(x) 
    
    # Picard iteration
    for i in range(100):
        A = get_A(x, T_ss_0, k, c_p, rho)
        qq = get_source(T_ss_0, 0, x, k, c_p, rho, q)
        T_ss = np.linalg.solve(-A, qq)
        err = np.linalg.norm(T_ss - T_ss_0)
        if verbose:
            print(f"iter={i} err={err:.4e}")
        if np.linalg.norm(T_ss - T_ss_0) < 1e-8:
            if verbose:
                print(f"converged steady state in {i} iterations.")
            break 
        T_ss_0[:] = T_ss[:]
    return T_ss, x


def solve(k, c_p, rho, q, T0, w, nx, times, gen_nl_snapshots=False):
    """Solve the full-order model. 

    Args:
        k (function): Conductivity, k(array: x, array: T)
        c_p (function): Specific heat, c_p(array: x, array: T)
        rho (function): Density, rho(array: x)
        q (function): Heat generation rate, q(array: x, float: t)
        T0 (function): Initial condition, T0(array: x)
        w (float): Wall width
        nx (int): Number of spatial points
        times (array): Times at which solution is requested

    Returns:
        snapshots (array): Solution at each requested time, (nx, len(times))
        x (array): Points at which solution is evaluated, (nx)
        nl_snapshots (array): Snapshots of the term f(T) = A(T)*T at the same times.
    """
    x = np.linspace(0, w, nx)
    ic = T0(x)
    sol = solve_ivp(rhs, t_span=(0, times[-1]), y0=ic, t_eval=times, args=(x, k, c_p, rho, q), method="BDF")
    snapshots=sol.y
    
    if gen_nl_snapshots == True:
        nl_snapshots = np.zeros_like(snapshots)
        for i in range(len(times)):
            A = get_A(x, snapshots[:, i], k, c_p, rho)
            nl_snapshots[:, i] = A@snapshots[:, i]
        return snapshots, x, nl_snapshots
    else:
        return snapshots, x


def deim():
    pass

def solve_rom(𝛙, x, k, c_p, rho, q, T0, times, nonlinear=False, deim=False):
    """Solve the reduced-order model.

    Args:
        𝛙 (array): POD basis (nx, nt)
        x (array): Spatial points on which POD basis is defined
        k (function): Conductivity, k(array: x, array: T)
        c_p (function): Specific heat, c_p(array: x, array: T)
        rho (function): Density, rho(array: x)
        q (function): Heat generation rate, q(array: x, float: t)
        T0 (function): Initial condition, T0(array: x)
        times (array): Times at which solution is requested
    """

    # from 
    #   dy/dt = Ay + s
    # to
    #   dỹ/dt = (𝛙^T A 𝛙)ỹ + 𝛙^T s

    # Project initial condition
    T_tilde_0 = 𝛙.T@T0(x) 

    if nonlinear == False:

        # Get A evaluated at the initial condition
        A = get_A(x, T0(x), k, c_p, rho)
        #   and then project
        A_tilde = 𝛙.T@(A@𝛙)

        # Get the source at time t = 0
        s = get_source(T0, 0, x, k, c_p, rho, q)
        #   and then project
        s_tilde = 𝛙.T@s

        def rhs_rom(t, T_tilde, A_tilde, s_tilde):
            return A_tilde@T_tilde + s_tilde
        
        sol = solve_ivp(rhs_rom, t_span=(0, times[-1]), y0=T_tilde_0, t_eval=times, args=(A_tilde, s_tilde), method="BDF")

    else:

        if deim == True:

            # Use DEIM to approximate the nonlinear part of the system.
            # Here, we're taking f(T) = A(T)*T to be that part.  Hence, cp and rho are T-independent by assumption.
            pass

        else:
            # Project at every time step.

            def rhs_rom_nl(t, T_tilde, 𝛙, x, k, c_p, rho, q):
                # Get A evaluated at the initial condition
                A = get_A(x, 𝛙@T_tilde, k, c_p, rho)
                #   and then project
                A_tilde = 𝛙.T@(A@𝛙)
                # Get the source at time t = 0
                s = get_source(T0, t, x, k, c_p, rho, q)
                #   and then project
                s_tilde = 𝛙.T@s
                return A_tilde@T_tilde + s_tilde
            
            sol = solve_ivp(rhs_rom_nl, t_span=(0, times[-1]), y0=T_tilde_0, t_eval=times, args=(𝛙, x, k, c_p, rho, q), method="BDF")

    T_tilde = sol.y # [r, nt]
    T = 𝛙@T_tilde 

    return T



def get_pod_basis(snapshots, rank):

    U, S, V = np.linalg.svd(snapshots)

    return U[:, 0:rank], S[:, 0:rank]

def k_fuel(x, T_K, B=0.0) : 
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

def k_cladding(x, T) :
    """ Returns thermal conductivity (W/m) for Zr-4 at temperature T (K)
    
    Reference: INL/EXT-14-32591, Revision 1
    """
    return 7.511 + 2.088e-2*T - 1.45e-5*T**2 + 7.668e-9*T**3

    

def demo_problem_simple():

    # Problem definition
    nx = 101  # number of spatial cells
    w = 1.0   # width of wall
    def ic(x):
        y = x**0; y[x>0.5] = 0.0
        return y
    k = lambda x, T: 1.0*x**0  
    c_p = lambda x, T: 1*x**0
    rho = lambda x: 1*x**0
    q = lambda x, t: 1*x**0


    # Solve FOM and extract snapshots
    times = np.linspace(0, 10, 11)
    sol_fom, x = solve(k, c_p, rho, q, ic, 1.0, nx, times)
    plt.figure(1, figsize=(8, 4))
    plt.title("FOM-Generated Snapshots")
    for i in range(len(times)):
        plt.plot(x, sol_fom[:, i], color=plt.cm.copper(np.sqrt(i/len(times)))); 
    # Steady-state solution (for sanity check)
    A = get_A(x, ic(x), k, c_p, rho)
    qq = get_source(ic(x), 0, x, k, c_p, rho, q)
    T_ss = np.linalg.solve(-A, qq)
    plt.plot(x, T_ss, 'b--', label=r"$T(\infty)$")
    plt.xlabel("x"); plt.ylabel("T(x)");
    plt.legend()
    plt.tight_layout()

    # Generate the POD modes
    U, S, V = np.linalg.svd(sol_fom[:,:])  # or sol[:,1:]! 
    plt.figure(2, figsize=(8,4))
    plt.title("Singular Values of Snapshot Matrix")
    plt.semilogy(S, 'o')
    plt.tight_layout()
    plt.figure(3, figsize=(8,4))
    plt.title("POD Modes")
    r = 5
    𝛙 = U[:, :r]
    plt.plot(x, 𝛙);
    plt.legend(range(0,5))
    plt.tight_layout()

    # Solve the ROM
    sol_rom = solve_rom(𝛙, x, k, c_p, rho, q, ic, times)
    plt.figure(4, figsize=(8, 4))
    plt.title("ROM Solution")
    for i in range(len(times)):
        plt.plot(x, sol_rom[:, i], color=plt.cm.copper(np.sqrt(i/len(times)))); 
    plt.figure(5, figsize=(8, 4))
    plt.title("ROM Error (%)")
    for i in range(len(times)):
        plt.plot(x, 100*(abs(sol_rom[:, i]-sol_fom[:, i])/sol_fom[:, i]), color=plt.cm.copper(np.sqrt(i/len(times)))); 
    plt.show()

def demo_problem_nonlinear(nonlinear=False):

    # NOTE: zeroing in on an interesting temperature profile
    #       that requires the nonlinearity to be accounted for

    purple = "#512888"

    # Problem definition
    nx =101  # number of spatial cells
    w = 1.0   # width of wall
    def ic(x):
        y = 600*(1-x)
        y[-1] = 0
        return y

    k = k_cladding
    #k = lambda x, T: 15*x**0
    c_p = lambda x, T: 100*x**0
    rho = lambda x: 18*x**0
    q = lambda x, t: 15000*x**0

    fig, axs = plt.subplots(3, 2, figsize=(10, 8), layout='constrained')

    # Solve FOM and extract snapshots
    times = np.linspace(0, 1000, 21)
    t0 = time()
    sol_fom, x, nl_fom = solve(k, c_p, rho, q, ic, w, nx, times, gen_nl_snapshots=True)
    te = time()-t0
    print(f"FOM solved in {te:.4e} s")
    axs[0, 0].set_title("FOM-Generated Snapshots")
    print(sol_fom.shape)
    for i in range(len(times)):
        axs[0, 0].plot(x, sol_fom[:, i], color=plt.cm.copper(np.sqrt(i/len(times)))); 
    # Steady-state solution (for sanity check)
    T_ss, _ = solve_steady_state(k, c_p, rho, q, ic, w, nx)
    axs[0, 0].plot(x, T_ss, 'b--', label=r"$T(\infty)$")
    axs[0, 0].set_xlabel("x"); 
    axs[0, 0].set_ylabel("T(x)");
    axs[0, 0].legend()

    # Generate the POD modes
    t0 = time()
    U, S, V = np.linalg.svd(sol_fom[:,:])  # or sol[:,1:]! 
    te = time()-t0
    print(f"POD modes found in {te:.4e} s")
    axs[1, 0].set_title("Singular Values of Snapshot Matrix")
    axs[1, 0].semilogy(range(1, len(S)+1), S, 'o', color=purple)
    axs[2, 0].set_title("POD Modes")
    r = 8
    𝛙 = U[:, :r]
    axs[2, 0].plot(x, 𝛙);
    axs[2, 0].legend(range(0,5))
    axs[2, 0].set_xlabel("x"); 
    print(axs.shape)
    # Solve the ROM
    t0 = time()
    sol_rom = solve_rom(𝛙, x, k, c_p, rho, q, ic, times, nonlinear=nonlinear)
    te = time()-t0
    print(f"ROM solved in {te:.4e} s")
    axs[0,1].set_title("ROM Solution")
    for i in range(len(times)):
        axs[0,1].plot(x, sol_rom[:, i], color=plt.cm.copper(np.sqrt(i/len(times)))); 
    
    axs[1,1].set_title("ROM Error (%)")
    for i in range(len(times)):
        axs[1,1].plot(x, 100*(abs(sol_rom[:, i]-sol_fom[:, i])/sol_fom[:, i]), color=plt.cm.copper((i/len(times)))); 
    axs[1,1].set_xlabel("x")

    axs[2,1].set_title("ROM Error (%)")
    for i in range(len(x)):
        axs[2,1].plot(times, 100*(abs(sol_rom[i, :]-sol_fom[i,:])/sol_fom[i, :]), color=plt.cm.copper((i/len(x)))); 
    axs[2,1].set_xlabel("t")
    
    plt.tight_layout()
    #plt.show()
    
   # plt.clf()
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), layout='constrained')
    U, S, V = np.linalg.svd(nl_fom[:,:])  # or sol[:,1:]! 
    te = time()-t0
    print(f"POD modes found in {te:.4e} s")
    axs[0, 0].set_title("Singular Values of Snapshot Matrix")
    axs[0, 0].semilogy(range(1, len(S)+1), S, 'o', color=purple)
    axs[0, 0].set_title("POD Modes")
    r = 8
    𝛙 = U[:, :r]
    axs[1, 0].plot(x, 𝛙);
    axs[1, 0].legend(range(0,5))
    axs[1, 0].set_xlabel("x");  
    print(axs.shape)
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":

    demo_problem_nonlinear(nonlinear=True)


