""" Creates a Poincare section of the solution trajectories starting for a grid of initial conditons.
"""

import numpy as np
from seasonality_odes import getr
from rk4_solver import rk4solver
from seasonal_odes import np_odes

r = getr()

def newpoincgrid(f=np_odes, years=1000, dt=0.01, aS=2*0.7, nu=0.4, nmin=0.01, nmax=1.0, pmin=0.01, pmax=0.13, Nn=10, Np=10):
    """ Creates a grid of initial condition, simulates the system for a long time 
        and returns the poincare sections of each trajectory
    
    Args:
        f: rhs function of the ODE under investigation
        years: number of years the simulation should run for (tend = t0 + years)
        dt: time increments per year
        aS: double the summer length
        nu: intrinsic predator winter mortality rate
        nmin: lower end of the range for the initial n-values
        nmax: upper end of the range for the initial n-values
        pmin: lower end of the range for the initial p-values
        pmax: upper end of the range for the initial p-values
        Nn: number of grid points in the n-dimension
        Np: number of grid points in the p-dimension (in total we have Nn*Np initial grid points)
        
        Note:   the ODE-solver rk4solver starts automatically at t=0.25*r.
                In the rk4solver, the Poincare section is always taken
                at time t=t0 % r, so with the initial shift 0.25*r, always in the
                middle of the summer (0.25*r)
                The simulation runs for years amount of years, i.e. with the 
                initial shift until (years+0.25)*r 
        
    Returns:
        sol: a 3D numpy.ndarray (initial condition/trajectory ID, rows: n/p/tau, solutions for each time step)
    """
    
    
    # Parameters and initial conditions
    n_init = np.linspace(nmin, nmax, Nn).reshape((Nn, 1))
    p_init = np.linspace(pmin, pmax, Np).reshape((Np, 1))
    init_all = np.array(np.meshgrid(n_init, p_init)).T.reshape(-1, 2)
    
    #print(init_all)
    
    # now the computation of the solution
    sol = rk4solver(aS, nu, r, int(1/dt), years, init_all, f)
    
    #print(sol.shape)
    #print(sol)
    
    return sol