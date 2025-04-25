""" Runge-Kutta 4 solver for ODE equations with tau stored in the solutions array.
"""
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def rk4solver(a_s, growth_rate, time_resolution:int, t_end, init_all, odes):
    """Solve ODEs using the Runge-Kutta 4 method.

    Args:
        a_s: Length of the summer parameter for sqd function
        growth_rate: Growth rate of prey
        kappa: Sharpness of Squdel function, 0 squared, 1 sin
        t_end: final time point for differentiation, in years
        time_resolution: time steps for differentiation per year (growth_rate / dtau = time_resolution)
        init_all: 2D array with initial conditions for n, p
        odes: Function call to N-P ODE equations

    Return: a 3D numpy.ndarray (initial condition, rows: n/p/tau, solutions for each time step)
    """
    # tau = natural time scale (dimensionless time)
    # incremental steps for differentiation:
    dtau = np.round(growth_rate/time_resolution,20)
    # converting t_end in years into final time point for differentiation (tau = t*r)
    tau_end = t_end * growth_rate
    # number of total time steps over the whole differentiation time
    time_steps_total = int(np.round(tau_end / dtau))  
    # t = np.linspace(0, t_end, tspan) # time vector in t (years)
    # time vector for tau (tau = t*r): an ndarray of tau values at which the output is calculated (time steps)
    tau = np.linspace(0, tau_end, time_steps_total)


    num_initial_conditions = init_all.shape[0]
    # creating an empty array for the system of 2 ODEs, i.e. preallocate array for solutions
    solutions = np.empty((num_initial_conditions, 3, int(time_steps_total//time_resolution + 1)))

    # Calculate the step interval for saving values at mid-summer for sqd function
    save_interval = time_resolution // 4

    for j in prange(num_initial_conditions):
        # Retrieve the initial conditions and create an array to store it
        y0 = init_all[j]
        
        # Initialize y_out with + 1 space to include the initial conditions and tau (3,)
        y_out = np.empty((3, time_steps_total // time_resolution + 1)) * np.nan 

        # Set the initial conditions in y_out
        y_out[:2, 0] = y0  # Writing out the very first initial condition
        y_out[2, 0] = tau[0]  # Writing out the initial time
        
        # Start with the initial conditions
        y = y0.copy()  

        # Loop through each time step starting from 0
        for i in range(time_steps_total):
            t0 = tau[i]
            f1 = odes(a_s, growth_rate, t0, y)
            f2 = odes(a_s, growth_rate, t0 + dtau / 2, y + (dtau / 2) * f1)
            f3 = odes(a_s, growth_rate, t0 + dtau / 2, y + (dtau / 2) * f2)
            f4 = odes(a_s, growth_rate, t0 + dtau, y + dtau * f3)

            # Update y with the new values
            y = y + (dtau / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

            if y.any() < -0.0001:
                print(f"Negative value found at index {i} for initial condition {j}")
                break

            # Save the state at specified intervals
            # saving only one value per year (mid-summer, at 1/4 of growth rate)
            elif i % time_resolution == save_interval:
                index = i // time_resolution + 1  # +1 because index 0 is used for initial conditions
                y_out[:2, index] = y
                y_out[2, index] = tau[i]  # Store the corresponding time value

        # Store the complete y_out in the solutions array
        solutions[j] = y_out

    return solutions
