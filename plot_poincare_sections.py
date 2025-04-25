"""Making Poincare Sections for various Ts season lengths for the ODEs solutions
"""

import numpy as np
import matplotlib.pyplot as plt
from rk4_solver import rk4solver
from seasonal_odes import np_odes

# Specify desired initial conditions
n_init_number = 100
p_init_number = 50
n_init = np.linspace(0.05, 0.95, n_init_number).reshape((n_init_number, 1))
p_init = np.linspace(0.05, 0.55, p_init_number).reshape((p_init_number, 1))
init_all = np.array(np.meshgrid(n_init, p_init)).T.reshape(-1, 2)
# Specify desired parameter values for nu, growth_rate
nu = 3
growth_rate = 5.4
# Speicfy time parameters
# t_end is the end of simulation time, in years
t_end = 2000  
# time steps within one year
time_resolution = int(100)

def generate_and_save_plot(a_s, nu, growth_rate=growth_rate, time_resolution=time_resolution, t_end=t_end, init_all=init_all):
    """Generating Poincare Sections for a specific length of the summer (a_s, Ts). 
    Saves the plot as a PNG file.

    Args: 
        a_s: Length of the summer parameter for sqd function
        nu: Generalist predator density dependence (per mustelid)
        growth_rate: Growth rate of prey
        time_resolution: time steps for differentiation per year (growth_rate / dtau = time_resolution)
        t_end: final time point for differentiation, in years
        init_all: 2D array with initial conditions for n, p
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sols = rk4solver(a_s=a_s, nu=nu, growth_rate=growth_rate, time_resolution=time_resolution, t_end=t_end, init_all=init_all, odes=np_odes)
    #valid_indices = ~np.isnan(n_values) & ~np.isnan(p_values)
    ax.scatter(
        sols[:, 0, 1800::].flatten(),
        sols[:, 1, 1800::].flatten(),
        color="black",
        s=1,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel("n (prey biomass / unit area)")
    ax.set_ylabel("p (predator biomass / unit area)")
    ax.set_title(f"T_s = {a_s*0.5:.5f}, nu = 3")
    ax.grid(True)
    plt.savefig(f"/home/jullcifer/Modelling/PMaps/plot_{a_s:.5f}.png", dpi=300)
    plt.close(fig)

# Specify the range of a_s values to iterate over
a_s_values = np.round(np.linspace(0.010, 1.950, 100), 5)  

# Run and save a plot for each a_s value    
for each_a_s in a_s_values:
    generate_and_save_plot(each_a_s, nu, growth_rate=growth_rate, time_resolution=time_resolution, t_end=t_end, init_all=init_all)
