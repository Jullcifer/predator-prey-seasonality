"""Defines seasonally varying ODEs for prey n and predator p (Tyson 2016).
   Most of the equations parameters are defined here.
"""

import numpy as np
from numba import njit

@njit
def np_odes(    
    a_s,
    growth_rate,
    tau,
    inital_values,
    a=70,
    K=100,
    gamma=0.017,
    b=13.8,
    m=1.0,
    s=1.6,
    alpha=600,
    beta=13,
    mu=3.4,
    nu = 5.5,
    kappa=0.001,
    c = 0.4, 
):
    """ Takes in system's ecological parameters and solves the ODEs (Tyson 2016).

    Args:
        a_s: Length of the summer parameter for sqd function
        kappa: Sharpness of Squdel function, 0 squared, 1 sin
        growth_rate: Growth rate of prey
        tau: Dimensionless time
        inital_values: Initial values for dimensionless densities n and p
        a: Generalist saturation killing rate (rodents/mustelid/year)
        K: Carrying capacity
        gamma: Conversion factor (mustelid/rodent)
        b: Prey level for half-maximal generalist predation rate
        nu: Generalist predator density dependence (per mustelid)
        m: Predator death rate in the summer
        s: Generalist growth, maximal rate
        alpha: Predator maximum specialist predation rate
        beta: Prey level for half-maximal specialist predation rate
        mu: Predator death rate in the winter
        

    Returns:
        Numpy array of solutions dn_dtau, dp_dtau
    """

    # all 'tilde' values are non-dimensionalized
    omega = 2 * np.pi
    omega_tilde = omega / growth_rate
    # a_s = 1.2


    # non-dimensionalizing the parameters
    b_tilde = b / K
    alpha_tilde = alpha / a
    beta_tilde = beta / K
    s_tilde = s / growth_rate
    m_tilde = m / growth_rate
    mu_tilde = mu / growth_rate
    gamma_tilde = gamma * a / growth_rate
    nu_tilde = nu * growth_rate * K / a

    sqd = (1 + np.sin(omega_tilde * tau) / (np.sqrt(np.sin(omega_tilde * tau) ** 2 + 
        kappa**2 * np.cos(omega_tilde * tau) ** 2))) / 2
    # sqd function has a 1+ / 2 to adjust the range from (-1,1) to (0,1)
    # so when sqd=0 we get summer, when sqd is 1 we get winter

    n, p = inital_values
    dn_dtau = a_s * sqd * (n * (1 - n) - (n**2) * p / (b_tilde**2 + n**2)) + (
        2 - a_s
    ) * (1 - sqd) * (-alpha_tilde * n * p / (beta_tilde + n) + c * n * (1 - n))
    dp_dtau = a_s * sqd * (
        gamma_tilde * n**2 * p / (b_tilde**2 + n**2)
        + s_tilde * p / (1 + nu_tilde * p)
        - m_tilde * p
    ) + (2 - a_s) * (1 - sqd) * (
        gamma_tilde * alpha_tilde * n * p / (beta_tilde + n)  -  mu_tilde * p
    )
    return np.array([dn_dtau, dp_dtau])
