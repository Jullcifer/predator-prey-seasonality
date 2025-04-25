""" Defining and rescaling the parameters and the rhs function of our system 
    as well as a function to return the value of the reproduction rate r
"""

import numpy as np

# Parameters and rescaling

K = 100
a = 70
r = 5.4
kappa = 0.001
b = 13.8
alpha = 600
beta = 13
s = 1.6
m = 1 
mu = 3.4 
gamma = 0.017
nu = 0.4 


c = 0.4

b_tilde = b/K
alpha_tilde = alpha/a
beta_tilde = beta/K
s_tilde = s/r
m_tilde = m/r
mu_tilde = mu/r
gamma_tilde = gamma*a/r
nu_tilde = nu*r*K/a

omega = 2*np.pi
omega_tilde = omega/r



# Our rhs function f


def f_vect(t, X, aS, nu):
    """ The right hand side of our ode

    Args: 
        t: actually not used; but the time in our simulation (should be the same as X[:,2])
        X: 2D np array where X[i] are i different vectors with 3 entries (n,p,tau)
        aS: Double the summer length (between 0 and 2)
        nu: second bifurcation parameter; intrinsic predator winter death rate
        
    Returns:
        dydt: 2D np array in the same format as X[i], i.e. (dndt, dpdt, dtaudt) 
        for each of the i different (n,p,tau)

    """
    dydt = np.zeros_like(X)
    n = X[:,0]
    p = X[:,1]
    tau = X[:,2]
    nu_tilde = nu*r*K/a

    sqr = 0.5*(1 + np.sin(omega_tilde*tau)/np.sqrt(np.sin(omega_tilde*tau)**2 + kappa**2*np.cos(omega_tilde*tau)**2))

    dydt[:,0] = aS*sqr*(n*(1 - n) - n**2*p/(b_tilde**2 + n**2)) + (2 - aS)*(1 - sqr)*(c*n*(1-n) -alpha_tilde*n*p/(beta_tilde + n))
    dydt[:,1] = aS*sqr*(gamma_tilde*n**2*p/(b_tilde**2 + n**2) + s_tilde*p/(1 + nu_tilde*p) - m_tilde*p) + (2 - aS)*(1 - sqr)*(gamma_tilde*alpha_tilde*n*p/(beta_tilde + n) - mu_tilde*p)
    dydt[:,2] = 1

    return dydt

def getr():
    """ Returns the prey reproduction rate r
    """
    return r