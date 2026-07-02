# cases/test_cases/scalar_advection/inputs/initial_condition.py
from dingus.config import CaseCfg
import numpy as np

def initial_condition(case_config: CaseCfg, x, y):
    """
    Analytical function that defines the initial condition for the scalar advection case.
    """
    # Pre-allocate the solution array with the appropriate shape.
    solution = np.zeros((x.shape[0], x.shape[1], case_config.physics.num_eq))

    # Extract parameters from the case configuration
    gam      = case_config.physics.gamma
    mach_ref = case_config.physics.mach_ref

    # Assign state variables for simple wave advection
    rho = np.ones((x.shape[0], x.shape[1])) # Density     (constant)
    u   = np.sin(2*np.pi*x)                    # x-velocity  (sinusoidal wave)
    v   = np.zeros_like(u)                     # y-velocity
    #T   = np.ones_like(u)                      # temperature (constant)
    p   = np.ones_like(u)                      # pressure    (constant)

    # Temperature
    # T = p * gam * mach_ref**2 / rho

    # Compute the total energy from pressure / temperature
    e = p/(gam - 1.0)/rho + 0.5*(u**2 + v**2)

    # Gather the solution variables into output array
    solution[:,:,0] = rho
    solution[:,:,1] = rho * u
    solution[:,:,2] = rho * v
    solution[:,:,3] = rho * e

    return solution