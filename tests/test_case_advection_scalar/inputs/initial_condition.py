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

    # Create parameters for a gaussian distribution, centered on pi
    sigma = 0.1
    x0    = 0.25
    y0    = 0.50   # Can be used for a 2D gaussian distribuation rather than a "spike"

    # Assign state variables for simple wave advection
    #rho = np.exp(- (x - x0)**2/(2*sigma**2)) # Gaussian distribution of the transported property (in this case, say density) 1D
    Phi = np.exp(- (x - x0)**2/(2*sigma**2) - (y - y0)**2/(2*sigma**2)) # Gaussian distribution of the transported property (in this case, say density) 2D

    # Gather the solution variables into output array
    solution[:,:,0] = Phi

    return solution