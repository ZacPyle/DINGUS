# tests/testInputs/2D/initial_condition.py
from dingus.config import CaseCfg
import numpy as np

def initial_condition(case_config: CaseCfg, x, y):
    """
    Analytical initial condition for the 2D scalar advection test case.

    Uses the (case_config, x, y) signature expected by
    initialize_solution._call_ic_function for a 2D case. A single sinusoidal
    wave in x is imposed: u(x, y) = sin(2*pi*x).
    """
    # Pre-allocate to (nx, ny, num_eq) so the shape tracks the physics model.
    solution = np.zeros((x.shape[0], x.shape[1], case_config.physics.num_eq))
    solution[:, :, 0] = np.sin(2 * np.pi * x)

    return solution
