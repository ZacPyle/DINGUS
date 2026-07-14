# tests/test_case_advection_scalar/inputs/initial_condition_sinusoid.py
from dingus.config import CaseCfg
import numpy as np


def initial_condition(case_config: CaseCfg, x, y):
    '''
    Exactly-periodic sinusoidal initial condition for the scalar advection case.

    Why a sinusoid (vs a Gaussian) for a spectral-convergence study: a product of sines matches
    itself AND all of its derivatives across the periodic domain boundaries *exactly* (it has
    period 1 in each direction on the unit square). A Gaussian is only APPROXIMATELY periodic --
    it has a tiny but nonzero value at the seams -- which acts like a microscopic discontinuity
    and floors the L2 error around ~1e-7. Removing that seam mismatch lets the p-convergence
    continue exponentially all the way down toward machine precision.

    Inputs:
    - case_config : validated case configuration (provides num_eq).
    - x, y        : (nx, ny) arrays of physical quadrature-node coordinates within an element.

    Outputs:
    - solution : (nx, ny, num_eq) initial state; the transported scalar lives in slot 0.
    '''
    # Pre-allocate the solution array with the appropriate shape.
    solution = np.zeros((x.shape[0], x.shape[1], case_config.physics.num_eq))

    # Smooth, analytic, and exactly periodic on the unit square (period 1 in x and y).
    Phi = np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)

    # Assign the transported scalar.
    solution[:, :, 0] = Phi

    return solution
