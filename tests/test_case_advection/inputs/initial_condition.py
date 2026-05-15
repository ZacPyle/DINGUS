# cases/test_cases/scalar_advection/inputs/initial_condition.py
import numpy as np

def initial_condition(x, y):
    """
    Analytical function that defines the initial condition for the scalar advection case.
    """
    solution = np.zeros((x.shape[0], x.shape[1], 1))
    solution[:,:,0] = np.sin(2 * np.pi * x)

    return solution