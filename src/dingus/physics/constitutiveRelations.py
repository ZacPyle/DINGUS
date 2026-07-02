# src/dingus/Physics/constitutiveRelations.py
import numpy as np
from dingus.config import CaseCfg

def compute_pressure(input_sol: np.ndarray, case_cfg: CaseCfg) -> np.ndarray:
    '''
    Helper function to compute the (static) pressure from conservative variables. 
    NOTE: input_sol MUST be an array of the conservative variables of shape (num_nodes, num_eq) where the first column is density, then momentum components, then energy.
    '''

    # Precompute gamma - 1 for efficiency
    gamma_minus_1 = case_cfg.physics.gamma - 1.0

    # Compute the kinetic energy from momentum components: KE = 0.5 * rho * (u^2 + v^2 + w^2) = 0.5 * (rhou^2 + rhov^2 + rhow^2) / rho
    KE =  np.sum(input_sol[:,1:-1]**2, axis=1) / input_sol[:,0]

    # Compute the pressure using the ideal gas law: p = (gamma - 1) * (E - 0.5*KE)
    pressure = gamma_minus_1 * (input_sol[:,-1] - 0.5*KE)

    return pressure

def compute_temperature(input_sol: np.ndarray, case_cfg: CaseCfg) -> np.ndarray:
    '''
    Helper function to compute the temperature from conservative variables.
    NOTE: input_sol MUST be an array of the conservative variables of shape (num_nodes, num_eq) where the first column is density, then momentum components, then energy.
    '''

    # Precompute gamma * mach_ref^2 for efficiency
    gammaM2 = case_cfg.physics.gamma * case_cfg.physics.mach_ref**2

    # Compute the pressure from conservative variables
    pressure = compute_pressure(input_sol, case_cfg)

    # Compute the temperature using the ideal gas law: T = p / (rho * R) where R = gamma * mach_ref^2
    temperature = gammaM2 * pressure / input_sol[:,0]

    return temperature