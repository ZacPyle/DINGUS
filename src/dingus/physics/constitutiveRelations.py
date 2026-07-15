# src/dingus/Physics/constitutiveRelations.py
import numpy as np
from dingus.config import CaseCfg

def _assert_positive(arr: np.ndarray, name: str) -> None:
    '''
    Raise an informative error if any node has a non-positive 'name' (density or pressure).

    A negative density or pressure is always unphysical for the compressible equations, and it is
    the usual signature of an UNDER-RESOLVED or UNSTABLE run. Left unchecked it silently poisons the
    sound speed c = sqrt(gamma * p / rho) with a NaN, which then propagates into the wave speed / time
    step and crashes far from the real cause. Catching it here converts that mystery NaN into an
    immediate, diagnosable failure.
    '''
    amin = float(np.min(arr))
    if amin <= 0.0:
        n_bad = int(np.count_nonzero(arr <= 0.0))
        raise ValueError(
            f"Non-positive {name} detected at {n_bad} node(s) (min {name} = {amin:.3e}). "
            f"The compressible state is unphysical -- sound speed sqrt(gamma*p/rho) would be NaN. "
            f"This usually means the solution is under-resolved or unstable; try more elements, a "
            f"lower polynomial degree, or a smaller CFL."
        )

def compute_pressure(input_sol: np.ndarray, case_cfg: CaseCfg) -> np.ndarray:
    '''
    Helper function to compute the (static) pressure from conservative variables.
    NOTE: input_sol MUST be an array of the conservative variables of shape (num_nodes, num_eq) where the first column is density, then momentum components, then energy.
    '''

    # Precompute gamma - 1 for efficiency
    gamma_minus_1 = case_cfg.physics.gamma - 1.0

    # Density lives in slot 0. Guard it BEFORE dividing by it (rho <= 0 -> inf/NaN kinetic energy).
    rho = input_sol[..., 0]
    _assert_positive(rho, "density")

    # Compute the kinetic energy from momentum components: KE = 0.5 * rho * (u^2 + v^2 + w^2) = 0.5 * (rhou^2 + rhov^2 + rhow^2) / rho
    KE =  0.5 * np.sum(input_sol[...,1:-1]**2, axis=-1) / rho

    # Compute the pressure using the ideal gas law: p = (gamma - 1) * (rhoE - KE)
    pressure = gamma_minus_1 * (input_sol[...,-1] - KE)

    # Guard pressure BEFORE it feeds the sound speed sqrt(gamma * p / rho) downstream.
    _assert_positive(pressure, "pressure")

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
    temperature = gammaM2 * pressure / input_sol[...,0]

    return temperature

def compute_viscosity(input_sol: np.ndarray, case_cfg: CaseCfg) -> np.ndarray:
    '''
    Dimensionless dynamic viscosity mu* = mu / mu_ref, shape = sol.shape[:-1] (one per node).

    Two models, selected by case_cfg.physics.viscosity_model:

    - 'constant'   : mu* = 1 everywhere. Isolates the discretization from the viscosity model during
                     verification (the MMS and Taylor-Green tests assume it).

    - 'sutherland' : mu* increases with temperature per Sutherland's law, in nondimensional form

                        mu* = (T*)^(3/2) * (1 + S*) / (T* + S*),   S* = sutherland_constant / ref_temperature

                     where T* = T / T_ref is the nondimensional temperature (compute_temperature),
                     sutherland_constant is the DIMENSIONAL Sutherland temperature S (110.4 K for air,
                     configurable), and ref_temperature is the DIMENSIONAL temperature at T* = 1. By
                     construction mu* = 1 at T* = 1, so it is consistent with the mu_ref used in the
                     nondimensionalization (Re, Pr all reference this same mu_ref).

    Because every viscous quantity (the stress tensor, the heat-flux coefficient kappa, and the viscous
    dt limit) reads mu through THIS function, switching the model here propagates everywhere automatically.
    '''

    match case_cfg.physics.viscosity_model:
        case 'constant':
            return np.ones_like(input_sol[..., 0])

        case 'sutherland':
            T_star = compute_temperature(input_sol, case_cfg)                            # T* = T / T_ref
            S_star = case_cfg.physics.sutherland_constant / case_cfg.physics.ref_temperature   # nondim S*
            return T_star**1.5 * (1.0 + S_star) / (T_star + S_star)

        case _:
            raise ValueError(f"Unknown viscosity_model: '{case_cfg.physics.viscosity_model}'.")