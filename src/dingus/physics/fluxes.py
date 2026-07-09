# src/dingus/physics/fluxes.py

from dingus.config import CaseCfg
from dingus.physics.constitutiveRelations import compute_pressure
import numpy as np

"""
This module computes the PHYSICAL fluxes in each element of the mesh and the NUMERICAL (interface)
fluxes used to couple neighboring elements at each element interface. This is done modularly, based 
on the physics model and dimensionality specified in the input configuration. The conservation 
equations are assumed to be of the form:

    q_t + div(F_vec) = 0,     F_vec = (F_1, F_2, ..., F_ndim)   e.g. F_vec = (F, G, H) in 3D,

where q is the solution vector of the conserved variables and F_vec is an array of the physical
flux vectors (see Kopriva Ch. 5). The DG method requires two flux quantities:

    1. The VOLUME flux F_vec(q), evaluated at every quadrature node inside an element. This
        feeds the volume (stiffness) term of the residual.

    2. The NUMERICAL flux F*(q^-, q^+, n) at element interfaces, a SINGLE-valued function of
        the two (generally different) traces q^- and q^+ from the elements on either side of a
        shared face, plus the face normal n. This feeds the surface term of the residual and is
        what makes DG stable/conservative (Kopriva Ch. 5.3 for the Riemann/numerical flux idea).

The numerical flux (or Riemann flux) may be computed using a Riemann solver of the user's choice:
    - Upwind (Godunov)
    - Local Lax-Friedrichs (Rusanov)
    - Roe
The Riemann solver is specified in the input file. If no solver is specified, the default is the 
local Lax-Friedrichs (Rusanov) solver, which is robust and works for any hyperbolic system.

In summary: the flux computations depend on:
    - The physics model (scalar_advection, euler, navier_stokes, etc.)
    - The Riemann solver (upwind, rusanov, roe, etc.)
    - The problem's dimensionality (1D, 2D, 3D) 
"""

def _max_wave_speed_one_face_side(sol: np.ndarray, normal: np.ndarray, case_cfg: CaseCfg):
    # Extract components from solution vector
    rho      = sol[..., 0]
    momentum = sol[..., 1:-1]
    pressure = compute_pressure(sol, case_cfg)

    # Compute the speed of sound
    c = np.sqrt(case_cfg.physics.gamma * pressure / rho)

    # Project the velocity onto face-normal vector
    vel_mag_n = np.sum(momentum * normal, axis = -1) / rho  # u \cdot n / rho

    # compute maximum wavespeed
    return np.abs(vel_mag_n) + c

def max_wave_speed(sol: np.ndarray, case_cfg: CaseCfg) -> float:
    '''
    Global maximum signal (wave) speed in the field, used to size the explicit time step via
    the CFL condition (Kopriva Ch. 8). Scalar advection: constant |a|.

    Inputs:
    - sol      : (..., num_eq) conserved variables (unused for scalar advection).
    - case_cfg : case configuration object.

    Outputs:
    - s_max : float, maximum wave speed in the field.
    '''

    match case_cfg.physics.model:
        case 'scalar_advection':
            return float(np.linalg.norm(case_cfg.physics.advection_velocity))

        case 'euler':
            # Extract density and momentum from solution vector
            rho      = sol[...,0]
            momentum = sol[...,1:-1]

            # Compute the speed of sound and velocity magnitude
            pressure = compute_pressure(sol, case_cfg)
            c        = np.sqrt(case_cfg.physics.gamma * pressure / rho)
            vel_mag  = np.linalg.norm(momentum, axis=-1) / rho

            # Compute the max wave speed (velocity_mag + speed of sound)
            return float(np.max(vel_mag + c))
        
        case 'navier-stokes':
            raise NotImplementedError(
                f"max_wave_speed() not yet implemented for physics model '{case_cfg.physics.model}'."
            )

        case _:
            raise ValueError(f"Unknown physics model: '{case_cfg.physics.model}'.")


def _max_abs_face_speed(q_minus : np.ndarray,
                        q_plus  : np.ndarray,
                        normal  : np.ndarray,
                        case_cfg: CaseCfg) -> np.ndarray:
    '''
    PHYSICS helper: the maximum |characteristic speed| NORMAL to a face, per face node. This is
    the scalar dissipation coefficient used by the (universal) Lax-Friedrichs / Rusanov solver.

    Scalar advection: the only normal wave speed is a . n, so this is |a . n|.
    Euler/NSE later: max over the two states of |q . n| + c (flow speed + sound speed).

    Inputs / Outputs mirror `numerical_flux` face arrays; returns shape (M, 1) so it broadcasts
    cleanly against the (M, num_eq) state jump (q_plus - q_minus).
    '''

    match case_cfg.physics.model:
        case 'scalar_advection':
            a   = case_cfg.physics.advection_velocity     # (ndim,)
            a_n = normal @ a                              # (M,)
            return np.abs(a_n)[:, None]                   # (M, 1)

        case 'euler':
            # compute the maximum wavespeed between the two faces using the interior (minus) and exterior (plus) states
            lam = np.maximum(_max_wave_speed_one_face_side(q_minus, normal, case_cfg),
                             _max_wave_speed_one_face_side(q_plus , normal, case_cfg))
            
            return lam[:, None]
        
        case 'navier-stokes':
            raise NotImplementedError(
                f"_max_abs_face_speed() not yet implemented for '{case_cfg.physics.model}'."
            )

        case _:
            raise ValueError(f"Unknown physics model: '{case_cfg.physics.model}'.")
        
def _roe_dissipation(q_minus : np.ndarray,
                     q_plus  : np.ndarray,
                     normal  : np.ndarray,
                     case_cfg: CaseCfg) -> np.ndarray:
    '''
    PHYSICS helper: the full Roe matrix dissipation |A_Roe| (q_plus - q_minus), per face node.
    Unlike Rusanov, this is inherently equation-set-specific (it needs the eigen-decomposition
    of the flux Jacobian), so it dispatches on the physics model.

    This is the natural home for your future Navier-Stokes Roe solver: build the Roe-averaged
    state from (q_minus, q_plus), form the eigenvalues/eigenvectors of the normal flux Jacobian,
    and return |A_Roe| . (q_plus - q_minus). For scalar advection "Roe" degenerates to upwind,
    so it is intentionally left unimplemented here.

    Outputs: (M, num_eq) dissipation vector (already includes the state jump).
    '''

    match case_cfg.physics.model:
        case 'scalar_advection':
            raise NotImplementedError(
                "Roe solver is not meaningful for scalar advection; use 'upwind'/'rusanov' instead."
            )
        
        case 'euler':
            raise NotImplementedError(
                "Roe dissipation not yet implemented — this is where the NSE/Euler Roe solver goes."
            )
        
        case 'navier-stokes':
            raise NotImplementedError(
                "Roe dissipation not yet implemented — this is where the NSE/Euler Roe solver goes."
            )       

        case _:
            raise ValueError(f"Unknown physics model: '{case_cfg.physics.model}'.")
        
def _compute_scalar_flux(sol : np.ndarray, case_cfg : CaseCfg) -> np.ndarray:
    '''
    Callable function to compute the flux in a scalar advection equation: 

        q_t + div(F_vec)  = 0    where F_vec = a * q, and a is the advection velocity vector. Thus:
        q_t + a . grad(q) = 0     <=>     q_t + sum_d (a_d q)_{x_d} = 0

    Inputs:
    - sol      : (..., num_eq) array of conserved variables. The leading axes are arbitrary
                 (e.g. (P+1, P+1) for a full 2D element, or (P+1,) for a single face line), so
                 this works for both volume and face evaluations in any dimensionality.
    - case_cfg : validated case configuration (selects the physics model + parameters).

    Outputs:
    - flux : (..., num_eq, ndim) physical flux; flux[..., d] is the F_d component.
    '''

    # Extract the advection velocity vector from the configuration
    a = case_cfg.physics.advection_velocity       # (ndim,)

    # F_d = a_d * u for every direction d, in one broadcast:
    #   sol[..., None] : (..., num_eq, 1)   (add a trailing dimension axis)
    #   a              : (ndim,)            (broadcasts along that new last axis)
    #   product        : (..., num_eq, ndim)
    flux = sol[..., None] * a
    return flux

def _compute_euler_flux(sol : np.ndarray, case_cfg : CaseCfg) -> np.ndarray:
    '''
    Callable function to compute the euler (inviscid) fluxes: 

        q_t + div(F_vec) = 0,    
        
    where F_vec = (F_1, ..., F_ndim), where F_d is the physical inviscid flux in direction d.
    E.g. F_1 = [rho u, rho u u + p, rho u v, rho u w, (E+p) u]^T in 3D

    Inputs:
    - sol      : (..., num_eq) array of conserved variables. The leading axes are arbitrary
                 (e.g. (P+1, P+1) for a full 2D element, or (P+1,) for a single face line), so
                 this works for both volume and face evaluations in any dimensionality.
    - case_cfg : validated case configuration (selects the physics model + parameters).

    Outputs:
    - flux : (..., num_eq, ndim) physical flux; flux[..., d] is the F_d component.
    '''

    # Extract rho, momentum, and energy components from the input solution
    rho      = sol[..., 0   ]      # shape (...,     )  density
    momentum = sol[..., 1:-1]      # shape (..., ndim)  momentum components
    rhoE     = sol[...,   -1]      # shape (...,     )  energy density

    # Compute the pressure for use in energy flux computation
    pressure = compute_pressure(sol, case_cfg)

    # extract the velocity vector from the momentum vector
    velocity = momentum / rho[..., None]  # remember to add dummy axis to rho

    # Create the flux components: rho*u_i,  rho*u_i_u_j + \delta_ij * p,  u_i(rhoE + p)
    mass_flux     = momentum[..., None, :]                                     # shape (..., 1 (single equation), ndim (Flux is a vector!))
    momentum_flux = np.einsum('...i,...j->...ij', momentum, velocity) \
                    + pressure[..., None, None] * np.eye(momentum.shape[-1])   # shape (..., ndim               , ndim (Flux is a vector!))
    energy_flux   = ((rhoE + pressure)[...,None] * velocity)[..., None, :]     # shape (..., 1 (single equation), ndim (Flux is a vector!))

    # Combine mass, momentum, and energy flux into a single flux vector
    flux = np.concatenate([mass_flux, momentum_flux, energy_flux], axis=-2)     # shape (..., numEq              , ndim (Flux is a vector!))
    
    return flux

def compute_volume_flux(sol : np.ndarray, case_cfg : CaseCfg) -> np.ndarray:
    '''
    Compute the physical (volume) flux F_vec of the governing equations at every quadrature node.
    The result carries one flux component per spatial dimension in the LAST axis: 
        'flux[..., d]' is the flux in physical direction d (d = 0->x, 1->y, 2->z).

    This function is a wrapper that calls internal helper functions to compute the following fluxes:
    - scalar advection flux
    - Euler flux
    - Navier-Stokes flux

    Note that the Navier-Stokes flux calls the Euler flux function to compute the inviscid flux 
    component of the Navier-Stokes fluxes.

    Inputs:
    - sol      : (..., num_eq) array of conserved variables. The leading axes are arbitrary
                 (e.g. (P+1, P+1) for a full 2D element, or (P+1,) for a single face line), so
                 this works for both volume and face evaluations in any dimensionality.
    - case_cfg : validated case configuration (selects the physics model + parameters).

    Outputs:
    - flux : (..., num_eq, ndim) physical flux; flux[..., d] is the F_d component. 
    '''

    # First, determine the physics model specified in the input configuration.
    match case_cfg.physics.model:
        case 'scalar_advection':
            return _compute_scalar_flux(sol, case_cfg)
            
        case 'euler':
            return _compute_euler_flux(sol, case_cfg)
            
        case 'navier-stokes':
            raise NotImplementedError("Navier-Stokes volume flux computation not yet implemented!")
            # TODO: Implement physical Navier-Stokes fluxes in each spatial direction. Do I need to separate
            # inviscid and viscous fluxes? I don't think so.
        
        case _:
            raise ValueError(f"Unknown physics model detected in volume flux computation: '{case_cfg.physics.model}'.")
        
def compute_numerical_flux(q_minus : np.ndarray,
                           q_plus  : np.ndarray,
                           normal  : np.ndarray,
                           case_cfg: CaseCfg) -> np.ndarray:
    
    '''
    Compute the NUMERICAL (interface) flux at a single face: a single-valued approximation to the 
    physical normal flux F_vec at the element interface(s). This is the DG analogue of a finite-volume
    Riemann solver, and in fact is often called a "Riemann flux" in literature (Kopriva Ch. 5.3). 

    Almost every common solver has the form
               ___ central average ___             __ dissipation __
        F*.n  =  1/2 ( F(q^-).n + F(q^+).n )  -  1/2 * D * (q^+ - q^-)
    and only the dissipation D changes between solvers:
        central : D = 0            (no dissipation; unstable on its own)
        LLF     : D = |lambda_max| (scalar max normal wave speed)  -- universal, any physics
        upwind  : for LINEAR advection this equals LLF (rusanov) with D = |a.n| (exact upwinding)
        roe     : D = |A_Roe|      (matrix; equation-set specific -> see the function "_roe_dissipation")

    Note: the sign convention MUST match the DG residual: 
    - q_minus is the trace from the "left" element (element that owns the face)
    - q_plus is the trace from the "right" element (element that does NOT own the face)
    - normal is the UNIT normal vector pointing OUT of the left element (toward the right element)

    Inputs: 
    - q_minus : (..., num_eq) array of conserved variables from the left/owner element.
    - q_plus  : (..., num_eq) array of conserved variables from the right/neighbor element.
    - normal  : (..., ndim) array of unit normal vectors at the face, pointing outward of the left/owner element
    - case_cfg: case configuration (selects the physics model + parameters).

    Outputs:
    - f_star_n : (..., num_eq) array of numerical fluxes at the face, projected onto the face normal. 
    '''

    # The 'averaged' flux: reuse the physics flux, then dot with the face normal.
    # compute_volume_flux returns (M, num_eq, ndim); contracting the ndim axis with `normal` gives F.n.
    Fn_minus = np.einsum('med,md->me', compute_volume_flux(q_minus, case_cfg), normal)   # (M, num_eq)
    Fn_plus  = np.einsum('med,md->me', compute_volume_flux(q_plus,  case_cfg), normal)   # (M, num_eq)
    central  = 0.5 * (Fn_minus + Fn_plus)

    # Compute the dissipation term specific to the chosen Riemann solver
    match case_cfg.physics.riemann_solver:
        case 'upwind' | 'LLF':
            # Scalar (Lax-Friedrichs / Rusanov) dissipation. For linear scalar advection physics models, this
            # reproduces EXACT upwinding with lam = |a.n|, the formula collapses to
            #   a.n * (q_minus if a.n >= 0 else q_plus).
            lam = _max_abs_face_speed(q_minus, q_plus, normal, case_cfg)   # (M, 1)
            return central - 0.5 * lam * (q_plus - q_minus)

        case 'roe':
            # Matrix dissipation; physics-specific (returns the full D.(q_plus - q_minus) already).
            return central - 0.5 * _roe_dissipation(q_minus, q_plus, normal, case_cfg)

        case _:
            raise ValueError(
                f"Unknown Riemann solver: '{case_cfg.physics.riemann_solver}'. "
                "Expected one of: 'upwind', 'LLF', 'roe'."
            )