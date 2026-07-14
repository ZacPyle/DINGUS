# src/dingus/physics.viscousFluxes.py

import numpy as np
from dingus.config import CaseCfg
from dingus.physics.constitutiveRelations import compute_pressure
from dingus.physics.constitutiveRelations import compute_viscosity

"""
This module computes the viscous fluxes of the Navier-Stokes equations. For spatial dimension, d, 
the components of the viscous flux vector F_Vec_Visc are:
     mass       : 0
     mom_i : tau_id
     energy     : sum_j u_j tau_jd  +  heat_flux_d          (viscous work + heat conduction)

Where the non-dimensional tau_ij (viscous stress tensor) and heat_flux_d are:
     tau_ij      = (mu / Re) * ( du_i/dx_j + du_j/dx_i - (2/3) delta_ij div(u) )
     heat_flux_d = kappa * dT/dx_d,      kappa = mu* / ( Re * Pr * (gamma-1) * M_ref^2 )

Here, mu is the dimensionless viscosity mu = mu_dim / mu_ref. FOR NOW WE ASSERT MU = 1; LATER 
THIS WILL BE COMPUTED VIA SUTHERLAND'S LAW
"""

def compute_velocity_gradient(sol: np.ndarray, grad_q: np.ndarray) -> np.ndarray:
    '''
    Velocity gradient tensor du_i/dx_d, recovered from the CONSERVED-variable gradient by the
    chain rule:  du_i/dx_d = (d(rho u_i)/dx_d - u_i drho/dx_d) / rho.

    Inputs:
        - sol      : (..., num_eq)        conserved variables [rho, rho*u_i, rho*E]
        - grad_q   : (..., num_eq, ndim)  gradient of the conserved variables

    Outputs:
        - grad_vel : (..., ndim, ndim)    grad_vel[..., i, d] = du_i/dx_d
    '''

    rho       = sol[..., 0]                    # (...,)
    vel       = sol[..., 1:-1] / rho[..., None]  # (..., ndim)
    grad_rho  = grad_q[..., 0   , :]           # (..., ndim)
    grad_mom  = grad_q[..., 1:-1, :]           # (..., ndim, ndim)

    return (grad_mom - vel[..., :, None] * grad_rho[..., None, :]) / rho[..., None, None]

def compute_temperature_gradient(sol: np.ndarray, grad_q: np.ndarray, case_cfg: CaseCfg) -> np.ndarray:
    '''
    Temperature gradient dT/dx_d, recovered from the CONSERVED-variable gradient by the chain rule:

        dp/dx_d = (gamma-1)[ d(rhoE)/dx_d - sum_i u_i d m_i/dx_d + 1/2 |u|^2 d rho/dx_d ]
        dT/dx_d = gamma M^2 / rho * ( dp/dx_d - (p/rho) d rho/dx_d )        [T = gamma M^2 p/rho]

    Inputs:
        - sol      : (..., num_eq)        conserved variables
        - grad_q   : (..., num_eq, ndim)  gradient of the conserved variables
        - case_cfg : validated case configuration (gamma, mach_ref).

    Outputs:
        - grad_T   : (..., ndim)
    '''

    gamma = case_cfg.physics.gamma
    MRef2 = case_cfg.physics.mach_ref**2

    rho       = sol[..., 0]
    vel       = sol[..., 1:-1] / rho[..., None]
    grad_rho  = grad_q[..., 0   , :]
    grad_mom  = grad_q[..., 1:-1, :]
    grad_rhoE = grad_q[..., -1  , :]

    vel_dot_grad_mom = np.einsum('...i,...id->...d', vel, grad_mom)     # sum_i u_i dm_i/dx_d
    vel_sq           = np.sum(vel * vel, axis=-1)                       # |u|^2
    p                = compute_pressure(sol, case_cfg)
    grad_p           = (gamma - 1.0) * (grad_rhoE - vel_dot_grad_mom + 0.5 * vel_sq[..., None] * grad_rho)

    return (gamma * MRef2 / rho[..., None]) * (grad_p - (p / rho)[..., None] * grad_rho)

def compute_heat_flux(sol: np.ndarray, grad_q: np.ndarray, case_cfg: CaseCfg) -> np.ndarray:
    '''
    The heat-conduction term of the viscous ENERGY flux, kappa * grad(T) / Re, with

        kappa = mu* / ( Re * Pr * (gamma-1) * M_ref^2 ).

    This is exactly the piece of F_visc[..., -1, :] that carries heat, isolated so an ADIABATIC wall
    can subtract its wall-normal component and drive  k dT/dn = 0  exactly (see boundaryConditions).

    Inputs / Outputs mirror compute_viscous_flux; returns (..., ndim).
    '''

    gamma = case_cfg.physics.gamma
    Re    = case_cfg.physics.Re
    Pr    = case_cfg.physics.Pr
    MRef2 = case_cfg.physics.mach_ref**2

    mu     = compute_viscosity(sol, case_cfg)                          # (...,)
    kappa  = mu / (Pr * (gamma - 1.0) * MRef2)                         # (...,)
    grad_T = compute_temperature_gradient(sol, grad_q, case_cfg)       # (..., ndim)

    return (1.0 / Re) * kappa[..., None] * grad_T

def compute_viscous_flux(sol: np.ndarray, grad_q: np.ndarray, case_cfg: CaseCfg) -> np.ndarray:
    '''
    The viscous flux vector, F_Vec_Visc, for the non-dimensional Navier-Stokes equations.

    Inputs:
        - sol      : (..., num_eq)        conserved variables [rho, rho*u_i, rho*E]  (num_eq = 2 + ndim)
        - grad_q   : (..., num_eq, ndim)  gradient of the CONSERVED variables: grad_q[..., k, d] = d q_k/dx_d
        - case_cfg : validated case configuration (Re, Pr, mach_ref, gamma).

        Outputs:
        - F_visc   : (..., num_eq, ndim)  viscous flux; F_visc[..., k, d] is the direction-d flux of eqn k.
    '''

    Re    = case_cfg.physics.Re
    ndim  = case_cfg.mesh.ndim

    # Extract conservative variables
    rho  = sol[..., 0]                   # (...,)
    mom  = sol[..., 1:-1]                # (..., ndim) rho*u_i
    vel  = mom / rho[..., None]          # (..., ndim) u_i

    # Recover the velocity gradient tensor and its trace (the velocity divergence) from grad_q
    grad_vel = compute_velocity_gradient(sol, grad_q)   # (..., ndim, ndim)
    div_u    = np.einsum('...ii->...', grad_vel)        # du_i/dx_i

    # Compute the stress tensor
    mu     = compute_viscosity(sol, case_cfg)    # (...,) dimensionless mu
    I      = np.eye(ndim)
    strain = grad_vel + np.swapaxes(grad_vel, -1, -2) - (2.0 / 3.0) * div_u[..., None, None] * I
    tau    = mu[..., None, None] * strain

    # Energy flux = viscous work (sum_j u_j tau_jd) + heat conduction. Only the work term picks up the
    # 1/Re here; compute_heat_flux already carries its own 1/Re.
    work   = np.einsum('...j,...jd->...d', vel, tau)               # (..., ndim)
    energy = (1.0 / Re) * work + compute_heat_flux(sol, grad_q, case_cfg)

    # Construct F_Vec_Visc
    mass       = np.zeros_like(rho)[..., None, None] * np.zeros(ndim)   # (..., 1, ndim) of zeros
    F_Vec_Visc = np.concatenate([mass, (1.0 / Re) * tau, energy[..., None, :]], axis=-2)

    return F_Vec_Visc