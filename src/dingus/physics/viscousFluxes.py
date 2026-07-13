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

    gamma = case_cfg.physics.gamma
    Re    = case_cfg.physics.Re
    Pr    = case_cfg.physics.Pr
    MRef2 = case_cfg.physics.mach_ref**2
    ndim  = case_cfg.mesh.ndim

    # Extract conservative variables
    rho  = sol[..., 0]                   # (...,)
    mom  = sol[..., 1:-1]                # (..., ndim) rho*u_i
    vel  = mom / rho[..., None]          # (..., ndim) u_i

    # Extract gradients
    grad_rho  = grad_q[..., 0   , :]     # (..., ndim)
    grad_mom  = grad_q[..., 1:-1, :]     # (..., ndim, ndim)
    grad_rhoE = grad_q[..., -1  , :]     # (..., ndim)

    # Compute velcity gradient tensor: du_i/dx_d = (d_mom_i/dx_d - u_i * drho/dx_d) / rho  (Chain rule)
    grad_vel = (grad_mom - vel[..., :, None] * grad_rho[..., None, :]) / rho[..., None, None]

    # Compute the velocity divergence: du_i/dx_i = [du/dx, dv/dy, dw/dz]
    div_u    = np.einsum('...ii->...', grad_vel)

    # Compute the temperature gradient via the chain rule:
    #   dp/dx_d = (gamma-1)[ d(rhoE)/dx_d - sum_i u_i d m_i/dx_d + 1/2 |u|^2 d rho/dx_d ]
    #   dT/dx_d = gamma M^2 / rho * ( dp/dx_d - (p/rho) d rho/dx_d )        [T = gamma M^2 p/rho]
    vel_dot_grad_mom = np.einsum('...i,...id->...d', vel, grad_mom)     # sum_i u_i dm_i/dx_d
    vel_sq           = np.sum(vel * vel, axis=-1)                       # |u|^2
    p                = compute_pressure(sol, case_cfg)
    grad_p           = (gamma - 1.0) * (grad_rhoE - vel_dot_grad_mom + 0.5 * vel_sq[..., None] * grad_rho)
    grad_T           = (gamma * MRef2 / rho[..., None]) * (grad_p - (p / rho)[..., None] * grad_rho)

    # Compute the stress tensor
    mu     = compute_viscosity(sol, case_cfg)    # (...,) dimensionless mu
    I      = np.eye(ndim)
    strain = grad_vel + np.swapaxes(grad_vel, -1, -2) - (2.0 / 3.0) * div_u[..., None, None] * I
    tau    = mu[..., None, None] * strain

    # Compute the 
    kappa  = mu / (Pr * (gamma - 1.0) * MRef2)          # (...,) heat-flux coefficient
    work   = np.einsum('...j,...jd->...d', vel, tau)    # sum_j u_j tau_jd  (viscous work)
    energy = work + kappa[..., None] * grad_T           # (..., ndim)

    # DIVIDE BY RE 
    # Construct F_Vec_Visc
    mass = np.zeros_like(grad_rho)[..., None, :]
    F_Vec_Visc = (1.0 / Re) * np.concatenate([mass, tau, energy[..., None, :]], axis=-2)
    
    return F_Vec_Visc