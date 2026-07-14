# src/dingus/timeIntegrators/timeIntegration.py

import numpy as np
from dingus.config import CaseCfg
from dingus.physics import fluxes
from dingus.physics.constitutiveRelations import compute_viscosity
from dingus.spatialOperators.residual import compute_residual

'''
Explicit time integration of the semi-discrete DG system  dq/dt = R(q, t), where R is the
spatial residual (compute_residual writes R into each element's `.residual`).

The time integration falls into two major tasks:
  1. compute_dt : determine the CFL-limited stable time step from the fastest wave speed 
                  and grid spacing.
  2. time_step  : advance the solution one step with the scheme chosen in the control file
                  (euler / rk2 / rk4), dispatched like the other modules.

The runner owns the while-loop over time; this module just sizes and takes single steps.

Each element carries q in 'e.solution' and R in 'e.residual', so a "stage" means: 
    - Set every element's '.solution' to the stage value
    - Call compute_residual
    - Then harvest each '.residual' (COPY it -- the next call overwrites it).
'''

def compute_dt(mesh, case_cfg: CaseCfg) -> float:
    '''
    Computes the CFL-limited stable time step from a metric-based (contravariant) estimate:

        dt = CFL / max_over_elements[ lambda * (2P+1) / h_min ]

    where, per element:
      - lambda   : fastest signal (wave) speed in that element (fluxes.max_wave_speed).
                   Recomputed every step, since it changes with the solution (and can vary
                   element-to-element for nonlinear physics like Euler/NSE).

      - (2P+1)   : the high-order node-clustering factor. Legendre nodes bunch up near element
                   edges with O(1/P^2) spacing, so the operator's spectral radius grows with P;
                   (2P+1) is the standard DGSEM scaling (Gassner-Kopriva).

      - h_min    : a metric-based length scale, h_min = 1 / max_nodes( sum_i |grad(xi^i)| ),
                   where grad(xi^i) are the contravariant vectors (contravar_xi/eta/zeta) with
                   units 1/length. This is PURE GEOMETRY -- it never changes during the run, so
                   it is precomputed once per element in compute_element_metrics and stored as
                   e.h_min. Because it comes from the metric terms, it automatically shrinks on
                   stretched or curved elements and works unchanged in 1D/2D/3D.

    Combining a per-element wave speed with a per-element length scale (rather than global maxima)
    means the element with the tightest local  lambda x resolution  correctly sets the step --
    important once fast regions and fine regions live in different elements (Euler/NSE).

    Inputs:
    - mesh     : constructed Mesh (elements carry the precomputed e.h_min and current e.solution).
    - case_cfg : validated case configuration (provides the CFL number).

    Outputs:
    - dt : float, the stable time step for the current solution state.
    '''

    # Extract the polynomial order from the mesh object
    P_factor = 2 * mesh.el_poly_order +1

    # Initialize time step as 1/inf
    inv_dt = 0.0

    # Extract the CFL from the case configuration object
    cfl = case_cfg.time_stepping.cfl

    # Determine if viscous time scales come into play (aka is this a navier-stokes model)
    is_nse = case_cfg.physics.model == 'navier-stokes'

    for e in mesh.elements:
        ##### Advective (inviscid) time step #####
        # Grab the maximum wavespeed per element and compute the inviscid time step size
        lam        = fluxes.max_wave_speed(e.solution, case_cfg) 
        inv_dt_adv = lam * P_factor / e.h_min   

        ##### Diffusive (invisid) time step #####
        inv_dt_visc = 0.0
        if is_nse:
            rho    = e.solution[..., 0]
            mu     = compute_viscosity(e.solution, case_cfg)
            nu_max = np.max( mu / (rho * case_cfg.physics.Re)) * \
                     max(4.0 / 3.0, case_cfg.physics.gamma / case_cfg.physics.Pr)
            
            inv_dt_visc = nu_max * P_factor**2 / e.h_min**2

        # use the smallest time step: inviscid or viscous
        inv_dt = max(inv_dt, inv_dt_adv + inv_dt_visc)

    # Compute the time step estimate
    dt = cfl / inv_dt

    return dt

def _euler_step(mesh, case_cfg: CaseCfg, t: float, dt: float) -> None:
    '''
    Forward Euler time integrator:

            q{n+1} = q^n + dt * R(q^n, t)

    This is first order accurate and quite dissipative. I recommend only using this for 
    debugging purposes.
    '''

    compute_residual(mesh, case_cfg, t)
    for e in mesh.elements:
        e.solution = e.solution + dt * e.residual

def _rk2_step(mesh, case_cfg: CaseCfg, t: float, dt: float) -> None:
    '''
    2-stage Runge-Kutta time stepper. This is second order accurate and requires 2 evaulations
    of the residuals per time step.

    The RK state times (t, t+dt/2) are passed into compute_residual() just in 
    case unsteady boundary conditions are used. If the boundary conditions are steady, these
    times do not affect the solution at all.  
    '''

    # solution at current time step
    q0 = [e.solution.copy() for e in mesh.elements]

    # Stage 1: k1 = R(q0, t)
    compute_residual(mesh, case_cfg, t)
    k1 = [e.residual.copy() for e in mesh.elements]

    # Stage 2: k2 = R(q0, t+dt)
    for e, q, k in zip(mesh.elements, q0, k1):
        e.solution = q + 0.5 * dt * k
    compute_residual(mesh, case_cfg, t + 0.5 * dt)
    k2 = [e.residual.copy() for e in mesh.elements]

    # Use the "mid-point" (or stage 2) residual to take the full time step
    for e, q, k in zip(mesh.elements, q0, k2):
        e.solution = q + dt * k

def _rk4_step(mesh, case_cfg: CaseCfg, t: float, dt: float) -> None:
    '''
    4-stage Runge-Kutta time stepper. This is fourth order accurate and requires 4 evaluations
    of the residual per time step.

    The RK state times (t, t+dt/2, t+dt/2, t+dt) are passed into compute_residual() just in 
    case unsteady boundary conditions are used. If the boundary conditions are steady, these
    times do not affect the solution at all. 
    '''

    # solution at current time step
    q0 = [e.solution.copy() for e in mesh.elements]

    # Stage 1: k1 = R(q0, t)
    compute_residual(mesh, case_cfg, t)
    k1 = [e.residual.copy() for e in mesh.elements]

    # Stage 2: k2 = R(q0 + dt/2 * k1, t + dt/2)
    for e, q, k in zip(mesh.elements, q0, k1):
        # Update solution saved onto the element
        e.solution = q + 0.5 * dt * k
    compute_residual(mesh, case_cfg, t + 0.5*dt)
    k2 = [e.residual.copy() for e in mesh.elements]

    # Stage 3: k3 = R(q0 + dt/2 * k2, t + dt/2)
    for e, q, k in zip(mesh.elements, q0, k2):
        e.solution = q + 0.5 * dt * k
    compute_residual(mesh, case_cfg, t + 0.5 * dt)
    k3 = [e.residual.copy() for e in mesh.elements]

    # Stage 4: k4 = R(q0 + dt * k3, t + dt) 
    for e, q, k in zip(mesh.elements, q0, k3):
        e.solution = q + dt * k
    compute_residual(mesh, case_cfg, t + dt)
    k4 = [e.residual.copy() for e in mesh.elements]

    # Compute the weighted residuals from each RK stage and advance the solution one time step
    for e, q, a, b, c, d in zip(mesh.elements, q0, k1, k2, k3, k4):
        e.solution = q + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d)

def time_step(mesh, case_cfg: CaseCfg, t: float, dt: float) -> None:
    '''
    Advance one step step using the integrator specified in the control file. 
    This just a wrapper function that will call the appropriate time integrator.
    '''

    match case_cfg.time_stepping.time_integrator:
        case 'euler':
            _euler_step(mesh, case_cfg, t, dt)
        case 'rk2':
            _rk2_step(mesh, case_cfg, t, dt)
        case 'rk4':
            _rk4_step(mesh, case_cfg, t, dt)
        case _:
            raise ValueError(f"Unknown time integrator: '{case_cfg.time_stepping.time_integrator}'.")