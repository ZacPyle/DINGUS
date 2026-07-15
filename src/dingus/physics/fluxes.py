# src/dingus/physics/fluxes.py

from dingus.config import CaseCfg
from dingus.physics.constitutiveRelations import compute_pressure
from dingus.physics.viscousFluxes import compute_viscous_flux
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

# ---------------------------------------------------------------------------------------------
# Entropy fix
# ---------------------------------------------------------------------------------------------
# Entropy-fix strength: the acoustic eigenvalues |u.n +/- c| are floored below
# eps * (|u.n| + c) to keep the Roe dissipation from vanishing at sonic points
# (which would otherwise admit entropy-violating expansion shocks). eps ~ 0.1 is standard.
_ROE_ENTROPY_FIX_EPS = 0.1

def _harten_entropy_fix(lam: np.ndarray, delta: np.ndarray) -> np.ndarray:
    '''
    Harten's smooth entropy fix for |lambda|. Where |lambda| >= delta it returns |lambda|
    unchanged; where |lambda| < delta it replaces the kink at zero with the parabola
    (lambda^2 + delta^2) / (2 delta), so the dissipation stays bounded away from zero through
    a sonic point. 'lam' and 'delta' are (M,) arrays; 'delta > 0'.

    For reference, see:
        Harten, A. (1983), "High Resolution Schemes for Hyperbolic Conservation Laws," 
            Journal of Computational Physics, 49(3), 357–393. 

        Toro, E. F. (2009), Riemann Solvers and Numerical Methods for Fluid Dynamics, 
            3rd ed., Springer
    '''
    abs_lam = np.abs(lam)
    return np.where(abs_lam >= delta, abs_lam, 0.5 * (lam**2 + delta**2) / delta)


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

        case 'euler'|'navier-stokes':
            # Extract density and momentum from solution vector
            rho      = sol[...,0]
            momentum = sol[...,1:-1]

            # Compute the speed of sound and velocity magnitude
            pressure = compute_pressure(sol, case_cfg)
            c        = np.sqrt(case_cfg.physics.gamma * pressure / rho)
            vel_mag  = np.linalg.norm(momentum, axis=-1) / rho

            # Compute the max wave speed (velocity_mag + speed of sound)
            return float(np.max(vel_mag + c))
        
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

        case 'euler'|'navier-stokes':
            # compute the maximum wavespeed between the two faces using the interior (minus) and exterior (plus) states
            lam = np.maximum(_max_wave_speed_one_face_side(q_minus, normal, case_cfg),
                             _max_wave_speed_one_face_side(q_plus , normal, case_cfg))
            
            return lam[:, None]

        case _:
            raise ValueError(f"Unknown physics model: '{case_cfg.physics.model}'.")
        
def _roe_dissipation_euler(q_minus : np.ndarray,
                           q_plus  : np.ndarray,
                           normal  : np.ndarray,
                           case_cfg: CaseCfg) -> np.ndarray:
    '''
    Roe matrix dissipation |A_Roe| . (q_plus - q_minus) for the compressible Euler equations,
    written dimension-agnostically (works for ndim = 1, 2, 3). The jump across the face is
    decomposed into the three Euler wave families sharing the Roe-averaged state:

        - acoustic waves    : eigenvalue u_n -/+ c   (genuinely nonlinear; entropy-fixed)
        - entropy/contact   : eigenvalue u_n         (linearly degenerate)
        - shear wave(s)     : eigenvalue u_n         (tangential velocity jump)

    and each wave is dissipated by its own |eigenvalue|. See Toro, 'Riemann Solvers and 
    Numerical Methods for Fluid Dynamics', Ch. 11 and Roe (1981).

    Inputs mirror _roe_dissipation; returns (M, num_eq) = |A_Roe| . (q_plus - q_minus), which
    the numerical flux uses as   F* = central - 0.5 * D.
    '''

    gamma = case_cfg.physics.gamma

    # Extract left (interior) state, constructing primitive variables from conserved variables
    rho_L = q_minus[..., 0   ]
    vel_L = q_minus[..., 1:-1] / rho_L[..., None]
    p_L   = compute_pressure(q_minus, case_cfg)
    H_L   = (q_minus[..., -1] + p_L) / rho_L          # total enthalpy: h = e + p/rho = (rhoE + p) / rho

    # Extract right (exterior) state, constructing primitive variables from conserved variables
    rho_R = q_plus[..., 0   ]
    vel_R = q_plus[..., 1:-1] / rho_R[..., None]
    p_R   = compute_pressure(q_plus, case_cfg)
    H_R   = (q_plus[..., -1] + p_R) / rho_R          # total enthalpy: h = e + p/rho = (rhoE + p) / rho

    # Compute the rho average weights
    sq_L   = np.sqrt(rho_L)
    sq_R   = np.sqrt(rho_R)
    inv_sq = 1.0 / (sq_L + sq_R)

    # Compute the rho averages
    rho = sq_L * sq_R                           # Roe-averaged density
    vel = (sq_L[..., None] * vel_L + sq_R[..., None] * vel_R) * inv_sq[..., None]
    H   = (sq_L            * H_L   + sq_R            * H_R  ) * inv_sq
    un = np.sum(vel * normal, axis=-1)          # Roe-averaged normal velocity
    q2 = np.sum(vel * vel, axis=-1)
    c  = np.sqrt((gamma-1.0) * (H - 0.5 * q2))  # Roe-averaged sound speed (M,)

    # Compute the jumps across the interface
    drho = rho_R - rho_L
    dp   = p_R   - p_L
    dvel = vel_R - vel_L
    dun  = np.sum(dvel * normal, axis=-1)  # normal velocity jump
    dvt  = dvel - dun[..., None] * normal  # tangential velocity jump

    # Compute wave strengths
    inv_c2 = 1.0 / (c * c)
    a_acoustic_minus = 0.5 * (dp - rho * c * dun) * inv_c2   # wave with eigenvalue un - c
    a_acoustic_plus  = 0.5 * (dp + rho * c * dun) * inv_c2   # wave with eigenvalue un + c
    a_entropy        = drho - dp * inv_c2                    # entropy wave

    # Compute the eigenvalues and add the smooth entropy fix
    delta = _ROE_ENTROPY_FIX_EPS * (np.abs(un) + c)
    abs_lam_minus = _harten_entropy_fix(un - c, delta)
    abs_lam_plus  = _harten_entropy_fix(un + c, delta)
    abs_lam_mid   = np.abs(un)

    ##### Construct the dissipation terms #####

    # Density component
    d_rho = abs_lam_minus * a_acoustic_minus + \
            abs_lam_mid   * a_entropy        + \
            abs_lam_plus  * a_acoustic_plus
    
    # Momentum components. Acoustic eigenvectors contain vel +/- c*n, the middle field
    # contains the entropy wave (alpha * vel) and the shear wave (rho * tangential jump)
    d_mom = ((abs_lam_minus * a_acoustic_minus)[..., None] * (vel - c[..., None] * normal) +
             (abs_lam_plus  * a_acoustic_plus )[..., None] * (vel + c[..., None] * normal) +
             abs_lam_mid[..., None] * (a_entropy[..., None] * vel + rho[..., None] * dvt))
    
    # Energy component
    d_E = (abs_lam_minus * a_acoustic_minus * (H - un * c) +
           abs_lam_plus  * a_acoustic_plus  * (H + un * c) +
           abs_lam_mid   * (0.5 * a_entropy * q2 + rho * np.sum(vel * dvt, axis=-1)))

    return np.concatenate([d_rho[..., None], d_mom, d_E[..., None]], axis=-1)

def characteristic_ghost_state(q_int   : np.ndarray,
                               q_ext   : np.ndarray,
                               normal  : np.ndarray,
                               case_cfg: CaseCfg) -> np.ndarray:
    '''
    The CHARACTERISTIC far-field ghost state for the compressible Euler equations:

        q_bc = 1/2 (q_int + q_ext)  +  1/2 sign(A_n) (q_int - q_ext)

    where sign(A_n) = R sign(Lambda) L applies the SIGN of each characteristic speed of the normal
    flux Jacobian, evaluated at the Roe average of (q_int, q_ext). Per characteristic wave: an OUTGOING
    one (lambda > 0) is taken from the INTERIOR (it carries information out of the domain), an INCOMING
    one (lambda < 0) from the EXTERIOR reference (it carries information in). This imposes exactly as
    many conditions as there are incoming characteristics -> non-reflecting, and it self-selects the
    count per node/regime (supersonic out -> q_int, supersonic in -> q_ext, subsonic -> a genuine blend).

    This is the SAME Roe-averaged wave decomposition as _roe_dissipation_euler, with sign(lambda) in
    place of |lambda| (and no entropy fix -- a stationary characteristic, lambda = 0, simply drops out,
    leaving the average, which is correct). Inputs mirror _roe_dissipation_euler; returns (M, num_eq).
    '''

    gamma = case_cfg.physics.gamma

    # Interior ("minus") and exterior ("plus") primitive states
    rho_L = q_int[..., 0]; vel_L = q_int[..., 1:-1] / rho_L[..., None]
    p_L   = compute_pressure(q_int, case_cfg); H_L = (q_int[..., -1] + p_L) / rho_L
    rho_R = q_ext[..., 0]; vel_R = q_ext[..., 1:-1] / rho_R[..., None]
    p_R   = compute_pressure(q_ext, case_cfg); H_R = (q_ext[..., -1] + p_R) / rho_R

    # Roe averages (identical to the Roe flux)
    sq_L, sq_R = np.sqrt(rho_L), np.sqrt(rho_R)
    inv_sq = 1.0 / (sq_L + sq_R)
    rho = sq_L * sq_R
    vel = (sq_L[..., None] * vel_L + sq_R[..., None] * vel_R) * inv_sq[..., None]
    H   = (sq_L * H_L + sq_R * H_R) * inv_sq
    un  = np.sum(vel * normal, axis=-1)
    q2  = np.sum(vel * vel, axis=-1)
    c   = np.sqrt((gamma - 1.0) * (H - 0.5 * q2))

    # Jumps q_ext - q_int and the wave strengths (identical to the Roe flux)
    drho = rho_R - rho_L
    dp   = p_R - p_L
    dvel = vel_R - vel_L
    dun  = np.sum(dvel * normal, axis=-1)
    dvt  = dvel - dun[..., None] * normal
    inv_c2 = 1.0 / (c * c)
    a_ac_minus = 0.5 * (dp - rho * c * dun) * inv_c2
    a_ac_plus  = 0.5 * (dp + rho * c * dun) * inv_c2
    a_entropy  = drho - dp * inv_c2

    # SIGN of each characteristic speed (no entropy fix): +1 outgoing, -1 incoming, 0 stationary.
    s_minus = np.sign(un - c)
    s_plus  = np.sign(un + c)
    s_mid   = np.sign(un)

    # D_sign = sign(A_n) (q_ext - q_int), assembled from the same eigenvectors as the Roe dissipation.
    d_rho = s_minus * a_ac_minus + s_mid * a_entropy + s_plus * a_ac_plus
    d_mom = ((s_minus * a_ac_minus)[..., None] * (vel - c[..., None] * normal) +
             (s_plus  * a_ac_plus )[..., None] * (vel + c[..., None] * normal) +
             s_mid[..., None] * (a_entropy[..., None] * vel + rho[..., None] * dvt))
    d_E   = (s_minus * a_ac_minus * (H - un * c) +
             s_plus  * a_ac_plus  * (H + un * c) +
             s_mid   * (0.5 * a_entropy * q2 + rho * np.sum(vel * dvt, axis=-1)))
    D_sign = np.concatenate([d_rho[..., None], d_mom, d_E[..., None]], axis=-1)

    # q_bc = 1/2 (q_int + q_ext) - 1/2 sign(A_n)(q_ext - q_int)
    return 0.5 * (q_int + q_ext) - 0.5 * D_sign

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
        
        case 'euler'|'navier-stokes':
            return _roe_dissipation_euler(q_minus, q_plus, normal, case_cfg)    

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

def compute_inviscid_flux(sol : np.ndarray, case_cfg : CaseCfg) -> np.ndarray:
    '''
    The INVISCID (hyperbolic) half of the physical flux for the active physics model. For
    'navier-stokes' this is just the Euler flux -- the viscous flux is a separate, additive piece.

    Kept separate from compute_volume_flux because the two halves need DIFFERENT treatment at an
    interface: the inviscid half goes through the Riemann solver (central + upwind dissipation),
    while the viscous half is averaged centrally (BR1 -- there is no upwind direction for diffusion)
    and, at a wall, is replaced outright by a BC-specific flux. See compute_numerical_flux.

    Inputs / Outputs mirror compute_volume_flux.
    '''

    match case_cfg.physics.model:
        case 'scalar_advection':
            return _compute_scalar_flux(sol, case_cfg)

        case 'euler' | 'navier-stokes':
            return _compute_euler_flux(sol, case_cfg)

        case _:
            raise ValueError(f"Unknown physics model detected in inviscid flux computation: '{case_cfg.physics.model}'.")

def compute_volume_flux(sol : np.ndarray, case_cfg : CaseCfg, grad_q : np.ndarray | None = None) -> np.ndarray:
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

    # The inviscid flux is common to every model; Navier-Stokes then subtracts the viscous flux.
    flux = compute_inviscid_flux(sol, case_cfg)

    if case_cfg.physics.model == 'navier-stokes':
        flux = flux - compute_viscous_flux(sol, grad_q, case_cfg)

    return flux

def compute_numerical_flux(q_minus            : np.ndarray,
                           q_plus             : np.ndarray,
                           normal             : np.ndarray,
                           case_cfg           : CaseCfg,
                           grad_minus          : np.ndarray | None = None,
                           grad_plus           : np.ndarray | None = None,
                           viscous_normal_flux : np.ndarray | None = None,
                           inviscid_normal_flux: np.ndarray | None = None) -> np.ndarray:

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

    For NAVIER-STOKES the flux carries a second, PARABOLIC half. It does NOT go through the Riemann
    solver (diffusion has no upwind direction); BR1 simply averages it centrally:

        F*.n  =  [ inviscid Riemann flux above ]  -  1/2 ( F_visc(q^-,g^-).n + F_visc(q^+,g^+).n )

    At a WALL that central average is meaningless -- the exterior state there is a reflection built
    for the Riemann solver, not a physical neighbour, and an adiabatic wall must additionally carry
    zero normal heat flux. Callers therefore pass `viscous_normal_flux` to REPLACE the averaged
    viscous term with a BC-specific one (see boundaryConditions.wall_viscous_normal_flux).

    Note: the sign convention MUST match the DG residual:
    - q_minus is the trace from the "left" element (element that owns the face)
    - q_plus is the trace from the "right" element (element that does NOT own the face)
    - normal is the UNIT normal vector pointing OUT of the left element (toward the right element)

    Inputs:
    - q_minus : (..., num_eq) array of conserved variables from the left/owner element.
    - q_plus  : (..., num_eq) array of conserved variables from the right/neighbor element.
    - normal  : (..., ndim) array of unit normal vectors at the face, pointing outward of the left/owner element
    - case_cfg: case configuration (selects the physics model + parameters).
    - grad_minus, grad_plus   : (..., num_eq, ndim) conserved-variable gradients from either side (NSE only).
    - viscous_normal_flux     : (..., num_eq) optional OVERRIDE for the viscous normal flux F_visc.n at
                                this face (NSE only). When given it is used verbatim, in place of the BR1
                                central average of the two sides. This is how the wall BCs are imposed.
    - inviscid_normal_flux    : (..., num_eq) optional OVERRIDE for the inviscid (hyperbolic) normal flux
                                at this face, used verbatim in place of the central + Riemann dissipation.
                                This is how a CHARACTERISTIC far-field BC imposes F(q_bc).n directly.

    Outputs:
    - f_star_n : (..., num_eq) array of numerical fluxes at the face, projected onto the face normal.
    '''

    # --- Hyperbolic half: the caller's override, or central average of the INVISCID flux + Riemann
    #     dissipation. compute_inviscid_flux returns (M, num_eq, ndim); contracting ndim with `normal`
    #     gives F.n.
    if inviscid_normal_flux is not None:
        f_star = inviscid_normal_flux
    else:
        Fn_minus = np.einsum('med,md->me', compute_inviscid_flux(q_minus, case_cfg), normal)   # (M, num_eq)
        Fn_plus  = np.einsum('med,md->me', compute_inviscid_flux(q_plus , case_cfg), normal)   # (M, num_eq)
        central  = 0.5 * (Fn_minus + Fn_plus)

        # Compute the dissipation term specific to the chosen Riemann solver
        match case_cfg.physics.riemann_solver:
            case 'upwind' | 'LLF':
                # Scalar (Lax-Friedrichs / Rusanov) dissipation. For linear scalar advection physics models, this
                # reproduces EXACT upwinding with lam = |a.n|, the formula collapses to
                #   a.n * (q_minus if a.n >= 0 else q_plus).
                lam    = _max_abs_face_speed(q_minus, q_plus, normal, case_cfg)   # (M, 1)
                f_star = central - 0.5 * lam * (q_plus - q_minus)

            case 'roe':
                # Matrix dissipation; physics-specific (returns the full D.(q_plus - q_minus) already).
                f_star = central - 0.5 * _roe_dissipation(q_minus, q_plus, normal, case_cfg)

            case _:
                raise ValueError(
                    f"Unknown Riemann solver: '{case_cfg.physics.riemann_solver}'. "
                    "Expected one of: 'upwind', 'LLF', 'roe'."
                )

    # --- Parabolic half: BR1 central viscous flux, or the caller's wall override ------------------
    if case_cfg.physics.model == 'navier-stokes':
        if viscous_normal_flux is None:
            Fv_minus            = np.einsum('med,md->me', compute_viscous_flux(q_minus, grad_minus, case_cfg), normal)
            Fv_plus             = np.einsum('med,md->me', compute_viscous_flux(q_plus , grad_plus , case_cfg), normal)
            viscous_normal_flux = 0.5 * (Fv_minus + Fv_plus)

        f_star = f_star - viscous_normal_flux

    return f_star