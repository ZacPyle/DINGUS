# src/dingus/boundaryConditions/boundaryConditions.py

import importlib.util
import numpy as np
from pathlib import Path
from dingus.config import BCCfg, CaseCfg
from dingus.mesh import mortar_class
from dingus.physics.constitutiveRelations import compute_temperature
from dingus.physics.viscousFluxes import compute_heat_flux, compute_viscous_flux
from dingus.physics.constitutiveRelations import compute_viscosity

'''
Boundary conditions in a DG scheme are imposed WEAKLY, through the numerical (interface) flux.
At a boundary face the scheme still evaluates a Riemann problem between two states:

    q_minus : the interior trace (from the element that owns the boundary face)
    q_plus  : an EXTERIOR "ghost" state that encodes the boundary condition

There is no physical element on the outside, so `q_plus` is *manufactured* from the BC type.
Feeding (q_minus, q_plus) into the same `compute_numerical_flux` used for interior interfaces
then enforces the condition automatically (e.g. upwinding decides whether the interior or the
prescribed exterior state is used, based on the flow direction).

So given a boundary mortar and its interior trace, this module will return q_plus.
The surface term calls it ONLY for boundary mortars; interior mortars get q_plus from the
neighbor element directly.

----------------------------------------------------------------------------------------------
WALLS (Navier-Stokes)
----------------------------------------------------------------------------------------------
A viscous wall cannot be expressed as ONE ghost state, because the two halves of the NSE flux
want different things from it. This module therefore exposes THREE wall quantities, each used by
exactly one place in the residual:

  1. `exterior_state`          -> the ghost for the INVISCID (Riemann) flux.
        All four wall types share it: reflect the NORMAL momentum, keep rho and rhoE.
        Since |mom| is unchanged, the ghost pressure equals the interior pressure (so the wall can
        never manufacture a negative pressure), the central mass and energy fluxes vanish
        identically, and the momentum flux collapses to the physical wall traction p*n as u.n -> 0.
        The no-slip condition is deliberately NOT imposed here: an impermeable wall carries no
        convective tangential momentum flux (u.n = 0), so no-slip is a purely VISCOUS statement.
        (Reflecting the full momentum vector instead would inject a spurious O(lambda * u_t)
        tangential flux through the Riemann dissipation.) This is the standard split -- cf. Trixi's
        pairing of `boundary_condition_slip_wall` with its Navier-Stokes wall BC.

  2. `wall_state`              -> the DIRICHLET data (rho, rho*u_w, rhoE_w) that the wall imposes.
        This is what no-slip / isothermal actually mean:
            no-slip   : u_w = prescribed wall velocity (0 for a stationary wall)
            slip      : u_w = interior velocity with its normal component removed (symmetry plane)
            isothermal: T_w = prescribed wall temperature
            adiabatic : T_w = interior temperature (an adiabatic wall constrains the heat FLUX, not T)

  3. `gradient_exterior_state` -> the ghost for the BR1 GRADIENT pass.
        BR1 wants the central trace q* = 1/2 (q- + q+) to equal the Dirichlet data q_w exactly, so
        the ghost is the mirror of the interior about the wall state:  q+ = 2 q_w - q-.
        Do NOT feed this mirror to the Riemann solver: its energy component can go pressure-negative.

  4. `wall_viscous_normal_flux`-> the viscous normal flux F_visc.n AT the wall, replacing the BR1
        central average (which would otherwise average in the reflected ghost). Built from the WALL
        state and the INTERIOR gradient; for an adiabatic wall the wall-normal heat flux is then
        subtracted off so that k dT/dn = 0 holds exactly.
'''

# The four viscous wall types. All are impermeable (u.n = 0); they differ only in how the tangential
# velocity and the temperature are constrained, which is entirely a VISCOUS-flux distinction.
_WALL_TYPES = ('adiabatic_slip_wall'   , 'adiabatic_no_slip_wall',
               'isothermal_slip_wall'  , 'isothermal_no_slip_wall')

# ---------------------------------------------------------------------------------------------
# Wall penalty (interior-penalty / BR2-style stabilization)
# ---------------------------------------------------------------------------------------------
# A Dirichlet wall imposed WEAKLY through the BR1 central trace, with nothing penalizing the jump
# (q^- - q_wall), is NOT energy-stable: there is no mechanism damping a boundary-condition violation,
# so one grows. This is not a hypothetical -- without this term a Couette run started from rest decays
# toward the exact solution until t~4, then diverges exponentially (growth rate ~0.4) and kills itself
# on a negative pressure by t~12. It bites BOTH stationary and moving walls.
#
# The cure is the standard interior-penalty term: add a dissipative flux proportional to the BC
# violation, which drives the interior trace onto the wall state:
#
#       F_visc*.n  =  F_v(q_wall, grad^-).n  -  sigma * (q^- - q_wall)
#       sigma      =  C * (P+1)^2 * mu / (Re * h)
#
# The (P+1)^2 / h scaling is the usual one: it must dominate the inverse-trace-inequality constant of
# the polynomial basis, which grows like P^2/h. Because sigma multiplies (q^- - q_wall), the whole term
# vanishes identically for any solution that SATISFIES the boundary condition -- so it is consistent,
# it cannot perturb the exact solution, and it leaves the spectral convergence rate untouched. It only
# ever acts on the error.
#
# Note the density row penalizes nothing (rho_wall == rho^- by construction, so the jump is exactly 0),
# and on an ADIABATIC wall the energy row reduces to the kinetic-energy jump alone (T_wall == T^- there,
# so no temperature is being imposed) -- both fall out automatically, which is a good sign the penalty
# is acting on precisely the constraints the wall actually imposes.
_WALL_PENALTY = 2.0


def wall_penalty_sigma(q_minus  : np.ndarray,
                       case_cfg : CaseCfg   ,
                       h_elem   : float     ) -> np.ndarray:
    '''
    The interior-penalty coefficient sigma = C (P+1)^2 mu / (Re h) at each face node, shape (M,).

    Inputs:
    - q_minus  : (M, num_eq) interior face trace (only used for the local viscosity mu).
    - case_cfg : validated case configuration (poly_deg, Re).
    - h_elem   : a length scale for the element owning the face (its metric-based h_min).
    '''
    P  = case_cfg.mesh.poly_deg
    Re = case_cfg.physics.Re
    mu = compute_viscosity(q_minus, case_cfg)            # (M,)

    return _WALL_PENALTY * (P + 1) ** 2 * mu / (Re * h_elem)

def is_wall(bc: BCCfg) -> bool:
    '''True if this boundary condition is one of the four wall types.'''
    return bc.type in _WALL_TYPES

# ---------------------------------------------------------------------------------------------
# Prescribed-state (full Dirichlet) boundary conditions
# ---------------------------------------------------------------------------------------------
# Two BC types impose a FULL exterior state as the ghost, differing only in HOW that state is supplied:
#   uniform_inflow : a CONSTANT state, given directly as bc.state.
#   prescribed     : a spatio-temporally VARYING state g(x, t), from a user function boundary_state().
# Both then feed the SAME machinery: inviscid ghost = target; gradient ghost = 2*target - q^- (so the
# BR1 central trace lands on the target exactly); and, for NSE, the viscous flux + interior penalty
# built around the target (prescribed_viscous_normal_flux). A prescribed BC is, in effect, a "wall"
# whose target state is an arbitrary user function instead of the impermeable wall reflection -- which
# is why it reuses wall_penalty_sigma and the shared _dirichlet_viscous_normal_flux core below.
_PRESCRIBED_TYPES = ('uniform_inflow', 'prescribed')

# The user's boundary_state module is imported ONCE and cached, keyed by (file, function name): the
# surface term calls it every RK stage, so re-importing per call would dominate the runtime. Mirrors
# sourceTerms._load_source_function.
_BOUNDARY_FN_CACHE: dict = {}

def is_prescribed(bc: BCCfg) -> bool:
    '''True if this BC imposes a full prescribed exterior state (uniform_inflow or prescribed).'''
    return bc.type in _PRESCRIBED_TYPES

def _load_boundary_function(state_file, state_function: str):
    '''
    Dynamically load `state_function` (default 'boundary_state') from the user's state_file, cached.
    Mirrors sourceTerms._load_source_function. state_file must already be an absolute/resolvable path
    (the runner resolves it relative to the case directory, exactly as it does for source_file).
    '''
    state_file = Path(state_file)
    key        = (str(state_file.resolve()) if state_file.is_file() else str(state_file), state_function)

    if key in _BOUNDARY_FN_CACHE:
        return _BOUNDARY_FN_CACHE[key]

    if not state_file.is_file():
        raise FileNotFoundError(f"Boundary state file not found! Specified path is: '{state_file}'")

    spec = importlib.util.spec_from_file_location("user_boundary_state", state_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load a module spec from the boundary state file: '{state_file}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, state_function):
        raise AttributeError(f"No function '{state_function}()' found in: '{state_file}'")

    fn = getattr(module, state_function)
    _BOUNDARY_FN_CACHE[key] = fn
    return fn

def prescribed_target_state(bc         : BCCfg              ,
                            q_minus    : np.ndarray         ,
                            face_coords: np.ndarray | None  ,
                            case_cfg   : CaseCfg            ,
                            t          : float=0.0          ) -> np.ndarray:
    '''
    The full state the BC imposes at the face nodes, shape (M, num_eq).

    - uniform_inflow : broadcast the constant bc.state (face_coords unused).
    - prescribed     : evaluate the user's boundary_state(cfg, x, y, t) at the face-node coordinates.

    Inputs:
    - bc          : the BCCfg (uniform_inflow or prescribed).
    - q_minus     : (M, num_eq) interior face trace (used for shape / broadcasting).
    - face_coords : (M, ndim) physical coordinates of the face nodes. Required for 'prescribed'.
    - case_cfg    : validated case configuration.
    - t           : current time (for unsteady prescribed BCs).
    '''

    if bc.type == 'uniform_inflow':
        return np.ones_like(q_minus) * bc.state

    # prescribed: evaluate the user function at the face-node coordinates.
    if face_coords is None:
        raise ValueError("A 'prescribed' boundary condition needs the face-node coordinates, but "
                         "prescribed_target_state was called without them.")

    fn   = _load_boundary_function(bc.state_file, bc.state_function)
    ndim = case_cfg.mesh.ndim
    match ndim:
        case 1:
            state = fn(case_cfg, face_coords[..., 0], t)
        case 2:
            state = fn(case_cfg, face_coords[..., 0], face_coords[..., 1], t)
        case 3:
            state = fn(case_cfg, face_coords[..., 0], face_coords[..., 1], face_coords[..., 2], t)
        case _:
            raise ValueError(f"Unsupported dimensionality for a prescribed boundary: {ndim}")

    return np.asarray(state, dtype=float)

def wall_state(bc       : BCCfg     ,
               q_minus  : np.ndarray,
               normal   : np.ndarray,
               case_cfg : CaseCfg   ) -> np.ndarray:
    '''
    The DIRICHLET state the wall imposes on the fluid: q_w = [rho, rho*u_w, rhoE_w].

    Density is never prescribed at a wall (it has no boundary condition of its own -- there is no
    diffusive flux of mass), so it is taken from the interior trace. The velocity and temperature
    carry the actual condition:

        slip     : u_w = u- - (u-.n) n     (normal component removed -> symmetry / Euler wall)
        no-slip  : u_w = bc.wall_velocity  (0 unless a moving wall is prescribed, e.g. Couette)
        isothermal : T_w = bc.wall_temperature
        adiabatic  : T_w = T-              (temperature is a RESULT at an adiabatic wall; the
                                            zero-heat-flux condition is imposed on the flux instead,
                                            in wall_viscous_normal_flux)

    The energy is then rebuilt consistently from (rho, u_w, T_w) via p_w = rho T_w / (gamma M^2):

        rhoE_w = p_w / (gamma - 1) + 1/2 rho |u_w|^2

    Inputs:
    - bc       : the BCCfg for this wall.
    - q_minus  : (M, num_eq) interior face trace.
    - normal   : (M, ndim) OUTWARD unit normal at the face.
    - case_cfg : validated case configuration.

    Outputs:
    - q_wall   : (M, num_eq) Dirichlet wall state.
    '''

    gamma   = case_cfg.physics.gamma
    gammaM2 = gamma * case_cfg.physics.mach_ref**2
    ndim    = case_cfg.mesh.ndim

    rho = q_minus[..., 0   ]                      # (M,)
    mom = q_minus[..., 1:-1]                      # (M, ndim)
    vel = mom / rho[..., None]                    # (M, ndim)

    # --- Velocity condition ---------------------------------------------------------------------
    if 'no_slip' in bc.type:
        # The fluid takes the wall's own velocity. A stationary wall (the default) gives u_w = 0.
        u_wall = np.zeros(ndim) if bc.wall_velocity is None else np.asarray(bc.wall_velocity, dtype=float)
        u_w    = np.broadcast_to(u_wall, vel.shape).copy()          # (M, ndim)
    else:
        # Slip wall / symmetry plane: only impermeability, u.n = 0. The tangential velocity slides
        # freely, so it is inherited from the interior.
        u_w = vel - np.sum(vel * normal, axis=-1)[..., None] * normal

    # --- Temperature condition ------------------------------------------------------------------
    if bc.type.startswith('isothermal'):
        T_w = np.full_like(rho, float(bc.wall_temperature))
    else:
        T_w = compute_temperature(q_minus, case_cfg)                # adiabatic: inherit interior T

    # --- Rebuild the conserved state from (rho, u_w, T_w) ---------------------------------------
    p_w    = rho * T_w / gammaM2
    rhoE_w = p_w / (gamma - 1.0) + 0.5 * rho * np.sum(u_w * u_w, axis=-1)

    return np.concatenate([rho[..., None], rho[..., None] * u_w, rhoE_w[..., None]], axis=-1)

def wall_viscous_normal_flux(bc        : BCCfg     ,
                             q_minus   : np.ndarray,
                             grad_minus: np.ndarray,
                             normal    : np.ndarray,
                             case_cfg  : CaseCfg   ,
                             h_elem    : float     ) -> np.ndarray:
    '''
    The viscous normal flux F_visc.n AT a wall face, used in place of the BR1 central average.

    BR1's boundary recipe: evaluate the viscous flux with the BOUNDARY (wall) state and the INTERIOR
    gradient. The gradient itself gets no ghost -- the wall's job is to prescribe the state, and the
    stresses it feels are whatever the interior solution produces.

    Two things are then layered on top:

    1. The ADIABATIC wall, whose condition IS a statement about a flux: the wall-normal heat flux must
       vanish. We get it exactly by computing the heat flux from the interior gradient and subtracting
       its normal component back out of the energy flux, leaving k dT/dn = 0 while the viscous WORK
       term (u . tau . n, nonzero on a moving or slip wall) survives.

    2. The PENALTY, -sigma (q^- - q_wall), WITHOUT WHICH THE SCHEME IS UNSTABLE. Weakly-imposed
       Dirichlet data with no penalty has nothing damping a boundary-condition violation, so one grows
       without bound (see the _WALL_PENALTY note above). The penalty supplies exactly that missing
       dissipation. It is proportional to the BC violation, so it vanishes identically on any solution
       that satisfies the wall condition -- it acts only on the error.

    Inputs:
    - bc         : the BCCfg for this wall.
    - q_minus    : (M, num_eq) interior face trace.
    - grad_minus : (M, num_eq, ndim) interior conserved-variable gradient at the face.
    - normal     : (M, ndim) OUTWARD unit normal at the face.
    - case_cfg   : validated case configuration.
    - h_elem     : length scale of the element owning this face (its h_min), for the penalty scaling.

    Outputs:
    - Fv_n : (M, num_eq) viscous flux at the wall, projected onto the outward normal.
    '''

    q_wall = wall_state(bc, q_minus, normal, case_cfg)                              # (M, num_eq)
    return _dirichlet_viscous_normal_flux(q_wall, q_minus, grad_minus, normal, case_cfg, h_elem,
                                          remove_normal_heat=bc.type.startswith('adiabatic'))

def _dirichlet_viscous_normal_flux(target            : np.ndarray,
                                   q_minus           : np.ndarray,
                                   grad_minus        : np.ndarray,
                                   normal            : np.ndarray,
                                   case_cfg          : CaseCfg   ,
                                   h_elem            : float     ,
                                   remove_normal_heat: bool = False) -> np.ndarray:
    '''
    The viscous normal flux F_visc.n for ANY full-state Dirichlet boundary, given the TARGET state the
    boundary imposes (a wall state, or a prescribed g). Shared by wall_viscous_normal_flux and
    prescribed_viscous_normal_flux -- a wall is just a Dirichlet BC whose target is the impermeable
    reflection, and everything downstream of "what state does the boundary impose" is identical:

        F_visc*.n = F_v(target, grad^-).n  -  sigma (q^- - target)     [ - normal heat flux if adiabatic ]

    The flux uses the TARGET state with the INTERIOR gradient (BR1's boundary recipe); the penalty
    -sigma(q^- - target) supplies the dissipation that a weakly-imposed Dirichlet BC needs to be stable
    (see the _WALL_PENALTY note). remove_normal_heat zeroes k dT/dn for an adiabatic wall (a prescribed
    BC prescribes T, so it conducts normally and never sets this).
    '''
    Fv_n = np.einsum('med,md->me', compute_viscous_flux(target, grad_minus, case_cfg), normal)

    if remove_normal_heat:
        # Remove the wall-normal heat flux: k dT/dn = 0. The energy equation is the last slot.
        q_heat        = compute_heat_flux(target, grad_minus, case_cfg)             # (M, ndim)
        Fv_n[..., -1] = Fv_n[..., -1] - np.sum(q_heat * normal, axis=-1)

    # Interior penalty: dissipatively drive the interior trace onto the target. The sign is set by the
    # strong form -- the residual adds -(1/J) S (f* - f_int) with f* = f_inviscid* - F_visc*.n, so
    # SUBTRACTING sigma (q^- - target) here puts -sigma (q^- - target) into dq/dt: restoring, as required.
    sigma = wall_penalty_sigma(q_minus, case_cfg, h_elem)                           # (M,)
    return Fv_n - sigma[:, None] * (q_minus - target)

def prescribed_viscous_normal_flux(bc         : BCCfg            ,
                                   q_minus    : np.ndarray       ,
                                   grad_minus : np.ndarray       ,
                                   normal     : np.ndarray       ,
                                   case_cfg   : CaseCfg          ,
                                   h_elem     : float            ,
                                   face_coords: np.ndarray | None,
                                   t          : float=0.0        ) -> np.ndarray:
    '''
    The viscous normal flux F_visc.n at a prescribed-state (uniform_inflow / prescribed) boundary,
    used in place of the BR1 central average -- the analogue of wall_viscous_normal_flux, with the
    prescribed target g in place of the wall state and no heat-flux removal (a prescribed BC sets T).
    '''
    target = prescribed_target_state(bc, q_minus, face_coords, case_cfg, t)
    return _dirichlet_viscous_normal_flux(target, q_minus, grad_minus, normal, case_cfg, h_elem,
                                          remove_normal_heat=False)

def exterior_state(mort       : mortar_class.SpectralMortar,
                   q_minus    : np.ndarray                 ,
                   case_cfg   : CaseCfg                    ,
                   t          : float=0.0                  ,
                   normal     : np.ndarray | None = None   ,
                   face_coords: np.ndarray | None = None   ) -> np.ndarray:
    '''
    Returns the exterior ("ghost") state q_plus at a boundary mortar, to be paired with the
    interior trace q_minus in the numerical flux. This is the ghost for the INVISCID (Riemann)
    flux -- the BR1 gradient pass uses `gradient_exterior_state` instead.

    Inputs:
    - mort        : the boundary SpectralMortar
    - q_minus     : (P+1, num_eq) interior face trace at the P+1 face nodes.
    - case_cfg    : validated case configuration.
    - t           : current time (Only required for unsteady boundary conditions).
    - normal      : (P+1, ndim) OUTWARD unit normal at the face. Required by the wall BCs, which must
                    know which momentum component to reflect.
    - face_coords : (P+1, ndim) physical coordinates of the face nodes. Required by 'prescribed'.

    Outputs:
    - q_plus   : (P+1, num_eq) exterior state.
    '''

    # Extract the boundary condition configuration object from the mortar
    bc = mort.boundary_condition

    match bc.type:
        case 'uniform_inflow' | 'prescribed':
            # Full-state Dirichlet: the ghost IS the prescribed state (constant, or a user function).
            q_plus = prescribed_target_state(bc, q_minus, face_coords, case_cfg, t)

        case 'outflow':
            # The exterior state equals the interior state, so the numerical flux reduces to
            # the interior physical flux
            q_plus = q_minus

        case 'periodic':
            # Periodic faces are NOT handled here — they are paired with their partner mortar in
            # apply_boundary_conditions and treated like an interior interface, pulling q_plus from
            # the partner element's trace. So exterior_state should never be called on them.
            raise NotImplementedError(
                "Periodic BCs are handled via mortar pairing in the surface term, not exterior_state."
            )

        case ('adiabatic_slip_wall'  | 'adiabatic_no_slip_wall' |
              'isothermal_slip_wall' | 'isothermal_no_slip_wall'):
            # Every wall is IMPERMEABLE, and impermeability is the only thing the inviscid flux can
            # see (see the module docstring). Reflect the normal momentum and leave rho / rhoE alone:
            #
            #   mom+ = mom- - 2 (mom-.n) n   =>   |mom+| = |mom-|   =>   p+ = p-  (positivity-safe)
            #
            # The central mass flux 1/2(mom-.n + mom+.n) and energy flux then vanish identically, and
            # the surviving momentum flux tends to the physical wall traction p*n as u.n -> 0.
            if normal is None:
                raise ValueError(
                    f"Boundary condition '{bc.type}' needs the face normal to reflect the momentum, "
                    f"but exterior_state was called without one."
                )

            mom   = q_minus[..., 1:-1]                                    # (M, ndim)
            mom_n = np.sum(mom * normal, axis=-1)[..., None]              # (M, 1) normal momentum

            q_plus            = q_minus.copy()
            q_plus[..., 1:-1] = mom - 2.0 * mom_n * normal

        case _:
             raise NotImplementedError(
                f"Boundary condition type '{bc.type}' not yet implemented."
            )

    return q_plus

def gradient_exterior_state(mort       : mortar_class.SpectralMortar,
                            q_minus    : np.ndarray                 ,
                            case_cfg   : CaseCfg                    ,
                            t          : float=0.0                  ,
                            normal     : np.ndarray | None = None   ,
                            face_coords: np.ndarray | None = None   ) -> np.ndarray:
    '''
    Returns the exterior state used by the BR1 GRADIENT pass at a boundary mortar.

    The gradient pass builds the central trace q* = 1/2 (q- + q+) and lifts (q* - q-). For a Dirichlet
    boundary we want that trace to BE the imposed state (the wall state, or the prescribed g) -- that is
    what makes the condition exact on grad(q) rather than a half-strength suggestion. So we hand back the
    mirror of the interior trace about the target state:

        q+ = 2 q_target - q-      =>      q* = 1/2 (q- + q+) = q_target   exactly.

    This mirror is ONLY safe here: its energy component can imply a negative pressure (it is a
    reflection in CONSERVED variables), which would break the Riemann solver. The inviscid flux gets
    the pressure-preserving reflection (walls) or the raw target (prescribed) from `exterior_state`.

    BCs with nothing extra to say about the gradient (outflow) reuse their inviscid ghost.

    Inputs / Outputs mirror `exterior_state`.
    '''

    bc = mort.boundary_condition

    if is_wall(bc):
        if normal is None:
            raise ValueError(
                f"Boundary condition '{bc.type}' needs the face normal to build the wall state, "
                f"but gradient_exterior_state was called without one."
            )
        q_wall = wall_state(bc, q_minus, normal, case_cfg)
        return 2.0 * q_wall - q_minus

    if is_prescribed(bc):
        target = prescribed_target_state(bc, q_minus, face_coords, case_cfg, t)
        return 2.0 * target - q_minus

    return exterior_state(mort, q_minus, case_cfg, t, normal, face_coords)
