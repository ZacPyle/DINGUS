# tests/test_case_couette_flow/inputs/initial_condition_couette.py
from dingus.config import CaseCfg
import numpy as np

'''
COMPRESSIBLE COUETTE FLOW -- an EXACT steady solution of the Navier-Stokes equations, and the
standard validation case for the no-slip and isothermal wall boundary conditions.

Setup: a channel 0 <= y <= H, periodic in x. The bottom wall is stationary, the top wall is dragged
tangentially at speed U, and BOTH walls are held at fixed temperatures (T_BOTTOM, T_TOP). The flow is
driven entirely by the moving WALL -- there is no pressure gradient and no body force, which is
exactly why this case is a clean test of the wall BCs and nothing else.

Derivation (steady, fully developed: v = 0, d/dx = 0, constant mu):

  mass      : d(rho v)/dy = 0                                    -> satisfied identically (v = 0)
  x-momentum: d(tau_xy)/dy = 0,  tau_xy = mu du/dy               -> u'' = 0    -> u is LINEAR
  y-momentum: dp/dy = 0                                          -> p is CONSTANT across the channel
              (tau_yy = mu(2 v' - 2/3 div u) = 0, since v = 0 and div u = du/dx = 0)
  energy    : d/dy( u tau_xy + kappa dT/dy ) = 0
              -> mu (u')^2 + kappa T'' = 0                       -> T'' = -(mu/kappa) (U/H)^2

The energy balance is the interesting one: the wall does WORK on the fluid, that work is dissipated
as heat at a uniform rate mu (u')^2, and conduction carries it out to the two walls. That is what
bends T into a parabola. With kappa = mu / (Pr (gamma-1) M^2) (the code's nondimensionalization,
T = gamma M^2 p / rho), integrating twice with eta = y/H gives:

    u(y) = U * eta
    v(y) = 0
    p    = P0                                                    (constant)
    T(y) = T_BOTTOM + (T_TOP - T_BOTTOM) * eta
                    + (Pr (gamma-1) M^2 U^2 / 2) * eta (1 - eta)   <-- VISCOUS HEATING
    rho(y) = gamma M^2 P0 / T(y)

Note the split of duties, which is what makes this a sharp test:
  - the LINEAR u profile pins the NO-SLIP walls (a slip wall would give uniform flow; a bad
    moving-wall BC would give the wrong slope),
  - the eta(1-eta) bump pins the viscous WORK term AND the ISOTHERMAL walls (an adiabatic wall would
    let the channel heat up without bound instead of reaching this balance),
  - the linear part of T pins wall-to-wall heat CONDUCTION.

The parameters below deliberately make the viscous-heating bump (~0.036) comparable to the wall-to-wall
temperature difference (0.05), so BOTH energy mechanisms are exercised at the same order.

IMPORTANT -- P0 is a FREE CONSTANT. Nothing in the steady solution fixes it: u and T do not depend on
it at all, and it only sets the density level rho = gamma M^2 P0 / T. In a CLOSED domain (periodic in
x, walls top and bottom) the actual P0 is pinned by the conserved total MASS instead, so a run relaxing
to steady state settles at whatever P0 its initial mass implies. Compare u and T -- they are P0-free.
'''

# --- Channel / wall parameters (keep in sync with control.yaml and NarrowChannel.mesh) ------------
H        = 1.0     # channel height; the mesh spans 0 <= y <= H
U_WALL   = 1.0     # tangential speed of the TOP wall (the bottom wall is stationary)
T_BOTTOM = 1.0     # isothermal temperature of the bottom wall
T_TOP    = 1.05    # isothermal temperature of the top wall
P0       = None    # base pressure; None -> derived below so that rho = 1 at the bottom wall


def _base_pressure(case_config: CaseCfg) -> float:
    '''The free pressure constant P0, fixed here by normalizing rho = 1 at the bottom wall.'''
    if P0 is not None:
        return P0
    gammaM2 = case_config.physics.gamma * case_config.physics.mach_ref**2
    return T_BOTTOM / gammaM2          # rho(0) = gamma M^2 P0 / T_BOTTOM = 1


def viscous_heating_amplitude(case_config: CaseCfg) -> float:
    '''
    The coefficient of the eta(1-eta) temperature bump: Pr (gamma-1) M^2 U^2 / 2.

    This is the peak temperature rise caused by the wall shearing the fluid. It is exported so the
    test can assert it is actually LARGE enough to be measuring something -- if it were negligible,
    "T matches" would only be testing the linear conduction profile.
    '''
    gamma = case_config.physics.gamma
    Pr    = case_config.physics.Pr
    M2    = case_config.physics.mach_ref**2
    return 0.5 * Pr * (gamma - 1.0) * M2 * U_WALL**2


def exact_velocity(y) -> np.ndarray:
    '''u(y) = U * y/H  -- the linear profile the no-slip walls must produce.'''
    return U_WALL * (y / H)


def exact_temperature(case_config: CaseCfg, y) -> np.ndarray:
    '''T(y): linear wall-to-wall conduction plus the viscous-heating parabola.'''
    eta = y / H
    return (T_BOTTOM + (T_TOP - T_BOTTOM) * eta
            + viscous_heating_amplitude(case_config) * eta * (1.0 - eta))


def exact_solution(case_config: CaseCfg, x, y) -> np.ndarray:
    '''
    The exact steady Couette state as CONSERVED variables [rho, rho u, rho v, rho E], shaped
    (..., num_eq) to match x/y. This is both the initial condition and the reference the tests
    measure against.
    '''
    gamma   = case_config.physics.gamma
    gammaM2 = gamma * case_config.physics.mach_ref**2

    p   = _base_pressure(case_config)                       # constant across the channel
    T   = exact_temperature(case_config, y)
    rho = gammaM2 * p / T                                   # T = gamma M^2 p / rho
    u   = exact_velocity(y)
    v   = np.zeros_like(u)

    solution = np.zeros(np.shape(x) + (case_config.physics.num_eq,))
    solution[..., 0] = rho
    solution[..., 1] = rho * u
    solution[..., 2] = rho * v
    solution[..., 3] = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return solution


def initial_condition(case_config: CaseCfg, x, y) -> np.ndarray:
    '''Initialize with the EXACT steady solution: the residual test then asks the solver to confirm
    that it is, in fact, steady (dq/dt = 0). The relaxation test overrides this with fluid at rest.'''
    return exact_solution(case_config, x, y)
