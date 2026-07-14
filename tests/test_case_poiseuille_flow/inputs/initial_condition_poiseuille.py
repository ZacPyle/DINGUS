# tests/test_case_poiseuille_flow/inputs/initial_condition_poiseuille.py
from dingus.config import CaseCfg
import numpy as np

'''
FORCED (compressible) POISEUILLE FLOW -- an EXACT steady solution of the Navier-Stokes equations with
a constant body force, and the companion validation case to Couette.

Setup: a channel 0 <= y <= H, periodic in x, with BOTH walls stationary isothermal no-slip walls. The
flow is driven not by the walls (they are at rest) but by a uniform body force G in +x -- see
inputs/source_term.py. Where Couette tests a MOVING no-slip wall, this tests a STATIONARY one, plus
the source-term machinery.

Derivation (steady, fully developed: v = 0, d/dx = 0, constant mu, and G from source_term.py):

  x-momentum: (1/Re) mu u'' + G = 0            ->  u'' = -G Re     ->  u is a PARABOLA
              with u(0) = u(H) = 0 (no slip on BOTH walls):

                    u(y) = (G Re / 2) y (H - y),      u_max = G Re H^2 / 8  at the centreline

  y-momentum: dp/dy = 0                        ->  p is CONSTANT across the channel
  energy    : the body force does work at a rate G*u. Expanding d/dy(u tau_xy) = u'^2 + u u'' and
              substituting u'' = -G Re, the  -G u  from the momentum balance CANCELS the +G u of the
              force's work exactly, leaving pure viscous dissipation:

                    kappa T'' = -u'^2,     kappa = 1 / (Pr (gamma-1) M^2)

              With u' = (G Re / 2)(H - 2y), the right-hand side is a squared linear, so integrating
              twice gives a QUARTIC temperature profile. With T(0) = T(H) = T_WALL:

                    T(y) = T_WALL + (A / 48) ( H^4 - (H - 2y)^4 ),   A = Pr (gamma-1) M^2 (G Re / 2)^2

That cancellation is the point of this case: it only comes out right if the source term's ENERGY
component (G*u, the rate of work done by the force) is correct. Get it wrong and T is visibly off.

  rho(y) = gamma M^2 P0 / T(y)      (from T = gamma M^2 p / rho, with p = P0 constant)

As in Couette, P0 is a FREE CONSTANT: u and T do not depend on it. In this closed (periodic + walled)
domain the true P0 is pinned by the conserved total MASS, so compare u and T, which are P0-free.
'''

# --- Channel parameters (keep in sync with control.yaml, source_term.py, and NarrowChannel.mesh) ---
H      = 1.0     # channel height; the mesh spans 0 <= y <= H
T_WALL = 1.0     # both walls are isothermal at this temperature
P0     = None    # base pressure; None -> derived below so that rho = 1 at the walls


def _base_pressure(case_config: CaseCfg) -> float:
    '''The free pressure constant P0, fixed here by normalizing rho = 1 at the (equal-temperature) walls.'''
    if P0 is not None:
        return P0
    gammaM2 = case_config.physics.gamma * case_config.physics.mach_ref**2
    return T_WALL / gammaM2          # rho(wall) = gamma M^2 P0 / T_WALL = 1


def _driving_force() -> float:
    '''The body force G, imported from the source-term module so there is ONE source of truth for it.'''
    from pathlib import Path
    import importlib.util
    src  = Path(__file__).resolve().parent / "source_term.py"
    spec = importlib.util.spec_from_file_location("poiseuille_source", src)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.G


def exact_velocity(case_config: CaseCfg, y) -> np.ndarray:
    '''u(y) = (G Re / 2) y (H - y) -- the parabola the two stationary no-slip walls must produce.'''
    G  = _driving_force()
    Re = case_config.physics.Re
    return 0.5 * G * Re * y * (H - y)


def centreline_velocity(case_config: CaseCfg) -> float:
    '''u_max = G Re H^2 / 8, at y = H/2.'''
    return _driving_force() * case_config.physics.Re * H**2 / 8.0


def viscous_heating_amplitude(case_config: CaseCfg) -> float:
    '''
    The centreline temperature rise, A H^4 / 48, caused by viscous dissipation of the work the body
    force does on the fluid. Exported so the test can assert it is big enough to actually be measuring
    the energy balance rather than round-off.
    '''
    gamma = case_config.physics.gamma
    Pr    = case_config.physics.Pr
    M2    = case_config.physics.mach_ref**2
    Re    = case_config.physics.Re
    A     = Pr * (gamma - 1.0) * M2 * (0.5 * _driving_force() * Re)**2
    return A * H**4 / 48.0


def exact_temperature(case_config: CaseCfg, y) -> np.ndarray:
    '''T(y) = T_WALL + (A/48)( H^4 - (H-2y)^4 ) -- the quartic viscous-heating profile.'''
    gamma = case_config.physics.gamma
    Pr    = case_config.physics.Pr
    M2    = case_config.physics.mach_ref**2
    Re    = case_config.physics.Re
    A     = Pr * (gamma - 1.0) * M2 * (0.5 * _driving_force() * Re)**2
    return T_WALL + (A / 48.0) * (H**4 - (H - 2.0 * y)**4)


def exact_solution(case_config: CaseCfg, x, y) -> np.ndarray:
    '''The exact steady Poiseuille state as CONSERVED variables [rho, rho u, rho v, rho E].'''
    gamma   = case_config.physics.gamma
    gammaM2 = gamma * case_config.physics.mach_ref**2

    p   = _base_pressure(case_config)                  # constant across the channel
    T   = exact_temperature(case_config, y)
    rho = gammaM2 * p / T                              # T = gamma M^2 p / rho
    u   = exact_velocity(case_config, y)
    v   = np.zeros_like(u)

    solution = np.zeros(np.shape(x) + (case_config.physics.num_eq,))
    solution[..., 0] = rho
    solution[..., 1] = rho * u
    solution[..., 2] = rho * v
    solution[..., 3] = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return solution


def initial_condition(case_config: CaseCfg, x, y) -> np.ndarray:
    '''Initialize with the EXACT steady solution; the residual test then confirms dq/dt = 0.'''
    return exact_solution(case_config, x, y)
