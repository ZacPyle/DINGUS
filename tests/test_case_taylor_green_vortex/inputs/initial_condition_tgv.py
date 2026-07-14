# tests/test_nse_taylor_green/inputs/initial_condition_taylor_green.py
from dingus.config import CaseCfg
import numpy as np

# --- Taylor-Green parameters (domain is the periodic unit square [0,1] x [0,1]) -------------
K = 2.0 * np.pi     # wavenumber: exactly one Taylor-Green vortex cell across the unit square.
                    # The analytic incompressible KE decay rate is 4 * K^2 / Re -- the test imports
                    # K from this module to build that reference, so keep this the single source.


def initial_condition(case_config: CaseCfg, x, y):
    '''
    2D compressible Taylor-Green vortex initial condition. The Taylor-Green vortex is an EXACT
    solution of the INCOMPRESSIBLE Navier-Stokes equations on a periodic box, with velocity
    amplitude decaying as exp(-2 K^2 t / Re) and kinetic energy as exp(-4 K^2 t / Re). Run at low
    reference Mach number the compressible solver reproduces this decay closely (the residual is the
    O(M^2) compressibility correction), which makes it the standard end-to-end NSE verification case.

    Nondimensional low-Mach construction (matches constitutiveRelations: T = gamma * M^2 * p / rho):
        rho = 1
        u   =  cos(K x) sin(K y)                 (velocity amplitude O(1))
        v   = -sin(K x) cos(K y)
        p   = P0 + 1/4 (cos 2K x + cos 2K y),    P0 = 1 / (gamma * M^2)
    Choosing P0 = 1/(gamma M^2) makes the sound speed sqrt(gamma P0 / rho) = 1/M, so the flow Mach
    number |u|/c equals the reference Mach M. Thus a small mach_ref keeps the flow near-incompressible
    (the regime where the analytic decay rate applies). The pressure perturbation 1/4 (cos2Kx+cos2Ky)
    is the incompressible Taylor-Green pressure that balances the initial velocity field.

    Inputs:
    - case_config : validated configuration (provides gamma, mach_ref, num_eq).
    - x, y        : (nx, ny) physical quadrature-node coordinates within an element.

    Outputs:
    - solution : (nx, ny, num_eq) initial conserved state [rho, rho*u, rho*v, rhoE].
    '''
    gamma = case_config.physics.gamma
    M     = case_config.physics.mach_ref
    P0    = 1.0 / (gamma * M * M)          # base pressure -> sound speed 1/M -> flow Mach = M

    rho = np.ones_like(x)
    u   =  np.cos(K * x) * np.sin(K * y)
    v   = -np.sin(K * x) * np.cos(K * y)
    p   = P0 + 0.25 * (np.cos(2.0 * K * x) + np.cos(2.0 * K * y))

    solution = np.zeros((x.shape[0], x.shape[1], case_config.physics.num_eq))
    solution[..., 0] = rho
    solution[..., 1] = rho * u
    solution[..., 2] = rho * v
    solution[..., 3] = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return solution
