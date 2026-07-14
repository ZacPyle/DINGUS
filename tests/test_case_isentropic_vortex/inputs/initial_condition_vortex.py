# tests/test_case_advection_scalar/inputs/initial_condition_sinusoid.py
from dingus.config import CaseCfg
import numpy as np

# --- Vortex / freestream parameters (domain is [0,10] x [0,10]) ----------------------------
X_C, Y_C = 8.0, 8.0     # vortex center (domain center)
U_INF    = 1.0          # freestream x-velocity  (carries the vortex diagonally)
V_INF    = 0.0          # freestream y-velocity
BETA     = 5.0          # vortex strength
R        = 1.0          # vortex radius (length scale)
RHO_INF  = 1.0          # freestream density  (nondimensionalized to 1)
P_INF    = 1.0          # freestream pressure (nondimensionalized to 1)


def initial_condition(case_config: CaseCfg, x, y):
    '''
    Isentropic (Euler) vortex initial condition is the standard smooth verification case for the
    compressible Euler equations. A vortex is superimposed on a uniform freestream; the vortex 
    is an EXACT solution that simply convects at the freestream velocity (u_inf, v_inf) without 
    changing shape. On a periodic domain, if the freestream carries it a whole number of domain 
    lengths in 'final_time', the exact solution at final_time equals this initial condition.
    Thus, the round-trip L2 error is a pure accuracy measure that falls spectrally with poly_deg
    (the Euler analogue of the sinusoid round-trip used in the scalar advection test).

    Decay-at-the-boundary caveat: the analytic vortex is an exact solution on an INFINITE domain. 
    On a periodic box it is only exactly periodic once the swirl has decayed to the freestream at 
    the boundaries. With center (5,5), R=1, beta=5 on the [0,10]^2 domain, the perturbation at
    the nearest boundary is ~1e-5, which sets the error floor.

    Conserved variables (ordering [rho, rho*u, rho*v, rhoE]) from the isentropic relations, using
    scaled coordinates Xs=(x-xc)/R, Ys=(y-yc)/R and r^2 = Xs^2 + Ys^2:
        du   = -(beta/2pi) * Ys * exp((1-r^2)/2)          u = U_INF + du
        dv   =  (beta/2pi) * Xs * exp((1-r^2)/2)          v = V_INF + dv
        T    = 1 - (gamma-1) beta^2 / (8 gamma pi^2) * exp(1-r^2)      (isentropic temperature)
        rho  = T^(1/(gamma-1)),   p = T^(gamma/(gamma-1)) = rho^gamma
        rhoE = p/(gamma-1) + 0.5 rho (u^2 + v^2)
    The pressure relation here matches the solver's p = (gamma-1)(E - KE), so this state drops
    straight into compute_volume_flux / compute_pressure.

    Inputs:
    - case_config : validated case configuration (provides gamma and num_eq).
    - x, y        : (nx, ny) arrays of physical quadrature-node coordinates within an element.

    Outputs:
    - solution : (nx, ny, num_eq) initial conserved state.
    '''
    # Pre-allocate the solution array with the appropriate shape.
    solution = np.zeros((x.shape[0], x.shape[1], case_config.physics.num_eq))

    # Specific heat ratio
    gamma = case_config.physics.gamma

    # Scale and center the coordinates about the vortex center
    xs = (x - X_C) / R
    ys = (y - Y_C) / R
    r2 = xs**2 + ys**2

    # Create the shape function to the vortex decays to 0 in the freestream
    shape_func = np.exp((1.0 - r2)/2.0)

    # Compute the velocity field
    u = U_INF - (BETA / (2.0 * np.pi)) * ys * shape_func
    v = V_INF + (BETA / (2.0 * np.pi)) * xs * shape_func

    # Set temperature such that pressure gradient balances radial momentum of vortex (holds it together)
    T = 1.0 - (gamma - 1.0) * BETA**2 / (8.0 * gamma * np.pi**2) * np.exp(1.0 - r2)

    # Compute density and pressure based on T according to the isentropic relations
    rho     = RHO_INF * T**(  1.0 / (gamma - 1.0))
    pressure = P_INF  * T**(gamma / (gamma - 1.0))

    # Compute the energy density using the pressure and velocity field
    rhoE = pressure / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    # Bundle the solution
    solution[...,0] = rho
    solution[...,1] = rho * u
    solution[...,2] = rho * v
    solution[...,3] = rhoE

    return solution
