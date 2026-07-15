# tests/test_case_mms_viscous/inputs/mms_manufactured.py
'''
Method of Manufactured Solutions (MMS) for the compressible Navier-Stokes viscous terms.

THE IDEA. Pick a smooth q_exact(x, y) -- it need NOT solve anything -- and push it through the PDE.
Whatever is left over,

        S := d/dt q_exact + div( F(q_exact) )        (here d/dt = 0: the manufactured solution is steady)

is, by construction, exactly the forcing that MAKES q_exact an exact solution of the FORCED system
q_t + div(F) = S. Run the solver with that S, and the discrete solution must reproduce q_exact -- with
an error that vanishes at the scheme's formal order. This is the ONLY test that gives a rigorous
order-of-accuracy number for the VISCOUS discretization: Couette/Poiseuille have polynomial-degree <= 2
exact solutions, which a high-order scheme represents almost exactly, so they confirm correctness but
never exercise the high-order truncation error. A manufactured solution built from sin/cos of 2*pi*x is
NOT representable at any finite polynomial degree, so its residual genuinely measures the operator.

WHY SYMPY. The source S is the divergence of the full NSE flux -- inviscid Euler flux MINUS the viscous
flux (stress tensor + heat conduction), including every cross-derivative. Deriving that by hand is
exactly the kind of algebra whose silent error would corrupt the order-of-accuracy number MMS exists to
produce. So we build it symbolically, mirroring the code's flux definitions TERM FOR TERM
(physics/fluxes.py::_compute_euler_flux and physics/viscousFluxes.py::compute_viscous_flux), and let
sympy differentiate. The derivation is self-validating: if it were wrong, S != div(F) even continuously,
and the residual test would FLOOR instead of converging -- so spectral convergence is simultaneous proof
that both the scheme AND this derivation are correct.

PERIODIC on the unit square [0,1]^2. This deliberately avoids any boundary condition: the test isolates
the INTERIOR operator (volume flux, BR1 gradient pass, interior-interface coupling, source term). A
separate Dirichlet-MMS test will exercise the boundary treatment once a spatially-varying Dirichlet BC
exists -- ordered that way so a boundary bug cannot hide behind an interior bug, or vice versa.

Constants (gamma, Re, Pr, mach_ref) are kept SYMBOLIC and lambdified as extra arguments, so the compiled
source works for any case configuration without re-deriving.
'''
import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------------------------
# 1. Symbols and the manufactured PRIMITIVE state
# ---------------------------------------------------------------------------------------------
_x, _y                    = sp.symbols('x y', real=True)
_gamma, _Re, _Pr, _Msq    = sp.symbols('gamma Re Pr Msq', positive=True)   # Msq = mach_ref**2
_TWO_PI                   = 2 * sp.pi

# Manufactured primitives -- periodic on [0,1]^2, with rho and p bounded safely away from zero, both
# velocity components varying, and distinct spatial structure per variable so that EVERY viscous term is
# active: tau_xx, tau_yy (from u_x, v_y), the shear tau_xy (from u_y, v_x), the dilatation (u_x + v_y),
# and heat conduction in BOTH directions (T varies with x and y).
_rho = 2.0 + 0.3 * sp.sin(_TWO_PI * _x) * sp.cos(_TWO_PI * _y)     # in [1.7, 2.3] > 0
_u   = 0.5 + 0.2 * sp.sin(_TWO_PI * _x) * sp.sin(_TWO_PI * _y)
_v   = -0.3 + 0.2 * sp.cos(_TWO_PI * _x) * sp.cos(_TWO_PI * _y)
_p   = 5.0 + 0.5 * sp.cos(_TWO_PI * _x) * sp.sin(_TWO_PI * _y)     # in [4.5, 5.5] > 0

# ---------------------------------------------------------------------------------------------
# 2. Conserved state q_exact = [rho, rho u, rho v, rho E]
# ---------------------------------------------------------------------------------------------
_rhoE  = _p / (_gamma - 1) + sp.Rational(1, 2) * _rho * (_u**2 + _v**2)
_Q     = [_rho, _rho * _u, _rho * _v, _rhoE]

# ---------------------------------------------------------------------------------------------
# 3. Physical flux  F = F_euler - F_viscous   (matching compute_volume_flux for 'navier-stokes')
# ---------------------------------------------------------------------------------------------
# --- Euler (inviscid) flux, per _compute_euler_flux: [rho u_d ; rho u_i u_d + p delta_id ; (rhoE+p) u_d]
_Fe = {
    'mass':   [_rho * _u,                    _rho * _v                   ],
    'momx':   [_rho * _u * _u + _p,          _rho * _u * _v              ],
    'momy':   [_rho * _u * _v,               _rho * _v * _v + _p         ],
    'energy': [(_rhoE + _p) * _u,            (_rhoE + _p) * _v           ],
}

# --- Viscous flux, per compute_viscous_flux with mu = 1 (compute_viscosity returns 1).
#     Stress tensor tau_ij = du_i/dx_j + du_j/dx_i - (2/3) (div u) delta_ij   (the mu=1 strain), and the
#     momentum block of F_visc is tau/Re. The energy block is (u . tau)/Re + (kappa grad T)/Re, with
#     kappa = 1 / (Pr (gamma-1) Msq) and T = gamma Msq p / rho.
_ux, _uy = sp.diff(_u, _x), sp.diff(_u, _y)
_vx, _vy = sp.diff(_v, _x), sp.diff(_v, _y)
_divu    = _ux + _vy

_tau_xx = 2 * _ux - sp.Rational(2, 3) * _divu
_tau_yy = 2 * _vy - sp.Rational(2, 3) * _divu
_tau_xy = _uy + _vx                                  # = tau_yx (symmetric)

_T      = _gamma * _Msq * _p / _rho
_kappa  = 1 / (_Pr * (_gamma - 1) * _Msq)
_qh_x   = _kappa * sp.diff(_T, _x)                   # kappa dT/dx  (the 1/Re is applied below)
_qh_y   = _kappa * sp.diff(_T, _y)

_work_x = _u * _tau_xx + _v * _tau_xy                # (u . tau)_x
_work_y = _u * _tau_xy + _v * _tau_yy                # (u . tau)_y

_Fv = {
    'mass':   [sp.Integer(0),                sp.Integer(0)               ],
    'momx':   [_tau_xx / _Re,                _tau_xy / _Re               ],
    'momy':   [_tau_xy / _Re,                _tau_yy / _Re               ],
    'energy': [(_work_x + _qh_x) / _Re,      (_work_y + _qh_y) / _Re     ],
}

# --- Total flux F = F_euler - F_viscous
_F = {k: [_Fe[k][0] - _Fv[k][0], _Fe[k][1] - _Fv[k][1]] for k in _Fe}

# ---------------------------------------------------------------------------------------------
# 4. Source term  S = div(F)  (steady manufactured solution, so d/dt q_exact = 0)
# ---------------------------------------------------------------------------------------------
_S = {k: sp.diff(_F[k][0], _x) + sp.diff(_F[k][1], _y) for k in _F}

# ---------------------------------------------------------------------------------------------
# 5. Lambdify to numpy. Constants ride along as arguments so one compile serves any configuration.
# ---------------------------------------------------------------------------------------------
_ARGS       = (_x, _y, _gamma, _Re, _Pr, _Msq)
_q_funcs    = [sp.lambdify(_ARGS, expr, 'numpy') for expr in _Q]
_S_funcs    = [sp.lambdify(_ARGS, _S[k], 'numpy') for k in ('mass', 'momx', 'momy', 'energy')]


def _consts(case_config):
    return (case_config.physics.gamma, case_config.physics.Re,
            case_config.physics.Pr, case_config.physics.mach_ref**2)


def _stack(funcs, x, y, consts) -> np.ndarray:
    '''Evaluate a list of 4 lambdified scalar functions at (x, y) and stack into (..., 4), broadcasting
    any component that came out constant up to the shape of x.'''
    cols = [np.broadcast_to(f(x, y, *consts), np.shape(x)) for f in funcs]
    return np.stack(cols, axis=-1)


def exact_solution(case_config, x, y) -> np.ndarray:
    '''The manufactured conserved state q_exact = [rho, rho u, rho v, rho E] at (x, y), shape (..., 4).'''
    return _stack(_q_funcs, x, y, _consts(case_config))


def initial_condition(case_config, x, y) -> np.ndarray:
    '''Initialize with the exact manufactured solution (the residual test then confirms dq/dt -> 0).'''
    return exact_solution(case_config, x, y)


def boundary_state(case_config, x, y, t) -> np.ndarray:
    '''
    The prescribed-BC boundary state = the manufactured solution at the face nodes, shape (..., 4).
    Used by the Dirichlet-MMS test, which imposes q_exact on the boundary via `prescribed` BCs to
    exercise the boundary operator. Steady, so t is ignored; the extra arg matches the BC signature.
    '''
    return exact_solution(case_config, x, y)


def source_term(case_config, q, x, y, t) -> np.ndarray:
    '''
    The manufactured source S = div(F(q_exact)), shape (..., 4). It is a fixed function of space (the
    manufactured solution is steady), so the conserved state q and time t are ignored -- they are part
    of the signature only because a general source S(q, x, t) may use them.
    '''
    return _stack(_S_funcs, x, y, _consts(case_config))
