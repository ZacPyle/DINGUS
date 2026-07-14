# tests/SingleTests/test_viscous_flux.py
'''
Unit test for the compressible Navier-Stokes viscous flux (viscousFluxes.compute_viscous_flux),
which builds F_visc(q, grad_q) = [ 0 ; tau ; viscous work + heat conduction ].

We test the FLUX ALGEBRA IN ISOLATION by feeding hand-built ANALYTIC grad_q arrays -- NOT the output
of the BR1 gradient pass. That separation is deliberate: if a full NSE run is wrong later, these
tests tell us whether the bug is in the flux formula or in the gradient/interface machinery. Each
scenario has a closed-form answer:

  1. ZERO gradient        grad_q = 0                 ->  F_visc == 0            (no gradients, no flux)
  2. tau SYMMETRY         random state + grad_q      ->  momentum block is symmetric (tau_id == tau_di)
  3. PURE SHEAR           u = (U, 0), du/dy = S,     ->  tau_xy = tau_yx = mu S / Re, tau_xx=tau_yy=0,
                          rho, p constant                energy flux = (0, U S / Re), no heat flux
  4. PURE CONDUCTION      u = 0, rho const, dp/dx=P   ->  tau = 0, and the ONLY nonzero flux is the
                          (so T varies)                  energy heat flux = gamma P / (Re Pr (gamma-1) rho)
                                                         (note M_ref^2 cancels between kappa and grad_T)

Scenarios 3 and 4 are manufactured single-node states with grad_q constructed analytically from the
prescribed primitive gradients via the chain rule d(rho u_i)/dx and d(rho E)/dx. Constant viscosity
(mu* = 1) is assumed, matching the current compute_viscosity.
'''
import numpy as np
import pytest

from dingus.config import CaseCfg
from dingus.physics.viscousFluxes import compute_viscous_flux

# TIER: unit -- see tests/conftest.py for what the markers mean.
pytestmark = pytest.mark.unit


# Minimal validated NSE configuration (2D). The viscous flux only reads gamma/Re/Pr/mach_ref and
# mesh.ndim; the mesh_file string is never opened here.
CFG = CaseCfg.model_validate({
    'mesh':           {'mesh_format': 'HOHQMesh', 'mesh_file': 'unused.mesh', 'ndim': 2, 'poly_deg': 2, 'quad_type': 'LG'},
    'physics':        {'model': 'navier-stokes', 'Re': 100.0, 'Pr': 0.71, 'mach_ref': 0.5, 'gamma': 1.4, 'riemann_solver' : 'LLF'},
    'time_stepping':  {'time_integrator': 'rk4', 'cfl': 0.5, 'final_time': 1.0, 'start_time': 0.0},
    'initialization': {'IC_method': 'analytical', 'IC_file': 'initial_condition.py'},
    'io':             {'output_format': 'vtk', 'output_dir': './outputs/'}
})
GAMMA = CFG.physics.gamma
RE    = CFG.physics.Re
PR    = CFG.physics.Pr
M2    = CFG.physics.mach_ref**2


def _state(rho, u, v, p):
    '''2D conserved state [rho, rho u, rho v, rho E] from primitives.'''
    rhoE = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    return np.array([rho, rho * u, rho * v, rhoE])


def test_zero_gradient_gives_zero_flux():
    '''Rung 1: with no gradients there is no viscous flux, for an arbitrary state.'''
    q      = _state(rho=1.2, u=0.7, v=-0.4, p=1.5)
    grad_q = np.zeros((q.size, 2))
    F      = compute_viscous_flux(q, grad_q, CFG)
    assert np.allclose(F, 0.0), f"zero gradient must give zero viscous flux, got\n{F}"


def test_stress_tensor_is_symmetric():
    '''Rung 2: the momentum block of F_visc equals (1/Re) tau, which must be symmetric.'''
    rng    = np.random.default_rng(0)
    q      = _state(rho=1.0 + rng.random(), u=rng.normal(), v=rng.normal(), p=1.0 + rng.random())
    grad_q = rng.normal(size=(q.size, 2))                 # arbitrary (unphysical is fine for tau symmetry)

    F   = compute_viscous_flux(q, grad_q, CFG)
    tau = F[1:1 + 2, :]                                   # momentum rows -> (ndim, ndim) stress block
    assert np.allclose(tau, tau.T), f"stress tensor must be symmetric, got\n{tau}"


def test_pure_shear():
    '''Rung 3: uniform-shear flow gives the exact off-diagonal stress and viscous work, no heat flux.'''
    rho0, U, S, p0 = 1.0, 2.0, 3.0, 1.0                   # u = (U, 0), du/dy = S; rho, p constant
    q = _state(rho0, U, 0.0, p0)

    # grad_q from the chain rule: d(rho u)/dy = rho0 S ; d(rho E)/dy = rho0 U S ; everything else 0.
    grad_q = np.array([[0.0, 0.0],
                       [0.0, rho0 * S],
                       [0.0, 0.0],
                       [0.0, rho0 * U * S]])

    F = compute_viscous_flux(q, grad_q, CFG)
    expected = np.array([[0.0,      0.0     ],   # mass
                         [0.0,      S / RE  ],   # x-momentum: tau_xx=0, tau_xy = S/Re
                         [S / RE,   0.0     ],   # y-momentum: tau_yx = S/Re, tau_yy=0
                         [0.0,      U * S / RE]])# energy: work only, no heat (T constant)
    assert np.allclose(F, expected), f"pure-shear flux mismatch:\n{F}\nexpected\n{expected}"


def test_pure_heat_conduction():
    '''Rung 4: quiescent flow with a pressure/temperature gradient gives ONLY heat conduction.'''
    rho0, p0, P = 1.0, 1.0, 0.5                           # u = 0, rho constant, dp/dx = P (so dT/dx != 0)
    q = _state(rho0, 0.0, 0.0, p0)

    # grad_q: only d(rho E)/dx = dp/dx / (gamma-1) is nonzero (kinetic energy is zero).
    grad_q = np.array([[0.0,             0.0],
                       [0.0,             0.0],
                       [0.0,             0.0],
                       [P / (GAMMA - 1), 0.0]])

    F = compute_viscous_flux(q, grad_q, CFG)

    # Momentum + work must vanish (no velocity); only the energy heat flux survives. M_ref^2 cancels
    # between kappa (~1/M^2) and grad_T (~M^2), leaving gamma P / (Re Pr (gamma-1) rho).
    q_heat = GAMMA * P / (RE * PR * (GAMMA - 1.0) * rho0)
    expected = np.array([[0.0,    0.0],
                         [0.0,    0.0],
                         [0.0,    0.0],
                         [q_heat, 0.0]])
    assert np.allclose(F, expected), f"pure-conduction flux mismatch:\n{F}\nexpected\n{expected}"
