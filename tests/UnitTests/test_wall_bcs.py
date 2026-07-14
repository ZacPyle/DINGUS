# tests/SingleTests/test_wall_bcs.py
'''
Unit tests for the four viscous WALL boundary conditions (boundaryConditions.py):

    adiabatic_slip_wall     adiabatic_no_slip_wall
    isothermal_slip_wall    isothermal_no_slip_wall

A viscous wall cannot be expressed as one ghost state, because the two halves of the NSE interface
flux want contradictory things from it. So the module exposes three pieces, and each is pinned here
against a closed-form answer:

  1. `exterior_state`           -- the INVISCID (Riemann) ghost. All four walls share it: reflect the
                                   NORMAL momentum, keep rho and rhoE. Tested for the two properties
                                   that make it a wall and make it safe:
                                       (a) zero mass flux through the wall  (impermeability)
                                       (b) ghost pressure == interior pressure  (positivity-safe;
                                           |mom| is unchanged by a reflection)
                                   Note it does NOT impose no-slip -- an impermeable wall carries no
                                   convective TANGENTIAL momentum flux, so no-slip is purely viscous.
                                   Reflecting the full momentum vector instead would inject a spurious
                                   O(lambda * u_t) tangential flux through the Riemann dissipation.

  2. `wall_state`               -- the DIRICHLET data the wall imposes. This is where the four types
                                   actually differ:
                                       no-slip    -> u_w = prescribed wall velocity (0 if stationary)
                                       slip       -> u_w = interior u with its normal part removed
                                       isothermal -> T_w = prescribed wall temperature
                                       adiabatic  -> T_w = interior temperature (T is a RESULT there)

  3. `gradient_exterior_state`  -- the BR1 gradient-pass ghost, q+ = 2 q_w - q-, whose whole purpose
                                   is that the central trace 1/2(q- + q+) lands on q_w EXACTLY. That
                                   identity is the test.

  4. `wall_viscous_normal_flux` -- the viscous flux AT the wall. The sharp test is thermal: drive a
                                   pure wall-normal temperature gradient past a stationary no-slip
                                   wall and check that an ADIABATIC wall conducts exactly zero energy
                                   while an ISOTHERMAL wall conducts the full Fourier flux
                                   -k dT/dn (equal and opposite for the two normal directions).
'''
import numpy as np
import pytest

from dingus.config import BCCfg, CaseCfg
from dingus.boundaryConditions.boundaryConditions import (exterior_state, gradient_exterior_state,
                                                          is_wall, wall_penalty_sigma, wall_state,
                                                          wall_viscous_normal_flux)
from dingus.physics.constitutiveRelations import compute_pressure, compute_temperature
from dingus.physics.fluxes import compute_inviscid_flux

# TIER: unit -- see tests/conftest.py for what the markers mean.
pytestmark = pytest.mark.unit


# Element length scale for the wall penalty. Any positive value works for these tests -- the penalty is
# proportional to the BC violation, so it is exactly zero whenever the trace already satisfies the wall.
H_ELEM = 0.1

CFG = CaseCfg.model_validate({
    'mesh':           {'mesh_format': 'HOHQMesh', 'mesh_file': 'unused.mesh', 'ndim': 2, 'poly_deg': 2, 'quad_type': 'LG'},
    'physics':        {'model': 'navier-stokes', 'Re': 100.0, 'Pr': 0.71, 'mach_ref': 0.5, 'gamma': 1.4, 'riemann_solver': 'LLF'},
    'time_stepping':  {'time_integrator': 'rk4', 'cfl': 0.5, 'final_time': 1.0, 'start_time': 0.0},
    'initialization': {'IC_method': 'analytical', 'IC_file': 'initial_condition.py'},
    'io':             {'output_format': 'vtk', 'output_dir': './outputs/'}
})
GAMMA   = CFG.physics.gamma
RE      = CFG.physics.Re
PR      = CFG.physics.Pr
GAMMAM2 = GAMMA * CFG.physics.mach_ref**2

ALL_WALLS = ['adiabatic_slip_wall', 'adiabatic_no_slip_wall',
             'isothermal_slip_wall', 'isothermal_no_slip_wall']


class _FakeMortar:
    '''The BC dispatch only ever reads mort.boundary_condition, so that is all we need to stand up.'''
    def __init__(self, bc: BCCfg):
        self.boundary_condition = bc


def _bc(bc_type, **kwargs) -> BCCfg:
    '''Build a validated BCCfg, supplying a wall temperature for the isothermal types.'''
    if bc_type.startswith('isothermal'):
        kwargs.setdefault('wall_temperature', 1.3)
    return BCCfg.model_validate({'type': bc_type, **kwargs})


def _state(rho, u, v, p):
    '''2D conserved state [rho, rho u, rho v, rho E] from primitives, as a (1, 4) face trace.'''
    rhoE = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    return np.array([[rho, rho * u, rho * v, rhoE]])


# A generic interior trace with BOTH a normal and a tangential velocity component, so that a
# reflection is distinguishable from a full reversal. The wall is y = const, outward normal +y.
Q_MINUS = _state(rho=1.2, u=0.7, v=-0.4, p=1.5)
NORMAL  = np.array([[0.0, 1.0]])


@pytest.mark.parametrize('bc_type', ALL_WALLS)
def test_inviscid_ghost_is_impermeable(bc_type):
    '''
    (1a) The central mass flux through ANY wall must vanish identically: the wall lets no mass through.
    With the normal-momentum reflection, 1/2 (mom-.n + mom+.n) = 1/2 (mom.n - mom.n) = 0 exactly.
    '''
    mort   = _FakeMortar(_bc(bc_type))
    q_plus = exterior_state(mort, Q_MINUS, CFG, 0.0, NORMAL)

    Fn_minus = np.einsum('med,md->me', compute_inviscid_flux(Q_MINUS, CFG), NORMAL)
    Fn_plus  = np.einsum('med,md->me', compute_inviscid_flux(q_plus,  CFG), NORMAL)
    mass_flux = 0.5 * (Fn_minus + Fn_plus)[..., 0]

    assert np.allclose(mass_flux, 0.0), f"'{bc_type}' leaks mass through the wall: {mass_flux}"


@pytest.mark.parametrize('bc_type', ALL_WALLS)
def test_inviscid_ghost_preserves_pressure(bc_type):
    '''
    (1b) The reflection changes the DIRECTION of the momentum but not its magnitude, so the ghost's
    kinetic energy -- and hence its pressure -- equals the interior's. This is what makes the wall
    ghost positivity-safe: it can never hand the Riemann solver a negative pressure.
    '''
    mort   = _FakeMortar(_bc(bc_type))
    q_plus = exterior_state(mort, Q_MINUS, CFG, 0.0, NORMAL)

    assert np.allclose(compute_pressure(q_plus, CFG), compute_pressure(Q_MINUS, CFG))
    assert np.allclose(q_plus[..., 0], Q_MINUS[..., 0]), "wall must not change the density"

    # And the reflection is a genuine reflection: tangential momentum survives, normal flips.
    assert np.allclose(q_plus[..., 1],  Q_MINUS[..., 1]), "tangential momentum must be untouched"
    assert np.allclose(q_plus[..., 2], -Q_MINUS[..., 2]), "normal momentum must be reversed"


@pytest.mark.parametrize('bc_type', ALL_WALLS)
def test_wall_state_imposes_impermeability(bc_type):
    '''(2) Every wall type is impermeable: the Dirichlet wall state has u.n = 0.'''
    q_w  = wall_state(_bc(bc_type), Q_MINUS, NORMAL, CFG)
    u_w  = q_w[..., 1:-1] / q_w[..., 0][..., None]
    u_n  = np.sum(u_w * NORMAL, axis=-1)

    assert np.allclose(u_n, 0.0), f"'{bc_type}' wall state has nonzero normal velocity {u_n}"


def test_no_slip_wall_state_takes_the_wall_velocity():
    '''(2) A no-slip wall hands the fluid its OWN velocity -- zero when stationary, U when moving.'''
    q_w = wall_state(_bc('adiabatic_no_slip_wall'), Q_MINUS, NORMAL, CFG)
    assert np.allclose(q_w[..., 1:-1], 0.0), "a stationary no-slip wall must drive the velocity to zero"

    # A moving wall (the driven plate of Couette flow) slides tangentially.
    U   = 0.35
    q_w = wall_state(_bc('adiabatic_no_slip_wall', wall_velocity=[U, 0.0]), Q_MINUS, NORMAL, CFG)
    u_w = q_w[..., 1:-1] / q_w[..., 0][..., None]
    assert np.allclose(u_w, np.array([[U, 0.0]])), f"moving no-slip wall must impose u = ({U}, 0), got {u_w}"


def test_slip_wall_state_keeps_the_tangential_velocity():
    '''(2) A slip wall removes ONLY the normal velocity -- the fluid slides freely along it.'''
    q_w = wall_state(_bc('adiabatic_slip_wall'), Q_MINUS, NORMAL, CFG)
    u_w = q_w[..., 1:-1] / q_w[..., 0][..., None]
    u_i = Q_MINUS[..., 1:-1] / Q_MINUS[..., 0][..., None]

    assert np.allclose(u_w[..., 0], u_i[..., 0]), "slip wall must NOT touch the tangential velocity"
    assert np.allclose(u_w[..., 1], 0.0),         "slip wall must remove the normal velocity"


def test_isothermal_wall_state_imposes_the_wall_temperature():
    '''(2) An isothermal wall pins T; an adiabatic wall inherits the interior T (it pins a FLUX instead).'''
    T_wall = 1.3
    q_w    = wall_state(_bc('isothermal_no_slip_wall', wall_temperature=T_wall), Q_MINUS, NORMAL, CFG)
    assert np.allclose(compute_temperature(q_w, CFG), T_wall)

    q_w = wall_state(_bc('adiabatic_no_slip_wall'), Q_MINUS, NORMAL, CFG)
    assert np.allclose(compute_temperature(q_w, CFG), compute_temperature(Q_MINUS, CFG)), \
        "an adiabatic wall constrains the heat flux, not the temperature -- T must come from the interior"


@pytest.mark.parametrize('bc_type', ALL_WALLS)
def test_gradient_ghost_makes_the_central_trace_land_on_the_wall_state(bc_type):
    '''
    (3) THE defining property of the BR1 gradient ghost. The gradient pass builds q* = 1/2 (q- + q+);
    the mirror q+ = 2 q_w - q- exists precisely so that q* == q_w EXACTLY. If this identity ever
    breaks, no-slip / isothermal silently degrade to half-strength conditions.
    '''
    mort   = _FakeMortar(_bc(bc_type))
    q_plus = gradient_exterior_state(mort, Q_MINUS, CFG, 0.0, NORMAL)
    q_star = 0.5 * (Q_MINUS + q_plus)
    q_w    = wall_state(_bc(bc_type), Q_MINUS, NORMAL, CFG)

    assert np.allclose(q_star, q_w), f"'{bc_type}': BR1 central trace {q_star} != wall state {q_w}"


def _pure_normal_temperature_gradient(rho, p, dT_dy):
    '''
    grad_q for a state at rest with UNIFORM rho and a wall-normal temperature gradient dT/dy.
    T = gamma M^2 p / rho, so at constant rho a temperature gradient is just a pressure gradient:
        dp/dy = rho * dT_dy / (gamma M^2),   and with u = 0,  d(rhoE)/dy = dp/dy / (gamma - 1).
    '''
    dp_dy  = rho * dT_dy / GAMMAM2
    grad_q = np.zeros((1, 4, 2))
    grad_q[0, 3, 1] = dp_dy / (GAMMA - 1.0)      # d(rhoE)/dy ; drho and dmom are zero
    return grad_q


@pytest.mark.parametrize('normal', [np.array([[0.0, 1.0]]), np.array([[0.0, -1.0]])])
def test_adiabatic_wall_conducts_no_heat(normal):
    '''
    (4) The adiabatic condition IS a statement about a flux: k dT/dn = 0. Drive a strong wall-normal
    temperature gradient into a STATIONARY no-slip wall (so the viscous work term is identically zero
    and the energy flux is pure conduction) and demand that exactly nothing gets through -- for BOTH
    orientations of the outward normal, since a sign error would only show up on one of them.
    '''
    rho, p, dT_dy = 1.0, 1.5, 3.0
    q_minus = _state(rho=rho, u=0.0, v=0.0, p=p)
    grad_q  = _pure_normal_temperature_gradient(rho, p, dT_dy)

    Fv_n = wall_viscous_normal_flux(_bc('adiabatic_no_slip_wall'), q_minus, grad_q, normal, CFG, H_ELEM)

    assert np.allclose(Fv_n[..., -1], 0.0), \
        f"adiabatic wall conducted heat ({Fv_n[..., -1]}) -- the wall-normal heat flux was not removed"


@pytest.mark.parametrize('normal', [np.array([[0.0, 1.0]]), np.array([[0.0, -1.0]])])
def test_isothermal_wall_conducts_the_full_fourier_flux(normal):
    '''
    (4) The counterpart: an ISOTHERMAL wall must conduct, and by exactly Fourier's law. With the wall
    held at the interior temperature (so the wall state is thermally identical to the interior and the
    only thing left is the flux itself), the energy flux must be

        F_energy . n  =  kappa (dT/dy) n_y / Re,      kappa = mu / (Pr (gamma-1) M^2),  mu = 1

    which is equal and opposite for the two normal orientations. Comparing against the adiabatic case
    above, this is precisely the term the adiabatic wall removes.
    '''
    rho, p, dT_dy = 1.0, 1.5, 3.0
    q_minus = _state(rho=rho, u=0.0, v=0.0, p=p)
    grad_q  = _pure_normal_temperature_gradient(rho, p, dT_dy)
    T_int   = float(compute_temperature(q_minus, CFG)[0])

    bc   = _bc('isothermal_no_slip_wall', wall_temperature=T_int)
    Fv_n = wall_viscous_normal_flux(bc, q_minus, grad_q, normal, CFG, H_ELEM)

    kappa    = 1.0 / (PR * (GAMMA - 1.0) * CFG.physics.mach_ref**2)
    expected = (kappa * dT_dy / RE) * normal[0, 1]

    assert np.allclose(Fv_n[..., -1], expected), \
        f"isothermal wall heat flux {Fv_n[..., -1]} != Fourier flux {expected}"


# -------------------------------------------------------------------------------------------------
# 5. THE WALL PENALTY -- the term whose absence made the scheme blow up.
# -------------------------------------------------------------------------------------------------
# A Dirichlet wall imposed weakly with NO penalty is not energy-stable: nothing damps a
# boundary-condition violation, so one grows. Without this term a Couette run started from rest decayed
# toward the exact solution until t~4, then diverged exponentially and died on a negative pressure by
# t~12 -- for BOTH stationary and moving walls. These tests pin the two properties that make the fix
# correct rather than merely stabilizing:
#
#   (a) it is CONSISTENT: exactly zero whenever the trace already satisfies the wall condition, so it
#       cannot perturb the exact solution or degrade the spectral convergence rate; and
#   (b) it is RESTORING: it opposes the violation (dissipative), rather than reinforcing it -- a sign
#       error here would turn the cure into a stronger version of the disease.

@pytest.mark.parametrize('bc_type', ALL_WALLS)
def test_penalty_vanishes_when_the_wall_condition_is_satisfied(bc_type):
    '''
    (a) CONSISTENCY. Build a trace that already SATISFIES the wall (u = u_wall, T = T_wall), so that
    q_wall == q_minus. The penalty must then be identically zero -- which is what guarantees it does not
    pollute the exact solution or the measured order of accuracy.
    '''
    bc = _bc(bc_type)

    # A trace that satisfies this wall exactly: take any state, then REPLACE it with its own wall state.
    q_sat = wall_state(bc, Q_MINUS, NORMAL, CFG)

    # wall_state is idempotent on a satisfying trace, so the jump -- and hence the penalty -- is zero.
    assert np.allclose(wall_state(bc, q_sat, NORMAL, CFG), q_sat), \
        f"'{bc_type}': wall_state is not idempotent, so a satisfying trace still shows a violation"

    grad_q = np.zeros((1, 4, 2))
    Fv_sat = wall_viscous_normal_flux(bc, q_sat, grad_q, NORMAL, CFG, H_ELEM)

    # With zero gradient the viscous flux is zero, so anything left over IS the penalty.
    assert np.allclose(Fv_sat, 0.0), (
        f"'{bc_type}': the penalty is nonzero ({Fv_sat}) on a trace that already satisfies the wall. "
        f"It must be proportional to the BC violation, or it will corrupt the exact solution."
    )


def test_penalty_opposes_the_violation():
    '''
    (b) RESTORING SIGN. This is the one that matters: a sign error would make the penalty AMPLIFY the
    boundary-condition violation instead of damping it -- a more energetic version of the very bug it
    is meant to fix, and one that no consistency test could see.

    The strong form adds  -(1/J) S (f* - f_int)  to dq/dt, with  f* = f_inviscid* - F_visc*.n. So the
    penalty's contribution to dq/dt carries the OPPOSITE sign to its contribution to F_visc*.n. For the
    scheme to be restoring (dq/dt driving q^- toward q_wall) the penalty must therefore enter F_visc*.n
    as  -sigma (q^- - q_wall)  -- i.e. it must point AGAINST the violation. Check exactly that.
    '''
    # Fluid at rest against a wall dragged in +x: the violation is a pure +x momentum deficit.
    U       = 0.4
    bc      = _bc('adiabatic_no_slip_wall', wall_velocity=[U, 0.0])
    q_minus = _state(rho=1.0, u=0.0, v=0.0, p=1.5)      # fluid at rest -- violates the moving wall
    grad_q  = np.zeros((1, 4, 2))                       # zero gradient: the flux is the PENALTY alone

    Fv_n   = wall_viscous_normal_flux(bc, q_minus, grad_q, NORMAL, CFG, H_ELEM)
    q_wall = wall_state(bc, q_minus, NORMAL, CFG)
    sigma  = wall_penalty_sigma(q_minus, CFG, H_ELEM)

    # With grad_q = 0 the viscous flux vanishes, so Fv_n IS -sigma (q^- - q_wall).
    expected = -sigma[:, None] * (q_minus - q_wall)
    assert np.allclose(Fv_n, expected), f"penalty is {Fv_n}, expected -sigma (q- - q_wall) = {expected}"

    # And spell out the physical direction: the fluid is SHORT of x-momentum relative to the wall
    # (q^- - q_wall < 0 in the x-momentum slot), so the penalty flux must be POSITIVE there -- the wall
    # pushing momentum into the fluid. If this sign flipped, the wall would suck momentum out and run away.
    assert q_minus[0, 1] - q_wall[0, 1] < 0.0, "test setup: the fluid should be short of x-momentum"
    assert Fv_n[0, 1] > 0.0, (
        "the penalty is driving x-momentum the WRONG WAY: the fluid is slower than the wall, so the "
        "penalty must push momentum INTO the fluid. A sign error here amplifies the violation."
    )


def test_penalty_grows_with_polynomial_degree_and_shrinks_with_element_size():
    '''
    The penalty must scale like C (P+1)^2 mu / (Re h): it has to dominate the inverse-trace-inequality
    constant of the basis, which grows like P^2/h. Too small and the scheme stays unstable; the scaling
    is what makes one constant C work across degrees and meshes rather than needing hand-tuning.
    '''
    q = _state(rho=1.0, u=0.0, v=0.0, p=1.5)

    s_h_coarse = float(wall_penalty_sigma(q, CFG, 0.2)[0])
    s_h_fine   = float(wall_penalty_sigma(q, CFG, 0.1)[0])
    assert np.isclose(s_h_fine, 2.0 * s_h_coarse), "sigma must scale like 1/h"

    cfg_hi = CFG.model_copy(deep=True)
    cfg_hi.mesh.poly_deg = 2 * (CFG.mesh.poly_deg + 1) - 1        # (P+1) -> 2(P+1)
    s_p_hi = float(wall_penalty_sigma(q, cfg_hi, H_ELEM)[0])
    s_p_lo = float(wall_penalty_sigma(q, CFG, H_ELEM)[0])
    assert np.isclose(s_p_hi, 4.0 * s_p_lo), "sigma must scale like (P+1)^2"


def test_isothermal_wall_requires_a_wall_temperature():
    '''An isothermal wall with no temperature is a user error, and must be caught at config time.'''
    with pytest.raises(ValueError, match='wall_temperature'):
        BCCfg.model_validate({'type': 'isothermal_no_slip_wall'})


def test_is_wall_recognises_every_wall_type_and_nothing_else():
    for bc_type in ALL_WALLS:
        assert is_wall(_bc(bc_type)), f"'{bc_type}' should be recognised as a wall"

    assert not is_wall(BCCfg.model_validate({'type': 'outflow'}))
    assert not is_wall(BCCfg.model_validate({'type': 'periodic', 'partner': 'other'}))
