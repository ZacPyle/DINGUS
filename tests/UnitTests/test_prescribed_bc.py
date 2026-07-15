# tests/UnitTests/test_prescribed_bc.py
'''
Unit tests for the PRESCRIBED-state boundary conditions (boundaryConditions.py):

    uniform_inflow : a CONSTANT prescribed state, given as bc.state.
    prescribed     : a spatio-temporally VARYING state g(x, t), from a user boundary_state() function.

Both impose a full exterior state and share one machinery (the difference is only HOW the state is
supplied), so these tests pin the pieces each uses:

  1. `prescribed_target_state` -- the imposed state at the face nodes. For uniform_inflow it is the
        constant broadcast; for prescribed it is the user function evaluated at the face COORDINATES
        (so a transposed/mis-mapped coordinate array would be caught).
  2. `exterior_state`          -- the inviscid ghost IS that target (no reflection, unlike a wall).
  3. `gradient_exterior_state` -- the BR1 gradient ghost q+ = 2*target - q-, so the central trace
        1/2(q- + q+) lands on the target EXACTLY (the identity that makes it a hard Dirichlet condition).
  4. The interior PENALTY inside prescribed_viscous_normal_flux is proportional to (q- - target), so it
        vanishes when the trace already satisfies the BC (consistency) and points AGAINST a violation
        (restoring sign) -- the same two properties that matter for the wall penalty.
'''
import numpy as np
import pytest
from pathlib import Path

from dingus.config import BCCfg, CaseCfg
from dingus.boundaryConditions.boundaryConditions import (exterior_state, gradient_exterior_state,
                                                          is_prescribed, prescribed_target_state,
                                                          prescribed_viscous_normal_flux,
                                                          wall_penalty_sigma)

CFG = CaseCfg.model_validate({
    'mesh':           {'mesh_format': 'HOHQMesh', 'mesh_file': 'unused.mesh', 'ndim': 2, 'poly_deg': 2, 'quad_type': 'LG'},
    'physics':        {'model': 'navier-stokes', 'Re': 10.0, 'Pr': 0.71, 'mach_ref': 1.0, 'gamma': 1.4, 'riemann_solver': 'roe'},
    'time_stepping':  {'time_integrator': 'rk4', 'cfl': 0.5, 'final_time': 1.0, 'start_time': 0.0},
    'initialization': {'IC_method': 'analytical', 'IC_file': 'ic.py'},
    'io':             {'output_format': 'vtk', 'output_dir': './o/'},
})
GAMMA  = CFG.physics.gamma
H_ELEM = 0.1

# A boundary_state fixture that depends on BOTH x and y, so a coordinate mix-up is detectable.
_BC_FILE = Path(__file__).resolve().parent / "testInputs" / "prescribed_state_fixture.py"


def _state(rho, u, v, p):
    rhoE = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    return np.array([[rho, rho * u, rho * v, rhoE]])


class _FakeMortar:
    def __init__(self, bc): self.boundary_condition = bc


def _write_fixture():
    _BC_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BC_FILE.write_text(
        "import numpy as np\n"
        "def boundary_state(cfg, x, y, t):\n"
        "    rho = 1.0 + 0.1 * x + 0.2 * y\n"
        "    u   = 0.3 * y\n"
        "    v   = 0.1 * x\n"
        "    p   = 2.0 + 0.05 * x\n"
        "    g   = cfg.physics.gamma\n"
        "    E   = p / (g - 1.0) + 0.5 * rho * (u * u + v * v)\n"
        "    return np.stack([rho, rho * u, rho * v, E], axis=-1)\n"
    )


@pytest.fixture(autouse=True)
def _fixture_file():
    _write_fixture()
    yield


def test_is_prescribed_recognizes_both_types_and_nothing_else():
    assert is_prescribed(BCCfg.model_validate({'type': 'uniform_inflow', 'state': [1.0, 0.0, 0.0, 2.5]}))
    assert is_prescribed(BCCfg.model_validate({'type': 'prescribed', 'state_file': str(_BC_FILE)}))
    assert not is_prescribed(BCCfg.model_validate({'type': 'outflow'}))
    assert not is_prescribed(BCCfg.model_validate({'type': 'adiabatic_no_slip_wall'}))


def test_uniform_inflow_target_is_the_constant_broadcast():
    bc      = BCCfg.model_validate({'type': 'uniform_inflow', 'state': [1.2, 0.3, -0.1, 5.0]})
    q_minus = np.ones((4, 4))
    target  = prescribed_target_state(bc, q_minus, None, CFG)   # face_coords unused for a constant
    assert target.shape == (4, 4)
    assert np.allclose(target, np.array([1.2, 0.3, -0.1, 5.0]))


def test_prescribed_target_is_the_function_at_the_face_coordinates():
    '''The user function is evaluated at the FACE-NODE coordinates -- checked against a hand build.'''
    bc      = BCCfg.model_validate({'type': 'prescribed', 'state_file': str(_BC_FILE)})
    coords  = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])     # distinguishes x from y
    q_minus = np.ones((3, 4))
    target  = prescribed_target_state(bc, q_minus, coords, CFG)

    x, y = coords[:, 0], coords[:, 1]
    rho  = 1.0 + 0.1 * x + 0.2 * y
    assert np.allclose(target[:, 0], rho),          f"density not evaluated at the face coords: {target[:, 0]}"
    assert np.allclose(target[:, 1], rho * 0.3 * y), "x-momentum used the wrong coordinate"
    assert np.allclose(target[:, 2], rho * 0.1 * x), "y-momentum used the wrong coordinate"


def test_prescribed_needs_coordinates():
    bc = BCCfg.model_validate({'type': 'prescribed', 'state_file': str(_BC_FILE)})
    with pytest.raises(ValueError, match="coordinates"):
        prescribed_target_state(bc, np.ones((3, 4)), None, CFG)


def test_inviscid_ghost_is_the_target():
    '''Unlike a wall, the prescribed inviscid ghost is the target itself -- no reflection.'''
    bc     = BCCfg.model_validate({'type': 'prescribed', 'state_file': str(_BC_FILE)})
    coords = np.array([[0.2, 0.4], [0.6, 0.8]])
    q_minus = _state(1.0, 0.0, 0.0, 3.0) * np.ones((2, 1))
    q_plus = exterior_state(_FakeMortar(bc), q_minus, CFG, 0.0, None, coords)
    target = prescribed_target_state(bc, q_minus, coords, CFG)
    assert np.allclose(q_plus, target)


def test_gradient_ghost_makes_central_trace_land_on_the_target():
    '''THE identity: q+ = 2*target - q-  =>  1/2(q- + q+) = target exactly (hard Dirichlet on grad q).'''
    bc      = BCCfg.model_validate({'type': 'prescribed', 'state_file': str(_BC_FILE)})
    coords  = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    q_minus = _state(1.1, 0.2, -0.3, 2.5) * np.ones((3, 1))
    q_plus  = gradient_exterior_state(_FakeMortar(bc), q_minus, CFG, 0.0, None, coords)
    q_star  = 0.5 * (q_minus + q_plus)
    target  = prescribed_target_state(bc, q_minus, coords, CFG)
    assert np.allclose(q_star, target)


def test_penalty_vanishes_when_the_bc_is_satisfied():
    '''Consistency: with q- == target and zero gradient, the viscous flux (all penalty) is zero.'''
    bc      = BCCfg.model_validate({'type': 'prescribed', 'state_file': str(_BC_FILE)})
    coords  = np.array([[0.3, 0.7], [0.7, 0.3]])
    target  = prescribed_target_state(bc, np.ones((2, 4)), coords, CFG)
    grad    = np.zeros((2, 4, 2))
    normal  = np.array([[0.0, 1.0], [0.0, 1.0]])
    Fv = prescribed_viscous_normal_flux(bc, target, grad, normal, CFG, H_ELEM, coords)   # q- == target
    assert np.allclose(Fv, 0.0), f"penalty nonzero on a satisfied BC: {Fv}"


def test_penalty_opposes_the_violation():
    '''Restoring sign: with zero gradient the flux is exactly -sigma (q- - target), pointing against it.'''
    bc      = BCCfg.model_validate({'type': 'prescribed', 'state_file': str(_BC_FILE)})
    coords  = np.array([[0.4, 0.6]])
    target  = prescribed_target_state(bc, np.ones((1, 4)), coords, CFG)
    q_minus = target + np.array([[0.0, -0.05, 0.0, 0.0]])       # fluid short of x-momentum vs target
    grad    = np.zeros((1, 4, 2))
    normal  = np.array([[0.0, 1.0]])

    Fv    = prescribed_viscous_normal_flux(bc, q_minus, grad, normal, CFG, H_ELEM, coords)
    sigma = wall_penalty_sigma(q_minus, CFG, H_ELEM)
    assert np.allclose(Fv, -sigma[:, None] * (q_minus - target))
    assert Fv[0, 1] > 0.0, "penalty must push x-momentum INTO the fluid when it is short of the target"
