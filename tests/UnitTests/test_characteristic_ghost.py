# tests/UnitTests/test_characteristic_ghost.py
'''
Unit tests for the CHARACTERISTIC far-field ghost state (fluxes.characteristic_ghost_state):

    q_bc = 1/2 (q_int + q_ext) + 1/2 sign(A_n) (q_int - q_ext)

which takes each characteristic wave from the INTERIOR if it is outgoing (lambda > 0) and from the
EXTERIOR reference if it is incoming (lambda < 0). The tests pin the behaviour in each flow regime --
the whole point is that ONE formula self-selects the right number of imposed conditions per node:

  supersonic outflow (all lambda > 0)  -> q_bc == q_int   (extrapolate everything; impose nothing)
  supersonic inflow  (all lambda < 0)  -> q_bc == q_ext   (impose everything)
  subsonic           (mixed signs)     -> a genuine blend, imposing exactly the incoming count
  consistency        (q_int == q_ext)  -> q_bc == q_int   (no spurious blend)

The subsonic case is checked in CHARACTERISTIC variables: the outgoing Riemann invariants must match
the interior and the incoming ones must match the exterior -- that IS the non-reflecting property.
'''
import numpy as np
import pytest

from dingus.config import CaseCfg
from dingus.physics.fluxes import characteristic_ghost_state
from dingus.physics.constitutiveRelations import compute_pressure

CFG = CaseCfg.model_validate({
    'mesh':           {'mesh_format': 'HOHQMesh', 'mesh_file': 'u.mesh', 'ndim': 2, 'poly_deg': 2, 'quad_type': 'LG'},
    'physics':        {'model': 'euler', 'gamma': 1.4, 'riemann_solver': 'roe'},
    'time_stepping':  {'time_integrator': 'rk4', 'cfl': 0.5, 'final_time': 1.0, 'start_time': 0.0},
    'initialization': {'IC_method': 'analytical', 'IC_file': 'ic.py'},
    'io':             {'output_format': 'vtk', 'output_dir': './o/'},
})
GAMMA = CFG.physics.gamma
NORMAL = np.array([[1.0, 0.0]])          # outward normal +x; uₙ = u


def _state(rho, u, v, p):
    rhoE = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    return np.array([[rho, rho * u, rho * v, rhoE]])


def _sound_speed(q):
    rho = q[..., 0]
    return np.sqrt(GAMMA * compute_pressure(q, CFG) / rho)


def test_supersonic_outflow_extrapolates_everything():
    '''uₙ > c > 0: every characteristic is outgoing, so the ghost is the INTERIOR state (impose nothing).'''
    q_int = _state(1.0, 3.0, 0.2, 1.0)                 # u = 3, c ~ 1.18 -> uₙ = 3 > c: supersonic out
    q_ext = _state(0.5, 1.0, 0.0, 0.4)                 # arbitrary far-field; must be ignored
    assert _sound_speed(q_int)[0] < 3.0, "test setup: interior must be supersonic outflow"
    q_bc = characteristic_ghost_state(q_int, q_ext, NORMAL, CFG)
    assert np.allclose(q_bc, q_int), f"supersonic outflow must extrapolate the interior, got {q_bc}"


def test_supersonic_inflow_imposes_everything():
    '''uₙ < -c: every characteristic is incoming, so the ghost is the EXTERIOR reference (full Dirichlet).'''
    q_int = _state(1.0, -3.0, 0.1, 1.0)                # uₙ = -3, c ~ 1.18: supersonic inflow
    q_ext = _state(1.2, -2.5, 0.0, 1.5)               # the imposed far-field
    assert _sound_speed(q_int)[0] < 3.0, "test setup: interior must be supersonic inflow"
    q_bc = characteristic_ghost_state(q_int, q_ext, NORMAL, CFG)
    assert np.allclose(q_bc, q_ext), f"supersonic inflow must impose the exterior, got {q_bc}"


def test_consistency_no_spurious_blend():
    '''q_int == q_ext must give exactly q_int, in every regime (P+ + P- = I).'''
    for q in [_state(1.0, 0.3, -0.2, 1.0), _state(1.0, -0.4, 0.1, 2.0), _state(0.8, 2.5, 0.0, 0.5)]:
        q_bc = characteristic_ghost_state(q, q.copy(), NORMAL, CFG)
        assert np.allclose(q_bc, q), f"identical states must give the state itself, got {q_bc}"


def test_subsonic_outflow_imposes_exactly_the_back_pressure():
    '''
    Subsonic outflow (0 < uₙ < c): exactly ONE incoming characteristic (uₙ - c). So the ghost must
    match the INTERIOR on the three outgoing waves (entropy, shear, uₙ + c acoustic) and the EXTERIOR
    on the one incoming wave. We verify that in characteristic terms via the Riemann invariants:

        outgoing entropy   :  s = p / rho^gamma           -> from interior
        outgoing acoustic  :  J+ = uₙ + 2c/(gamma-1)      -> from interior
        outgoing shear     :  u_t                          -> from interior
        incoming acoustic  :  J- = uₙ - 2c/(gamma-1)      -> from exterior
    '''
    q_int = _state(1.0, 0.5, 0.3, 1.0)                 # u = 0.5, c ~ 1.18 -> 0 < uₙ < c: subsonic out
    q_ext = _state(0.7, 0.2, -0.1, 0.6)               # far-field with a different (back) pressure
    assert 0.0 < 0.5 < _sound_speed(q_int)[0], "test setup: interior must be subsonic outflow"

    q_bc = characteristic_ghost_state(q_int, q_ext, NORMAL, CFG)

    def invariants(q):
        rho = q[..., 0]; u = q[..., 1] / rho; v = q[..., 2] / rho
        p = compute_pressure(q, CFG); c = np.sqrt(GAMMA * p / rho)
        s  = p / rho**GAMMA                              # entropy
        Jp = u + 2 * c / (GAMMA - 1.0)                   # uₙ + 2c/(g-1)  (outgoing here)
        Jm = u - 2 * c / (GAMMA - 1.0)                   # uₙ - 2c/(g-1)  (incoming here)
        return s, Jp, Jm, v

    s_bc, Jp_bc, Jm_bc, ut_bc = invariants(q_bc)
    s_i,  Jp_i,  _,    ut_i   = invariants(q_int)
    _,    _,     Jm_e, _      = invariants(q_ext)

    # NOTE: these hold to LINEAR (acoustic) order -- the blend linearizes about the Roe average, so a
    # finite jump leaves an O(jump^2) residual. Use a modest tolerance rather than exact equality.
    assert np.isclose(Jm_bc[0], Jm_e[0], rtol=0.05), "incoming J- must come from the EXTERIOR (back-pressure)"
    assert np.isclose(Jp_bc[0], Jp_i[0], rtol=0.05), "outgoing J+ must come from the INTERIOR"
    assert np.isclose(s_bc[0],  s_i[0],  rtol=0.05), "outgoing entropy must come from the INTERIOR"
    assert np.isclose(ut_bc[0], ut_i[0], rtol=0.05), "outgoing shear (u_t) must come from the INTERIOR"


def test_ghost_is_physical_in_the_subsonic_blend():
    '''The blended ghost must stay physical (positive density and pressure) -- it feeds a flux next.'''
    q_int = _state(1.0, 0.5, 0.3, 1.0)
    q_ext = _state(0.7, 0.2, -0.1, 0.6)
    q_bc  = characteristic_ghost_state(q_int, q_ext, NORMAL, CFG)
    assert q_bc[..., 0] > 0.0, "blended ghost density must be positive"
    assert compute_pressure(q_bc, CFG)[0] > 0.0, "blended ghost pressure must be positive"
