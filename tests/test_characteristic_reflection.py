# tests/test_characteristic_reflection.py
'''
The payoff test for the CHARACTERISTIC (non-reflecting) far-field boundary condition: does it actually
let waves LEAVE the domain instead of bouncing them back?

Setup: a quiescent gas at rest (u = 0) on the unit square, with a small Gaussian PRESSURE pulse at the
centre. The pulse radiates outward as an acoustic wave and reaches the boundaries. We track the domain
acoustic energy

        E(t) = integral [ (p - p0)^2 / (2 rho0 c0^2)  +  1/2 rho |u|^2 ] dV

and compare, at a time by which the wave has reached the boundary, how much energy REMAINS:

  - `characteristic` boundaries  -> the wave passes through; almost no energy is left behind (~0.1%).
  - PERIODIC boundaries          -> the energy-conserving CONTROL: nothing can leave, so ~all of it
                                    stays (~99.6%), and what little is lost bounds the interior
                                    numerical dissipation.

The periodic box is the right reference precisely BECAUSE it conserves energy: it isolates "did the
energy leave through the boundary?" from "did the scheme dissipate it?". The ~99.6% it retains proves
the interior dissipation is negligible, so the characteristic BC's ~0.1% is genuine ABSORPTION, not
dissipation eating everything.

(Two references that are NOT used, and why: a plain `outflow` BC is ill-posed at a stagnant boundary
-- pure extrapolation with u.n ~ 0 has no defined outgoing direction and simply goes unstable. A
reflecting SLIP WALL does bounce the wave back, but a weakly-imposed reflecting wall is itself
numerically LOSSY -- the Riemann dissipation acts on the reflection jump each bounce -- so it retains
only ~34% here and muddies "reflection" with "wall dissipation". The periodic control avoids both.)

Run under BOTH Riemann solvers: the characteristic BC imposes F(q_bc).n directly, bypassing the
interior solver, so its absorption must be identical for roe and LLF -- which the test asserts.

This is Euler (acoustics need no viscosity) and it time-marches, so it is a slow test.
'''
import numpy as np
import pytest
from pathlib import Path

from dingus.config import CaseCfg, BCCfg
from dingus.mesh import mesh_class
from dingus.physics.constitutiveRelations import compute_pressure
from dingus.timeIntegrators.timeIntegration import compute_dt, time_step

# Reuse a unit-square mesh (5x5) with Bottom/Right/Top/Left boundaries; the BCs are overridden below.
MESH   = Path(__file__).resolve().parent / "test_case_taylor_green_vortex" / "inputs" / "Square.mesh"
GAMMA  = 1.4
RHO0   = 1.0
P0     = 1.0
C0     = np.sqrt(GAMMA * P0 / RHO0)          # background sound speed
AMP    = 0.015                               # pulse amplitude (small -> ~linear acoustics, stays positive)
SIGMA  = 0.08                                # pulse width
POLY   = 5
T_END  = 1.0                                 # the corner is ~0.71/c0 ~ 0.6 away; by t=1 the wave has hit


def _cfg(solver: str) -> CaseCfg:
    return CaseCfg.model_validate({
        'mesh':           {'mesh_format': 'HOHQMesh', 'mesh_file': str(MESH), 'ndim': 2, 'poly_deg': POLY, 'quad_type': 'LG'},
        'physics':        {'model': 'euler', 'gamma': GAMMA, 'riemann_solver': solver},
        'time_stepping':  {'time_integrator': 'rk4', 'cfl': 0.6, 'final_time': T_END, 'start_time': 0.0},
        'initialization': {'IC_method': 'analytical', 'IC_file': 'ic.py'},
        'io':             {'output_format': 'vtk', 'output_dir': './o/'},
    })


def _build(bc_type: str, solver: str):
    '''Build the mesh with the given boundary type on all four sides and the centred pressure pulse.'''
    cfg = _cfg(solver)
    if bc_type == 'characteristic':
        char = BCCfg.model_validate({'type': 'characteristic', 'state': [RHO0, 0.0, 0.0, P0 / (GAMMA - 1.0)]})
        cfg.boundary_conditions = {name: char for name in ('Bottom', 'Right', 'Top', 'Left')}
    else:   # 'periodic' -- the energy-conserving control
        cfg.boundary_conditions = {
            'Bottom': BCCfg.model_validate({'type': 'periodic', 'partner': 'Top'}),
            'Top':    BCCfg.model_validate({'type': 'periodic', 'partner': 'Bottom'}),
            'Left':   BCCfg.model_validate({'type': 'periodic', 'partner': 'Right'}),
            'Right':  BCCfg.model_validate({'type': 'periodic', 'partner': 'Left'}),
        }

    mesh = mesh_class.Mesh()
    mesh.read_mesh(MESH)
    mesh.construct_mesh(cfg)
    for e in mesh.elements:
        x, y = e.quad_node_coords[..., 0], e.quad_node_coords[..., 1]
        dp   = AMP * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / (2 * SIGMA**2))   # Gaussian pressure bump
        q = np.zeros(x.shape + (4,))
        q[..., 0] = RHO0 + dp / C0**2          # isentropic density perturbation
        q[..., 3] = (P0 + dp) / (GAMMA - 1.0)  # u = v = 0
        e.solution = q
    return mesh, cfg


def _acoustic_energy(mesh, cfg) -> float:
    '''Domain-integrated linear acoustic energy: pressure-disturbance + kinetic, quadrature-weighted.'''
    w = mesh.quad_weights[:, None] * mesh.quad_weights[None, :]
    E = 0.0
    for e in mesh.elements:
        q   = e.solution
        p   = compute_pressure(q, cfg)
        ke  = 0.5 * (q[..., 1]**2 + q[..., 2]**2) / q[..., 0]
        E  += np.sum(e.jacobian_det * w * ((p - P0)**2 / (2 * RHO0 * C0**2) + ke))
    return float(E)


def _run(bc_type: str, solver: str):
    '''March to T_END, returning (peak energy, final energy).'''
    mesh, cfg = _build(bc_type, solver)
    E_peak = _acoustic_energy(mesh, cfg)
    t = 0.0
    while t < T_END:
        dt = min(compute_dt(mesh, cfg), T_END - t)
        time_step(mesh, cfg, t, dt)
        t += dt
        E_peak = max(E_peak, _acoustic_energy(mesh, cfg))
    return E_peak, _acoustic_energy(mesh, cfg)


@pytest.fixture(scope='module')
def energies():
    '''
    Run each UNIQUE simulation exactly once -- characteristic and periodic, under roe and LLF (4 runs) --
    and cache (peak, final) acoustic energy keyed by (bc_type, solver). The tests below then just read
    these results, so the expensive marching is not duplicated across the absorption and
    solver-independence checks (which would otherwise re-run the characteristic cases). The fixture is
    lazy and module-scoped: it only executes when a (slow-gated) test that requests it actually runs, so
    a bare `pytest` never pays for it.
    '''
    out = {}
    for solver in ('roe', 'LLF'):
        for bc_type in ('characteristic', 'periodic'):
            out[(bc_type, solver)] = _run(bc_type, solver)
    return out


@pytest.mark.numerics
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize('solver', ['roe', 'LLF'])
def test_characteristic_bc_absorbs_an_acoustic_pulse(energies, solver):
    '''
    The characteristic BC must let the acoustic pulse LEAVE (tiny residual energy), while the periodic
    energy-conserving control keeps ~all of it. Checked under both Riemann solvers -- the characteristic
    result must be the same, since the BC imposes F(q_bc).n directly and does not use the interior solver.
    '''
    char_peak, char_final = energies[('characteristic', solver)]
    per_peak,  per_final  = energies[('periodic',       solver)]

    char_retained = char_final / char_peak
    per_retained  = per_final  / per_peak

    # 1. CONTROL: the periodic box conserves energy (nothing can leave), so it must retain nearly all of
    #    it. This simultaneously proves the interior dissipation is negligible -- WITHOUT it, assertion 2
    #    would be vacuous (a heavily dissipative scheme would also leave ~nothing behind).
    assert per_retained > 0.90, (
        f"[{solver}] the periodic energy-conserving control retained only {per_retained*100:.1f}%; the "
        f"scheme is dissipating so much that the absorption test cannot be trusted."
    )

    # 2. the characteristic boundary ABSORBS: almost nothing is left after the wave reaches the edge.
    assert char_retained < 0.02, (
        f"[{solver}] characteristic BC retained {char_retained*100:.2f}% of the acoustic energy -- the "
        f"pulse is reflecting instead of leaving."
    )

    # 3. the discriminating comparison: with the SAME interior scheme, the characteristic boundary lets
    #    at least ~20x less energy remain than the (loss-free) periodic boundary -> the difference is the
    #    boundary genuinely radiating energy out, not dissipation.
    assert char_final < per_final / 20.0, (
        f"[{solver}] characteristic left {char_final:.2e} vs periodic {per_final:.2e} -- the boundary is "
        f"not clearly radiating energy out of the domain."
    )


@pytest.mark.numerics
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_characteristic_absorption_is_solver_independent(energies):
    '''The absorbed-energy result must be (near) identical for roe and LLF -- the BC bypasses the
    interior Riemann solver, so only round-off from the different interior dissipation should differ.'''
    _, final_roe = energies[('characteristic', 'roe')]
    _, final_llf = energies[('characteristic', 'LLF')]
    assert np.isclose(final_roe, final_llf, rtol=1e-6), (
        f"characteristic absorption differs between solvers: roe={final_roe:.6e}, LLF={final_llf:.6e}; "
        f"the BC should be solver-independent."
    )
