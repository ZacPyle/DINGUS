# tests/SingleTests/test_freestream_preservation.py
'''
FREE-STREAM PRESERVATION for the compressible Navier-Stokes solver.

A UNIFORM state is an exact solution of the Navier-Stokes equations: it has zero gradients (so the
viscous flux vanishes) and zero flux divergence (so the inviscid part vanishes), hence dq/dt = 0
EXACTLY. Marching it must therefore leave it bit-for-bit unchanged. Anything that drifts is a bug.

This lives on its own rather than inside the Taylor-Green script because it is a different KIND of
statement -- it is not about the vortex at all -- and because it is the one test that drives the FULL
pipeline (gradient pass -> volume flux -> interface flux -> dt -> RK4) and demands EXACT zero from it.
The related unit tests each cover only one link of that chain:

    test_gradient_pass.test_gradient_constant_field_is_zero   -> the gradient pass alone
    test_viscous_flux.test_zero_gradient_gives_zero_flux      -> the flux algebra alone, at one node

Neither exercises the SURFACE term (whether the strong-form correction f* - f_interior actually cancels
across an interface) nor the time integrator. This one does.

It is a weak check on the straight-sided meshes used today -- but it is the canonical METRIC IDENTITY
test, and it becomes the sharpest tool in the box the moment curved / isoparametric elements land
(roadmap Phase 5): a curved element whose metric terms do not satisfy the discrete geometric
conservation law will fail free-stream preservation, and essentially nothing else will catch it.
'''
import numpy as np
import pytest
from pathlib import Path

from dingus.config import load_case_yaml
from dingus.mesh import mesh_class
from dingus.timeIntegrators.timeIntegration import compute_dt, time_step

# TIER: numerics (the discrete metric identity), and `slow` because it marches NSE through many RK4
# steps. See tests/conftest.py for what the markers mean.
pytestmark = [pytest.mark.numerics, pytest.mark.slow]


# Borrows the Taylor-Green case directory purely as a convenient, validated 2D Navier-Stokes
# configuration (mesh + physics). Nothing about this test is Taylor-Green specific.
CASE_DIR  = Path(__file__).resolve().parent / "test_case_taylor_green_vortex"
CTRL_FILE = CASE_DIR / "control.yaml"


def _build_mesh(poly_deg: int = 4):
    cfg = load_case_yaml(CTRL_FILE)
    cfg.mesh.poly_deg = poly_deg
    mesh = mesh_class.Mesh()
    mesh.read_mesh(CASE_DIR / cfg.mesh.mesh_file)
    mesh.construct_mesh(cfg)
    return mesh, cfg


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_nse_uniform_flow_is_preserved():
    '''A constant state has zero residual, so time-marching NSE must leave it unchanged to round-off.'''
    mesh, cfg = _build_mesh(poly_deg=4)     # coarse on purpose: the property is P-independent and cheap

    # Uniform conserved state (set directly -- a trivial constant field, not a "case" worth an IC file).
    gamma = cfg.physics.gamma
    rho, u, v, p = 1.2, 0.3, -0.1, 1.5
    q0 = np.array([rho, rho * u, rho * v, p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)])
    for e in mesh.elements:
        e.solution = np.ones(e.quad_node_coords.shape[:-1] + (q0.size,)) * q0
    init = [e.solution.copy() for e in mesh.elements]

    t, T = 0.0, 0.05
    while t < T:
        dt = min(compute_dt(mesh, cfg), T - t)
        time_step(mesh, cfg, t, dt)
        t += dt

    drift = max(np.abs(e.solution - i0).max() for e, i0 in zip(mesh.elements, init))
    assert drift < 1e-11, f"uniform NSE flow drifted by {drift:.2e}; the viscous path should add zero"
