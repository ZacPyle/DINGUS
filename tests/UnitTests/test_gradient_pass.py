# tests/SingleTests/test_gradient_pass.py
'''
Unit test for the BR1 gradient pass (residual._compute_gradient), which computes g = grad(q) and
stores it in e.grad_q. This is step 1 of Navier-Stokes: the auxiliary/gradient variable that the
(future) viscous fluxes will consume. NOTHING here computes or uses a viscous flux -- we verify the
gradient operator in ISOLATION so that when the viscous flux is written later, its grad_q input is
already trusted.

The gradient operator has an EXACT answer, so we test it directly (the analogue of the Roe
scratchpad checks) with a three-rung manufactured-solution ladder of increasing strictness:

  1. CONSTANT field   q = 1        ->  grad_q == 0 to machine precision.
        This is the discrete metric-identity / free-stream check: it catches almost any bug in the
        volume term's contravariant contraction, and confirms the surface term adds nothing for a
        continuous field.

  2. LINEAR field     q = a x + b y ->  grad_q == (a, b) exactly (to round-off).
        The collocation derivative is exact for a linear field, so this pins the volume term's
        metric scaling. Checked on INTERIOR elements only (see note on the surface term below).

  3. SMOOTH field     q = sin(2 pi x) cos(2 pi y)  ->  grad_q converges SPECTRALLY to the analytic
        gradient as poly_deg increases. This is the high-order accuracy check.

Why the mesh + field must be consistent: for any CONTINUOUS field the two interface traces at an
interior face are equal (q- == q+), so the BR1 central correction (q* - q-) = 1/2 (q+ - q-) is
exactly zero there -- the surface term only fires at DOMAIN boundaries. We therefore use the
periodic scalar-advection Square mesh with:
  - constant / smooth-PERIODIC fields  -> consistent on ALL faces (periodic partners match), so the
    whole-domain gradient is clean; and
  - the (non-periodic) linear field    -> checked only on interior elements, where the neighbor
    trace is the true field. (Boundary elements would see a periodic partner shifted by a domain
    length, which is intentionally not the analytic field.)

The surface term's coupling of genuinely DISCONTINUOUS DG data is exercised end-to-end later by the
Navier-Stokes MMS / Taylor-Green convergence tests; here it is verified to stay consistent (zero for
continuous data) and not corrupt the interior.
'''
import numpy as np
import pytest
from pathlib import Path

from dingus.config import load_case_yaml
from dingus.mesh import mesh_class
from dingus.spatialOperators import residual

# TIER: unit -- see tests/conftest.py for what the markers mean.
pytestmark = pytest.mark.unit


# Reuse the known-good periodic, axis-aligned 2D case (scalar advection: num_eq = 1, so grad_q is
# (P+1, P+1, 1, 2)). _compute_gradient is physics-agnostic, so the scalar case is the simplest driver.
CASE_DIR  = Path(__file__).resolve().parent.parent / "test_case_scalar_advection"
CTRL_FILE = CASE_DIR / "control.yaml"


def _build_mesh(poly_deg: int):
    '''Load the periodic scalar-advection case, override poly_deg, and construct the mesh.'''
    cfg = load_case_yaml(CTRL_FILE)
    cfg.mesh.poly_deg = poly_deg
    # This gradient-accuracy test OWNS its mesh: it needs the finer 5x5 Square mesh to reach ~1e-6 by the
    # top swept degree. A coarse 2x2 converges ~2x slower per degree and would miss that threshold. The
    # gradient pass does no time-marching, so a coarser mesh buys no speed. control.yaml's mesh is free.
    cfg.mesh.mesh_file = "./inputs/Square.mesh"
    mesh = mesh_class.Mesh()
    mesh.read_mesh(CASE_DIR / cfg.mesh.mesh_file)
    mesh.construct_mesh(cfg)
    return mesh, cfg


def _set_field(mesh, f) -> None:
    '''Set e.solution to the scalar field f(x, y) sampled at each element's quadrature nodes.'''
    for e in mesh.elements:
        c = e.quad_node_coords                       # (P+1, P+1, 2)
        e.solution = f(c[..., 0], c[..., 1])[..., None]   # (P+1, P+1, 1)


def _grad_l2_error(mesh, gx, gy, interior_only: bool = False) -> float:
    '''
    Quadrature-weighted relative L2 error of e.grad_q against the analytic gradient (gx, gy).
    interior_only restricts the sum to elements whose faces are all interior (no domain boundary).
    '''
    num = den = 0.0
    for e in mesh.elements:
        if interior_only and not all(None not in m.connected_elements for m in e.connected_mortars):
            continue
        c     = e.quad_node_coords
        w     = e.jacobian_det * mesh.quad_weights[:, None] * mesh.quad_weights[None, :]
        exact = np.stack([gx(c[..., 0], c[..., 1]), gy(c[..., 0], c[..., 1])], axis=-1)  # (P+1,P+1,2)
        num  += np.sum(w[..., None] * (e.grad_q[..., 0, :] - exact)**2)
        den  += np.sum(w[..., None] * exact**2)
    return float(np.sqrt(num / den))


def test_gradient_constant_field_is_zero():
    '''Rung 1: grad of a constant field must vanish to machine precision (metric identity).'''
    mesh, cfg = _build_mesh(poly_deg=4)
    _set_field(mesh, lambda x, y: np.ones_like(x))
    residual._compute_gradient(mesh, cfg)

    worst = max(np.abs(e.grad_q).max() for e in mesh.elements)
    assert worst < 1e-11, f"grad of a constant field should be 0, got max |grad_q| = {worst:.2e}"


def test_gradient_linear_field_is_exact_interior():
    '''Rung 2: grad of a linear field a x + b y must equal (a, b) exactly on interior elements.'''
    a, b = 2.0, -3.0
    mesh, cfg = _build_mesh(poly_deg=4)

    # Guard: the exactness check needs at least one fully-interior element.
    n_interior = sum(all(None not in m.connected_elements for m in e.connected_mortars)
                     for e in mesh.elements)
    if n_interior == 0:
        pytest.skip("mesh has no fully-interior elements to check linear exactness")

    _set_field(mesh, lambda x, y: a * x + b * y)
    residual._compute_gradient(mesh, cfg)

    err = _grad_l2_error(mesh,
                         gx=lambda x, y: np.full_like(x, a),
                         gy=lambda x, y: np.full_like(x, b),
                         interior_only=True)
    assert err < 1e-11, f"grad of a linear field should be exact on interior elements, got L2 err {err:.2e}"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_gradient_smooth_field_converges_spectrally():
    '''Rung 3: grad of a smooth periodic field must converge spectrally with poly_deg.'''
    # Periodic on [0,1]^2 so partner traces match on every face (surface term stays consistent).
    field = lambda x, y: np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    gx    = lambda x, y:  2 * np.pi * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
    gy    = lambda x, y: -2 * np.pi * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    degrees = [3, 4, 5, 6]
    errors  = []
    for P in degrees:
        mesh, cfg = _build_mesh(poly_deg=P)
        _set_field(mesh, field)
        residual._compute_gradient(mesh, cfg)
        errors.append(_grad_l2_error(mesh, gx, gy))

    slope = float(np.polyfit(degrees, np.log10(errors), 1)[0])

    # 1. finite and strictly decreasing with poly_deg
    assert all(np.isfinite(errors)), f"non-finite gradient error in {errors}"
    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1)), \
        f"gradient error not decreasing with poly_deg: {errors}"

    # 2. spectral (exponential) decay: log10(error) drops at least ~0.6 per degree
    assert slope < -0.6, f"gradient convergence slope {slope:.3f}/degree is not spectral"

    # 3. sanity: the finest degree is already very accurate
    assert errors[-1] < 1e-4, f"finest-degree gradient error {errors[-1]:.2e} unexpectedly large"
