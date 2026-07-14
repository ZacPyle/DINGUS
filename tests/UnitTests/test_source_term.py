# tests/SingleTests/test_source_term.py
'''
Unit tests for the SOURCE TERM machinery (sourceTerms.py + its hook at the end of compute_residual).

A source term turns  q_t + div(F) = 0  into  q_t + div(F) = S(q, x, t), so the semi-discrete residual
picks up one extra, purely LOCAL term:

        q_t = -(1/J) [ div F~ ]  +  S

Two things need it: forced Poiseuille (a periodic channel has no mean pressure drop, so the driving
pressure gradient is supplied as a body force) and MMS (the leftover of pushing a manufactured solution
through the PDE). Both are worthless if S is added in the wrong place, so the tests pin exactly that:

  1. NO SOURCE IS THE DEFAULT     A case that declares no source must be bit-for-bit identical to before
                                  the feature existed. (Regression guard for every existing case.)

  2. THE SOURCE LANDS UNSCALED    On a UNIFORM field the flux divergence is exactly zero, so the residual
                                  must equal S EXACTLY -- no metric Jacobian, no sign flip. This is the
                                  test that catches adding S before the -1/J scaling (which would divide
                                  it by J and silently break every curved/stretched element) or with the
                                  wrong sign.

  3. S IS EVALUATED AT THE RIGHT   The fixture source depends on q, x, y AND t. Comparing against the
     (q, x, t)                     value rebuilt independently from each element's own quadrature-node
                                   coordinates catches a transposed or mis-mapped coordinate array, a
                                   stale state, or an ignored time argument -- none of which a constant
                                   source would reveal.
'''
import numpy as np
import pytest
from pathlib import Path

from dingus.config import load_case_yaml
from dingus.mesh import mesh_class
from dingus.spatialOperators.residual import compute_residual

# TIER: unit -- see tests/conftest.py for what the markers mean.
pytestmark = pytest.mark.unit


CASE_DIR  = Path(__file__).resolve().parent.parent / "test_case_scalar_advection"
CTRL_FILE = CASE_DIR / "control.yaml"
SRC_FILE  = Path(__file__).resolve().parent / "testInputs" / "source_term_manufactured.py"


def _build_mesh(with_source: bool):
    '''Load the scalar-advection case and build its mesh, optionally attaching the fixture source term.'''
    cfg = load_case_yaml(CTRL_FILE)
    cfg.mesh.poly_deg = 3                       # small and cheap; the source path is P-independent
    if with_source:
        cfg.source.source_method = 'analytical'
        cfg.source.source_file   = str(SRC_FILE)

    mesh = mesh_class.Mesh()
    mesh.read_mesh(CASE_DIR / cfg.mesh.mesh_file)
    mesh.construct_mesh(cfg)
    return mesh, cfg


def _set_uniform(mesh, value: float):
    '''A constant field: div(F) = a . grad(q) = 0 exactly, so the residual is the source alone.'''
    for e in mesh.elements:
        e.solution = np.full(e.quad_node_coords.shape[:-1] + (1,), value)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_no_source_is_the_default_and_changes_nothing():
    '''A uniform field with NO source term has zero residual -- the pre-existing behaviour, unchanged.'''
    mesh, cfg = _build_mesh(with_source=False)
    assert cfg.source.source_method == 'none', "cases must default to the unforced equations"

    _set_uniform(mesh, 2.5)
    compute_residual(mesh, cfg, t=0.0)

    worst = max(np.abs(e.residual).max() for e in mesh.elements)
    assert worst < 1e-12, f"unforced uniform advection must have zero residual, got {worst:.2e}"


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize('t', [0.0, 0.7])
def test_source_is_added_unscaled_and_at_the_right_nodes(t):
    '''
    The core test. On a uniform field div(F) = 0, so  residual == S(q, x, y, t)  EXACTLY.

    Rebuilding the expected S here from each element's own quadrature coordinates (rather than trusting
    the value the solver passed in) is what makes this catch a metric-Jacobian scaling, a sign flip, a
    transposed coordinate array, or a dropped time argument.
    '''
    mesh, cfg = _build_mesh(with_source=True)
    q0 = 2.5
    _set_uniform(mesh, q0)

    compute_residual(mesh, cfg, t=t)

    for e in mesh.elements:
        x, y     = e.quad_node_coords[..., 0], e.quad_node_coords[..., 1]
        expected = q0 * (1.0 + x + 2.0 * y + 3.0 * t)[..., None]     # the fixture's S, rebuilt here

        assert np.allclose(e.residual, expected, atol=1e-11), (
            f"residual != source at t={t}. Max deviation "
            f"{np.abs(e.residual - expected).max():.3e}. A source must be added AFTER the -1/J scaling, "
            f"with a + sign, at the node's own coordinates."
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_source_is_not_divided_by_the_jacobian():
    '''
    A sharper form of the scaling check, isolated because it is the easy mistake: if S were added before
    the residual's -1/J scaling it would come out as -S/J. On this mesh J is a nonzero constant per
    element, so that bug would show up as a uniform WRONG FACTOR -- which "allclose to expected" above
    already catches, but this states the failure mode explicitly against the actual Jacobian.
    '''
    mesh, cfg = _build_mesh(with_source=True)
    _set_uniform(mesh, 1.0)
    compute_residual(mesh, cfg, t=0.0)

    e        = mesh.elements[0]
    x, y     = e.quad_node_coords[..., 0], e.quad_node_coords[..., 1]
    expected = (1.0 + x + 2.0 * y)[..., None]
    wrong    = -expected * e.jacobian_det_inv[..., None]              # what the misplaced-scaling bug gives

    assert np.allclose(e.residual, expected, atol=1e-11)
    assert not np.allclose(e.residual, wrong), "the source term is being scaled by the metric Jacobian"
