# tests/UnitTests/test_face_node_flip.py
'''
Unit test for the FACE-NODE-ORDERING FLIP on general (non-axis-aligned) meshes.

On an axis-aligned mesh, two elements sharing an edge order its nodes the same way, so the neighbor's
prolonged face trace lines up with the local one node-for-node. On a general/curved mesh (e.g. a
cylinder) that is NOT true: for some shared edges the two elements traverse the nodes in OPPOSITE order
("orientation-reversed" mortars, encoded by a negative right-element side index in ISM-V2). The surface
terms must FLIP the neighbor's trace to realign it, and the outward normal must be taken in the LOCAL
element's node order.

This test drives a small cylinder mesh (which has reversed mortars) and checks the two invariants the
surface terms rely on, directly at every interior mortar:

  1. TRACE CONSISTENCY. For a field that is EXACTLY representable (linear), the prolonged trace from the
     left element must equal the flipped trace from the right element to round-off. A general smooth
     field would only match to interpolation-error, so LINEAR is used to isolate the flip from that.
     A wrong flip shows up as an O(field-variation) mismatch on the reversed mortars.

  2. NORMAL CONSISTENCY. The outward unit normals the two elements compute for the shared edge must be
     node-wise OPPOSITE (equal and opposite at the same physical point) once the flip is accounted for.
     This checks that the normal is built in each element's own node order (from its own metric), not
     the neighbor's.

If the mesh had NO reversed mortars the test would be vacuous, so it also asserts that some exist.
'''
import numpy as np
import pytest
from pathlib import Path

from dingus.config import CaseCfg, BCCfg
from dingus.mesh import mesh_class
from dingus.spatialOperators.residual import _prolong_to_face_2d, _prolong_metric_to_face_2d, _FACE_2D

pytestmark = pytest.mark.unit

MESH = Path(__file__).resolve().parent / "testInputs" / "2D" / "CoarseCylinder.mesh"


def _build():
    '''Load the coarse cylinder mesh (Outer box + Cylinder hole). BC types are irrelevant here -- only
    interior mortars are inspected -- so everything is `outflow` to satisfy construction.'''
    cfg = CaseCfg.model_validate({
        'mesh':           {'mesh_format': 'HOHQMesh', 'mesh_file': str(MESH), 'ndim': 2, 'poly_deg': 4, 'quad_type': 'LG'},
        'physics':        {'model': 'euler', 'gamma': 1.4, 'riemann_solver': 'roe'},
        'time_stepping':  {'time_integrator': 'rk4', 'cfl': 0.5, 'final_time': 1.0, 'start_time': 0.0},
        'initialization': {'IC_method': 'analytical', 'IC_file': 'ic.py'},
        'io':             {'output_format': 'vtk', 'output_dir': './o/'},
        'boundary_conditions': {'Outer':    BCCfg.model_validate({'type': 'outflow'}),
                                'Cylinder': BCCfg.model_validate({'type': 'outflow'})},
    })
    mesh = mesh_class.Mesh()
    mesh.read_mesh(MESH)
    mesh.construct_mesh(cfg)
    return mesh


def _face_normal(e, face_id, I_min, I_max):
    '''Outward unit normal at a face, in e's own node order (built from e's metric) -- exactly as the
    surface terms do it.'''
    Ja  = e.jacobian_det[..., None] * (e.contravar_xi if _FACE_2D[face_id]['axis'] == 0 else e.contravar_eta)
    Jaf = _prolong_metric_to_face_2d(Ja, face_id, I_min, I_max)
    return e.face_sign_map[face_id - 1] * Jaf / np.linalg.norm(Jaf, axis=-1)[:, None]


def _interior_mortars(mesh):
    for mort in mesh.mortars:
        left, right = mort.connected_elements
        if left is not None and right is not None:
            yield mort, left, right


def test_mesh_actually_has_reversed_mortars():
    '''Guard: a mesh with no reversed mortars would make the flip tests below vacuous.'''
    mesh = _build()
    n_rev = sum(mort.orientation_reversed for mort, _, _ in _interior_mortars(mesh))
    assert n_rev > 0, "the coarse cylinder mesh should contain orientation-reversed mortars"


def test_flip_gives_consistent_traces_for_a_linear_field():
    '''
    A LINEAR field is exactly representable, so with the correct flip the two elements' prolonged traces
    at every shared edge must agree to round-off -- ESPECIALLY on the reversed mortars, where a missing
    or spurious flip would leave an O(1) mismatch.
    '''
    mesh = _build()
    I_min, I_max = mesh.face_interp_min, mesh.face_interp_max

    for e in mesh.elements:
        c = e.quad_node_coords
        e.solution = (1.0 + 0.3 * c[..., 0] - 0.2 * c[..., 1])[..., None]   # linear -> exact

    worst_all = worst_reversed = 0.0
    for mort, left, right in _interior_mortars(mesh):
        fL = left.connected_mortars.index(mort) + 1
        fR = right.connected_mortars.index(mort) + 1
        qL = _prolong_to_face_2d(left.solution,  fL, I_min, I_max)
        qR = _prolong_to_face_2d(right.solution, fR, I_min, I_max)
        if mort.orientation_reversed:
            qR = qR[::-1]
        d = float(np.abs(qL - qR).max())
        worst_all = max(worst_all, d)
        if mort.orientation_reversed:
            worst_reversed = max(worst_reversed, d)

    assert worst_reversed < 1e-11, (
        f"reversed-mortar trace mismatch {worst_reversed:.2e} -- the face-node flip is wrong on "
        f"orientation-reversed mortars."
    )
    assert worst_all < 1e-11, f"interior trace mismatch {worst_all:.2e} for a linear (exact) field"


def test_element_normals_are_node_wise_opposite():
    '''
    The two elements sharing an edge must produce equal-and-opposite outward normals at the same
    physical node (after the flip). This confirms the normal is taken in each element's OWN node order.
    '''
    mesh = _build()
    I_min, I_max = mesh.face_interp_min, mesh.face_interp_max

    worst = 0.0
    for mort, left, right in _interior_mortars(mesh):
        fL = left.connected_mortars.index(mort) + 1
        fR = right.connected_mortars.index(mort) + 1
        nL = _face_normal(left,  fL, I_min, I_max)
        nR = _face_normal(right, fR, I_min, I_max)
        if mort.orientation_reversed:
            nR = nR[::-1]
        worst = max(worst, float(np.abs(nL + nR).max()))   # opposite -> nL + nR = 0

    assert worst < 1e-11, (
        f"the two elements' outward normals are not node-wise opposite (worst {worst:.2e}) -- the "
        f"normal is being taken in the wrong node order on reversed mortars."
    )
