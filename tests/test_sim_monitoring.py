import dingus.mesh.mesh_class as mesh_class
import dingus.coreNumerics.quadrature as quadrature
import dingus.IO.meshPlotter as meshPlotter
import pytest
from pathlib import Path
from pprint import pprint

TESTS_INPUT  = Path(__file__).resolve().parent / "testInputs"
TESTS_OUTPUT = Path(__file__).resolve().parent / "testOutputs"

MESH_TEST_CASES = [
    ("Square_ISMV2.mesh", "LG" ),
    ("Square_ISMV2.mesh", "LGL"),
    ("Square2.mesh"     , "LG" ),
    ("Square2.mesh"     , "LGL")
]

# Create a pytest fixture. This is a callable function that serves as an
# input to test functions
@pytest.fixture(params=MESH_TEST_CASES,
                ids=[cases[0] for cases in MESH_TEST_CASES],
                scope="module")
def mesh_fixture(request):
    # Grab the mesh file name and quadrature type from the parameter list
    mesh_file, quad_type = request.param

    m = mesh_class.Mesh()
    m.read_mesh(TESTS_INPUT / "2D" / mesh_file)
    m.construct_elements()
    m.construct_mortars()
    m.link_elements_and_mortars()
    m.quad_type = quad_type
    quadrature.Compute_Quadrature_Nodes_And_Weights(m)
    m.compute_element_metrics()

    return m, mesh_file

def test_delaunay_tri(mesh_fixture):
    """
    Tests the generation of a Delaunay triangulation 
    """

    test_mesh, mesh_file = mesh_fixture

    # Build the triangulation
    test_mesh.build_delaunay_tri()

    assert test_mesh.delaunay_coords is not None
    assert test_mesh.delaunay_tri is not None

    