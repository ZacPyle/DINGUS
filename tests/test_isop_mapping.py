import numpy as np
import dingus.mesh.mesh_class as mesh_class
import dingus.coreNumerics.quadrature as quadrature
import dingus.IO.meshPlotter as meshPlotter
import pytest
from pathlib import Path
from pprint import pprint

TESTS_INPUT  = Path(__file__).resolve().parent / "testInputs" 
TESTS_OUTPUT = Path(__file__).resolve().parent / "testOutputs"

MESH_TEST_CASES = [
    ("Square_ISMV2.mesh", "LG"),
    ("Square2.mesh",      "LG"),
]

@pytest.fixture(params=MESH_TEST_CASES,
                ids=[case[0] for case in MESH_TEST_CASES],
                scope="module")
def mesh_fixture(request):
    mesh_file, quad_type = request.param

    m = mesh_class.Mesh()
    m.read_mesh(TESTS_INPUT / "2D" / mesh_file)
    m.construct_elements()
    m.construct_mortars()
    m.link_elements_and_mortars()
    m.quad_type = quad_type
    quadrature.Compute_Quadrature_Nodes_And_Weights(m)

    return m, mesh_file

def test_isop_map_2d(mesh_fixture):
    '''
    Tests the isoparametric mapping of quadrature nodes from the reference element [-1, 1] to
    an arbitrary 1D range within each element.
    '''

    test_mesh, mesh_file = mesh_fixture

    # Compute the isoparametric mapping
    #test_mesh.compute_isoparametric_mapping()
    test_mesh.compute_element_metrics()

    # Plot the mesh to visually inspect the mapping of quadrature nodes
    ax = meshPlotter.plot_mesh(test_mesh, show = False)

    # Check that the figure was created successfully
    assert ax is not None

    # Save the output figure, named after the mesh file for easy identification
    stem = Path(mesh_file).stem
    fig  = ax.figure
    fig.savefig(TESTS_OUTPUT / "2D" / f"QuadratureTest_2D_{stem}.png", dpi=200)