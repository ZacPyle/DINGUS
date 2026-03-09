import numpy as np
import dingus.mesh.mesh_class as mesh_class
import dingus.coreNumerics.quadrature as quadrature
import dingus.IO.meshPlotter as meshPlotter
from pathlib import Path
from pprint import pprint

TESTS_INPUT  = Path(__file__).resolve().parent / "testInputs" 
TESTS_OUTPUT = Path(__file__).resolve().parent / "testOutputs"

# Inistantiate the Mesh object
my_mesh = mesh_class.Mesh()

# Read in a mesh file
my_mesh.read_mesh(TESTS_INPUT / "2D" / "Square_ISMV2.mesh")

# Build the mesh
my_mesh.construct_mesh()

# Assert LG quadrature; this should be set in the control file but for now
# we hardcode it for testing.
my_mesh.quad_type = "LG"

# Compute quadrature nodes and weights
quadrature.Compute_Quadrature_Nodes_And_Weights(my_mesh)

# def test_isop_map_1d(test_mesh : mesh_class.Mesh = my_mesh):
#     '''
#     Tests the isoparametric mapping of quadrature nodes from the reference element [-1, 1] to
#     an arbitrary 1D range within each element.
#     '''

#     # Compute the isoparametric mapping
#     test_mesh.compute_isoparametric_mapping()

def test_isop_map_2d(test_mesh : mesh_class.Mesh = my_mesh):
    '''
    Tests the isoparametric mapping of quadrature nodes from the reference element [-1, 1] to
    an arbitrary 1D range within each element.
    '''

    # Compute the isoparametric mapping
    test_mesh.compute_isoparametric_mapping()

    # Plot the mesh to visually inspect the mapping of quadrature nodes
    ax = meshPlotter.plot_mesh(test_mesh, show = False)

    # Check that the figure was created successfully
    assert ax is not None

    # Save the output for inspection
    fig = ax.figure
    fig.savefig(TESTS_OUTPUT / "2D" /"QuadratureTest_2D.png", dpi=200)