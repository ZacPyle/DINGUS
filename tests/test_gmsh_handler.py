from dingus.mesh.gmsh_handler import read_gmsh
from dingus.mesh.general_mesh_handler import plot_mesh
from pathlib import Path
import numpy as np

TESTS_INPUT  = Path(__file__).resolve().parent / "testInputs" 
TESTS_OUTPUT = Path(__file__).resolve().parent / "testOutputs"

def test_read_1d_line_mesh():
    # Read in the mesh
    m        = read_gmsh(TESTS_INPUT / "1D" / "advection.msh")

    assert "line" in m["cells"]              # Check that 'line' cell block exists
    assert m["cells"]["line"].shape[1] == 2  # Check that each line contains 2 nodes
    assert m["points"].shape[1] == 3         # Check that each point has 3 coordinates

    xs = m["points"][:, 0]
    assert np.isclose(xs.min(), 0.0)         # Check that minimum x-coordinate is 0.0
    assert np.isclose(xs.max(), 1.0)         # Check that maximum x-coordinate is 1.0

def test_plot_1d_line_mesh():
    # Read in the mesh
    m        = read_gmsh(TESTS_INPUT / "1D" / "advection.msh")

    # Plot the mesh
    ax = plot_mesh(m, show=False)
    assert ax is not None
    fig = ax.figure
    fig.savefig(TESTS_OUTPUT / "1D" /"gmsh1D.png", dpi=200)

def test_read_2d_quad_mesh():
    # Read in the mesh
    m        = read_gmsh(TESTS_INPUT / "2D" / "advection.msh")

    assert "quad" in m["cells"]              # Check that 'quad' cell block exists
    assert m["cells"]["quad"].shape[1] == 4  # Check that each quad contains 4 nodes
    assert m["points"].shape[1] == 3         # Check that each point has 3 coordinates

    xs = m["points"][:, 0]
    assert np.isclose(xs.min(), 0.0)         # Check that minimum x-coordinate is 0.0
    assert np.isclose(xs.max(), 1.0)         # Check that maximum x-coordinate is 1.0

def test_plot_2d_quad_mesh():
    # Read in the mesh
    m        = read_gmsh(TESTS_INPUT / "2D" / "advection.msh")

    # Plot the mesh
    ax = plot_mesh(m, show=False)
    assert ax is not None
    fig = ax.figure
    fig.savefig(TESTS_OUTPUT / "2D" /"gmsh2D_quad.png", dpi=200)