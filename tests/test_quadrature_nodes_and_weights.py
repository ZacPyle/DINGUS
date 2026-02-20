import numpy as np
from dingus.coreNumerics import quadrature
from dingus.mesh import mesh_class
from pathlib import Path
from pprint import pprint

TESTS_INPUT  = Path(__file__).resolve().parent / "testInputs" 
TESTS_OUTPUT = Path(__file__).resolve().parent / "testOutputs"

# Inistantiate the Mesh object
my_mesh = mesh_class.Mesh()

# Read in a mesh file
my_mesh.read_mesh(TESTS_INPUT / "2D" / "Square_ISMV2.mesh")

# Arrays of LG, LGL, CG, and CGL quadrature nodes an 8th order polynomial
ref_nodes = {}
ref_nodes["LG"]  = np.array([-0.96816024, -0.83603111, -0.61337143, -0.32425342,  0.0,
                              0.32425342,  0.61337143,  0.83603111,  0.96816024])
ref_nodes["LGL"] = np.array([-1.0       , -0.89975800, -0.67718628, -0.36311746,  0.0,
                              0.36311746,  0.67718628,  0.89975800,  1.0       ])
ref_nodes["CG"]  = np.array([-0.98480775, -0.86602540, -0.64278761, -0.34202014,  0.0,
                              0.34202014,  0.64278761,  0.86602540,  0.98480775])
ref_nodes["CGL"] = np.array([-1.0       , -0.92387953, -0.70710678, -0.38268343,  0.0,
                              0.38268343,  0.70710678,  0.92387953,  1.0       ])

# Arrays of LG, LGL, CG, and CGL quadrature weights an 8th order polynomial
ref_weights = {}
ref_weights["LG"]  = np.array([0.08127439, 0.18064816, 0.26061070, 0.31234708, 0.33023936,
                                0.31234708, 0.26061070, 0.18064816, 0.08127439])
ref_weights["LGL"] = np.array([0.02777778, 0.16549536, 0.27453871, 0.34642851, 0.37151927,
                                0.34642851, 0.27453871, 0.16549536, 0.02777778])
ref_weights["CG"]  = np.array([0.34906585, 0.34906585, 0.34906585, 0.34906585, 0.34906585,
                                0.34906585, 0.34906585, 0.34906585, 0.34906585])
ref_weights["CGL"] = np.array([0.19634954, 0.39269908, 0.39269908, 0.39269908, 0.39269908,
                                0.39269908, 0.39269908, 0.39269908, 0.19634954])

def test_LG_quadrature(test_mesh : mesh_class.Mesh = my_mesh):
    """
    Tests the computation of Legendre-Gauss quadrature nodes and weights.
    """

    # Assert LG quadrature; this should be set in the control file but for now
    # we hardcode it for testing.
    my_mesh.quad_type = "LG"

    quadrature.Compute_Quadrature_Nodes_And_Weights(my_mesh)

    # Check that quadrature nodes and weights have actually been computed and saved
    assert len(my_mesh.quad_nodes  ) == (my_mesh.el_poly_order + 1)
    assert len(my_mesh.quad_weights) == (my_mesh.el_poly_order + 1)

    # Check accuracy of the computed nodes and weights
    assert np.allclose(my_mesh.quad_nodes  , ref_nodes  ["LG"], atol=1e-14)
    assert np.allclose(my_mesh.quad_weights, ref_weights["LG"], atol=1e-14)

def test_LGL_quadrature(test_mesh : mesh_class.Mesh = my_mesh):
    """
    Tests the computation of Legendre-Gauss-Lobatto quadrature nodes and weights.
    """

    # Assert LG quadrature; this should be set in the control file but for now
    # we hardcode it for testing.
    my_mesh.quad_type = "LGL"

    quadrature.Compute_Quadrature_Nodes_And_Weights(my_mesh)

    # Check that quadrature nodes and weights have actually been computed and saved
    assert len(my_mesh.quad_nodes  ) == (my_mesh.el_poly_order + 1)
    assert len(my_mesh.quad_weights) == (my_mesh.el_poly_order + 1)

    # Check accuracy of the computed nodes and weights
    assert np.allclose(my_mesh.quad_nodes  , ref_nodes  ["LGL"], atol=1e-14)
    assert np.allclose(my_mesh.quad_weights, ref_weights["LGL"], atol=1e-14)

def test_CG_quadrature(test_mesh : mesh_class.Mesh = my_mesh):
    """
    Tests the computation of Legendre-Gauss quadrature nodes and weights.
    """

    # Assert CG quadrature; this should be set in the control file but for now
    # we hardcode it for testing.
    my_mesh.quad_type = "CG"

    quadrature.Compute_Quadrature_Nodes_And_Weights(my_mesh)

    # Check that quadrature nodes and weights have actually been computed and saved
    assert len(my_mesh.quad_nodes  ) == (my_mesh.el_poly_order + 1)
    assert len(my_mesh.quad_weights) == (my_mesh.el_poly_order + 1)

    # Check accuracy of the computed nodes and weights
    assert np.allclose(my_mesh.quad_nodes  , ref_nodes  ["CG"], atol=1e-14)
    assert np.allclose(my_mesh.quad_weights, ref_weights["CG"], atol=1e-14)

def test_CGL_quadrature(test_mesh : mesh_class.Mesh = my_mesh):
    """
    Tests the computation of Legendre-Gauss-Lobatto quadrature nodes and weights.
    """

    # Assert CG quadrature; this should be set in the control file but for now
    # we hardcode it for testing.
    my_mesh.quad_type = "CGL"

    quadrature.Compute_Quadrature_Nodes_And_Weights(my_mesh)

    # Check that quadrature nodes and weights have actually been computed and saved
    assert len(my_mesh.quad_nodes  ) == (my_mesh.el_poly_order + 1)
    assert len(my_mesh.quad_weights) == (my_mesh.el_poly_order + 1)

    # Check accuracy of the computed nodes and weights
    assert np.allclose(my_mesh.quad_nodes  , ref_nodes  ["CGL"], atol=1e-14)
    assert np.allclose(my_mesh.quad_weights, ref_weights["CGL"], atol=1e-14)