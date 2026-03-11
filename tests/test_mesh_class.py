from dingus.mesh import element_class
from dingus.mesh import mesh_class
from dingus.mesh import mortar_class
from pathlib import Path
from pprint import pprint
import pytest
import numpy as np

TESTS_INPUT  = Path(__file__).resolve().parent / "testInputs" 
TESTS_OUTPUT = Path(__file__).resolve().parent / "testOutputs"

# Expected values for each mesh file
# (mesh_file, num_nodes, num_mortars, num_elements, bc_poly_order,
#  el0_bc_side0, el4_bc_side1, el_last_bc_side2,
#  mort0_bc, mort14_bc, mort_last_bc, mort_last_idx)
MESH_TEST_CASES = [
    ("Square_ISMV2.mesh", 36, 60, 25, 8,
     "Bottom", "Right", "Right",
     "Bottom", "Right", "Top", 59),
    ("Square2.mesh",      36, 60, 25, 6,
     None, None, "Left",        
     None, None, "Left", 59), 
]

# # Inistantiate the Mesh object
# my_mesh = mesh_class.Mesh()

# # Read in a mesh file
# my_mesh.read_mesh(TESTS_INPUT / "2D" / "Square_ISMV2.mesh")
@pytest.fixture(params=MESH_TEST_CASES,
                ids=[case[0] for case in MESH_TEST_CASES],
                scope="module")
def mesh_and_expected(request):
    """
    Fixture that builds a fully constructed Mesh object and bundles it with
    the expected values for that mesh file. Runs once per entry in MESH_TEST_CASES.
    """
    (mesh_file, num_nodes, num_mortars, num_elements, bc_poly_order,
     el0_bc0, el4_bc1, el_last_bc2,
     mort0_bc, mort14_bc, mort_last_bc, mort_last_idx) = request.param

    # Build the mesh once and share it across all tests in this parametrize run
    m = mesh_class.Mesh()
    m.read_mesh(TESTS_INPUT / "2D" / mesh_file)
    m.construct_elements()
    m.construct_mortars()
    m.link_elements_and_mortars()

    expected = {
        "num_nodes"     : num_nodes,
        "num_mortars"   : num_mortars,
        "num_elements"  : num_elements,
        "bc_poly_order" : bc_poly_order,
        "el0_bc0"       : el0_bc0,
        "el4_bc1"       : el4_bc1,
        "el_last_bc2"   : el_last_bc2,
        "mort0_bc"      : mort0_bc,
        "mort14_bc"     : mort14_bc,
        "mort_last_bc"  : mort_last_bc,
        "mort_last_idx" : mort_last_idx,
    }

    return m, expected

def test_mesh_class_basic_info(mesh_and_expected):
    """
    Tests the reading of a mesh file and construction of Element and Mortar objects within
    the Mesh class.
    """

    test_mesh, exp = mesh_and_expected

    # Test basic mesh attributes
    assert test_mesh.mesh_format   == "ism-v2"
    assert test_mesh.dim           == 2
    assert test_mesh.num_nodes     == exp["num_nodes"]
    assert test_mesh.num_elements  == exp["num_elements"]
    assert test_mesh.num_mortars   == exp["num_mortars"]
    assert test_mesh.bc_poly_order == exp["bc_poly_order"]

def test_construct_elements(mesh_and_expected):
    """
    Tests the construct_element method of the Mesh class. Thie method should create a collection
    of SpectralElement objects within the attribute self.elements of the Mesh class.
    """
    
    test_mesh, exp = mesh_and_expected

    # Test various attributes of the constructed elements
    assert len(test_mesh.elements) == test_mesh.num_elements
    assert max([e.id_global for e in test_mesh.elements]) == test_mesh.num_elements
    assert min([e.id_global for e in test_mesh.elements]) == 1
    assert all([e.id_global != 0 for e in test_mesh.elements])
    assert all([e.el_type == "quad" for e in test_mesh.elements])
    assert all([e.poly_order == test_mesh.el_poly_order for e in test_mesh.elements])
    assert all([isinstance(e.node_ids, np.ndarray) for e in test_mesh.elements])
    assert all([e.node_ids.shape == (4,) for e in test_mesh.elements])
    assert all([e.node_coords.shape == (4, 3) for e in test_mesh.elements])
    assert all([0 not in e.node_ids for e in test_mesh.elements])
    assert test_mesh.elements[0 ].boundary_condition_names[0] == exp["el0_bc0"]
    assert test_mesh.elements[4 ].boundary_condition_names[1] == exp["el4_bc1"]
    assert test_mesh.elements[-1].boundary_condition_names[1] == exp["el_last_bc2"]
    # assert test_mesh.elements[0 ].mortar_curvature[0] == 0
    # assert test_mesh.elements[4 ].mortar_curvature[1] == 0
    # assert test_mesh.elements[-1].mortar_curvature[1] == 0

def test_construct_mortars(mesh_and_expected):
    """
    Tests the construct_mortars method of the Mesh class. This method should create a collection
    of Mortar objects within the attribute self.mortars of the Mesh class.
    """

    test_mesh, exp = mesh_and_expected

    # Test various attributes of the constructed mortars
    assert len(test_mesh.mortars) == test_mesh.num_mortars
    assert max([mort.id_global for mort in test_mesh.mortars]) == test_mesh.num_mortars
    assert min([mort.id_global for mort in test_mesh.mortars]) == 1
    assert all([mort.id_global != 0 for mort in test_mesh.mortars])
    assert all([mort.mortar_type == "edge" for mort in test_mesh.mortars])
    assert all([mort.poly_order == test_mesh.el_poly_order for mort in test_mesh.mortars])
    assert all([isinstance(mort.node_ids, np.ndarray) for mort in test_mesh.mortars])
    assert all([mort.node_ids.shape == (2,) for mort in test_mesh.mortars])
    assert all([mort.node_coords.shape == (2, 3) for mort in test_mesh.mortars])
    assert all([0 not in mort.node_ids for mort in test_mesh.mortars])

def test_link_element_and_mortars(mesh_and_expected):
    """
    Tests the link_elements_and_mortars method of the Mesh class. This method should link
    Element and Mortar objects together based on the connectivity information read in from
    the mesh file.
    """

    test_mesh, exp = mesh_and_expected

    # Test that elements have been connected to mortars
    for mort in test_mesh.mortars:
        assert hasattr(mort, "connected_elements"), \
            f"Mortar {mort.id_global} is missing 'connected_elements' attribute!"
        assert any([isinstance(el, element_class.SpectralElement) for el in mort.connected_elements]), \
            f"Mortar {mort.id_global} has no connected Element objects!"
        
    # Test that the mortars have been connected to elements
    for el in test_mesh.elements:
        assert hasattr(el, "connected_mortars"), \
            f"Element {el.id_global} is missing 'connected_mortars' attribute!"
        assert all([isinstance(mort, mortar_class.SpectralMortar) for mort in el.connected_mortars]), \
            f"Element {el.id_global} is missing a connected Mortar object!"
        
    # Test several mortars to see that the boundary conditions are properly extracted from the element objects
    # assert test_mesh.mortars[0                   ].bc_name == exp["mort0_bc"]
    # assert test_mesh.mortars[14                  ].bc_name == exp["mort14_bc"]
    # assert test_mesh.mortars[-1].bc_name == exp["mort_last_bc"]

# ---------- NOT IMPLEMENTED ---------------------
# def test_apply_mortar_curvature(test_mesh : mesh_class.Mesh = my_mesh):
#     """
#     Tests the apply_mortar_curvature method of the Mesh class. 
#     """

#     # Construct elements and mortars if not already constructed
#     if not hasattr(test_mesh, "elements"):
#         test_mesh.construct_elements()
#     if not hasattr(test_mesh, "mortars"):
#         test_mesh.construct_mortars()

#     raise NotImplementedError("Applying mortar curvature not yet implemented!")

# def test_apply_boundary_conditions(test_mesh : mesh_class.Mesh = my_mesh):
#     """
#     Tests the apply_boundary_conditions method of the Mesh class. 

#     """

#     # Construct elements and mortars if not already constructed
#     if not hasattr(test_mesh, "elements"):
#         test_mesh.construct_elements()
#     if not hasattr(test_mesh, "mortars"):
#         test_mesh.construct_mortars()

#     raise NotImplementedError("Applying boundary conditions not yet implemented!")