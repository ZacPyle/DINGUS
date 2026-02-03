from dingus.mesh import mesh_class
from pathlib import Path
from pprint import pprint
import numpy as np

TESTS_INPUT  = Path(__file__).resolve().parent / "testInputs" 
TESTS_OUTPUT = Path(__file__).resolve().parent / "testOutputs"

# Inistantiate the Mesh object
my_mesh = mesh_class.Mesh()

# Read in a mesh file
my_mesh.read_mesh(TESTS_INPUT / "2D" / "Square_ISMV2.mesh")

def test_mesh_class_basic_info(test_mesh : mesh_class.Mesh = my_mesh):
    """
    Tests the reading of a mesh file and construction of Element and Mortar objects within
    the Mesh class.
    """
    # Inistantiate the Mesh object
    test_mesh = mesh_class.Mesh()

    # Read in a mesh file
    test_mesh.read_mesh(TESTS_INPUT / "2D" / "Square_ISMV2.mesh")

    # Test basic mesh attributes
    assert test_mesh.mesh_format   == "ism-v2"
    assert test_mesh.dim           == 2
    assert test_mesh.num_nodes     == 36
    assert test_mesh.num_elements  == 25
    assert test_mesh.num_mortars   == 60
    assert test_mesh.bc_poly_order == 8

def test_construct_elements(test_mesh : mesh_class.Mesh = my_mesh):
    """
    Tests the construct_element method of the Mesh class. Thie method should create a collection
    of SpectralElement objects within the attribute self.elements of the Mesh class.
    """

    # Construct the elements if not already constructed
    if not hasattr(test_mesh, "elements"): test_mesh.construct_elements()

    # Make sure the elements have been constructed
    assert hasattr(test_mesh, "elements")

    # Test various attributes of the constructed elements
    assert len(test_mesh.elements) == test_mesh.num_elements
    assert max([e.id_global for e in test_mesh.elements]) == test_mesh.num_elements
    assert min([e.id_global for e in test_mesh.elements]) == 1
    assert all([ e.id_global != 0 for e in test_mesh.elements])              # No zero element IDs
    assert all([e.el_type == "quad" for e in test_mesh.elements])
    assert all([e.poly_order == test_mesh.el_poly_order for e in test_mesh.elements])
    assert all([isinstance(e.node_ids, np.ndarray) for e in test_mesh.elements])
    assert all([e.node_ids.shape == (4,) for e in test_mesh.elements])       # 4 nodes per quad element
    assert all([e.node_coords.shape == (4, 3) for e in test_mesh.elements])  # 4 nodes per element, 3 coords per node
    assert all([0 not in e.node_ids for e in test_mesh.elements])            # No zero node IDs

def test_construct_mortars(test_mesh : mesh_class.Mesh = my_mesh):
    """
    Tests the construct_mortars method of the Mesh class. This method should create a collection
    of Mortar objects within the attribute self.mortars of the Mesh class.
    """

    # Construct the mortars if not already constructed
    if not hasattr(test_mesh, "mortars"): test_mesh.construct_mortars()

    # Make sure the mortars have been constructed
    assert hasattr(test_mesh, "mortars")

    # Test various attributes of the constructed mortars
    #breakpoint()
    assert len(test_mesh.mortars) == test_mesh.num_mortars
    assert max([mort.id_global for mort in test_mesh.mortars]) == test_mesh.num_mortars
    assert min([mort.id_global for mort in test_mesh.mortars]) == 1
    assert all([ mort.id_global != 0 for mort in test_mesh.mortars])              # No zero mortar IDs
    assert all([mort.mortar_type == "edge" for mort in test_mesh.mortars])
    assert all([mort.poly_order == test_mesh.el_poly_order for mort in test_mesh.mortars])
    assert all([isinstance(mort.node_ids, np.ndarray) for mort in test_mesh.mortars])
    assert all([mort.node_ids.shape == (2,) for mort in test_mesh.mortars])
    assert all([mort.node_coords.shape == (2, 3) for mort in test_mesh.mortars])  # 2 nodes per mortar, 3 coords per node
    assert all([0 not in mort.node_ids for mort in test_mesh.mortars])            # No zero node IDs

# def test_link_element_and_mortars(test_mesh : mesh_class.Mesh = my_mesh):
#     """
#     Tests the link_elements_and_mortars method of the Mesh class. This method should link
#     Element and Mortar objects together based on the connectivity information read in from
#     the mesh file.
#     """

#     # Construct elements and mortars if not already constructed
#     if not hasattr(test_mesh, "elements"):
#         test_mesh.construct_elements()
#     if not hasattr(test_mesh, "mortars"):
#         test_mesh.construct_mortars()

#     # Link the elements and mortars
#     test_mesh.link_elements_and_mortars()

#     raise NotImplementedError("Element - Mortar linking not yet implemented!")

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