from pathlib import Path
from pprint import pprint
import pytest
import dingus.mesh.hohqmesh_handler as hoq
import numpy as np

TESTS_INPUT  = Path(__file__).resolve().parent / "testInputs" 
TESTS_OUTPUT = Path(__file__).resolve().parent / "testOutputs"

# Define the expected values for each mesh file as a list of tuples:
# (mesh_file, num_nodes, num_mortars, num_elements, poly_order)
ISM_V2_TEST_CASES = [
    ("Square_ISMV2.mesh", 36, 60, 25, 8),
    ("Square2.mesh",      36, 60, 25, 6),
]

@pytest.mark.parametrize("mesh_file, num_nodes, num_mortars, num_elements, poly_order", ISM_V2_TEST_CASES)
def test_read_hohqmesh_ism_v2(mesh_file, num_nodes, num_mortars, num_elements, poly_order):
    """
    Test the reading of a HOHQMesh mesh file in ISM-V2 format.
    """

    # Call the reader; output is a dictionary
    meshData = hoq.read_hohqmesh(TESTS_INPUT / "2D" / mesh_file)
    
    # Check that the header was read in correctly
    assert meshData["format"]       == "ISM-V2"
    assert meshData["num_nodes"]    == num_nodes
    assert meshData["num_mortars"]  == num_mortars
    assert meshData["num_elements"] == num_elements
    assert meshData["poly_order"]   == poly_order

    # Check that the nodes were read in correctly
    assert meshData["nodes"].shape   == (num_nodes, 3)
    

    # Check that all mortars were read in
    assert meshData["mortars"].shape == (num_mortars, 6)

    # Check that the element data (node IDs, mortar curves, BCs) were read in correctly
    assert meshData["elementNodeIDs"].shape      == (num_elements, 4)
    assert meshData["mortarCurvatureType"].shape == (num_elements, 4)
    assert meshData["elementBCNames"].shape      == (num_elements, 4)

# ---------- NOT IMPLEMENTED ---------------------

# def test_read_hohqmesh_ism():
#     """
#     Test the reading of a HOHQMesh mesh file in ISM format.
#     """
#     try:
#         hoq.read_hohqmesh(TESTS_INPUT / "2D" / "Square_ISM.mesh")
#     except Exception as e:
#         errorType = type(e)
#         errorMsg  = str(e)
#         assert errorType == NotImplementedError
#         assert errorMsg == "ISM and ISM-MM format not implemented for HOHQMesh mesh files!"

# def test_read_hohqmesh_ism_mm():
#     """
#     Test the reading of a HOHQMesh mesh file in ISM-MM format.
#     """
#     try:
#         hoq.read_hohqmesh(TESTS_INPUT / "2D" / "Square_ISMMM.mesh")
#     except Exception as e:
#         errorType = type(e)
#         errorMsg  = str(e)
#         assert errorType == NotImplementedError
#         assert errorMsg == "ISM and ISM-MM format not implemented for HOHQMesh mesh files!"

# def test_read_hohqmesh_abaqus():
#     """
#     Test the reading of a HOHQMesh mesh file in ABAQUS format.
#     """
#     try:
#         hoq.read_hohqmesh(TESTS_INPUT / "2D" / "Square_ABAQUS.mesh")
#     except Exception as e:
#         errorType = type(e)
#         errorMsg  = str(e)
#         assert errorType == NotImplementedError
#         assert errorMsg == "ABAQUS format not implemented for HOHQMesh mesh files!"