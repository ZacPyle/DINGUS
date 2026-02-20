from pathlib import Path
from pprint import pprint
import dingus.mesh.hohqmesh_handler as hoq
import numpy as np

TESTS_INPUT  = Path(__file__).resolve().parent / "testInputs" 
TESTS_OUTPUT = Path(__file__).resolve().parent / "testOutputs"

def test_read_hohqmesh_ism():
    """
    Test the reading of a HOHQMesh mesh file in ISM format.
    """
    try:
        hoq.read_hohqmesh(TESTS_INPUT / "2D" / "Square_ISM.mesh")
    except Exception as e:
        errorType = type(e)
        errorMsg  = str(e)
        assert errorType == NotImplementedError
        assert errorMsg == "ISM and ISM-MM format not implemented for HOHQMesh mesh files!"

def test_read_hohqmesh_ism_mm():
    """
    Test the reading of a HOHQMesh mesh file in ISM-MM format.
    """
    try:
        hoq.read_hohqmesh(TESTS_INPUT / "2D" / "Square_ISMMM.mesh")
    except Exception as e:
        errorType = type(e)
        errorMsg  = str(e)
        assert errorType == NotImplementedError
        assert errorMsg == "ISM and ISM-MM format not implemented for HOHQMesh mesh files!"

def test_read_hohqmesh_abaqus():
    """
    Test the reading of a HOHQMesh mesh file in ABAQUS format.
    """
    try:
        hoq.read_hohqmesh(TESTS_INPUT / "2D" / "Square_ABAQUS.mesh")
    except Exception as e:
        errorType = type(e)
        errorMsg  = str(e)
        assert errorType == NotImplementedError
        assert errorMsg == "ABAQUS format not implemented for HOHQMesh mesh files!"

def test_read_hohqmesh_ism_v2():
    """
    Test the reading of a HOHQMesh mesh file in ISM-V2 format.
    """

    # Call the reader; output is a dictionary
    meshData = hoq.read_hohqmesh(TESTS_INPUT / "2D" / "Square_ISMV2.mesh")
    
    # Check that the header was read in correctly
    assert meshData["format"]         == "ISM-V2"
    assert meshData["num_nodes"]      == 36
    assert meshData["num_mortars"] == 60
    assert meshData["num_elements"]   == 25
    assert meshData["poly_order"]  == 8

    # Check that the nodes were read in correctly
    assert meshData["nodes"].shape   == (meshData["num_nodes"], 3)
    

    # Check that all mortars were read in
    assert meshData["mortars"].shape == (meshData["num_mortars"], 6)

    # Check that the element data (node IDs, mortar curves, BCs) were read in correctly
    assert meshData["elementNodeIDs"].shape      == (meshData["num_elements"], 4)
    assert meshData["mortarCurvatureType"].shape == (meshData["num_elements"], 4)
    assert meshData["elementBCNames"].shape      == (meshData["num_elements"], 4)