# tests/test_initial_condition_compute.py
from dingus.config import load_case_yaml as file_reader
from dingus.config import CaseCfg
from pathlib import Path
from pprint import pprint
from dingus.InitialConditions.initialize_solution import initialize
import dingus.coreNumerics.quadrature as quadrature
import dingus.IO.meshPlotter as meshPlotter
import dingus.mesh.mesh_class as mesh_class
import pytest

# Create path to the case directory
TESTS_INPUT  = Path(__file__).resolve().parent / "testInputs" / "2D" 
TESTS_OUTPUT = Path(__file__).resolve().parent / "testOutputs" / "2D" 

@ pytest.fixture(scope="module")
def initialize_mesh():

    # Create the 'case directory'
    CASE_DIR = TESTS_INPUT

    # Define input file name
    INPUT_FILE = CASE_DIR / "control.yaml"

    # Read in the configuration data from the input file and create a configuration object.
    case_config = file_reader(INPUT_FILE)

    # Extract data and construct mesh file name
    MESH_FILE = CASE_DIR / case_config.mesh.mesh_file 
    IC_FILE   = CASE_DIR / str(case_config.initialization.IC_file)

    # Read in the mesh file and construct the mesh object
    my_mesh = mesh_class.Mesh()
    my_mesh.read_mesh(MESH_FILE)

    # Construct elements and mortars, then link them
    my_mesh.construct_elements()
    my_mesh.construct_mortars()
    my_mesh.link_elements_and_mortars()

    # Construct quadrature nodes within each element and compute element metrics
    my_mesh.quad_type = str(case_config.mesh.quad_type)
    quadrature.Compute_Quadrature_Nodes_And_Weights(my_mesh)
    my_mesh.compute_element_metrics()

    # Compute the initial condition at the quadrature nodes
    initialize(case_config, IC_FILE, my_mesh)

    return my_mesh, case_config

def test_initial_condition_population(initialize_mesh):
    test_mesh, case_config = initialize_mesh

    assert all(e.solution is not None for e in test_mesh.elements)
    assert test_mesh.elements[0].solution.shape == (test_mesh.el_poly_order+1, test_mesh.el_poly_order+1, case_config.physics.num_eq)
