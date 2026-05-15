# test/test_input_file_reader.py
from dingus.config import load_case_yaml as file_reader
from dingus.config import CaseCfg
from pathlib import Path
from pprint import pprint
import pytest

# Create path to the case directory
TESTS_INPUT  = Path(__file__).resolve().parent / "testInputs" / "2D" 
TESTS_OUTPUT = Path(__file__).resolve().parent / "testOutputs" / "2D" 

# Define input file name
INPUT_FILE = "control.yaml"

# Create a pytest fixture (callable function) that reads the input file
# and returns the configuration data as a configuration object.
@pytest.fixture(scope="module")
def case_config():
    """
    Fixture that reads in the input file and creates a configuration object.
    """
    return file_reader(TESTS_INPUT / INPUT_FILE)

def test_read_input_file_basic(case_config):
    assert case_config is not None
    assert isinstance(case_config, CaseCfg)
    assert hasattr(case_config, "physics")
    assert hasattr(case_config, "mesh")
    assert hasattr(case_config, "time_stepping")

def test_read_input_file_physics(case_config):
    assert case_config.physics.model    == "scalar_advection"
    assert case_config.physics.gamma    == 1.4
    assert case_config.physics.Re       == 1000
    assert case_config.physics.Pr       == 0.71
    assert case_config.physics.mach_ref == 1.0

def test_read_input_file_mesh(case_config):
    assert case_config.mesh.mesh_format == "HOHQMesh"
    assert case_config.mesh.mesh_file   == "Square_ISMV2.mesh"
    assert case_config.mesh.ndim        == 2
    assert case_config.mesh.poly_deg    == 8
    assert case_config.mesh.quad_type   == "LG"

def test_read_input_file_time_stepping(case_config):
    assert case_config.time_stepping.time_integrator == "euler"
    assert case_config.time_stepping.cfl             == 0.2
    assert case_config.time_stepping.final_time      == 1.0

def test_read_input_file_initialization(case_config):
    assert case_config.initialization.IC_method == "analytical"
    assert case_config.initialization.IC_file    == "initial_condition.py"

def test_read_input_file_io(case_config):
    assert case_config.io.plot_uniform_grid == False
    assert case_config.io.output_format     == 'hdf5'
    assert case_config.io.uniform_grid_res  == 150
    assert case_config.io.output_interval   == 0.1
    assert case_config.io.monitor_run       == False
    assert case_config.io.monitor_interval  == 1.0