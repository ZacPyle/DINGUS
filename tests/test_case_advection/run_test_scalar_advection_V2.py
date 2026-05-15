# cases/test_cases/scalar_advection/run_test_scalar_advection.py
from dingus.config import load_case_yaml as file_reader
from dingus.config import CaseCfg
from dingus.InitialConditions.initialize_solution import initialize
from dingus.mesh import element_class
from dingus.mesh import mesh_class
from dingus.mesh import mortar_class
from pathlib import Path
from pprint import pprint
import dingus.coreNumerics.quadrature as quadrature
import dingus.IO.meshPlotter as meshPlotter
import numpy as np
import pytest

##########################################################
# TEST FUNCTIONS -----------------------------------------
##########################################################

def test_case_config(test_config):
    # Top level of case configuration
    assert test_config is not None
    assert isinstance(test_config, CaseCfg)
    assert hasattr(test_config, "physics")
    assert hasattr(test_config, "mesh")
    assert hasattr(test_config, "time_stepping")

    # Physics configuration
    assert test_config.physics.model    == "scalar_advection"
    assert test_config.physics.gamma    == 1.4
    assert test_config.physics.Re       == 1000
    assert test_config.physics.Pr       == 0.71
    assert test_config.physics.mach_ref == 1.0

    # Mesh configuration
    assert test_config.mesh.mesh_format == "HOHQMesh"
    assert test_config.mesh.mesh_file   == "Square_ISMV2.mesh"
    assert test_config.mesh.ndim        == 2
    assert test_config.mesh.poly_deg    == 8
    assert test_config.mesh.quad_type   == "LG"

    # Time stepping configuration
    assert test_config.time_stepping.time_integrator == "euler"
    assert test_config.time_stepping.cfl             == 0.2
    assert test_config.time_stepping.final_time      == 1.0

    # Initialization configuration
    assert test_config.initialization.IC_method == "analytical"
    assert test_config.initialization.IC_file    == "initial_condition.py"

    # IO configuration
    assert test_config.io.plot_uniform_grid == False
    assert test_config.io.output_format     == 'hdf5'
    assert test_config.io.uniform_grid_res  == 150
    assert test_config.io.output_interval   == 0.1
    assert test_config.io.monitor_run       == False
    assert test_config.io.monitor_interval  == 1.0

    return

def test_mesh_elements_and_mortars(test_mesh):
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
        
    return

def test_mesh_quadrature(test_mesh):
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

    # Check that quadrature nodes and weights have actually been computed and saved
    assert len(test_mesh.quad_nodes  ) == (test_mesh.el_poly_order + 1)
    assert len(test_mesh.quad_weights) == (test_mesh.el_poly_order + 1)

    # Check accuracy of the computed nodes and weights
    assert np.allclose(test_mesh.quad_nodes  , ref_nodes  [test_mesh.quad_type], atol=1e-14)
    assert np.allclose(test_mesh.quad_weights, ref_weights[test_mesh.quad_type], atol=1e-14)
    return

def test_mesh_isop_mapping(test_mesh, test_config):
    # Plot the mesh to visually inspect the mapping of quadrature nodes
    ax = meshPlotter.plot_mesh(test_mesh, show = False)

    # Check that the figure was created successfully
    assert ax is not None

    # Save the output figure, named after the mesh file for easy identification
    stem     = test_config.name
    fig      = ax.figure
    fig_path = test_config.io.output_dir / 'Figures'
    if not fig_path.exists(): fig_path.mkdir(parents=True) 
    fig_name = fig_path / f"QuadratureTest_{stem}.png"
    fig.savefig(str(fig_path / f"QuadratureTest_{stem}.png"), dpi=200)
    return

def test_mesh_delaunay_tri(test_mesh):
    assert test_mesh.delaunay_coords is not None
    assert test_mesh.delaunay_tri is not None
    return

def test_initialization(test_mesh, test_config):
    assert all(e.solution is not None for e in test_mesh.elements)
    assert test_mesh.elements[0].solution.shape == (test_mesh.el_poly_order+1, test_mesh.el_poly_order+1, test_config.physics.num_eq)
    return

##########################################################
# FIXTURE DEFINITIONS-------------------------------------
##########################################################

@pytest.fixture(scope='module')
def build_case():
    ##########################################################
    # SET UP AND DEFINITIONS ---------------------------------
    ##########################################################
    # Create path to the case directory
    CASE_DIR    = Path(__file__).resolve().parent
    CASE_INPUTS = CASE_DIR / "inputs/"

    # Define input file name
    CTRL_FILE = "control.yaml"

    ##########################################################
    # EXECUTION BLOCK ----------------------------------------
    ##########################################################

    case_config = file_reader(CASE_INPUTS / CTRL_FILE)

    # Create the specified outputs directory it it does not exist
    if not case_config.io.output_dir.exists():
        case_config.io.output_dir.mkdir(parents=True)

    # Extract the data and construct the mesh file name
    MESH_FILE = CASE_INPUTS / case_config.mesh.mesh_file
    IC_FILE   = CASE_INPUTS / str(case_config.initialization.IC_file)

    # Read in mesh file and construct the mesh object
    my_mesh = mesh_class.Mesh()
    my_mesh.read_mesh(MESH_FILE)
    my_mesh.construct_mesh(case_config)

    # Compute the initial condition at the quadrature nodes
    initialize(case_config, IC_FILE, my_mesh)

    return case_config, my_mesh

