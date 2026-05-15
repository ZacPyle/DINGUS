# src/InitialConditions/initialize_solution.py
import importlib.util
import inspect
import numpy as np
import warnings
from dingus.config import CaseCfg
from dingus.mesh import mesh_class
from pathlib import Path

def _load_ic_module(IC_file: Path):
    """
    Dynamically load a user-defined Python module containing an analytical initialization of the solution.
    """

    # First, check if the specified initial condition file exists
    IC_file = Path(IC_file) # Should already be a Path object, but just make sure
    if not IC_file.is_file():
        raise FileNotFoundError(f"Initial condition file not found! Specified path is: '{IC_file}'")
    
    # Create a module spec from the specified initial condition file
    spec   = importlib.util.spec_from_file_location("user_ic", IC_file)

     # Ensure the module spec was successfully created
    if spec is None:
        raise ImportError(f"Could not create a module spec from the initial condition file: '{IC_file}'")
    if spec.loader is None:
        raise ImportError(f"Module spec for '{IC_file}' has no loader.")

    # Load the module from the spec and make contents available in this script.
    module = importlib.util.module_from_spec(spec)    
    spec.loader.exec_module(module)

    # Check that the function 'initial_condition()' is defined in the user-defined initial_condition.py
    if not hasattr(module, "initial_condition"):
        raise AttributeError(f"No function 'intial_condition()' found in: '{IC_file}'")

    return module

def _call_ic_function(ic_func, coords: np.ndarray) -> np.ndarray:
    """
    Call the user-defined initial condition function at the specified coordinates (should be quadrature nodes). This wrapper function
    if defined so the dimensionality of the input coordinates and initial_condition() function can be handled flexibly. For example,
    if a 3D case is being run but the user defines a 2D initial condition then initial_condition() may only accept x and y coordinates
    as input arguments. 

    The dimensionality handling is based on the signature of initial_condition(). Accepted signatures are:
    - initial_condition(x)        - for 1D cases
    - initial_condition(x, y)     - for 2D cases
    - initial_condition(x, y, z)  - for 3D cases
    """

    params   = inspect.signature(ic_func).parameters
    n_params = len(params)
    
    match n_params:
        case 1:
            return ic_func(coords[...,0]) # Pass x-coordinates only
        case 2:
            return ic_func(coords[...,0], coords[...,1]) # Pass x and y coordinates
        case 3:
            return ic_func(coords[...,0], coords[...,1], coords[...,2]) # Pass x, y, and z coordinates
        case _:
            raise TypeError(
                f"'initial_condition' must accept 1, 2, or 3 positional arguments "
                f"(got {n_params}: {list(params)})"
            ) 

def initialize(case_config: CaseCfg, IC_FILE: Path, my_mesh: mesh_class.Mesh) -> None:
    """
    Computes the initial condition of the solution at the quadrature nodes. This can be done based on a user-defined analytical
    function or by reading in a data file containing the initial condition values. The appropriate method is determined from the
    user input file.
    """

    # TODO: initialize
    # - analytical solution
    # - read in a data file at t = 0
    # - read in a restart file at t =/= 0

    # Extract the initial condition parameters from the case configuration object
    IC_method = case_config.initialization.IC_method

    # Create a dispatch dictionary to call appropriate mapping funciton based on dimensionality.
    IC_DISPATCH = {
        'analytical' : IC_from_function,
        'datafile'   : IC_from_data_file,
        'restart'    : IC_from_restart
    }
    if IC_method not in IC_DISPATCH:
        raise ValueError(f"Unsupported IC_method: '{IC_method}'. Supported methods are: 'analytical', 'datafile', and 'restart',")

    # Call appropriate function to compute the initial condition based on user-defined method
    IC_DISPATCH[IC_method](IC_FILE, my_mesh)
    
    return

def IC_from_function(IC_file: Path, my_mesh: mesh_class.Mesh):
    """
    Load a user-defined function from an initial_condition.py file. This function may be a simple mathematical function (e.g., sin(2*pi*x)),
    or a more complex logic-based function. 
    """

    # Ensure the IC_file input is a Path object
    IC_file = Path(IC_file)

    # Load the user-defined initial condition module and extract the function 'initial_condition()' from it.
    module  = _load_ic_module(IC_file)
    ic_func = module.initial_condition

    # Check that the mesh dimensionality and initial condition function are compatible.
    mesh_dim = my_mesh.dim
    ic_dim   = len(inspect.signature(ic_func).parameters)
    if mesh_dim < ic_dim:
        raise ValueError(
            f"Initial condition function expects {ic_dim} input arguments, but mesh is only {mesh_dim}D. "
            f"Please ensure the initial condition function is compatible with the mesh dimensionality."
        )
    if mesh_dim > ic_dim:
        warnings.warn(
            f"Initial condition function expects {ic_dim} input arguments, but mesh is {mesh_dim}D. "
            f"Assuming the initial condition function only depends on the first {ic_dim} coordinates."
        )
    # Call the user-defined initial condition function at the quadrature nodes to initialize the solution
    for e in my_mesh.elements:
        coords     = e.quad_node_coords
        e.solution = _call_ic_function(ic_func, coords)

    return

def IC_from_data_file(IC_file: Path, my_mesh: mesh_class.Mesh):
    # TODO: Implement IC_from_data_file() 
    raise NotImplementedError("IC_from_data_file() is not implemented yet.")
    return

def IC_from_restart(IC_file: Path, my_mesh: mesh_class.Mesh):
    # TODO: Implement IC_from_restart()
    raise NotImplementedError("IC_from_restart() is not implemented yet.")
    return