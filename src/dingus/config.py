# src/dingus/config.py
import warnings
import yaml
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import Any, List, Literal, Dict,  Optional

# Helper functions ########################################################################
def default_case_name() -> str:
    """
    Generate a default case name based on the current date and time. This function is 
    used as the default factory for the 'name' field in the 'CaseCfg' class. Alternatively, 
    users can specify choose to use a 'lambda' function within the CaseCfg definition. I 
    find the helper function approach to be more intuitive.
    """

    return datetime.now().strftime("%Y_%m_%d__%H:%M:%S")

# Configuration classes ###################################################################
class PhysicsCfg(BaseModel):
    # Define the parameters for physics modeling
    model     : Literal['scalar_advection',                               # Physics model / governing equations
                        'euler', 
                        'navier-stokes']    = Field(...,
                                                    description='Governing equation to solve. Must be explicitly set.')
    
    gamma     : float                       = Field(1.4, gt=1.0,          # Ratio of specific heats
                                                    description='Ratio of specific heats. Default is 1.4 for air')
      
    Re        : Optional[float]             = Field(None, gt = 0.0,       # Reynolds number (if applicable; Euler equations don't require it)
                                                    description='Reynolds number resulting from non-dimensionalization. Required for Navier-Stokes, ignored for Euler and scalar advection.')
    
    Pr        : float                       = Field(0.71, gt = 0.0,       # Prandtl number (if applicable)
                                                    description='Prandtl number resulting from non-dimensionalization. Default is 0.71. Required for Navier-Stokes, ignored for Euler and scalar advection.')
    
    mach_ref  : float                       = Field(1.0,  gt = 0.0,       # Reference Mach number (if applicable)
                                                    description='Reference Mach number resulting from non-dimensionalization. Default is 1.0. Required for Navier-Stokes, ignored for Euler and scalar advection.')
    
    num_eq    : Optional[int]               = Field(None,  gt = 0.0,       # Number of governing equations (determined automatically based on selected physics model and dimensionality)
                                                    description='Number of governing equations. This is determined automatically based on the selected physics model and dimensionality.')
    
    # Create a validator to print errors and warnings related to the selected physics parameters.
    @model_validator(mode='after')
    def check_physics_params(self) -> 'PhysicsCfg':

        # Errors ----------------------------------------------
        if self.model == 'navier-stokes' and self.Re is None:
            raise ValueError("Re (Reynolds number) must be specified when 'navier-stokes' is selected as the governing equation.")
        
        # Warnings --------------------------------------------
        if self.model == 'navier-stokes':
            if 'Pr' not in self.model_fields_set:
                warnings.warn("Pr (Prandtl number) not specified. Defaulting to Pr = 0.71.")

            if 'mach_ref' not in self.model_fields_set:
                warnings.warn("mach_ref (reference Mach number) not specified. Defaulting to mach_ref = 1.0.")

        if 'gamma' not in self.model_fields_set:
            warnings.warn("gamma (ratio of specific heats) not specified. Defaulting to gamma = 1.4.")
        return self

class MeshCfg(BaseModel):
    # Define the parameters for mesh configuration
    mesh_format : Literal['HOHQMesh','gmsh']    = Field(...,
                                                        description='Mesh format used for the simulation. Must be explicitly set.')    
    
    mesh_file   : str                           = Field(...,                     # Mesh file name
                                                        description='Mesh file name. Include path if not in the same directory as the case YAML file. Must be explicitly set.')

    ndim        : int                           = Field(...,                     # Spatial dimensions (1, 2, or 3)
                                                        description='Number of spatial dimensions (1, 2, or 3). Must be explicitly set.')
    
    poly_deg    : Optional[int]                 = Field(..., gt=-1,              # Polynomial degree for solution approximation within each element
                                                        description='Polynomial degree for solution approximation within each element. Must be explicitly set.')
    
    quad_type   : Optional[Literal['LG','LGL']] = Field('LG',                    # Quadrature type: Legendre-Gauss (LG) or Legendre-Gauss-Lobatto (LGL)
                                                        description='Quadrature type for numerical integration. Default is LG (Legendre-Gauss).')
    
    # Create a validator to print errors and warnings related to the selected physics parameters.
    @model_validator(mode='after')
    def check_mesh_params(self) -> 'MeshCfg':

        # Errors ----------------------------------------------
        
        if self.ndim not in [1, 2, 3]:
            raise ValueError(f"Unsupported ndim: '{self.ndim}'. Supported dimensionalities are 1, 2, or 3.")

        # Warnings --------------------------------------------
        
        if 'quad_type' not in self.model_fields_set:
            warnings.warn("quad_type not specified. Defaulting to 'LG' (Legendre-Gauss) quadrature.")

        if 'poly_deg' not in self.model_fields_set:
            warnings.warn("poly_deg (polynomial degree) not specified. Defaulting to the polynomial degree specified in the mesh file.")

        return self

class TimeIntegratorCfg(BaseModel):
    # Define the parameters for time integration
    time_integrator : Literal['euler','rk2', 'rk4']   = Field(...,                 # Time integrator
                                                              description='Time integration scheme to use. Must be explicitly set.')
    
    cfl             : float                           = Field(..., gt=0.0, lt=1.0, # CFL number for stable time step calculations
                                                              description='CFL number for stable time step calculations. Must be explicitly set.')
    
    final_time      : float                           = Field(..., gt=0.0,        # Final simulation time
                                                              description='End time for the simulation. Must be explicitly set.')
    
    # Create a validator to print errors and warnings related to the selected physics parameters.
    @model_validator(mode='after')
    def check_time_params(self) -> 'TimeIntegratorCfg':
        # Errors ---------------------------------------------

        return self
    
class ICCfg(BaseModel):
    # Define the parameters for the initial condition.
    IC_method : Optional[Literal['analytical','datafile','restart']] = Field('analytical',
                                                                             description='Method for computing the initial condition. Default is "analytical".')
    
    IC_file   : Optional[str]                                        = Field('initial_condition.py',
                                                                             description='File name for the initialization method. Default is "initial_condition.py". This file should be located in the same directory as the case YAML file.')
    
    # Create a validator to print errors and warnings related to the selected initialization parameters.
    @model_validator(mode='after')
    def check_initialization_params(self) -> 'ICCfg':
        # Errors ---------------------------------------------
        
        # Warnings -------------------------------------------
        if 'IC_method' not in self.model_fields_set:
            warnings.warn("IC_method not specified. Defaulting to 'analytical' method for computing the initial condition.")

        if 'IC_file' not in self.model_fields_set:
            warnings.warn("IC_file not specified. Defaulting to 'initial_condition.py'.")

        return self
    
class IOCfg(BaseModel):
    # Define the parameters for the IO configuration
    output_dir        : Path           = Field(Path('./outputs/'),
                                               description="Path to the directory you would like the output files to be stored. Defaults to './outputs/'")
    
    plot_uniform_grid : bool           = Field(False,
                                               description="'True' outputs data to be interpolated to a uniform grid. 'False' outputs data on quadrature nodes. Defaults to 'False'.")
    
    uniform_grid_res  : int            = Field(150, gt=0,
                                               description="Number of nodes in the x, y, and z directions used to create the uniform grid (if applicable).")
    
    output_format     : Literal['hdf5',
                                'txt',
                                'csv',
                                'mat',
                                'vtk'] = Field('hdf5',
                                               description="Format of the output files. Valid options are: 'hdf5', 'txt', 'csv', 'mat', and 'vtk'. Defaults to 'hdf5'.")
    
    output_interval   : float          = Field(1.0, gt=0.0,
                                               description="Time step interval that data will be written out. e.g., output_interval = 0.5 means data is written out every 0.5 time units. Defaults to 1.0.")
    
    monitor_run       : bool           = Field(False,
                                               description="'True' plots flowfield during simulation for observation. 'False' does not. WARNING: This may significantly increase simulation time! Defaults to 'False'.")

    monitor_interval  : float          = Field(1.0, gt=0.0,
                                               description="Time step interval that data will be plotted for observation. e.g., monitor_interval = 0.5 means data is plotted every 0.5 time units. Defaults to 1.0.")
    
    # Create a validator to print errors and warnings related to the selected initialization parameters.
    @model_validator(mode='after')
    def check_io_params(self) -> 'IOCfg':
        # Errors -------------------------------------------------------

        # Warnings -----------------------------------------------------
        if 'output_format' not in self.model_fields_set:
            warnings.warn("No output data format specified, defaulting to 'hdf5'.")

        if 'plot_uniform_grid' not in self.model_fields_set:
            warnings.warn("No preference specified to uniform vs quadrature grid in data output. Defaulting to quadrature grid.")

        if self.plot_uniform_grid == True and 'uniform_grid_res' not in self.model_fields_set:
            warnings.warn("No uniform grid resolution is specified for data output. Defaulting to 100 nodes in each dimension.")

        if 'output_dir' not in self.model_fields_set:
            warnings.warn("No output directory specified. Defaulting to './Output/'.")

        return self

class CaseCfg(BaseModel):
    """
    Top-level configuration object for a simulation case. 
    This class synthesizes the 'Physics', 'Mesh', and 'TimeIntegrator' classes 
    into a single configuration object that can be easily loaded from a 
    user-defined input file (e.g., a YAML file).
    """
    # Synthesize the case configuration from the preceeding components
    name          : str               = Field(default_factory=default_case_name,        # Name of the case (primarily used for output operations)
                                              description='Name of the simulation case. This is primarily used for output operations. If not specified, it will default to the date/time the simulation was started.')
    
    mesh          : MeshCfg           = Field(...,                                                  # Mesh configuration block. 
                                              description='Mesh configuration. Requires inputs that must be explicitly set within the mesh block of the case YAML file.')
    
    physics       : PhysicsCfg        = Field(...,                                                  # Physics configuration block.
                                              description='Physics configuration. Requires inputs that must be explicitly set within the physics block of the case YAML file.')
    
    time_stepping : TimeIntegratorCfg = Field(...,                                      # Time integration configuration block.
                                              description='Time integration configuration. Requires inputs that must be explicitly set within the time block of the case YAML file.')
    
    initialization : ICCfg            = Field(default_factory=ICCfg,
                                              description='Initialization configuration. Requires inputs that must be explicitly set within the initialization block of the case YAML file.')
    
    io             : IOCfg            = Field(default_factory=IOCfg,
                                              description='IO configuration. Requres inputs that must be explicitly set within the io block of the case YAML file.')
    
    misc: Dict[str, Any]     = Field(default_factory=dict,                                 # Miscellaneous user-defined parameters.
        description="Optional miscellaneous user-defined parameters.")
    
    # Validate the CaseCfg model and alter any entries before finalization
    @model_validator(mode='after')
    def derive_computed_fields(self) -> 'CaseCfg':
        # Derive the number of governing equations based on dimensionality and physics model
        ndim = self.mesh.ndim
        match self.physics.model:
            case 'scalar_advection': num_eq = 1
            case 'euler' | 'navier-stokes': num_eq = 2 + ndim
        self.physics.num_eq = num_eq

        # Set output_dir to match the case name if the user did not explicitly set a desired output directory
        if 'output_dir' not in self.io.model_fields_set:
            if 'name' not in self.model_fields_set:
                self.io.output_dir = f"./Output_{self.name}/"
            else:
                self.io.output_dir = f"./Output_{default_case_name()}/"

        return self

__all__ = ['PhysicsCfg', 'MeshCfg', 'TimeIntegratorCfg', 'ICCfg', 'IOCfg', 'CaseCfg']

def load_case_yaml(path) -> CaseCfg:
    """Load a YAML case file and return a validated `CaseCfg`.

    This function adapts common camelCase keys to the snake_case names
    used in the pydantic model so users may keep legacy/camelCase
    YAML files if they prefer.
    """
    p   = Path(path)                        # Path object for the YAML file you want to load
    raw = yaml.safe_load(p.read_text())     # Load the raw YAML data into a dictionary

    validated = CaseCfg.model_validate(raw) # Validate the raw data against the CaseCfg model

    return validated