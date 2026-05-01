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
    used as the default factory for the 'name' field in the 'CaseCnfg' class. Alternatively, 
    users can specify choose to use a 'lambda' function within the CaseCnfg definition. I 
    find the helper function approach to be more intuitive.
    """

    return datetime.now().strftime("%Y_%m_%d__%H:%M:%S")

# Configuration classes ###################################################################
class PhysicsCnfg(BaseModel):
    # Define the parameters for physics modeling
    equation  : Literal['scalar_advection',                               # Physics model / governing equations
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
    
    # Create a validator to print errors and warnings related to the selected physics parameters.
    @model_validator(mode='after')
    def check_physics_params(self) -> 'PhysicsCnfg':

        # Errors ----------------------------------------------
        if self.equation == 'navier-stokes' and self.Re is None:
            raise ValueError("Re (Reynolds number) must be specified when 'navier-stokes' is selected as the governing equation.")
        
        # Warnings --------------------------------------------
        if self.equation == 'navier-stokes':
            if 'Pr' not in self.model_fields_set:
                warnings.warn("Pr (Prandtl number) not specified. Defaulting to Pr = 0.71.")

            if 'mach_ref' not in self.model_fields_set:
                warnings.warn("mach_ref (reference Mach number) not specified. Defaulting to mach_ref = 1.0.")

        if 'gamma' not in self.model_fields_set:
            warnings.warn("gamma (ratio of specific heats) not specified. Defaulting to gamma = 1.4.")
        return self

class MeshCnfg(BaseModel):
    # Define the parameters for mesh configuration
    mesh_format : Literal['HOHQMesh','gmsh']    = Field(...,
                                                        description='Mesh format used for the simulation. Must be explicitly set.')    
    
    mesh_file   : Optional[str]                 = Field(...,                     # Mesh file name
                                                        description='Mesh file name. Include path if not in the same directory as the case YAML file. Must be explicitly set.')

    ndim        : int                           = Field(...,                     # Spatial dimensions (1, 2, or 3)
                                                        description='Number of spatial dimensions (1, 2, or 3). Must be explicitly set.')
    
    poly_deg    : Optional[int]                 = Field(..., gt=-1,              # Polynomial degree for solution approximation within each element
                                                        description='Polynomial degree for solution approximation within each element. Must be explicitly set.')
    
    quad_type   : Optional[Literal['LG','LGL']] = Field('LG',                    # Quadrature type: Legendre-Gauss (LG) or Legendre-Gauss-Lobatto (LGL)
                                                        description='Quadrature type for numerical integration. Default is LG (Legendre-Gauss).')
    
    # Create a validator to print errors and warnings related to the selected physics parameters.
    @model_validator(mode='after')
    def check_mesh_params(self) -> 'MeshCnfg':

        # Errors ----------------------------------------------
        if self.mesh_format not in self.model_fields_set:
            raise ValueError("mesh_format must be specified. Supported formats are 'HOHQMesh' and 'gmsh'.")
        
        if self.mesh_format not in ['HOHQMesh', 'gmsh']:
            raise ValueError(f"Unsupported mesh_format: '{self.mesh_format}'. Supported formats are 'HOHQMesh' and 'gmsh'.")
        
        if self.mesh_file not in self.model_fields_set:
            raise ValueError("mesh_file must be specified. Please provide the name of the mesh file to load, including the path if it is not located in the same directory as the case YAML file.")
        
        if self.ndim not in self.model_fields_set:
            raise ValueError("ndim (number of spatial dimensions) must be specified. Choose 1, 2, or 3.")
        
        if self.ndim not in [1, 2, 3]:
            raise ValueError(f"Unsupported ndim: '{self.ndim}'. Supported dimensionalities are 1, 2, or 3.")

        if self.quad_type not in ['LG', 'LGL']:
            raise ValueError(f"Unsupported quad_type: '{self.quad_type}'. Supported quadrature types are 'LG' (Legendre-Gauss) and 'LGL' (Legendre-Gauss-Lobatto).")
        
        # Warnings --------------------------------------------
        if self.quad_type not in self.model_fields_set:
            warnings.warn("quad_type not specified. Defaulting to 'LG' (Legendre-Gauss) quadrature.")

        if self.poly_deg not in self.model_fields_set:
            warnings.warn("poly_deg (polynomial degree) not specified. Defaulting to the polynomial degree specified in the mesh file.")

        return self

class TimeIntegratorCnfg(BaseModel):
    # Define the parameters for time integration
    time_integrator : Literal['euler','rk2', 'rk4']   = Field(...,                 # Time integrator
                                                              description='Time integration scheme to use. Must be explicitly set.')
    
    cfl             : float                           = Field(..., gt=0.0, lt=1.0, # CFL number for stable time step calculations
                                                              description='CFL number for stable time step calculations. Must be explicitly set.')
    
    final_time      : float                           = Field(..., gt=0.0,        # Final simulation time
                                                              description='End time for the simulation. Must be explicitly set.')
    
    # Create a validator to print errors and warnings related to the selected physics parameters.
    @model_validator(mode='after')
    def check_time_params(self) -> 'TimeIntegratorCnfg':
        # Errors ---------------------------------------------
        if self.time_integrator not in self.model_fields_set:
            raise ValueError("time_integrator must be specified. Supported options are 'euler', 'rk2', and 'rk4'.")
        
        if self.cfl not in self.model_fields_set:
            raise ValueError("cfl must be specified. Choose a value between 0 and 1.")
        
        if self.final_time not in self.model_fields_set:
            raise ValueError("final_time must be specified. Choose a positive value for the end time of the simulation.")

        return self
    
class CaseCnfg(BaseModel):
    """
    Top-level configuration object for a simulation case. 
    This class synthesizes the 'Physics', 'Mesh', and 'TimeIntegrator' classes 
    into a single configuration object that can be easily loaded from a 
    user-defined input file (e.g., a YAML file).
    """
    # Synthesize the case configuration from the preceeding components
    name    : str            = Field(default_factory=default_case_name,        # Name of the case (primarily used for output operations)
                                     description='Name of the simulation case. This is primarily used for output operations. If not specified, it will default to the date/time the simulation was started.')
    
    mesh    : MeshCnfg       = Field(...,                                                  # Mesh configuration block. 
                                     description='Mesh configuration. Requires inputs that must be explicitly set within the mesh block of the case YAML file.')
    
    physics : PhysicsCnfg    = Field(...,                                                  # Physics configuration block.
                                     description='Physics configuration. Requires inputs that must be explicitly set within the physics block of the case YAML file.')
    
    time_stepping : TimeIntegratorCnfg = Field(...,                                      # Time integration configuration block.
                                                 description='Time integration configuration. Requires inputs that must be explicitly set within the time block of the case YAML file.')
    
    misc: Dict[str, Any]     = Field(default_factory=dict,                                 # Miscellaneous user-defined parameters.
        description="Optional miscellaneous user-defined parameters.")

__all__ = ['PhysicsCnfg', 'MeshCnfg', 'TimeIntegratorCnfg', 'CaseCnfg']

def load_case_yaml(path: str) -> CaseCnfg:
    """Load a YAML case file and return a validated `CaseCnfg`.

    This function adapts common camelCase keys to the snake_case names
    used in the pydantic model so users may keep legacy/camelCase
    YAML files if they prefer.
    """
    p   = Path(path)                       # Path object for the YAML file you want to load
    raw = yaml.safe_load(p.read_text())    # Load the raw YAML data into a dictionary

    # Adapt mesh block / top-level keys with common alternatives
    mesh_block = raw.get("mesh", {}) if isinstance(raw.get("mesh", {}), dict) else {}

    adapted = {
        "name": raw.get("name", raw.get("caseName", "Case")),
        "mesh": {
            "mesh_format"     : mesh_block.get("mesh_format", mesh_block.get("meshFormat", raw.get("mesh_format", raw.get("meshFormat", "gmsh")))),
            "mesh_file"       : mesh_block.get("mesh_file"  , mesh_block.get("meshFile"  , raw.get("mesh_file"  , raw.get("meshFile"  , None  )))),
            "ndim"            : mesh_block.get("ndim"       , raw.get("ndim", 1)),
            "poly_deg"        : mesh_block.get("poly_deg"   , mesh_block.get("polyDeg"   , raw.get("poly_deg"   , raw.get("polyDeg"   , 4     )))),
            "quad_type"       : mesh_block.get("quad_type"  , mesh_block.get("quadType"  , raw.get("quad_type"  , raw.get("quadType"  , "LG"  )))),
        },
        "physics": {
            "equation"        : raw.get("physics", {}).get("equation", raw.get("physics", {}).get("equations"   , raw.get("equation"  , raw.get("equations", "euler")))),
            "gamma"           : raw.get("physics", {}).get("gamma"   , raw.get("gamma"  , 1.4)),
            "Re"              : raw.get("physics", {}).get("Re"      , raw.get("Re"     , None)),
            "Pr"              : raw.get("physics", {}).get("Pr"      , raw.get("Pr"     , None)),
            "mach_ref"        : raw.get("physics", {}).get("mach_ref", raw.get("physics", {}).get("machRef"     , raw.get("mach_ref"  , raw.get("machRef", None     )))),
        },
        "time": {
            "cfl"             : raw.get("time", {}).get("cfl"            , raw.get("cfl", 0.4)),
            "final_time"      : raw.get("time", {}).get("final_time"     , raw.get("time", {}).get("finalTime"     , raw.get("final_time"     , raw.get("finalTime"     , 1.0    )))),
            "time_integrator" : raw.get("time", {}).get("time_integrator", raw.get("time", {}).get("timeIntegrator", raw.get("time_integrator", raw.get("timeIntegrator", "euler")))),
        },
        "misc": raw.get("misc", raw.get("meta", {})),
    }

    return CaseCnfg.parse_obj(adapted)