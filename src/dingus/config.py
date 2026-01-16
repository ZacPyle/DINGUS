# src/dingus/config.py
from typing import List, Literal, Dict, Optional
from pydantic import BaseModel, Field
import yaml
from pathlib import Path

class PhysicsCfg(BaseModel):
    # Define the parameters for physics modeling
    equation  : Literal['euler', 'navier-stokes'] = 'navier-stokes'            # Physics model / governing equations
    gamma     : float                             = 1.4                        # Ratio of specific heats
    Re        : Optional[float]                   = None                       # Reynolds number (if applicable; Euler equations don't require it)
    Pr        : Optional[float]                   = None                       # Prandtl number (if applicable)
    mach_ref  : Optional[float]                   = None                       # Reference Mach number (if applicable)

class MeshCfg(BaseModel):
    # Define the parameters for mesh configuration
    mesh_format : Literal['gmsh']     = 'gmsh'                                 # Default mesh format
    mesh_file   : Optional[str]       = None                                   # Mesh file name
    ndim        : int                 = 1                                      # Spatial dimensions (1, 2, or 3)
    poly_deg    : int                 = 4                                      # Polynomial degree for solution approximation within each element
    quad_type   : Literal['LG','LGL'] = 'LG'                                   # Quadrature type: Legendre-Gauss (LG) or Legendre-Gauss-Lobatto (LGL)

class TimeCfg(BaseModel):
    # Define the parameters for time integration
    cfl             : float                     = 0.4                          # CFL number for stable time step calculations
    final_time      : float                     = 1.0                          # Final simulation time
    time_integrator : Literal['euler','rk4']   = 'euler'                       # Default time integration scheme

class CaseCfg(BaseModel):
    # Synthesize the case configuration from the preceeding components
    name    : Optional[str] = "Case"                                           # Default case name
    mesh    : MeshCfg       = MeshCfg()
    physics : PhysicsCfg    = PhysicsCfg()
    time    : TimeCfg       = TimeCfg()
    misc    : Dict          = Field(default_factory=dict)                      # Miscellaneous parameters

__all__ = ['PhysicsCfg', 'MeshCfg', 'TimeCfg', 'CaseCfg']

def load_case_yaml(path: str) -> CaseCfg:
    """Load a YAML case file and return a validated `CaseCfg`.

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

    return CaseCfg.parse_obj(adapted)