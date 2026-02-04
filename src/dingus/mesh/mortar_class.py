# src/dingus/mesh/mortar_class.py

from typing import List, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from dingus.mesh.element_class import SpectralElement

class SpectralMortar:
    """
    The Mortar class represents a single mortar within a mesh. A mortar is the interface
    between two neighboring elements, or between an element and a boundary. A Mortar object
    contains information on:
        - the nodes that make up the mortar
        - the mortar's curvature (straight, quadratic, spline, etc.)
        - the mortar's dimensionality
        - connectivity information to the neighboring elements
    """

    def __init__(self, mortar_id                   : int        = 0,
                       node_ids                    : np.ndarray = np.array([]),
                       node_coords                 : np.ndarray = np.array([]),
                       poly_order                  : int        = 0,
                       mortar_type                 : str        = ""
                       ) -> None:
        
        # Basic information
        self.id_global         = mortar_id
        self.node_ids          = node_ids
        self.node_coords       = node_coords
        self.poly_order        = poly_order
        self.mortar_type       = mortar_type       # 2D = 'edge', 3D = 'face'

        # Element / connectivity information
        self.connected_elements : List[Optional['SpectralElement']] = [None, None]
        self.connected_elements_face_map                            = None

        # Curvature information
        self.mortar_curvature  = None

        # Boundary condition information
        self.bc_name           = str

        # Spectral data (to be initialized later)
        self.quad_node_coords = None
        self.quad_weights     = None
        self.jacobian         = None
        self.jacobian_inv     = None
        self.metric_terms     = None

        # Solution data
        self.solution = None
        self.residual = None