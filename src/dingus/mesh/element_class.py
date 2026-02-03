# src/dingus/mesh/element_class.py

from pathlib import Path
from typing import Dict, Any, Union
from itertools import islice
import numpy as np

class SpectralElement:
    """
    The SpectralElement class represent a single element within a mesh. It contains information on:
        - the nodes that make up the element
        - the element's polynomial order
        - the element's dimensionality
        - connectivity information to neighboring elements
        - the mortars associated with each element face / side
    """

    def __init__(self, element_id  : int        = 0,
                       node_ids    : np.ndarray = np.array([]),
                       node_coords : np.ndarray = np.array([]),
                       poly_order  : int        = 0,
                       el_type     : str        = ""
                       ) -> None:
        
        # Basic information
        self.id_global   = element_id
        self.node_ids    = node_ids
        self.node_coords = node_coords
        self.poly_order  = poly_order
        self.el_type     = el_type

        # Mortar / connectivity information (to be initialized later)
        self.mortar_ids  = np.array([])

        # Spectral data (to be initialized later)
        self.quad_node_coords = None
        self.quad_weights     = None
        self.jacobian         = None
        self.jacobian_inv     = None
        self.metric_terms     = None
        
        # Solution data
        self.solution = None
        self.residual = None