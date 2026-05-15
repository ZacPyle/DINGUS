# src/dingus/mesh/element_class.py

from typing import List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from dingus.mesh.mortar_class import SpectralMortar

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
        self.mortar_ids                                           = np.array([])
        self.connected_mortars : List[Optional['SpectralMortar']] = [None, None, None, None]  # Hardcoded for 2D; generalize later
        #self.connected_mortars : List[Optional['SpectralMortar']] = []  # Hardcoded for 2D; generalize later
        self.face_sign_map                                        = np.array([]) 

        # Boundary condition information
        self.boundary_condition_names = np.array([])

        # Boundayr curvature information
        self.mortar_curvature = np.array([])

        # Spectral data (to be initialized later)
        self.quad_node_coords  = np.array([])
        self.quad_weights      = np.array([])
        self.jacobian          = np.array([])
        self.jacobian_inv      = np.array([])
        self.jacobian_det      = np.array([])
        self.covar_xi          = np.array([])
        self.covar_eta         = np.array([])
        self.covar_zeta        = np.array([])
        self.contravar_xi      = np.array([])
        self.contravar_eta     = np.array([])
        self.contravar_zeta    = np.array([])
        self.dXdXi             = np.array([])
        self.dXdEta            = np.array([])
        self.dXdZeta           = np.array([])
        self.dYdXi             = np.array([])
        self.dYdEta            = np.array([])
        self.dYdZeta           = np.array([])
        self.dZdXi             = np.array([])
        self.dZdEta            = np.array([])
        self.dZdZeta           = np.array([])
        self.normal_xi         = np.array([])
        self.normal_eta        = np.array([])
        self.normal_zeta       = np.array([])
        
        # Solution data
        self.solution = np.array([])
        self.residual = np.array([])