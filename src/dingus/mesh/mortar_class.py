# src/dingus/mesh/mortar_class.py

from typing import List, Optional, Union, TYPE_CHECKING
from unittest import case
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
        self.connected_elements_face_sign_map                       = np.array([])

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
        self.normal_vector    = np.array([])

        # Solution data
        self.solution = None
        self.residual = None

    def compute_mortar_face_sign_map(self, dim : int, el_face_left : int, el_face_right : int) -> None:
        '''
        Computes the face sign map for a given mortar. This is an array containing the sign of the outward-facing 
        normal vector at each face of the element the mortar is connected to: [left_el_sign, right_el_sign]. The
        cannonical norm direction is defined as outward-facing. The reference coordinate system is the xi, eta, zeta
        system defined in each element. Thus, the sign of the normal vector is determined by how the cannonical norm
        aligns with the per-element coordinate system. 

        Example sign conventions:
        1D: 
            face 1 (left , -xi): -1
            face 2 (right, +xi):  1

        2D: 
            face 1 (bottom, -eta): -1
            face 2 (right , +xi ):  1
            face 3 (top   , +eta):  1
            face 4 (left  , -xi ): -1

        3D: 
            # TODO: confirm these sign conventions for HOHQMesh 3D! This DOES follow the convention in Guus' thesis.
            face 1 (-eta ): -1
            face 2 (+eta ):  1
            face 3 (-zeta): -1
            face 4 (+xi  ):  1
            face 5 (+zeta):  1
            face 6 (-xi  ): -1
        '''

        # Create a dispatch dictionary to determine the face sign map based on dimensionality
        FACE_SIGNS = {
            1: {1: -1, 2: 1},
            2: {1: -1, 2: 1, 3:  1, 4: -1},
            3: {1: -1, 2: 1, 3: -1, 4:  1, 5: 1, 6: -1}
        }

        # Choose the appropriate face sign map based on dimensinoality
        signs = FACE_SIGNS[dim]

        # Assign the face according to the left and right element faces (a.k.a. the face that
        # the mortar defines for the left element and for the right element)
        left_sign  = signs[abs(el_face_left )] if el_face_left != 0 else 0
        right_sign = signs[abs(el_face_right)] if el_face_right != 0 else 0

        # Assign the face sign map for this mortar
        self.connected_elements_face_sign_map = np.array([left_sign, right_sign])