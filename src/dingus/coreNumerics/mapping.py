# src/dingus/coreNumerics/mapping.py
import numpy as np
from dingus.mesh import element_class
from dingus.mesh import mesh_class
from dingus.mesh import mortar_class
from typing import Tuple

'''
This module contains functions for computing the isoparametric mapping of quadrature nodes 
from the reference element [-1, 1]^D (where D is the dimensionality of the mesh) to the 
physical domain in each element. All mapping functions are dervived from Chapter 6 in David 
Kopriva's "Implementing Spectral Methods for Partial Differential Equations".
'''

def isop_map_1d(myMesh: 'mesh_class.Mesh') -> None:
    '''
    Map quadrature nodes from [-1, 1] to an arbitrary 1D range.

    The mapping is:
        x(xi) = (1/2) * [ x1(1-xi) + x2(1+xi) ]

    Inputs:
    - myMesh: Mesh object containing the element structure and computational domain quadrature nodes.

    Outputs:
    - myMesh: Mesh object with physical domain quadrature node field populated.
    '''

    # Create 1D arrays of computational domain coordinates (aka the quadrature nodes)
    xi = myMesh.quad_nodes

def isop_map_2d(myMesh: 'mesh_class.Mesh') -> None:
    '''
    Map quadrature nodes from [-1, 1]^2 to an arbitrary 2D range.

    The mapping is:
        x(xi, eta) = (1/4) * [ x1(1-xi)(1-eta) + x2(1+xi)(1-eta)
                               + x3(1+xi)(1+eta) + x4(1-xi)(1+eta) ]

    Inputs:
    - myMesh: Mesh object containing the element structure and computational domain quadrature nodes.
    '''

    # Create 1D arrays of computational domain coordinates (aka the quadrature nodes)
    xi  = myMesh.quad_nodes
    eta = myMesh.quad_nodes

    # Reshape to adhere to numpy broadcasting rules
    xi  = xi[:,None]
    eta = eta[None,:]
    
    # Pre-compute the (1 +/- xi) and (1 +/- eta) terms to avoid redundant computation per element
    N1 = (1 - xi) * (1 - eta) / 4   # Shape function for node 1 (bottom left)
    N2 = (1 + xi) * (1 - eta) / 4   # Shape function for node 2 (bottom right)
    N3 = (1 + xi) * (1 + eta) / 4   # Shape function for node 3 (top right)
    N4 = (1 - xi) * (1 + eta) / 4   # Shape function for node 4 (top left)

    # Stack into (P, P, 4) — one shape function value per node per corner
    N = np.stack([N1, N2, N3, N4], axis=-1)    

    # Loop over elements and compute mapping
    for e in myMesh.elements:
        # Grab physical domain coordinates of the element corners
        corners = e.node_coords

        # Matrix multiply shape functions with corner coordinates to get physical domain
        # coordinates of quadrature nodes
        e.quad_node_coords = N @ corners

def isop_map_3d(myMesh: 'mesh_class.Mesh') -> None:
    '''
    Map quadrature nodes from [-1, 1]^3 to an arbitrary 3D range.

    Inputs:
    - myMesh: Mesh object containing the element structure and computational domain quadrature nodes.

    Outputs:
    - myMesh: Mesh object with physical domain quadrature node field populated.
    '''

    # Create 1D arrays of computational domain coordinates (aka the quadrature nodes)
    xi  = myMesh.quad_nodes
    eta = myMesh.quad_nodes
    zeta = myMesh.quad_nodes