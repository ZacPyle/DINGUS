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
    
    # Pre-compute the (1 +/- xi) terms to avoid redundant computation per element
    N1 = (1 - xi) / 2   # Shape function for node 1 (left)
    N2 = (1 + xi) / 2   # Shape function for node 2 (right)

    # Loop over elements and compute mapping
    for e in myMesh.elements:
        # Grab physical domain coordinates of the element corners
        corners = e.node_coords

        # compute 1D mapping; no need for matrix multiplication in this simple case
        e.quad_node_coords = N1 * corners[0,0] + N2 * corners[1,0]

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
    xi   = myMesh.quad_nodes
    eta  = myMesh.quad_nodes
    zeta = myMesh.quad_nodes

def compute_mapping_derivatives_1d(myMesh: 'mesh_class.Mesh') -> None:
    '''
    Computes the derivative of the physical space coordinate with respect to the
    computational domain coordinate (dx/dxi) for a 1D mesh. The mapping is:

        x(xi) = (1/2) * [x1*(1-xi) + x2*(1+xi)]

    So the derivative is simply:

        dx/dxi = (1/2) * (-x1 + x2) = (x2 - x1) / 2

    which is just half the physical element width — constant within each element.

    Inputs:
    - myMesh: Mesh object with quadrature nodes and elements populated.
    '''

    # Shape function derivatives — constants in 1D, no broadcasting needed
    dN1dXi = -0.5
    dN2dXi =  0.5

    for e in myMesh.elements:
        x = e.node_coords[:, 0]   # (2,) — just the two endpoint x-coordinates

        # dx/dxi is a scalar constant per element, but we tile it to (P+1,) so
        # downstream code can index it at each quadrature node consistently
        dXdXi = (x[0]*dN1dXi + x[1]*dN2dXi)   # scalar = (x2 - x1) / 2

        # Tile to (P+1,) so it matches the shape of quad_nodes
        e.dXdXi = np.full(myMesh.quad_nodes.shape, dXdXi)

def compute_mapping_derivatives_2d(myMesh : 'mesh_class.Mesh') -> None:
    '''
    Computes the derivatives of the physical space coordinates with respect to the 
    computaitonal domain coordinate system (e.g., dx/dxi, dx/deta, ...) necessary 
    for mapping the physical domain equations to the computational domain. The 
    reference/computational -> physical coordinate system mapping is computed as:

        x(xi,eta) = (1/4)sum_{k=1}^4 x_k N_k

    where x(xi, eta) = x*iHat + y*jHat, x_k is the kth corner node, and N_k is the 
    face function defining the boundary curve between x_k and x_{k+1}. Thus, the 
    derivatives computed in this function are:

        dx_i/dxi_j = (1/4)sum_{k=1}^4 x_k dN_k/dxi_j
    
    The need for these derivatives, their derivation (see what I did there?), and 
    an extended on them can be found in Chapter 6 of Kopriva. This computes the 
    TWO DIMENSIONAL derivatives:
        - dx/dxi
        - dx/deta
        - dy/dxi
        - dy/deta

    Inputs: 
    - myMesh: Mesh object with physical domain quadrature node field populated.
    '''

    # Create 1D arrays of computational domain coordinates (aka the quadrature nodes)
    xi   = myMesh.quad_nodes
    eta  = myMesh.quad_nodes

    # Prep the shape of the 1D arrays for broadcasting
    xi  = xi [:, None]
    eta = eta[None, :]

    # Formulate the face function derivatives. These are constant across all elements
    # as they depend ONLY on the quadrature nodes
    dN1dXi  = -(1.0 - eta)
    dN2dXi  =   1.0 - eta
    dN3dXi  =   1.0 + eta
    dN4dXi  = -(1.0 + eta)
    dN1dEta = -(1.0 - xi)
    dN2dEta = -(1.0 + xi)
    dN3dEta =   1.0 + xi
    dN4dEta =   1.0 - xi

    # Loop through each element, computing and storing the derivatives
    for e in myMesh.elements:
        # Extract physical-domain nodal coordinates
        x = e.node_coords[:,0]
        y = e.node_coords[:,1]

        # Compute spatial derivatives
        dXdXi  = (1/4) * (x[0]*dN1dXi  + x[1]*dN2dXi  + x[2]*dN3dXi  + x[3]*dN4dXi )
        dXdEta = (1/4) * (x[0]*dN1dEta + x[1]*dN2dEta + x[2]*dN3dEta + x[3]*dN4dEta)
        dYdXi  = (1/4) * (y[0]*dN1dXi  + y[1]*dN2dXi  + y[2]*dN3dXi  + y[3]*dN4dXi )
        dYdEta = (1/4) * (y[0]*dN1dEta + y[1]*dN2dEta + y[2]*dN3dEta + y[3]*dN4dEta)

        # Save spatial derivatives
        e.dXdXi  = dXdXi
        e.dXdEta = dXdEta
        e.dYdXi  = dYdXi
        e.dYdEta = dYdEta

def compute_mapping_derivatives_3d(myMesh : 'mesh_class.Mesh') -> None:
    '''
    Computes the derivatives of the physical space coordinates with respect to the 
    computaitonal domain coordinate system (e.g., dx/dxi, dx/deta, ...) necessary 
    for mapping the physical domain equations to the computational domain. The 
    reference/computational -> physical coordinate system mapping is computed as:

        x(xi,eta) = (1/8)sum_{k=1}^8 x_k N_k

    where x(xi, eta) = x*iHat + y*jHat, x_k is the kth corner node, and N_k is the 
    face function defining the boundary curve between x_k and x_{k+1}. Thus, the 
    derivatives computed in this function are:

        dx_i/dxi_j = (1/8)sum_{k=1}^8 x_k dN_k/dxi_j
    
    The need for these derivatives, their derivation (see what I did there?), and 
    an extended on them can be found in Chapter 6 of Kopriva. This computes the 
    THREE DIMENSIONAL derivatives:
        - dx/dxi,  dx/deta,  dx/dzeta
        - dy/dxi,  dy/deta,  dy/dzeta
        - dz/dxi,  dz/deta,  dz/dzeta

    Inputs: 
    - myMesh: Mesh object with physical domain quadrature node field populated.
    '''

    # Create 1D arrays of computational domain coordinates (aka the quadrature nodes)
    xi   = myMesh.quad_nodes
    eta  = myMesh.quad_nodes
    zeta = myMesh.quad_nodes

    # Prep the shape of the 1D arrays for broadcasting
    xi   = xi  [:, None, None]
    eta  = eta [None, :, None]
    zeta = zeta[None, None, :]

    # Formulate the face function derivatives. These are constant across all elements
    # as they depend ONLY on the quadrature nodes

    # dN/dxi — eta and zeta terms remain, xi drops out
    dN1dXi  = -(1 - eta) * (1 - zeta)
    dN2dXi  =  (1 - eta) * (1 - zeta)
    dN3dXi  =  (1 + eta) * (1 - zeta)
    dN4dXi  = -(1 + eta) * (1 - zeta)
    dN5dXi  = -(1 - eta) * (1 + zeta)
    dN6dXi  =  (1 - eta) * (1 + zeta)
    dN7dXi  =  (1 + eta) * (1 + zeta)
    dN8dXi  = -(1 + eta) * (1 + zeta)

    # dN/deta — xi and zeta terms remain, eta drops out
    dN1dEta = -(1 - xi) * (1 - zeta)
    dN2dEta = -(1 + xi) * (1 - zeta)
    dN3dEta =  (1 + xi) * (1 - zeta)
    dN4dEta =  (1 - xi) * (1 - zeta)
    dN5dEta = -(1 - xi) * (1 + zeta)
    dN6dEta = -(1 + xi) * (1 + zeta)
    dN7dEta =  (1 + xi) * (1 + zeta)
    dN8dEta =  (1 - xi) * (1 + zeta)

    # dN/dzeta — xi and eta terms remain, zeta drops out
    dN1dZeta = -(1 - xi) * (1 - eta)
    dN2dZeta = -(1 + xi) * (1 - eta)
    dN3dZeta = -(1 + xi) * (1 + eta)
    dN4dZeta = -(1 - xi) * (1 + eta)
    dN5dZeta =  (1 - xi) * (1 - eta)
    dN6dZeta =  (1 + xi) * (1 - eta)
    dN7dZeta =  (1 + xi) * (1 + eta)
    dN8dZeta =  (1 - xi) * (1 + eta)

    # Loop through each element, computing and storing the derivatives
    for e in myMesh.elements:
        x = e.node_coords[:, 0]   # (8,)
        y = e.node_coords[:, 1]   # (8,)
        z = e.node_coords[:, 2]   # (8,)

        # Compute all nine mapping derivatives — each is (P+1, P+1, P+1)
        dXdXi   = (1/8) * (x[0]*dN1dXi  + x[1]*dN2dXi  + x[2]*dN3dXi  + x[3]*dN4dXi
                          +  x[4]*dN5dXi  + x[5]*dN6dXi  + x[6]*dN7dXi  + x[7]*dN8dXi )
        
        dXdEta  = (1/8) * (x[0]*dN1dEta + x[1]*dN2dEta + x[2]*dN3dEta + x[3]*dN4dEta
                          +  x[4]*dN5dEta + x[5]*dN6dEta + x[6]*dN7dEta + x[7]*dN8dEta)
        
        dXdZeta = (1/8) * (x[0]*dN1dZeta+ x[1]*dN2dZeta+ x[2]*dN3dZeta+ x[3]*dN4dZeta
                          +  x[4]*dN5dZeta+ x[5]*dN6dZeta+ x[6]*dN7dZeta+ x[7]*dN8dZeta)

        dYdXi   = (1/8) * (y[0]*dN1dXi  + y[1]*dN2dXi  + y[2]*dN3dXi  + y[3]*dN4dXi
                          +  y[4]*dN5dXi  + y[5]*dN6dXi  + y[6]*dN7dXi  + y[7]*dN8dXi )
        
        dYdEta  = (1/8) * (y[0]*dN1dEta + y[1]*dN2dEta + y[2]*dN3dEta + y[3]*dN4dEta
                          +  y[4]*dN5dEta + y[5]*dN6dEta + y[6]*dN7dEta + y[7]*dN8dEta)
        
        dYdZeta = (1/8) * (y[0]*dN1dZeta+ y[1]*dN2dZeta+ y[2]*dN3dZeta+ y[3]*dN4dZeta
                          +  y[4]*dN5dZeta+ y[5]*dN6dZeta+ y[6]*dN7dZeta+ y[7]*dN8dZeta)

        dZdXi   = (1/8) * (z[0]*dN1dXi  + z[1]*dN2dXi  + z[2]*dN3dXi  + z[3]*dN4dXi
                          +  z[4]*dN5dXi  + z[5]*dN6dXi  + z[6]*dN7dXi  + z[7]*dN8dXi )
        
        dZdEta  = (1/8) * (z[0]*dN1dEta + z[1]*dN2dEta + z[2]*dN3dEta + z[3]*dN4dEta
                          +  z[4]*dN5dEta + z[5]*dN6dEta + z[6]*dN7dEta + z[7]*dN8dEta)
        
        dZdZeta = (1/8) * (z[0]*dN1dZeta+ z[1]*dN2dZeta+ z[2]*dN3dZeta+ z[3]*dN4dZeta
                          +  z[4]*dN5dZeta+ z[5]*dN6dZeta+ z[6]*dN7dZeta+ z[7]*dN8dZeta)

        # Save spatial derivatives
        e.dXdXi   = dXdXi
        e.dXdEta  = dXdEta
        e.dXdZeta = dXdZeta
        e.dYdXi   = dYdXi
        e.dYdEta  = dYdEta
        e.dYdZeta = dYdZeta
        e.dZdXi   = dZdXi
        e.dZdEta  = dZdEta
        e.dZdZeta = dZdZeta