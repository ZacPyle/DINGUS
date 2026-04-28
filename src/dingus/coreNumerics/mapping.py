# src/dingus/coreNumerics/mapping.py
import numpy as np
from dingus.coreNumerics import interpolation
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

def isop_map_1d(myMesh: 'mesh_class.Mesh', e: 'element_class.SpectralElement') -> None:
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

    # Grab physical domain coordinates of the element corners
    corners = e.node_coords

    # compute 1D mapping; no need for matrix multiplication in this simple case
    e.quad_node_coords = N1 * corners[0,0] + N2 * corners[1,0]

def isop_map_2d(myMesh: 'mesh_class.Mesh', e: 'element_class.SpectralElement') -> None:
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

    # Grab physical domain coordinates of the element corners
    corners = e.node_coords

    # Matrix multiply shape functions with corner coordinates to get physical domain
    # coordinates of quadrature nodes
    e.quad_node_coords = N @ corners

def isop_map_3d(myMesh: 'mesh_class.Mesh', e: 'element_class.SpectralElement') -> None:
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

def compute_mapping_derivatives_1d(myMesh: 'mesh_class.Mesh', e: 'element_class.SpectralElement') -> None:
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

    x = e.node_coords[:, 0]   # (2,) — just the two endpoint x-coordinates

    # dx/dxi is a scalar constant per element, but we tile it to (P+1,) so
    # downstream code can index it at each quadrature node consistently
    dXdXi = (x[0]*dN1dXi + x[1]*dN2dXi)   # scalar = (x2 - x1) / 2

    # Tile to (P+1,) so it matches the shape of quad_nodes
    e.dXdXi = np.full(myMesh.quad_nodes.shape, dXdXi)

def compute_mapping_derivatives_2d(myMesh : 'mesh_class.Mesh', e: 'element_class.SpectralElement') -> None:
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

    # create 2D arrays of computational domain coordinates (P+1, P+1)
    xi, eta = np.meshgrid(myMesh.quad_nodes, myMesh.quad_nodes, indexing='ij') 

    # Formulate the face function derivatives. These are constant across all elements
    # as they depend ONLY on the quadrature nodes
    dN1dXi  = -(1.0 - eta)   # (P+1, P+1)
    dN2dXi  =   1.0 - eta  
    dN3dXi  =   1.0 + eta  
    dN4dXi  = -(1.0 + eta) 

    dN1dEta = -(1.0 - xi)     # (P+1, P+1)
    dN2dEta = -(1.0 + xi)  
    dN3dEta =   1.0 + xi   
    dN4dEta =   1.0 - xi   

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

def compute_mapping_derivatives_3d(myMesh : 'mesh_class.Mesh', e: 'element_class.SpectralElement') -> None:
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

    # create 3D arrays of computational domain coordinates (P+1, P+1, P+1)
    xi, eta, zeta = np.meshgrid(myMesh.quad_nodes, myMesh.quad_nodes, myMesh.quad_nodes, indexing='ij') 

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

def compute_jacobian_metrics_1d(e: 'element_class.SpectralElement') -> None:
    '''
    Computes the various Jacobian-related terms required for the physical <--> reference
    domain mapping, e.g., the Jacobian matrix, inverse Jacobian matrix, and Jacobian 
    matrix determinant. These are computed at each quadrature node within each element.
    For example, the Jacobian matrix of a 1D element is:

        J = [e.dXdXi]

    Inputs:
    - myMesh: Mesh object with mapping derivatives already computed and saved.
    '''

    # Extract the mapping derivatives
    dXdXi = e.dXdXi

    # Compute Jacobian metrics
    J    = dXdXi  # (P+1,)
    JDet = dXdXi  # (P+1,)   in 1D, det(J) = J = dx/dxi
    JInv = 1.0 / JDet

    # Save the Jacobian metrics to the element object
    e.jacobian     = J
    e.jacobian_det = JDet
    e.jacobian_inv = JInv

def compute_jacobian_metrics_2d(e: 'element_class.SpectralElement') -> None:
    '''
    Computes the various Jacobian-related terms required for the physical <--> reference
    domain mapping, e.g., the Jacobian matrix, inverse Jacobian matrix, and Jacobian 
    matrix determinant. These are computed at each quadrature node within each element.
    For example, the Jacobian matrix of a 3D hex element is:

        J = [e.dXdXi, e.dXdEta]
            [e.dYdXi, e.dYdEta]

    Inputs:
    - myMesh: Mesh object with mapping derivatives already computed and saved.
    '''

    # Extract the mapping derivatives
    dXdXi  = e.dXdXi         # (P+1, P+1)
    dXdEta = e.dXdEta        # (P+1, P+1)
    dYdXi  = e.dYdXi         # (P+1, P+1)
    dYdEta = e.dYdEta        # (P+1, P+1)

    # Compute the Jacobian determinant at each quadrature node for the 2x2 
    # Jacobian matrix
    JDet = dXdXi * dYdEta - dXdEta * dYdXi

    # Double check Jacobian determinant values. Negative or zero-value determinant means
    # no inverse exists / the mapping is not unique (linear alegrba 101). This should
    # NEVER be the case
    if np.any(JDet <= 0.0):
        raise ValueError(
            f'Element {e.id_global} has a non-positive jacobian determinant '
            f'(min det = {JDet.min():.6e}). Check mesh orientation.'
        )
    
    # Compute the Jacobian matrix for each quadrature node. Shape is:
    #   (          P+1,           P+1,               2,                   2)
    #   (quad node (x), quad node (y), jacobian (rows), jacobian (columns) )
    J = np.empty((*JDet.shape,2,2))
    J[..., 0, 0] = dXdXi
    J[..., 0, 1] = dXdEta
    J[..., 1, 0] = dYdXi
    J[..., 1, 1] = dYdEta

    # Compute the inverse Jacobian matrix for each quadrature node. Shape is:
    #   (          P+1,           P+1,               2,                   2)
    #   (quad node (x), quad node (y), jacobian (rows), jacobian (columns) )
    JInv = np.empty((*JDet.shape,2,2))
    JInv[..., 0, 0] =  dYdEta / JDet
    JInv[..., 0, 1] = -dXdEta / JDet
    JInv[..., 1, 0] = -dYdXi  / JDet
    JInv[..., 1, 1] =  dXdXi  / JDet

    # Save the Jacobian metrics to the element object
    e.jacobian     = J
    e.jacobian_det = JDet
    e.jacobian_inv = JInv

def compute_jacobian_metrics_3d(e: 'element_class.SpectralElement') -> None:
    '''
    Computes the various Jacobian-related terms required for the physical <--> reference
    domain mapping, e.g., the Jacobian matrix, inverse Jacobian matrix, and Jacobian 
    matrix determinant. These are computed at each quadrature node within each element.
    For example, the Jacobian matrix of a 3D hex element is:

        J = [e.dXdXi, e.dXdEta, e.dXdZeta]
            [e.dYdXi, e.dYdEta, e.dYdZeta]
            [e.dZdXi, e.dZdEta, e.dZdZeta]

    Inputs:
    - myMesh: Mesh object with mapping derivatives already computed and saved.
    '''

    # Extract the mapping derivatives
    dXdXi   = e.dXdXi         # (P+1, P+1, P+1)
    dXdEta  = e.dXdEta        # (P+1, P+1, P+1)
    dXdZeta = e.dXdZeta       # (P+1, P+1, P+1)
    dYdXi   = e.dYdXi         # (P+1, P+1, P+1)
    dYdEta  = e.dYdEta        # (P+1, P+1, P+1)
    dYdZeta = e.dYdZeta       # (P+1, P+1, P+1)
    dZdXi   = e.dZdXi         # (P+1, P+1, P+1)
    dZdEta  = e.dZdEta        # (P+1, P+1, P+1)
    dZdZeta = e.dZdZeta       # (P+1, P+1, P+1)

    # Compute the Jacobian matrix for each quadrature node. Shape is:
    #   (          P+1,           P+1,          P+1,               3,                   3)
    #   (quad node (x), quad node (y), quad node(z), jacobian (rows), jacobian (columns) )
    J = np.empty((*dXdXi.shape,3,3))
    J[..., 0, 0] = dXdXi
    J[..., 0, 1] = dXdEta
    J[..., 0, 2] = dXdZeta
    J[..., 1, 0] = dYdXi
    J[..., 1, 1] = dYdEta
    J[..., 1, 2] = dYdZeta
    J[..., 2, 0] = dZdXi
    J[..., 2, 1] = dZdEta 
    J[..., 2, 2] = dZdZeta
    
    # There is no 'clean' algorithm for the determinant of a 3x3 matrix like there is 
    # for a 2x2, so compute using Python functions
    JDet = np.linalg.det(J)

    # There is no 'clean' algorithm for the inverse of a 3x3 matrix like there is 
    # for a 2x2, so compute using Python functions
    JInv = np.linalg.inv(J)

    # Save the Jacobian metrics to the element object
    e.jacobian     = J
    e.jacobian_det = JDet
    e.jacobian_inv = JInv

def compute_covariant_and_contravariant_vectors_1d(e: 'element_class.SpectralElement') -> None:
    '''
    Computes the covariant and contravariant vectors at each quadrature node within each element. 
    Covariant vectors point along lines of constant xi / eta / zeta (aka 'tangent' vectors), and
    contravariant vecotrs point normal to lines of constant xi / eta / zeta. For a more detailed 
    description of these vectors, how to compute them, and what exactly they are used for please
    see Chapter 6.2 in Kopriva. 

    Inputs:
    - myMesh : Mesh object with the Jacobian metrics already computed and saved to each element.
    '''

    e.covar_xi     = e.dXdXi
    e.contravar_xi = np.ones((*e.dXdXi,1))

def compute_covariant_and_contravariant_vectors_2d(e: 'element_class.SpectralElement') -> None:
    '''
    Computes the covariant and contravariant vectors at each quadrature node within each element. 
    Covariant vectors point along lines of constant xi / eta / zeta (aka 'tangent' vectors), and
    contravariant vecotrs point normal to lines of constant xi / eta / zeta. For a more detailed 
    description of these vectors, how to compute them, and what exactly they are used for please
    see Chapter 6.2 in Kopriva. 

    Inputs:
    - myMesh : Mesh object with the Jacobian metrics already computed and saved to each element.
    '''

    # Preallocate the covariant and contravariant vector arrays
    covar_xi      = np.empty((*e.dXdXi.shape,2))
    covar_eta     = np.empty((*e.dXdEta.shape,2))
    contravar_xi  = np.empty((*e.dXdXi.shape,2))
    contravar_eta = np.empty((*e.dXdEta.shape,2))

    # Compute covariant vectors - xi
    covar_xi [...,0] = e.dXdXi
    covar_xi [...,1] = e.dYdXi

    # Compute covariant vectors - eta
    covar_eta[...,0] = e.dXdEta
    covar_eta[...,1] = e.dYdEta

    # Compute the contravariant vectors - xi
    contravar_xi[...,0] =        e.dYdEta / e.jacobian_det
    contravar_xi[...,1] = -1.0 * e.dXdEta / e.jacobian_det

    # compute the contravariant vectors - eta
    contravar_eta[...,0] = -1.0 * e.dYdXi / e.jacobian_det
    contravar_eta[...,1] =        e.dXdXi / e.jacobian_det

    # Save vectors to element
    e.covar_xi      = covar_xi
    e.covar_eta     = covar_eta
    e.contravar_xi  = contravar_xi
    e.contravar_eta = contravar_eta

def compute_covariant_and_contravariant_vectors_3d(e: 'element_class.SpectralElement') -> None:
    '''
    Computes the covariant and contravariant vectors at each quadrature node within each element. 
    Covariant vectors point along lines of constant xi / eta / zeta (aka 'tangent' vectors), and
    contravariant vecotrs point normal to lines of constant xi / eta / zeta. For a more detailed 
    description of these vectors, how to compute them, and what exactly they are used for please
    see Chapter 6.2 in Kopriva. 

    Inputs:
    - myMesh : Mesh object with the Jacobian metrics already computed and saved to each element.
    '''

    # Preallocate the covariant and contravariant vector arrays
    covar_xi       = np.empty((*e.dXdXi.shape,3))
    covar_eta      = np.empty((*e.dXdXi.shape,3))
    covar_zeta     = np.empty((*e.dXdXi.shape,3))
    contravar_xi   = np.empty((*e.dXdXi.shape,3))
    contravar_eta  = np.empty((*e.dXdXi.shape,3))
    contravar_zeta = np.empty((*e.dXdXi.shape,3))

    # Compute covariant vectors - xi
    covar_xi  [...,0] = e.dXdXi
    covar_xi  [...,1] = e.dYdXi
    covar_xi  [...,2] = e.dZdXi

    # Compute covariant vectors - eta
    covar_eta [...,0] = e.dXdEta
    covar_eta [...,1] = e.dYdEta
    covar_eta [...,2] = e.dZdEta

    # Compute covariant vectors - zeta
    covar_zeta[...,0] = e.dXdZeta
    covar_zeta[...,1] = e.dYdZeta
    covar_zeta[...,2] = e.dZdZeta

    # Compute the contravariant vectors - xi
    contravar_xi   = np.cross(covar_eta , covar_zeta) / e.jacobian_det

    # Compute the contravariant vectors - eta
    contravar_eta  = np.cross(covar_zeta, covar_xi  ) / e.jacobian_det

    # Compute the contravariant vectors - zeta
    contravar_zeta = np.cross(covar_xi  , covar_eta ) / e.jacobian_det

    # Save vectors to element
    e.covar_xi       = covar_xi
    e.covar_eta      = covar_eta
    e.covar_zeta     = covar_zeta
    e.contravar_xi   = contravar_xi
    e.contravar_eta  = contravar_eta
    e.contravar_zeta = contravar_zeta

def compute_normals_1d(myMesh : 'mesh_class.Mesh') -> None:
    '''
    Computes unit normal vectors at all quadrature nodes within each element (for future FVSE
    use), and the outward-facing unit normals at each mortar face (for DG flux assembly).

    Per Kopriva Eq. 6.46, the unit normal to a coordinate surface is simply the normalized
    contravariant vector associated with that surface:

        n_i = contravar_i / |contravar_i|

    The contravariant vectors are already normal to their respective coordinate surfaces:
        contravar_xi  is normal to xi  = const faces (left/right 'faces')

    Interior normals stored on each element:
        e.normal_xi  : shape (P+1, 2) — unit normals to xi  = const surfaces

    Face normals stored on each mortar:
        mort.normal_vector : shape (1, 2) — outward-facing unit normals along the face
    
    Inputs:
    - myMesh : Mesh object with covariant/contravariant vectors already computed and saved
               to each element
    '''

    # Interior normals
    for e in myMesh.elements:
        xi_mag = np.linalg.norm(e.contravar_xi, axis=-1, keepdims=True)   # (P+1, 1)
        e.normal_xi = e.contravar_xi / xi_mag                              # (P+1, 1)

    if myMesh.quad_type == "LGL":

        for mort in myMesh.mortars:
            owner_el, owner_sign = _get_mortar_owner(mort)
            face_idx = owner_el.connected_mortars.index(mort) + 1

            match face_idx:
                case 1: face_normal = owner_el.normal_xi[ 0, :]   # ξ = -1
                case 2: face_normal = owner_el.normal_xi[-1, :]   # ξ = +1

            mort.normal_vector = np.array(owner_sign * face_normal)

    elif myMesh.quad_type == "LG":

        I_min = interpolation.Polynomial_Interpolation_Matrix(
            myMesh.quad_nodes, np.array([-1.0])
        )  # (1,P+1)

        I_max = interpolation.Polynomial_Interpolation_Matrix(
            myMesh.quad_nodes, np.array([1.0])
        )

        for mort in myMesh.mortars:
            owner_el, owner_sign = _get_mortar_owner(mort)
            face_idx = owner_el.connected_mortars.index(mort) + 1

            match face_idx:
                case 1:
                    face_normal = np.einsum(
                        'mi,ic->c', I_min, owner_el.normal_xi
                    )
                case 2:
                    face_normal = np.einsum(
                        'mi,ic->c', I_max, owner_el.normal_xi
                    )

            mort.normal_vector = np.array(owner_sign * face_normal)

    else:
        raise NotImplementedError

def compute_normals_2d(myMesh : 'mesh_class.Mesh') -> None:
    '''
    Computes unit normal vectors at all quadrature nodes within each element (for future FVSE
    use), and the outward-facing unit normals at each mortar face (for DG flux assembly).

    Per Kopriva Eq. 6.46, the unit normal to a coordinate surface is simply the normalized
    contravariant vector associated with that surface:

        n_i = contravar_i / |contravar_i|

    The contravariant vectors are already normal to their respective coordinate surfaces:
        contravar_xi  is normal to xi  = const faces (left/right faces: 4 and 2)
        contravar_eta is normal to eta = const faces (bottom/top faces: 1 and 3)

    Array index convention: [i, j] = (xi_i, eta_j), consistent with isop_map_2d broadcasting.

    Interior normals stored on each element:
        e.normal_xi  : shape (P+1, P+1, 2) — unit normals to xi  = const surfaces
        e.normal_eta : shape (P+1, P+1, 2) — unit normals to eta = const surfaces

    Face normals stored on each mortar:
        mort.normal_vector : shape (P+1, 2) — outward-facing unit normals along the face
    
    Inputs:
    - myMesh : Mesh object with covariant/contravariant vectors already computed and saved
               to each element
    '''

    # Compute the interior unit normals at every quadrature node in each element. These are
    # Just normalized contravariant vectors. NOTE: These are NOT necessarily outward facing.
    for e in myMesh.elements:
        # Compute the magnitude of the vectors
        xi_mag  = np.linalg.norm(e.contravar_xi , axis=-1, keepdims=True) # (P+1, P+1, 1)
        eta_mag = np.linalg.norm(e.contravar_eta, axis=-1, keepdims=True) # (P+1, P+1, 1)

        # Normalize
        e.normal_xi  = e.contravar_xi / xi_mag
        e.normal_eta = e.contravar_eta / eta_mag

    # Extract face normals from the element interior normals and store on each mortar.
    # Array convention: index [i, j] = (xi_i, eta_j), so:
    #   face 1 (bottom, eta=-1): e.normal_eta[:, 0,  :]
    #   face 2 (right,  xi =+1): e.normal_xi [-1, :, :]
    #   face 3 (top,    eta=+1): e.normal_eta[:, -1, :]
    #   face 4 (left,   xi =-1): e.normal_xi [ 0, :, :]
    #
    # Note the difference in how we extract face normals depending on the quadrature type:
    #           - LGL: boundary nodes are exactly the first/last quadrature nodes,
    #                  so we can slice directly.
    #
    #           - LG:  boundary nodes are not in the quadrature set, so we must
    #                  interpolate to ±1 using a polynomial interpolation matrix.
    if myMesh.quad_type == "LGL":
        for mort in myMesh.mortars:
            owner_el, owner_sign = _get_mortar_owner(mort)
            face_idx = owner_el.connected_mortars.index(mort) + 1  # 1-based

            match face_idx:
                case 1: face_normal = owner_el.normal_eta[:,  0, :]   # (P+1, 2)
                case 2: face_normal = owner_el.normal_xi [-1, :, :]   # (P+1, 2)
                case 3: face_normal = owner_el.normal_eta[:, -1, :]   # (P+1, 2)
                case 4: face_normal = owner_el.normal_xi [ 0, :, :]   # (P+1, 2)

            mort.normal_vector = np.array(owner_sign * face_normal)   # (P+1, 2)
    elif myMesh.quad_type == "LG":
        # Build interpolation matrices from the LG nodes to ±1. These are the same
        # for every element and every mortar, so compute them once outside the loop.
        # I_min  maps from interior LG nodes to the left  boundary (xi or eta = -1)
        # I_max maps from interior LG nodes to the right boundary (xi or eta = +1)
        I_min  = interpolation.Polynomial_Interpolation_Matrix(myMesh.quad_nodes, np.array([-1.0]))  # (1, P+1)
        I_max  = interpolation.Polynomial_Interpolation_Matrix(myMesh.quad_nodes, np.array([ 1.0]))  # (1, P+1)

        for mort in myMesh.mortars:
            owner_el, owner_sign = _get_mortar_owner(mort)
            face_idx = owner_el.connected_mortars.index(mort) + 1  # 1-based

            # Interpolate the normal to the face boundary using einsum.
            # normal_eta shape: (P+1, P+1, 2) with axes (xi, eta, component)
            # normal_xi  shape: (P+1, P+1, 2) with axes (xi, eta, component)
            #
            # For faces where eta is fixed (faces 1 and 3), contract I along the eta
            # axis (axis 1) of normal_eta, leaving the xi axis free:
            #   'mj, ijc -> ic'  contracts j (eta) with I row, keeps i (xi) and c (component)
            #
            # For faces where xi is fixed (faces 2 and 4), contract I along the xi
            # axis (axis 0) of normal_xi, leaving the eta axis free:
            #   'mi, ijc -> jc'  contracts i (xi) with I row, keeps j (eta) and c (component)
            match face_idx:
                case 1: face_normal = np.einsum('mj, ijc -> ic', I_min, owner_el.normal_eta)  # (P+1, 2)
                case 2: face_normal = np.einsum('mi, ijc -> jc', I_max, owner_el.normal_xi )  # (P+1, 2)
                case 3: face_normal = np.einsum('mj, ijc -> ic', I_max, owner_el.normal_eta)  # (P+1, 2)
                case 4: face_normal = np.einsum('mi, ijc -> jc', I_min, owner_el.normal_xi )  # (P+1, 2)

            mort.normal_vector = np.array(owner_sign * face_normal)                           # (P+1, 2)

    else:
        raise NotImplementedError(
            f"Normal computation not implemented for quadrature type '{myMesh.quad_type}'. "
            f"Supported types are 'LG' and 'LGL'."
        )

def compute_normals_3d(myMesh : 'mesh_class.Mesh') -> None:
    '''
    Computes unit normal vectors at all quadrature nodes within each element (for future FVSE
    use), and the outward-facing unit normals at each mortar face (for DG flux assembly).

    Per Kopriva Eq. 6.46, the unit normal to a coordinate surface is simply the normalized
    contravariant vector associated with that surface:

        n_i = contravar_i / |contravar_i|

    The contravariant vectors are already normal to their respective coordinate surfaces:
        contravar_xi   is normal to xi   = const faces (faces: 6 and 4)
        contravar_eta  is normal to eta  = const faces (faces: 1 and 2)
        contravar_zeta is normal to zeta = const faces (faces: 3 and 5)

    Array index convention: [i, j, k] = (xi_i, eta_j, zeta_k), consistent with isop_map_3d broadcasting.

    Interior normals stored on each element:
        e.normal_xi  : shape (P+1, P+1, P+1, 3) unit normals to xi   = const surfaces
        e.normal_eta : shape (P+1, P+1, P+1, 3) unit normals to eta  = const surfaces
        e.normal_zeta: shape (P+1, P+1, P+1, 3) unit normals to zeta = const surfaces

    Face normals stored on each mortar:
        mort.normal_vector : shape (P+1, P+1,2) — outward-facing unit normals along the face
    
    Inputs:
    - myMesh : Mesh object with covariant/contravariant vectors already computed and saved
               to each element
    '''

    # ---------- interior normals ----------
    for e in myMesh.elements:

        xi_mag   = np.linalg.norm(e.contravar_xi  , axis=-1, keepdims=True)
        eta_mag  = np.linalg.norm(e.contravar_eta , axis=-1, keepdims=True)
        zeta_mag = np.linalg.norm(e.contravar_zeta, axis=-1, keepdims=True)

        e.normal_xi   = e.contravar_xi   / xi_mag
        e.normal_eta  = e.contravar_eta  / eta_mag
        e.normal_zeta = e.contravar_zeta / zeta_mag


    # ---------- face extraction ----------
    if myMesh.quad_type == "LGL":

        for mort in myMesh.mortars:

            owner_el, owner_sign = _get_mortar_owner(mort)
            face_idx = owner_el.connected_mortars.index(mort) + 1

            match face_idx:

                # zeta = -1
                case 1:
                    face_normal = owner_el.normal_zeta[:, :,  0, :]

                # xi = +1
                case 2:
                    face_normal = owner_el.normal_xi[-1, :, :, :]

                # zeta = +1
                case 3:
                    face_normal = owner_el.normal_zeta[:, :, -1, :]

                # xi = -1
                case 4:
                    face_normal = owner_el.normal_xi[ 0, :, :, :]

                # eta = -1
                case 5:
                    face_normal = owner_el.normal_eta[:,  0, :, :]

                # eta = +1
                case 6:
                    face_normal = owner_el.normal_eta[:, -1, :, :]

            mort.normal_vector = np.array(owner_sign * face_normal)


    elif myMesh.quad_type == "LG":

        I_min = interpolation.Polynomial_Interpolation_Matrix(
            myMesh.quad_nodes, np.array([-1.0])
        )

        I_max = interpolation.Polynomial_Interpolation_Matrix(
            myMesh.quad_nodes, np.array([1.0])
        )

        for mort in myMesh.mortars:

            owner_el, owner_sign = _get_mortar_owner(mort)
            face_idx = owner_el.connected_mortars.index(mort) + 1

            match face_idx:

                # zeta = -1
                case 1:
                    face_normal = np.einsum(
                        'mk, ijkc -> ijc',
                        I_min,
                        owner_el.normal_zeta
                    )

                # xi = +1
                case 2:
                    face_normal = np.einsum(
                        'mi, ijkc -> jkc',
                        I_max,
                        owner_el.normal_xi
                    )

                # zeta = +1
                case 3:
                    face_normal = np.einsum(
                        'mk, ijkc -> ijc',
                        I_max,
                        owner_el.normal_zeta
                    )

                # xi = -1
                case 4:
                    face_normal = np.einsum(
                        'mi, ijkc -> jkc',
                        I_min,
                        owner_el.normal_xi
                    )

                # eta = -1
                case 5:
                    face_normal = np.einsum(
                        'mj, ijkc -> ikc',
                        I_min,
                        owner_el.normal_eta
                    )

                # eta = +1
                case 6:
                    face_normal = np.einsum(
                        'mj, ijkc -> ikc',
                        I_max,
                        owner_el.normal_eta
                    )

            mort.normal_vector = np.array(owner_sign * face_normal)

    else:
        raise NotImplementedError(
            f"Normal computation not implemented for quadrature type '{myMesh.quad_type}'."
        )

def _get_mortar_owner(mort: 'mortar_class.SpectralMortar') -> tuple['element_class.SpectralElement', int]:
    '''
    Returns the owner element and its face sign for a given mortar. The left element
    (index 0) is preferred as the canonical owner; the right element (index 1) is used
    as a fallback for boundary mortars where the left element is None.

    Returns:
    - owner_el   : the owner SpectralElement object
    - owner_sign : +1 or -1 from connected_elements_face_sign_map
    '''

    # Extract connected elements
    left_el  = mort.connected_elements[0]
    right_el = mort.connected_elements[1]

    # Check if left element exists (i.e., this isn't a boundary mortar)
    if left_el is not None:
        return left_el, mort.connected_elements_face_sign_map[0]
    elif right_el is not None:
        return right_el, mort.connected_elements_face_sign_map[1]
    else:
        raise RuntimeError(
            f"Mortar {mort.id_global} has no connected elements — "
            "mesh connectivity may not have been established before computing normals."
            )