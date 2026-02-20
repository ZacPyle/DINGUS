# src/dingus/coreNumerics/quadrature.py
import numpy as np
from dingus.coreNumerics.polynomials import Legendre_Polynomial as legpoly
from dingus.mesh import mesh_class
from warnings import warn

'''
This module contains basic functions used in various quadrature procedures, including:
- Creating Legendre and Chebyshev polynomials
- Computing the derivatives of Legendre and Chebyshev polynomials
- Computing the roots of Legendre and Chebyshev polynomials
- Computing the weights of Legendre and Chebyshev polynomials
- Computing the quadrature points and weights for Legendre and Chebyshev polynomials

'''

def Compute_Quadrature_Nodes_And_Weights(myMesh: 'mesh_class.Mesh') -> None:
    """
    Computes the quadrature nodes and weights given a polynomial order and quadrature type
    (e.g. Legendre-Gauss, Legendre-Gauss-Lobatto). The polynomial order and quadrature type
    are specificed in the control file and saved to the mesh class object, which is an input 
    to this function. As the quadrature nodes and weights do not change per-element (on the 
    computational square), these nodes and weights are computed a single time and saved to the
    mesh class. This saves significant time in computing Legendre-Gauss-Lobatto quadrature 
    where iterative root-solving is required.

    Inputs:
    - myMesh: mesh class object containing the polynomial order and quadrature type

    Outputs: 
    - None (the quadrature nodes and weights are saved to the mesh class object)

    NOTE: This is a wrapper function that calls the appropriate quadrature node and weight
    calculator based on the specified quadrature type.
        - Legendre-Gauss          (LG)
        - Legendre-Gauss-Lobatto  (LGL)
        - Chebyshev-Gauss         (CG)
        - Chebyshev-Gauss-Lobatto (CGL)   
    """

    quadType = myMesh.quad_type

    # Check that the polynomial order is properly specified
    if not isinstance(myMesh.el_poly_order, int) or myMesh.el_poly_order <= 0:
        raise ValueError(
            "The element polynomial order must be a non-negative integer\n"
            f"\t input polynomial order is: {myMesh.el_poly_order!r}"
        )

    # Call appropriate quadrature node and weight calculator based on the specified quadrature type
    match quadType:
        case "LG":
            myMesh.quad_nodes, myMesh.quad_weights = Legendre_Gauss_Nodes_And_Weights(myMesh.el_poly_order)
        case "LGL":
            myMesh.quad_nodes, myMesh.quad_weights = Legendre_Gauss_Lobatto_Nodes_And_Weights(myMesh.el_poly_order)
        case "CG":
            myMesh.quad_nodes, myMesh.quad_weights = Chebyshev_Gauss_Nodes_And_Weights(myMesh.el_poly_order)
        case "CGL":
            myMesh.quad_nodes, myMesh.quad_weights = Chebyshev_Gauss_Lobatto_Nodes_And_Weights(myMesh.el_poly_order)
        case _:
            # Create a list of strings indicating the accepted quadrature types
            accepted_quad_types = ["LG  : (Legendre-Gauss)",
                                   "LGL : (Legendre-Gauss-Lobatto)",
                                   "CG  : (Chebyshev-Gauss)",
                                   "CGL : (Chebyshev-Gauss-Lobatto)"]
            
            # Add a tab and newline character to each entry in the accepted quad type list
            accepted_str = "\n\t".join(accepted_quad_types)

            # Print out error message indicating the unknown quadrature type and the accepted 
            # quadrature types
            
            raise ValueError(
                f"Unknown quadrature type specified in control file! Quadrature type: {quadType}\n"
                f"Accepted quadrature types are:\n\t{accepted_str}")

def Legendre_Gauss_Nodes_And_Weights(N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Legendre-Gauss quadrature nodes and weights given a polynomial order. This is
    done using the numpy package np.polynomial.legendre.leggauss function, which computes
    the roots of the Legendre polynomial using the Golub-Welsch algorithm. The nodes and
    weights are returned as numpy arrays.

    Inputs:
    - N: polynomial order of the quadrature (number of nodes is N + 1)

    Outputs:
    - quad_nodes: numpy array of quadrature nodes
    - quad_weights: numpy array of quadrature weights

    NOTE: LG quadrature integrates polynomials up to degree 2*N+1 exactly! This prevents
    aliasing in the nonlinear advective terms of the Navier-Stokes equations, which are of degree 
    2*N.
    """

    nodes, weights = np.polynomial.legendre.leggauss(N + 1)

    return nodes, weights    

def Legendre_Gauss_Lobatto_Nodes_And_Weights(N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Legendre-Gauss quadrature nodes and weights given a polynomial order. This is
    done by computing the roots of the derivative of the Legendre polynomial of degree 
    N, which can be done iteratively using the Newton-Raphson method.

    Inputs:
    - N: polynomial order of the quadrature (number of nodes is N + 1)

    Outputs:
    - quad_nodes: numpy array of quadrature nodes
    - quad_weights: numpy array of quadrature weights

    NOTE: LGL quadrature integrates polynomials up to degree 2*N-1 exactly! Thus, 
    it is expected that aliasing will occur in the nonlinear advective terms of the Navier-Stokes 
    equations, which are of degree 2*N.
    """

    # ------- Compute interior nodes with Newton-Raphson Method -------
    num_iters = 50

    # Initial guess for roots is the Chebyshev-Gauss-Lobatto nodes
    x, _ = Chebyshev_Gauss_Lobatto_Nodes_And_Weights(N)
    x    = x[1:-1]       # Remove endpoints from initial guess

    # Precompute the Legendre polynomial and derivative coefficients once outside the 
    # loop to save time. 
    # NOTE: In np.polynomial.legendre, the degree N Legendre polynomial is represented
    # as a series of Legendre basis functions:
    #        P_N(x) = coeffs[0]*P_0(x) + coeffs[1]*P_1(x) + ... + coeffs[N]*P_N(x)
    # NOT in the classic monomial basis:
    #        P_N(x) = coeffs[0]*x^N + coeffs[1]*x^(N-1) + ... + coeffs[N]*x^0
    # Thus, the coefficient array for an Nth order Legendre polynomial is ALWAYS an array of
    # length N + 1 with the Nth coefficient equal to 1 and the rest equal to 0.
    coeffs    = np.zeros(N +1)
    coeffs[N] = 1
    d_coeffs  = np.polynomial.legendre.legder(coeffs)
    d2_coeffs = np.polynomial.legendre.legder(d_coeffs)

    # Iteratively solve for the roots; 50 "should" be enough for convergence
    for _ in range(num_iters):
        # Compute the value of P'_N(x) and P''_N(x)
        dP_N  = np.polynomial.legendre.legval(x, d_coeffs)
        dP2_N = np.polynomial.legendre.legval(x, d2_coeffs)

        # Update the guess for the roots using the Newton-Raphson method
        dx = -dP_N / dP2_N
        x += dx

        # Check for convergence
        if np.max(np.abs(dx)) < 1e-15:
            break

    # if the loop did not converge, print a warning message
    else:
        warn(
            f"Potential problem with the Legendre-Gauss-Lobatto quadrature node computation.\n"
            f"The Newton-Raphson method did not converge after {num_iters} iterations.\n"
        )

    # Prepend / append the endpoints -1 and 1 to the node array
    nodes = np.concatenate(( [-1], x, [1]))
    
    # ------- Compute weights -------
    # Compute the Legendre polynomial values at the quadrature nodes
    P_N = legpoly(N, nodes)

    # Compute the quadrature weights (See "Implementing Spectral Methods for Partial Differential
    # Equations" by David Kopriva, Chapter 2 and 3 for reference)
    weights = 2 / (N * (N + 1) * P_N**2)

    return nodes, weights

def Chebyshev_Gauss_Nodes_And_Weights(N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Chebyshev-Gauss quadrature nodes and weights given a polynomial order. These
    can be computed from analytical equations; see "Implementing Spectral Methods for Partial 
    Differential Equations" by David Kopriva, Chapter 2 and 3 for reference.

    Inputs:
    - N: polynomial order of the quadrature (number of nodes is N + 1)

    Outputs:
    - quad_nodes: numpy array of quadrature nodes
    - quad_weights: numpy array of quadrature weights

    NOTE: Because Chebyshev-based quadrature uses a family of cosine functions as the basis, 
    this quadrature choice is only truly appropriate for problems with periodic boundary conditions.
    """
    
    i       = np.arange(N + 1)
    nodes   = -np.cos( (2*i + 1) / (2*N + 2) * np.pi )
    weights = np.full(N + 1, np.pi / (N + 1))

    return nodes, weights

def Chebyshev_Gauss_Lobatto_Nodes_And_Weights(N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Chebyshev-Gauss-Lobatto quadrature nodes and weights given a polynomial order. These
    can be computed from analytical equations; see "Implementing Spectral Methods for Partial 
    Differential Equations" by David Kopriva, Chapter 2 and 3 for reference.

    Inputs:
    - N: polynomial order of the quadrature (number of nodes is N + 1)

    Outputs:
    - quad_nodes: numpy array of quadrature nodes
    - quad_weights: numpy array of quadrature weights

    NOTE: Because Chebyshev-based quadrature uses a family of cosine functions as the basis, 
    this quadrature choice is only truly appropriate for problems with periodic boundary conditions.
    """
    if N == 1:
        return np.array([-1, 1]), np.array([1, 1])
    else:
        i       = np.arange(N + 1)
        nodes   = -np.cos(i / N * np.pi)
        weights = np.full(N + 1, np.pi / N)

        # Adjust the weights at the endpoints as required by Lobatto quadrature
        weights[0]  = weights[0]/2
        weights[-1] = weights[-1]/2

        return nodes, weights