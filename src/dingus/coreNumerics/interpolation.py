# src/dingus/coreNumerics/interpolation.py
import numpy as np
from typing import Optional

'''
This module contains basic functions used in various interpolation procedures, including:
- Creating a Lagrange polynomial interpolant
- Interpolating a generic set of 1D, 2D, or 3D data points to a different set of arbitrary 1D, 2D, or 3D points

All algorithms follow those presented in David Kopriva's "Implementing Spectral Methods
for Partial Differential Equations" (Springer, 2009), primarily Chapters 2 and 3.
'''

def Barycentric_Weights(x : np.ndarray) -> np.ndarray:
    '''
    Computes the barycentric weights for a set of interpolation nodes used to represent an Nth order polynomial, P_N(x). 
    The barycentric weights are defined as:

        w_j = 1 / Π_{i=0:N}{i≠j} (x_j - x_i)       j = 0, 1, ..., N

    where x_i are the interpolation nodes. This barycentric polynomial representation allows Lagrange polynomials to be
    evaluated and interpolated efficiently. For details, see Section 3.4 (Algorithm 30) Kopriva's book.

    Inputs:
    - x: (N+1,) array of interpolation nodes (e.g. LGL or LG quadrature points)
 
    Outputs:
    - w: (N+1,) array of barycentric weights
    '''
    
    # Initalize the array of weights.
    N = len(x)
    w = np.ones(N, dtype=float)

    for j in range(N):
        for i in range(N):
            if i != j:
                w[j] /= (x[j] - x[i])

    # Alternative implementation using vectorized operations (may be more efficient for large arrays):
    # diffs = x[:, None] - x[None, :]          # (N+1, N+1) matrix of (x_j - x_i)
    # np.fill_diagonal(diffs, 1.0)             # avoid divide-by-zero on diagonal
    # w = 1.0 / np.prod(diffs, axis=1)         # product across each row, skipping diagonal

    return w

def Lagrange_Interpolating_Polynomial(j: int, x_nodes: np.ndarray, w: np.ndarray, x_i: float) -> float:
    '''
    Evaluates the j-th Lagrange basis polynomial L_j at a single point xi using the
    barycentric form. See Algorithm 34 in Kopriva.

    Inputs:
    - j       : index of the desired basis polynomial (0-based)
    - x_nodes : (N+1,) array of interpolation nodes
    - w       : (N+1,) array of barycentric weights
    - x_i     : scalar evaluation point

    Outputs:
    - L_j(x_i): scalar value of the j-th Lagrange basis polynomial at xi
    '''

    # Initialization
    N        = len(x_nodes)
    L_j      = 0.0

    # Compute the Lagrange interpolation. Keep numerator and denominator separate to avoid numerical
    # instability.
    numerator   = 0.0
    denominator = 0.0
    exact_node  = False

    # Compute interpolation for input point x_interp[i]
    for k in range(N):
        # Create array of x_interp[i] - x_nodes[j] to check for potential division by zero
        x_diff = x_i - x_nodes[k]

        if np.abs(x_diff) < np.finfo(float).eps:  # Node we're interpolating to is exactly an input node
            L_j         = 1.0 if k == j else 0.0
            exact_node  = True
            break

        # Compute the numerator and denominator (only reaches this computation is exact node is NOT detected)
        denominator += w[k] / x_diff

        # non-zero numerator when k == j
        if k == j:
            numerator = w[k] / x_diff

    # Combine numerator and denominator for final interpolant value
    if not exact_node:
        L_j       = numerator / denominator
    
    return L_j

def Lagrange_Interpolation(x_nodes : np.ndarray, f_nodes : np.ndarray, w : np.ndarray, x_interp : np.ndarray) -> np.ndarray:
    '''
    Evaluates the Lagrange interpolating polynomial defined by a set of interpolation nodes, f_nodes(x_nodes), at a 
    new set of point(s), x_interp. The Lagrange interpolating polynomial is defined as:

        P_N(x) = (Σ_{j=0:N} f_j w_j / (x - x_j)) / (Σ_{j=0:N} w_j / (x - x_j))

    where w_j are the barycentric weights computed using the funciton Barycentric_Weights(), and f_j are the function
    values at the interpolation nodes x_j. This formula allows Barycentric weights to be computed ONCE and then stored
    for future interpolations at various points with various function values; this is possible because w_j depend ONLY
    on the interpolation nodes x_j. Thus, if we always interpolated from the same set of interpolation nodes (e.g. LGL
    or LG quadrature points within a spectral element), w_j would not change. For details, see Section 3.4 (Algorithm 31)
    in Kopriva's book. 

    Inputs:
    - x_nodes: (N+1,) array of interpolation nodes (e.g. LGL or LG quadrature points)
    - f_nodes: (N+1,) array of function values at the interpolation nodes, x_nodes
    - w: (N+1,) array of barycentric weights computed from the interpolation nodes, x_nodes
    - x_interp: (M,) array of points at which to evaluate the Lagrange interpolating polynomial

    Outputs:
    - f_interp: (M,) array of interpolated function values at the points x_interp
    '''

    # Initialize the array of interpolated function values at the new points.
    N        = len(x_nodes)
    M        = len(x_interp)
    f_interp = np.zeros(M, dtype=float)

    # Loop through each point in x_interp and compute the Lagrange interpolation. Keep numerator and denominator separate 
    # to avoid numerical instability.
    for i, x_i in enumerate(x_interp):
        # Intialize
        numerator   = 0.0
        denominator = 0.0
        exact_node  = False

        # Compute interpolation for input point x_interp[i]
        for j in range(N):
            # Create array of x_interp[i] - x_nodes[j] to check for potential division by zero
            x_diff = x_i - x_nodes[j]

            if np.abs(x_diff) < np.finfo(float).eps:  # Node we're interpolating to is exactly an input node
                f_interp[i] = f_nodes[j]
                exact_node  = True
                break

            # Compute the numerator and denominator (only reaches this computation is exact node is NOT detected)
            numerator   += f_nodes[j] * w[j] / x_diff
            denominator +=              w[j] / x_diff

        # Combine numerator and denominator for final interpolant value
        if not exact_node:
            f_interp[i] = numerator / denominator
    
    return f_interp

def Lagrange_Interpolant_Derivative(x_nodes : np.ndarray, f_nodes : np.ndarray, w : np.ndarray, x_interp : np.ndarray) -> np.ndarray:
    '''
    Computes the derivative of a Lagrang interpolating polynomial, defined by nodes x_nodes and function values f_nodes, at a new
    set of points, x_interp. The polynomial derivative is defined as

         p'(x) = [ sum_j  w_j/(x-x_j) * (f_j - p(x)) / (x - x_j) ]                          (Eq. 3.45)
                / [ sum_j  w_j/(x-x_j) ]

    At an a construction node (x_nodes), this equation simplifies to:

        p'(x_i) = -(1/w_j) sum_{j!=i} w_j (f_i - f_j) / (x_i - x_j)   j = 0, 1, 2, ..., N   (Eq. 3.46)

    For details, see Chapter 3.5 (Algorithm 36) in Kopriva's book.

    Inputs:
    - x_nodes: (N+1,) array of interpolation nodes (e.g. LGL or LG quadrature points)
    - f_nodes: (N+1,) array of function values at the interpolation nodes, x_nodes
    - w: (N+1,) array of barycentric weights computed from the interpolation nodes, x_nodes
    - x_interp: (M,) array of points at which to evaluate the Lagrange interpolating polynomial

    Outputs:
    - f_interp_deriv: (M,) array of interpolated function derivative values at the points x_interp
    '''

    # Initialize the array of interpolated function values at the new points.
    N              = len(x_nodes)
    M              = len(x_interp)
    f_interp_deriv = np.zeros(M, dtype=float)
    #exact_node     = False

    for i, x_i in enumerate(x_interp):
        # Check if any of the desired interpolation points lie on a construction node
        x_diff         = x_i - x_nodes
        exact_node_loc = np.where(np.abs(x_diff) < np.finfo(float).eps * max(1.0, np.max(np.abs(x_nodes))))[0]

        # Initialize numerator and denominator
        numerator   = 0.0
        denominator = 1.0

        if exact_node_loc.size > 0:  # The interpolation node lies on a construction node, x_nodes[k]
            # Extract index of construction node that the interpolation node matches and save denominator scalar
            k           = exact_node_loc[0]

            for j in range(N):
                if j != k:
                    f_interp_deriv[i] += (w[j] / w[k]) * (f_nodes[k] - f_nodes[j]) / (x_nodes[k] - x_nodes[j])
            
            # Combine numerator and denominator
            f_interp_deriv[i] *= -1.0

        else:
            # Use Eq. 3.45. Numpy vectorization is used to compute array products faster
            f_interp          = Lagrange_Interpolation(x_nodes, f_nodes, w, np.array([x_i]))[0]  # Remember, Lagrange_Interpolation accepts ndarray's only
            w_over_x          = w / x_diff
            f_interp_deriv[i] = np.dot(w_over_x, (f_interp - f_nodes)/x_diff) / np.sum(w_over_x)

    return f_interp_deriv

def Polynomial_Derivative_Matrix(x_nodes : np.ndarray, w : Optional[np.ndarray]=None) -> np.ndarray:
    '''
    Computes the standard derivative matrix used to compute the derivative of an Nth order Lagrange polynomial according to:

        P'_N(x_i) = D_{ij} @ f_j   i, j = 0, 1, 2, ..., N    (Eq. 3.47)

    where D_{ij} is the standard derivative matrix for an Nth order polynomial and f_j is the array of function values (weights) 
    at the construction points / nodes, x_j. This formulation is particularly useful for DG schemes because the derivative 
    matrix entries are formulated entirely from the Lagrange Polynomial; that is, all entries depend on x_i and x_j ONLY. Thus,
    if we evaluate the derivative at the same points within each element (e.g., the LG or LGL nodes in the computational domain
    [-1, 1]), the derivative matrix D_{ij} is the same in ALL elements. Only the weight array, f_j, changes. That means D_{ij} 
    can be precomputed ONCE, stored, and used for all elements. For details, see Section 3.5.2 (Algorithm 37) in Kopriva's book.

    Inputs:
    - x_nodes: (N+1,) array of interpolation nodes (e.g. LGL or LG quadrature points)
    - w: (N+1,) array of barycentric weights (optional — computed if not provided)

    Outputs: 
    - D: (N+1, N+1) polynomial derivative matrix
    '''

    # Initialize arrays
    N = len(x_nodes)
    D = np.zeros((N,N), dtype=float)

    # Compute barycentric weights if necessary
    if w is None:
        w = Barycentric_Weights(x_nodes)
    
    # Compute off-diagonal entries of derivative matrix (Kopriva, eq. 3.46, Alg. 37)
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (w[j]/w[i]) * (1/(x_nodes[i] - x_nodes[j]))

    # Diagonal entries: negative sum of each row (Kopriva Eq 2.43)
    for i in range(N):
        D[i, i] = -np.sum(D[i,:])

    return D

def Polynomial_Interpolation_Matrix(x_nodes: np.ndarray, x_interp: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    '''
    Computes the (M x N+1) polynomial interpolation matrix T, where entry T_ij = L_j(x_interp_i)
    gives the value of the j-th Lagrange basis polynomial at the i-th interpolation point. The
    interpolated values of any polynomial represented on x_nodes can then be computed as:

        f_interp = T @ f_nodes

    This is particularly useful in DG schemes when interpolating many different functions from
    the same set of nodes to the same set of target points (e.g. interpolating solution values
    to flux points, or to a finer grid for visualization), since T can be precomputed ONCE and
    reused for any f via a single matrix-vector multiply.

    Inputs:
    - x_nodes : (N+1,) array of construction nodes (e.g. LG or LGL quadrature points)
    - x_interp: (M,)   array of target interpolation points
    - w       : (N+1,) array of barycentric weights (optional — computed if not provided)

    Outputs:
    - T: (M, N+1) polynomial interpolation matrix
    '''

    # Initialize arrays
    N = len(x_nodes)
    M = len(x_interp)
    T = np.zeros((M, N), dtype=float)

    # Compute barycentric weights if not provided
    if w is None:
        w = Barycentric_Weights(x_nodes)

    # Compute entries T_ij = L_j(x_interp_i)
    for i, xi in enumerate(x_interp):
        for j in range(N):
            T[i, j] = Lagrange_Interpolating_Polynomial(j, x_nodes, w, xi)

    return T

def Interpolate_To_New_Points(T_ij : np.ndarray, f_j : np.ndarray) -> np.ndarray:
    '''
    This is a wrapper function that interpolates a polynomial from one set of points to another. This is done using 
    an interpolation matrix, T_{ij}, which is pre-computed using the function "Polynomial_Interpolation_Matrix()". 
    Because the interpolation matrix T_{ij} is constructed using the polynomial construction nodes (x_nodes) and 
    desired interpolation points (x_interp), the function Interpolate_To_New_Points() can ONLY interpolate to the
    points defined by the matrix T_{ij}. The primary uses for Interpolate_To_New_Points is:
        - Interpolation to a predefined grid for IO operations
        - Interpolation from one mesh to a coarser/finer mesh
        - Interpolation from LG to LGL point (primarily to capture boundary nodes and/or for staggered grid methods)

    Note: N is the polynomial order defined by the nodes we are interpolating FROM, and M is the numer of nodes we
          are interpolating TO

    Inputs:
    - T_ij     : (M, N+1) polynomial interpolation matrix
    - f_j      : (N+1,)    array of function values at nodes we are interpolating FROM

    Outputs:
    - f_interp : (M,) array of function values at nodes we are interpolating TO
    '''

    # Matrix vector operations are simple in Python :)
    f_interp = T_ij @ f_j
    
    return f_interp
