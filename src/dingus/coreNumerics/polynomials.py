# src/dingus/coreNumerics/polynomials.py
import numpy as np
from numpy.polynomial.legendre import legder, legval

'''
This module contains basic functions used in various interpolation procedures, including:
- Computing Legendre polynomials and derivatives
- Computing Chebyshev polynomials and derivatives
- Computing a polynomial derivative matrix
'''

def Legendre_Polynomial(N: int, x: np.ndarray) -> np.ndarray:
    """
    Computes the Legendre polynomial of degree N at the points specified in x. The Legendre
    polynomials are orthogonal polynomials that are commonly used in spectral methods for 
    solving partial differential equations. The Legendre polynomial of degree N is defined as:

    P_N(x) = (1/2^N) * sum_{k=0}^{N} (-1)^k * (N choose k) * (N+k choose k) * x^(N-k)

    Inputs:
    - N: degree of the Legendre polynomial
    - x: points at which to evaluate the Legendre polynomial

    Outputs:
    - P_N(x): values of the Legendre polynomial of degree N at the points specified in x
    """

    # Coefficients for order N Legendre polynomial (the Nth coefficient is 1, the rest are 0)
    coeffs    = np.zeros(N + 1)
    coeffs[N] = 1

    # Compute the Legendre polynomial using numpy's polynomial package
    P_N         = legval(x, coeffs)

    return P_N

def Legendre_Polynomial_And_Derivative(N: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Legendre polynomial of degree N and its derivative at the points specified in x. 
    Legendre polynomials are orthogonal polynomials commonly used in spectral methods for 
    solving partial differential equations. The Legendre polynomial of degree N is defined as:

    P_N(x) = (1/2^N) * sum_{k=0}^{N} (-1)^k * (N choose k) * (N+k choose k) * x^(N-k)

    Inputs:
    - N: degree of the Legendre polynomial
    - x: points at which to evaluate the Legendre polynomial

    Outputs:
    - P_N(x) : values of the Legendre polynomial of degree N at the points specified in x
    - dP_N(x): values of the derivative of the degree N Legendre polynomial at the points specified
               in x
    """

    # Coefficients for order N Legendre polynomial (the Nth coefficient is 1, the rest are 0)
    coeffs    = np.zeros(N + 1)
    coeffs[N] = 1

    # Compute the Legendre polynomial using numpy's polynomial package
    P_N         = legval(x, coeffs)
    dP_N        = legval(x, legder(coeffs))

    return P_N, dP_N

def Chebyshev_Polynomial(N: int, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Chebyshev_Polynomial is not yet implemented!")

def Chebyshev_Polynomial_And_Derivative(N: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("Chebyshev_Polynomial_And_Derivative is not yet implemented!")
