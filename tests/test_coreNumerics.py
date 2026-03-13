# tests/test_coreNumerics.py
import numpy as np
import pytest
from dingus.coreNumerics.interpolation import (
    Barycentric_Weights,
    Lagrange_Interpolating_Polynomial,
    Lagrange_Interpolation,
    Lagrange_Interpolant_Derivative,
    Polynomial_Derivative_Matrix,
    Polynomial_Interpolation_Matrix,
    Interpolate_To_New_Points,
)
from dingus.coreNumerics.quadrature import (
    Legendre_Gauss_Nodes_And_Weights,
    Legendre_Gauss_Lobatto_Nodes_And_Weights,
)

# -----------------------------------------------------------------------
# Shared fixtures: LG and LGL nodes/weights for a polynomial of order N=4
# -----------------------------------------------------------------------
N         = 4                                                    # polynomial order — 5 nodes
LG_NODES, LG_WEIGHTS   = Legendre_Gauss_Nodes_And_Weights(N)
LGL_NODES, LGL_WEIGHTS = Legendre_Gauss_Lobatto_Nodes_And_Weights(N)

# -----------------------------------------------------------------------
# Helper: a few simple test polynomials and their derivatives
# -----------------------------------------------------------------------
def poly_const(x):  return np.ones_like(x)                      # f(x) = 1,    f'(x) = 0
def poly_linear(x): return x                                     # f(x) = x,    f'(x) = 1
def poly_quad(x):   return x**2                                  # f(x) = x^2,  f'(x) = 2x
def poly_cubic(x):  return x**3                                  # f(x) = x^3,  f'(x) = 3x^2

def dpoly_const(x):  return np.zeros_like(x)
def dpoly_linear(x): return np.ones_like(x)
def dpoly_quad(x):   return 2 * x
def dpoly_cubic(x):  return 3 * x**2


# =======================================================================
# 1. Barycentric_Weights
# =======================================================================
class TestBarycentricWeights:

    def test_returns_correct_shape_LG(self):
        w = Barycentric_Weights(LG_NODES)
        assert w.shape == (N + 1,)

    def test_returns_correct_shape_LGL(self):
        w = Barycentric_Weights(LGL_NODES)
        assert w.shape == (N + 1,)

    def test_partition_of_unity(self):
        """
        The Lagrange basis polynomials sum to 1 everywhere.  Equivalently,
        interpolating f=1 (constant) must return 1 at every point.  This is
        a necessary (but not sufficient) condition that the weights are correct.
        """
        w        = Barycentric_Weights(LG_NODES)
        x_test   = np.linspace(-1, 1, 20)
        f_ones   = np.ones(N + 1)
        f_interp = Lagrange_Interpolation(LG_NODES, f_ones, w, x_test)
        assert np.allclose(f_interp, 1.0, atol=1e-14)

    def test_weights_nonzero(self):
        w = Barycentric_Weights(LG_NODES)
        assert np.all(w != 0.0)

    def test_LGL_endpoints_have_opposite_sign(self):
        """
        For LGL nodes on [-1,1] the outermost weights have opposite signs
        to the interior weights — a sanity check on sign alternation.
        """
        w = Barycentric_Weights(LGL_NODES)
        assert w[0] * w[1] < 0   # adjacent weights must alternate sign


# =======================================================================
# 2. Lagrange_Interpolating_Polynomial
# =======================================================================
class TestLagrangeInterpolatingPolynomial:

    def test_kronecker_delta_property(self):
        """
        L_j(x_i) = 1 if i == j, else 0  (the defining property of Lagrange basis polynomials)
        """
        w = Barycentric_Weights(LG_NODES)
        for j in range(N + 1):
            for i, xi in enumerate(LG_NODES):
                expected = 1.0 if i == j else 0.0
                assert np.isclose(
                    Lagrange_Interpolating_Polynomial(j, LG_NODES, w, xi),
                    expected,
                    atol=1e-14,
                ), f"Kronecker delta failed for j={j}, i={i}"

    def test_values_between_nodes(self):
        """
        At a non-node point, L_j should be strictly between -1 and 1 for well-behaved nodes.
        """
        w    = Barycentric_Weights(LGL_NODES)
        xi   = 0.123456   # arbitrary interior point
        vals = [Lagrange_Interpolating_Polynomial(j, LGL_NODES, w, xi) for j in range(N + 1)]
        assert np.isclose(sum(vals), 1.0, atol=1e-14)   # partition of unity

    def test_returns_scalar(self):
        w   = Barycentric_Weights(LG_NODES)
        val = Lagrange_Interpolating_Polynomial(0, LG_NODES, w, 0.5)
        assert isinstance(val, (float, np.floating))


# =======================================================================
# 3. Lagrange_Interpolation
# =======================================================================
class TestLagrangeInterpolation:

    def test_exact_at_nodes(self):
        """Interpolating back to the construction nodes must return f exactly."""
        w = Barycentric_Weights(LG_NODES)
        for poly in [poly_const, poly_linear, poly_quad, poly_cubic]:
            f        = poly(LG_NODES)
            f_interp = Lagrange_Interpolation(LG_NODES, f, w, LG_NODES)
            assert np.allclose(f_interp, f, atol=1e-14), f"Failed for {poly.__name__}"

    def test_constant_function(self):
        """A constant polynomial should be reproduced exactly everywhere."""
        w      = Barycentric_Weights(LG_NODES)
        f      = poly_const(LG_NODES)
        x_test = np.linspace(-1, 1, 50)
        assert np.allclose(Lagrange_Interpolation(LG_NODES, f, w, x_test), 1.0, atol=1e-14)

    def test_linear_function(self):
        """A degree-1 polynomial must be reproduced exactly by any N>=1 interpolant."""
        w      = Barycentric_Weights(LG_NODES)
        f      = poly_linear(LG_NODES)
        x_test = np.linspace(-1, 1, 50)
        assert np.allclose(
            Lagrange_Interpolation(LG_NODES, f, w, x_test),
            poly_linear(x_test),
            atol=1e-13,
        )

    def test_quadratic_function(self):
        """f(x) = x^2 must be reproduced exactly for N >= 2."""
        w      = Barycentric_Weights(LG_NODES)
        f      = poly_quad(LG_NODES)
        x_test = np.linspace(-1, 1, 50)
        assert np.allclose(
            Lagrange_Interpolation(LG_NODES, f, w, x_test),
            poly_quad(x_test),
            atol=1e-13,
        )

    def test_output_shape(self):
        w      = Barycentric_Weights(LG_NODES)
        f      = poly_quad(LG_NODES)
        x_test = np.linspace(-1, 1, 17)
        result = Lagrange_Interpolation(LG_NODES, f, w, x_test)
        assert result.shape == (17,)


# =======================================================================
# 4. Lagrange_Interpolant_Derivative
# =======================================================================
class TestLagrangeInterpolantDerivative:

    def test_derivative_of_constant_is_zero(self):
        w      = Barycentric_Weights(LG_NODES)
        f      = poly_const(LG_NODES)
        x_test = np.linspace(-0.9, 0.9, 20)
        dp     = Lagrange_Interpolant_Derivative(LG_NODES, f, w, x_test)
        assert np.allclose(dp, 0.0, atol=1e-13)

    def test_derivative_of_linear_is_one(self):
        w      = Barycentric_Weights(LG_NODES)
        f      = poly_linear(LG_NODES)
        x_test = np.linspace(-0.9, 0.9, 20)
        dp     = Lagrange_Interpolant_Derivative(LG_NODES, f, w, x_test)
        assert np.allclose(dp, 1.0, atol=1e-13)

    def test_derivative_of_quadratic(self):
        """f(x) = x^2  =>  f'(x) = 2x"""
        w      = Barycentric_Weights(LG_NODES)
        f      = poly_quad(LG_NODES)
        x_test = np.linspace(-0.9, 0.9, 20)
        dp     = Lagrange_Interpolant_Derivative(LG_NODES, f, w, x_test)
        assert np.allclose(dp, dpoly_quad(x_test), atol=1e-12)

    def test_derivative_at_construction_nodes(self):
        """Derivative evaluated exactly at the LG nodes (exact-node branch)."""
        w  = Barycentric_Weights(LG_NODES)
        f  = poly_quad(LG_NODES)
        dp = Lagrange_Interpolant_Derivative(LG_NODES, f, w, LG_NODES)
        assert np.allclose(dp, dpoly_quad(LG_NODES), atol=1e-12)

    def test_derivative_of_cubic(self):
        """f(x) = x^3  =>  f'(x) = 3x^2; exact for N >= 3."""
        w      = Barycentric_Weights(LG_NODES)
        f      = poly_cubic(LG_NODES)
        x_test = np.linspace(-0.9, 0.9, 20)
        dp     = Lagrange_Interpolant_Derivative(LG_NODES, f, w, x_test)
        assert np.allclose(dp, dpoly_cubic(x_test), atol=1e-11)

    def test_output_shape(self):
        w      = Barycentric_Weights(LG_NODES)
        f      = poly_quad(LG_NODES)
        x_test = np.linspace(-0.9, 0.9, 13)
        dp     = Lagrange_Interpolant_Derivative(LG_NODES, f, w, x_test)
        assert dp.shape == (13,)


# =======================================================================
# 5. Polynomial_Derivative_Matrix
# =======================================================================
class TestPolynomialDerivativeMatrix:

    def test_shape(self):
        D = Polynomial_Derivative_Matrix(LG_NODES)
        assert D.shape == (N + 1, N + 1)

    def test_row_sum_is_zero(self):
        """Each row of D must sum to zero (derivative of a constant is zero)."""
        D = Polynomial_Derivative_Matrix(LG_NODES)
        assert np.allclose(D.sum(axis=1), 0.0, atol=1e-13)

    def test_derivative_of_constant(self):
        D      = Polynomial_Derivative_Matrix(LG_NODES)
        f      = poly_const(LG_NODES)
        df     = D @ f
        assert np.allclose(df, 0.0, atol=1e-13)

    def test_derivative_of_linear(self):
        D  = Polynomial_Derivative_Matrix(LG_NODES)
        f  = poly_linear(LG_NODES)
        df = D @ f
        assert np.allclose(df, dpoly_linear(LG_NODES), atol=1e-13)

    def test_derivative_of_quadratic(self):
        D  = Polynomial_Derivative_Matrix(LG_NODES)
        f  = poly_quad(LG_NODES)
        df = D @ f
        assert np.allclose(df, dpoly_quad(LG_NODES), atol=1e-12)

    def test_derivative_of_cubic(self):
        D  = Polynomial_Derivative_Matrix(LG_NODES)
        f  = poly_cubic(LG_NODES)
        df = D @ f
        assert np.allclose(df, dpoly_cubic(LG_NODES), atol=1e-11)

    def test_optional_weights_match_computed(self):
        """Passing precomputed weights should give the same D as computing internally."""
        w  = Barycentric_Weights(LG_NODES)
        D1 = Polynomial_Derivative_Matrix(LG_NODES)
        D2 = Polynomial_Derivative_Matrix(LG_NODES, w=w)
        assert np.allclose(D1, D2, atol=1e-15)

    def test_LGL_nodes(self):
        """D should also work correctly for LGL nodes."""
        D  = Polynomial_Derivative_Matrix(LGL_NODES)
        f  = poly_quad(LGL_NODES)
        df = D @ f
        assert np.allclose(df, dpoly_quad(LGL_NODES), atol=1e-12)


# =======================================================================
# 6. Polynomial_Interpolation_Matrix
# =======================================================================
class TestPolynomialInterpolationMatrix:

    def test_shape(self):
        x_interp = np.linspace(-1, 1, 10)
        T        = Polynomial_Interpolation_Matrix(LG_NODES, x_interp)
        assert T.shape == (10, N + 1)

    def test_identity_at_construction_nodes(self):
        """
        When x_interp == x_nodes, T should be the identity matrix since
        L_j(x_i) = delta_ij.
        """
        T = Polynomial_Interpolation_Matrix(LG_NODES, LG_NODES)
        assert np.allclose(T, np.eye(N + 1), atol=1e-14)

    def test_row_sum_is_one(self):
        """Each row of T must sum to 1 (partition of unity)."""
        x_interp = np.linspace(-1, 1, 20)
        T        = Polynomial_Interpolation_Matrix(LG_NODES, x_interp)
        assert np.allclose(T.sum(axis=1), 1.0, atol=1e-14)

    def test_interpolation_matches_Lagrange_Interpolation(self):
        """T @ f must give the same result as Lagrange_Interpolation for any f."""
        w        = Barycentric_Weights(LG_NODES)
        f        = poly_quad(LG_NODES)
        x_interp = np.linspace(-0.9, 0.9, 15)
        T        = Polynomial_Interpolation_Matrix(LG_NODES, x_interp, w=w)
        assert np.allclose(
            T @ f,
            Lagrange_Interpolation(LG_NODES, f, w, x_interp),
            atol=1e-13,
        )

    def test_optional_weights_match_computed(self):
        w        = Barycentric_Weights(LG_NODES)
        x_interp = np.linspace(-1, 1, 10)
        T1       = Polynomial_Interpolation_Matrix(LG_NODES, x_interp)
        T2       = Polynomial_Interpolation_Matrix(LG_NODES, x_interp, w=w)
        assert np.allclose(T1, T2, atol=1e-15)


# =======================================================================
# 7. Interpolate_To_New_Points
# =======================================================================
class TestInterpolateToNewPoints:

    def test_constant_function(self):
        x_interp = np.linspace(-1, 1, 20)
        T        = Polynomial_Interpolation_Matrix(LG_NODES, x_interp)
        f        = poly_const(LG_NODES)
        assert np.allclose(Interpolate_To_New_Points(T, f), 1.0, atol=1e-14)

    def test_quadratic_function(self):
        x_interp = np.linspace(-0.9, 0.9, 25)
        T        = Polynomial_Interpolation_Matrix(LG_NODES, x_interp)
        f        = poly_quad(LG_NODES)
        assert np.allclose(
            Interpolate_To_New_Points(T, f),
            poly_quad(x_interp),
            atol=1e-13,
        )

    def test_output_shape(self):
        x_interp = np.linspace(-1, 1, 17)
        T        = Polynomial_Interpolation_Matrix(LG_NODES, x_interp)
        f        = poly_cubic(LG_NODES)
        result   = Interpolate_To_New_Points(T, f)
        assert result.shape == (17,)

    def test_matches_matrix_multiply(self):
        """Interpolate_To_New_Points should be identical to T @ f directly."""
        x_interp = np.linspace(-0.9, 0.9, 12)
        T        = Polynomial_Interpolation_Matrix(LG_NODES, x_interp)
        f        = poly_cubic(LG_NODES)
        assert np.allclose(Interpolate_To_New_Points(T, f), T @ f, atol=1e-15)

    def test_LG_to_LGL_interpolation(self):
        """
        A practical use case: interpolate from LG nodes to LGL nodes.
        For a polynomial of degree <= N this must be exact.
        """
        T = Polynomial_Interpolation_Matrix(LG_NODES, LGL_NODES)
        f = poly_cubic(LG_NODES)
        assert np.allclose(
            Interpolate_To_New_Points(T, f),
            poly_cubic(LGL_NODES),
            atol=1e-12,
        )