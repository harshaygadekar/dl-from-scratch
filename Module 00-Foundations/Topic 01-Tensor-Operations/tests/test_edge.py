"""
Topic 01: Edge Case Tests

Tests unusual inputs and boundary conditions.
Run with: pytest tests/test_edge.py -v
"""

import numpy as np
import pytest


class TestEdgeCaseBatchedMatmul:
    """Edge cases for batched matmul."""
    
    def test_empty_batch(self):
        """Empty batch should return empty result."""
        a = np.random.randn(0, 3, 4)
        b = np.random.randn(0, 4, 5)
        result = np.matmul(a, b)
        assert result.shape == (0, 3, 5)
    
    def test_single_element_matrices(self):
        """1x1 matrices should work."""
        a = np.array([[[5]]])  # (1, 1, 1)
        b = np.array([[[3]]])  # (1, 1, 1)
        result = np.matmul(a, b)
        assert result[0, 0, 0] == 15
    
    def test_large_batch(self):
        """Large batch should work."""
        a = np.random.randn(100, 2, 3)
        b = np.random.randn(100, 3, 2)
        result = np.matmul(a, b)
        assert result.shape == (100, 2, 2)
    
    def test_single_row_col(self):
        """Matrices with single row/column."""
        a = np.random.randn(2, 1, 5)  # Single row
        b = np.random.randn(2, 5, 1)  # Single column
        result = np.matmul(a, b)
        assert result.shape == (2, 1, 1)


class TestEdgeCaseBroadcasting:
    """Edge cases for broadcasting."""
    
    def test_scalar_array(self):
        """Scalar should broadcast to any shape."""
        scalar = np.array(5)
        arr = np.ones((3, 4, 5))
        result = scalar + arr
        assert result.shape == (3, 4, 5)
        assert np.all(result == 6)
    
    def test_all_ones_dimensions(self):
        """Arrays with all 1 dimensions."""
        a = np.ones((1, 1, 1))
        b = np.ones((5, 4, 3))
        result = a + b
        assert result.shape == (5, 4, 3)
    
    def test_higher_dimensional(self):
        """Test with 5D arrays."""
        a = np.ones((2, 1, 4, 1, 6))
        b = np.ones((1, 3, 1, 5, 1))
        result = a + b
        assert result.shape == (2, 3, 4, 5, 6)
    
    def test_negative_values(self):
        """Broadcasting with negative values."""
        a = np.array([-1, -2, -3])
        b = np.array([[1], [2]])
        result = a * b
        expected = np.array([[-1, -2, -3], [-2, -4, -6]])
        np.testing.assert_array_equal(result, expected)


class TestEdgeCaseTranspose:
    """Edge cases for transpose operations."""
    
    def test_1d_transpose(self):
        """1D array transpose should be no-op."""
        a = np.array([1, 2, 3])
        result = a.T
        np.testing.assert_array_equal(result, a)
    
    def test_transpose_view(self):
        """Transpose should be a view, not copy."""
        a = np.random.randn(3, 4)
        b = a.T
        assert np.shares_memory(a, b)
    
    def test_double_transpose(self):
        """Double transpose should return original."""
        a = np.random.randn(3, 4)
        result = a.T.T
        np.testing.assert_array_equal(result, a)


class TestEdgeCaseReductions:
    """Edge cases for reduction operations."""
    
    def test_sum_empty_array(self):
        """Sum of empty array should be 0."""
        a = np.array([])
        result = np.sum(a)
        assert result == 0
    
    def test_sum_single_element(self):
        """Sum of single element should be that element."""
        a = np.array([42])
        result = np.sum(a)
        assert result == 42
    
    def test_keepdims(self):
        """keepdims should preserve dimensions."""
        a = np.ones((3, 4, 5))
        result = np.sum(a, axis=1, keepdims=True)
        assert result.shape == (3, 1, 5)
    
    def test_multiple_axes(self):
        """Reducing over multiple axes."""
        a = np.ones((3, 4, 5))
        result = np.sum(a, axis=(0, 2))
        assert result.shape == (4,)
        assert np.all(result == 15)  # 3 * 5 = 15


class TestEdgeCaseDataTypes:
    """Edge cases for different data types."""
    
    def test_int_operations(self):
        """Integer operations should preserve type."""
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4, 5, 6], dtype=np.int32)
        result = a + b
        assert result.dtype == np.int32
    
    def test_mixed_types(self):
        """Mixed types should promote."""
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        result = a + b
        assert result.dtype == np.float64
    
    def test_complex_numbers(self):
        """Complex number operations."""
        a = np.array([1+2j, 3+4j])
        result = np.abs(a)
        expected = np.array([np.sqrt(5), 5.0])
        np.testing.assert_allclose(result, expected)
    
    def test_bool_array(self):
        """Boolean array operations."""
        a = np.array([True, False, True])
        b = np.array([False, True, True])
        result = a & b
        expected = np.array([False, False, True])
        np.testing.assert_array_equal(result, expected)


class TestEdgeCaseNumericalStability:
    """Edge cases for numerical stability."""
    
    def test_very_large_numbers(self):
        """Very large numbers should not overflow in operations."""
        a = np.array([1e300, 1e300])
        b = np.array([1e-300, 1e-300])
        result = a * b
        # Should be approximately 1.0
        np.testing.assert_allclose(result, [1.0, 1.0])
    
    def test_very_small_numbers(self):
        """Very small numbers should not underflow to zero inappropriately."""
        a = np.array([1e-300])
        result = a / 1e-300
        assert not np.isnan(result[0])
        np.testing.assert_allclose(result, [1.0])
    
    def test_nan_handling(self):
        """NaN should propagate correctly."""
        a = np.array([1.0, np.nan, 3.0])
        result = np.sum(a)
        assert np.isnan(result)
    
    def test_inf_handling(self):
        """Infinity should be handled correctly."""
        a = np.array([1.0, np.inf, 3.0])
        result = np.sum(a)
        assert np.isinf(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
