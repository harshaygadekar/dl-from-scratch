"""
Topic 01: Basic Tests

Tests core tensor operations for correctness.
Run with: pytest tests/test_basic.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add solutions to path
sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))


class TestBatchedMatmul:
    """Tests for batched matrix multiplication."""
    
    def test_shapes(self):
        """Verify output shape is correct."""
        a = np.random.randn(4, 3, 5)
        b = np.random.randn(4, 5, 2)
        result = np.matmul(a, b)
        assert result.shape == (4, 3, 2)
    
    def test_single_batch(self):
        """Test with batch size 1."""
        a = np.random.randn(1, 2, 3)
        b = np.random.randn(1, 3, 4)
        result = np.matmul(a, b)
        assert result.shape == (1, 2, 4)
    
    def test_identity_matrix(self):
        """Multiplying by identity should return input."""
        a = np.random.randn(3, 4, 4)
        identity = np.broadcast_to(np.eye(4), (3, 4, 4))
        result = np.matmul(a, identity)
        np.testing.assert_allclose(result, a, rtol=1e-5)
    
    def test_known_values(self):
        """Test with hand-computed values."""
        a = np.array([[[1, 2], [3, 4]]])  # (1, 2, 2)
        b = np.array([[[5, 6], [7, 8]]])  # (1, 2, 2)
        expected = np.array([[[19, 22], [43, 50]]])  # Known result
        result = np.matmul(a, b)
        np.testing.assert_array_equal(result, expected)


class TestBroadcasting:
    """Tests for broadcasting operations."""
    
    def test_scalar_add(self):
        """Adding scalar should work."""
        a = np.array([[1, 2], [3, 4]])
        result = a + 10
        expected = np.array([[11, 12], [13, 14]])
        np.testing.assert_array_equal(result, expected)
    
    def test_row_vector_add(self):
        """Adding row vector should broadcast across rows."""
        a = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        b = np.array([10, 20, 30])             # (3,)
        expected = np.array([[11, 22, 33], [14, 25, 36]])
        result = a + b
        np.testing.assert_array_equal(result, expected)
    
    def test_column_vector_add(self):
        """Adding column vector should broadcast across columns."""
        a = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        b = np.array([[10], [20]])             # (2, 1)
        expected = np.array([[11, 12, 13], [24, 25, 26]])
        result = a + b
        np.testing.assert_array_equal(result, expected)
    
    def test_3d_broadcast(self):
        """Test 3D broadcasting."""
        a = np.ones((2, 3, 4))
        b = np.ones((4,))
        result = a + b
        assert result.shape == (2, 3, 4)
        assert np.all(result == 2)
    
    def test_incompatible_shapes(self):
        """Incompatible shapes should raise error."""
        a = np.ones((3, 4))
        b = np.ones((3, 5))
        with pytest.raises((ValueError, Exception)):
            a + b


class TestOuterProduct:
    """Tests for outer product."""
    
    def test_shape(self):
        """Outer product shape should be (m, n)."""
        a = np.array([1, 2, 3])  # (3,)
        b = np.array([10, 20])    # (2,)
        result = np.outer(a, b)
        assert result.shape == (3, 2)
    
    def test_values(self):
        """Test known outer product values."""
        a = np.array([1, 2, 3])
        b = np.array([10, 20])
        expected = np.array([[10, 20], [20, 40], [30, 60]])
        result = np.outer(a, b)
        np.testing.assert_array_equal(result, expected)
    
    def test_broadcasting_equivalent(self):
        """Outer product should equal a[:, None] * b[None, :]."""
        a = np.random.randn(5)
        b = np.random.randn(7)
        outer_np = np.outer(a, b)
        outer_broadcast = a[:, None] * b[None, :]
        np.testing.assert_allclose(outer_np, outer_broadcast)


class TestSoftmax:
    """Tests for softmax implementation."""
    
    @staticmethod
    def softmax(x, axis=-1):
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def test_sums_to_one(self):
        """Softmax should sum to 1 along axis."""
        x = np.random.randn(4, 5)
        result = self.softmax(x, axis=1)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(4), rtol=1e-5)
    
    def test_all_positive(self):
        """Softmax output should be positive."""
        x = np.random.randn(10, 10)
        result = self.softmax(x)
        assert np.all(result > 0)
    
    def test_max_element_largest(self):
        """Largest input should have largest output."""
        x = np.array([[1.0, 2.0, 3.0]])
        result = self.softmax(x, axis=1)
        assert result[0, 2] > result[0, 1] > result[0, 0]
    
    def test_numerical_stability(self):
        """Should not overflow with large values."""
        x = np.array([[1000, 1001, 1002]])
        result = self.softmax(x, axis=1)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)


class TestNormalization:
    """Tests for vector normalization."""
    
    @staticmethod
    def normalize(x, axis=-1):
        norms = np.linalg.norm(x, axis=axis, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return x / norms
    
    def test_unit_length(self):
        """Normalized vectors should have unit length."""
        x = np.random.randn(10, 5)
        result = self.normalize(x, axis=1)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(10), rtol=1e-5)
    
    def test_zero_vector(self):
        """Zero vectors should be handled gracefully."""
        x = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
        result = self.normalize(x, axis=1)
        # Zero vector should remain zero (or at least not NaN)
        assert not np.any(np.isnan(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
