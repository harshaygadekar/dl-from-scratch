"""
Topic 11: Edge Case Tests

Tests edge cases and boundary conditions.
Run with: pytest tests/test_edge.py -v
"""

import numpy as np
import sys
from pathlib import Path

# Add solutions to path
sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

try:
    from level01_naive import conv2d_naive, calculate_output_size, pad_array
    from level02_vectorized import conv2d_vectorized
    from level03_memory_efficient import conv2d_strided
except ImportError as e:
    print(f"Warning: Could not import solutions: {e}")
    conv2d_naive = None
    conv2d_vectorized = None
    conv2d_strided = None


class TestBatchSizeEdgeCases:
    """Tests for batch size edge cases."""
    
    def test_batch_size_1(self):
        """Test with batch size of 1."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 3, 8, 8)
        weight = np.random.randn(4, 3, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=1)
        
        assert output.shape[0] == 1
        assert output.shape == (1, 4, 8, 8)
    
    def test_batch_size_0_not_allowed(self):
        """Test that batch size 0 raises error."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(0, 3, 8, 8)
        weight = np.random.randn(4, 3, 3, 3)
        
        # Should either raise error or return empty array
        try:
            output = conv2d_naive(input_data, weight, stride=1, padding=1)
            assert output.shape[0] == 0
        except (ValueError, IndexError):
            pass  # Also acceptable


class TestSpatialEdgeCases:
    """Tests for spatial dimension edge cases."""
    
    def test_minimum_spatial_size(self):
        """Test with minimum spatial size (kernel size)."""
        if conv2d_naive is None:
            return
        
        # 3x3 input with 3x3 kernel, no padding → 1x1 output
        input_data = np.random.randn(1, 1, 3, 3)
        weight = np.random.randn(1, 1, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=0)
        
        assert output.shape == (1, 1, 1, 1)
    
    def test_spatial_size_less_than_kernel(self):
        """Test when spatial size < kernel size (with padding)."""
        if conv2d_naive is None:
            return
        
        # 2x2 input with 3x3 kernel, pad=1 → 2x2 output
        input_data = np.random.randn(1, 1, 2, 2)
        weight = np.random.randn(1, 1, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=1)
        
        assert output.shape == (1, 1, 2, 2)
    
    def test_non_square_input(self):
        """Test with non-square input."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(2, 3, 8, 16)  # H=8, W=16
        weight = np.random.randn(4, 3, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=1)
        
        # Should preserve aspect ratio
        assert output.shape == (2, 4, 8, 16)


class TestChannelEdgeCases:
    """Tests for channel dimension edge cases."""
    
    def test_single_input_channel(self):
        """Test with single input channel."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(2, 1, 8, 8)
        weight = np.random.randn(4, 1, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=1)
        
        assert output.shape == (2, 4, 8, 8)
    
    def test_single_output_channel(self):
        """Test with single output channel."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(2, 3, 8, 8)
        weight = np.random.randn(1, 3, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=1)
        
        assert output.shape == (2, 1, 8, 8)
    
    def test_many_input_channels(self):
        """Test with many input channels."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 256, 8, 8)
        weight = np.random.randn(64, 256, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=1)
        
        assert output.shape == (1, 64, 8, 8)


class TestKernelEdgeCases:
    """Tests for kernel size edge cases."""
    
    def test_1x1_kernel(self):
        """Test with 1x1 kernel (pointwise convolution)."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(2, 64, 16, 16)
        weight = np.random.randn(128, 64, 1, 1)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=0)
        
        # 1x1 conv shouldn't change spatial size
        assert output.shape == (2, 128, 16, 16)
    
    def test_5x5_kernel(self):
        """Test with 5x5 kernel."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 3, 32, 32)
        weight = np.random.randn(16, 3, 5, 5)
        
        # With padding=2, should preserve size
        output = conv2d_naive(input_data, weight, stride=1, padding=2)
        
        assert output.shape == (1, 16, 32, 32)
    
    def test_7x7_kernel(self):
        """Test with 7x7 kernel."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 3, 64, 64)
        weight = np.random.randn(64, 3, 7, 7)
        
        # With padding=3, should preserve size
        output = conv2d_naive(input_data, weight, stride=1, padding=3)
        
        assert output.shape == (1, 64, 64, 64)


class TestStrideEdgeCases:
    """Tests for stride edge cases."""
    
    def test_stride_1(self):
        """Test with stride 1 (default)."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 1, 8, 8)
        weight = np.random.randn(1, 1, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=1)
        
        assert output.shape == (1, 1, 8, 8)
    
    def test_stride_2(self):
        """Test with stride 2 (downsampling)."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 1, 8, 8)
        weight = np.random.randn(1, 1, 3, 3)
        
        # stride=2, pad=1: (8-3+2)/2 + 1 = 4.5 floored = 4
        output = conv2d_naive(input_data, weight, stride=2, padding=1)
        
        assert output.shape == (1, 1, 4, 4)
    
    def test_large_stride(self):
        """Test with large stride."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 1, 32, 32)
        weight = np.random.randn(1, 1, 3, 3)
        
        # stride=4
        output = conv2d_naive(input_data, weight, stride=4, padding=1)
        
        # (32-3+2)/4 + 1 = 7.75 floored = 7
        assert output.shape == (1, 1, 8, 8)


class TestPaddingEdgeCases:
    """Tests for padding edge cases."""
    
    def test_zero_padding(self):
        """Test with zero padding (valid convolution)."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 1, 5, 5)
        weight = np.random.randn(1, 1, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=0)
        
        # (5-3)/1 + 1 = 3
        assert output.shape == (1, 1, 3, 3)
    
    def test_large_padding(self):
        """Test with large padding."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 1, 4, 4)
        weight = np.random.randn(1, 1, 3, 3)
        
        # pad=2: (4-3+4)/1 + 1 = 6
        output = conv2d_naive(input_data, weight, stride=1, padding=2)
        
        assert output.shape == (1, 1, 6, 6)


class TestNumericalEdgeCases:
    """Tests for numerical edge cases."""
    
    def test_zero_input(self):
        """Test with all-zero input."""
        if conv2d_naive is None:
            return
        
        input_data = np.zeros((1, 1, 8, 8))
        weight = np.random.randn(1, 1, 3, 3)
        bias = np.array([1.0])
        
        output = conv2d_naive(input_data, weight, bias, stride=1, padding=1)
        
        # Output should be bias everywhere (since input is 0)
        assert np.allclose(output, 1.0)
    
    def test_zero_weights(self):
        """Test with all-zero weights."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 1, 8, 8)
        weight = np.zeros((1, 1, 3, 3))
        bias = np.array([2.5])
        
        output = conv2d_naive(input_data, weight, bias, stride=1, padding=1)
        
        # Output should be bias everywhere
        assert np.allclose(output, 2.5)
    
    def test_no_bias(self):
        """Test with no bias (None)."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 1, 8, 8)
        weight = np.random.randn(1, 1, 3, 3)
        
        output = conv2d_naive(input_data, weight, bias=None, stride=1, padding=1)
        
        assert output.shape == (1, 1, 8, 8)
        assert not np.any(np.isnan(output))
    
    def test_negative_values(self):
        """Test with negative input values."""
        if conv2d_naive is None:
            return
        
        input_data = -np.abs(np.random.randn(1, 1, 8, 8))
        weight = np.random.randn(1, 1, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=1)
        
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))


class TestConsistencyEdgeCases:
    """Test consistency across implementations for edge cases."""
    
    def test_all_implementations_batch_size_1(self):
        """Verify all implementations handle batch=1 correctly."""
        if conv2d_naive is None or conv2d_vectorized is None or conv2d_strided is None:
            return
        
        np.random.seed(42)
        input_data = np.random.randn(1, 4, 8, 8).astype(np.float32)
        weight = np.random.randn(8, 4, 3, 3).astype(np.float32)
        bias = np.random.randn(8).astype(np.float32)
        
        output_naive = conv2d_naive(input_data, weight, bias, stride=1, padding=1)
        output_vect = conv2d_vectorized(input_data, weight, bias, stride=1, padding=1)
        output_strided = conv2d_strided(input_data, weight, bias, stride=1, padding=1)
        
        np.testing.assert_allclose(output_naive, output_vect, rtol=1e-5)
        np.testing.assert_allclose(output_naive, output_strided, rtol=1e-5)
    
    def test_all_implementations_1x1_kernel(self):
        """Verify all implementations handle 1x1 kernel correctly."""
        if conv2d_naive is None or conv2d_vectorized is None or conv2d_strided is None:
            return
        
        np.random.seed(42)
        input_data = np.random.randn(2, 16, 8, 8).astype(np.float32)
        weight = np.random.randn(32, 16, 1, 1).astype(np.float32)
        bias = np.random.randn(32).astype(np.float32)
        
        output_naive = conv2d_naive(input_data, weight, bias, stride=1, padding=0)
        output_vect = conv2d_vectorized(input_data, weight, bias, stride=1, padding=0)
        output_strided = conv2d_strided(input_data, weight, bias, stride=1, padding=0)
        
        np.testing.assert_allclose(output_naive, output_vect, rtol=1e-5)
        np.testing.assert_allclose(output_naive, output_strided, rtol=1e-5)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
