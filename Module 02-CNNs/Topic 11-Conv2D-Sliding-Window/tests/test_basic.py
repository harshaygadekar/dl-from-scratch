"""
Topic 11: Basic Tests

Tests core Conv2D functionality for correctness.
Run with: pytest tests/test_basic.py -v
"""

import numpy as np
import sys
from pathlib import Path

# Add solutions to path
sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

try:
    from level01_naive import conv2d_naive, calculate_output_size, pad_array
except ImportError as e:
    print(f"Warning: Could not import level01_naive: {e}")
    conv2d_naive = None

try:
    from level02_vectorized import conv2d_vectorized
except ImportError:
    conv2d_vectorized = None

try:
    from level03_memory_efficient import conv2d_strided
except ImportError:
    conv2d_strided = None


class TestOutputSizeCalculation:
    """Tests for output dimension calculations."""
    
    def test_basic_calculation(self):
        """Test basic output size formula."""
        # H=7, K=3, P=0, S=1 → H_out = (7-3)/1 + 1 = 5
        result = calculate_output_size(7, 3, 0, 1)
        assert result == 5
    
    def test_with_padding(self):
        """Test output size with padding."""
        # H=7, K=3, P=1, S=1 → H_out = (7-3+2)/1 + 1 = 7
        result = calculate_output_size(7, 3, 1, 1)
        assert result == 7
    
    def test_with_stride(self):
        """Test output size with stride."""
        # H=8, K=3, P=1, S=2 → H_out = (8-3+2)/2 + 1 = 4.5 floored = 4
        result = calculate_output_size(8, 3, 1, 2)
        assert result == 4
    
    def test_large_kernel(self):
        """Test with large kernel."""
        # H=32, K=7, P=3, S=1 → H_out = (32-7+6)/1 + 1 = 32
        result = calculate_output_size(32, 7, 3, 1)
        assert result == 32
    
    def test_1x1_kernel(self):
        """Test with 1x1 kernel."""
        # H=16, K=1, P=0, S=1 → H_out = (16-1)/1 + 1 = 16
        result = calculate_output_size(16, 1, 0, 1)
        assert result == 16


class TestPadding:
    """Tests for padding functionality."""
    
    def test_no_padding(self):
        """Test that padding=0 returns same array."""
        x = np.random.randn(2, 3, 4, 4)
        result = pad_array(x, 0)
        assert result.shape == x.shape
        np.testing.assert_array_equal(result, x)
    
    def test_padding_increases_size(self):
        """Test that padding increases spatial dimensions."""
        x = np.random.randn(2, 3, 4, 4)
        result = pad_array(x, 1)
        assert result.shape == (2, 3, 6, 6)
    
    def test_padding_values(self):
        """Test that padding adds zeros."""
        x = np.ones((1, 1, 3, 3))
        result = pad_array(x, 1)
        
        # Check corners are zero
        assert result[0, 0, 0, 0] == 0
        assert result[0, 0, 0, -1] == 0
        assert result[0, 0, -1, 0] == 0
        assert result[0, 0, -1, -1] == 0
        
        # Check center is preserved
        assert result[0, 0, 1, 1] == 1
    
    def test_padding_2(self):
        """Test with padding=2."""
        x = np.random.randn(1, 1, 4, 4)
        result = pad_array(x, 2)
        assert result.shape == (1, 1, 8, 8)


class TestConv2DShapes:
    """Tests for Conv2D output shapes."""
    
    def test_basic_shape(self):
        """Test basic Conv2D output shape."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(2, 3, 5, 5)
        weight = np.random.randn(4, 3, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=0)
        
        # (5-3)/1 + 1 = 3
        assert output.shape == (2, 4, 3, 3)
    
    def test_same_convolution(self):
        """Test 'same' convolution preserves spatial size."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 1, 7, 7)
        weight = np.random.randn(2, 1, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=1)
        
        # With padding=1, output should be same as input
        assert output.shape == (1, 2, 7, 7)
    
    def test_strided_convolution(self):
        """Test strided convolution reduces size."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 1, 8, 8)
        weight = np.random.randn(1, 1, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=2, padding=1)
        
        # (8-3+2)/2 + 1 = 4.5 floored = 4
        assert output.shape == (1, 1, 4, 4)
    
    def test_multi_channel_input(self):
        """Test with multiple input channels."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(2, 64, 8, 8)
        weight = np.random.randn(128, 64, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=1)
        
        assert output.shape == (2, 128, 8, 8)
    
    def test_batch_processing(self):
        """Test with larger batch size."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(16, 3, 32, 32)
        weight = np.random.randn(64, 3, 3, 3)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=1)
        
        assert output.shape == (16, 64, 32, 32)


class TestConv2DValues:
    """Tests for Conv2D output values."""
    
    def test_known_values_simple(self):
        """Test with hand-computed values."""
        if conv2d_naive is None:
            return
        
        # Simple 3x3 input with 2x2 kernel
        input_data = np.array([[[[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9]]]], dtype=np.float32)
        
        # Identity kernel (just sums the window)
        weight = np.ones((1, 1, 2, 2), dtype=np.float32)
        
        output = conv2d_naive(input_data, weight, stride=1, padding=0)
        
        # Expected: [[1+2+4+5, 2+3+5+6], [4+5+7+8, 5+6+8+9]] = [[12, 16], [24, 28]]
        expected = np.array([[[[12, 16],
                               [24, 28]]]], dtype=np.float32)
        
        np.testing.assert_array_equal(output, expected)
    
    def test_identity_kernel(self):
        """Test 1x1 identity kernel."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 1, 5, 5)
        weight = np.ones((1, 1, 1, 1))
        
        output = conv2d_naive(input_data, weight, stride=1, padding=0)
        
        # 1x1 conv with weight=1 should preserve values
        np.testing.assert_allclose(output, input_data, rtol=1e-5)
    
    def test_bias_addition(self):
        """Test that bias is correctly added."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(1, 1, 4, 4)
        weight = np.ones((2, 1, 2, 2))
        bias = np.array([1.0, 2.0])
        
        output = conv2d_naive(input_data, weight, bias, stride=1, padding=0)
        
        # Check output has 2 channels
        assert output.shape[1] == 2
        
        # Output[0] should be different from output[1] due to different biases
        assert not np.allclose(output[0, 0], output[0, 1])


class TestImplementationConsistency:
    """Test that all implementations produce same results."""
    
    def test_naive_vs_vectorized(self):
        """Compare naive and vectorized implementations."""
        if conv2d_naive is None or conv2d_vectorized is None:
            return
        
        np.random.seed(42)
        input_data = np.random.randn(2, 4, 8, 8).astype(np.float32)
        weight = np.random.randn(8, 4, 3, 3).astype(np.float32)
        bias = np.random.randn(8).astype(np.float32)
        
        output_naive = conv2d_naive(input_data, weight, bias, stride=1, padding=1)
        output_vectorized = conv2d_vectorized(input_data, weight, bias, stride=1, padding=1)
        
        np.testing.assert_allclose(output_naive, output_vectorized, rtol=1e-5)
    
    def test_naive_vs_strided(self):
        """Compare naive and strided implementations."""
        if conv2d_naive is None or conv2d_strided is None:
            return
        
        np.random.seed(42)
        input_data = np.random.randn(2, 4, 8, 8).astype(np.float32)
        weight = np.random.randn(8, 4, 3, 3).astype(np.float32)
        bias = np.random.randn(8).astype(np.float32)
        
        output_naive = conv2d_naive(input_data, weight, bias, stride=1, padding=1)
        output_strided = conv2d_strided(input_data, weight, bias, stride=1, padding=1)
        
        np.testing.assert_allclose(output_naive, output_strided, rtol=1e-5)
    
    def test_vectorized_vs_strided(self):
        """Compare vectorized and strided implementations."""
        if conv2d_vectorized is None or conv2d_strided is None:
            return
        
        np.random.seed(42)
        input_data = np.random.randn(4, 8, 16, 16).astype(np.float32)
        weight = np.random.randn(16, 8, 3, 3).astype(np.float32)
        bias = np.random.randn(16).astype(np.float32)
        
        output_vectorized = conv2d_vectorized(input_data, weight, bias, stride=1, padding=1)
        output_strided = conv2d_strided(input_data, weight, bias, stride=1, padding=1)
        
        np.testing.assert_allclose(output_vectorized, output_strided, rtol=1e-5)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
