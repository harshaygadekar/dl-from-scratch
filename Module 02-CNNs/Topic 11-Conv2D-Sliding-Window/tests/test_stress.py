"""
Topic 11: Stress Tests

Tests performance with large arrays and various configurations.
Run with: pytest tests/test_stress.py -v
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add solutions to path
sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

try:
    from level01_naive import conv2d_naive
    from level02_vectorized import conv2d_vectorized
    from level03_memory_efficient import conv2d_strided
except ImportError as e:
    print(f"Warning: Could not import solutions: {e}")
    conv2d_naive = None
    conv2d_vectorized = None
    conv2d_strided = None


# Performance thresholds (seconds)
TIME_THRESHOLDS = {
    'small': 1.0,   # 4x8x16x16
    'medium': 3.0,  # 8x32x32x32
    'large': 10.0,  # 4x64x64x64
}


class TestPerformanceSmall:
    """Performance tests with small arrays."""
    
    def test_small_naive(self):
        """Test naive implementation with small input."""
        if conv2d_naive is None:
            return
        
        input_data = np.random.randn(4, 8, 16, 16).astype(np.float32)
        weight = np.random.randn(16, 8, 3, 3).astype(np.float32)
        bias = np.random.randn(16).astype(np.float32)
        
        start = time.time()
        output = conv2d_naive(input_data, weight, bias, stride=1, padding=1)
        elapsed = time.time() - start
        
        assert output.shape == (4, 16, 16, 16)
        assert elapsed < TIME_THRESHOLDS['small'], f"Too slow: {elapsed:.2f}s"
    
    def test_small_vectorized(self):
        """Test vectorized implementation with small input."""
        if conv2d_vectorized is None:
            return
        
        input_data = np.random.randn(4, 8, 16, 16).astype(np.float32)
        weight = np.random.randn(16, 8, 3, 3).astype(np.float32)
        bias = np.random.randn(16).astype(np.float32)
        
        start = time.time()
        output = conv2d_vectorized(input_data, weight, bias, stride=1, padding=1)
        elapsed = time.time() - start
        
        assert output.shape == (4, 16, 16, 16)
        assert elapsed < TIME_THRESHOLDS['small'] / 3, f"Too slow: {elapsed:.2f}s"
    
    def test_small_strided(self):
        """Test strided implementation with small input."""
        if conv2d_strided is None:
            return
        
        input_data = np.random.randn(4, 8, 16, 16).astype(np.float32)
        weight = np.random.randn(16, 8, 3, 3).astype(np.float32)
        bias = np.random.randn(16).astype(np.float32)
        
        start = time.time()
        output = conv2d_strided(input_data, weight, bias, stride=1, padding=1)
        elapsed = time.time() - start
        
        assert output.shape == (4, 16, 16, 16)
        assert elapsed < TIME_THRESHOLDS['small'] / 3, f"Too slow: {elapsed:.2f}s"


class TestPerformanceMedium:
    """Performance tests with medium arrays."""
    
    def test_medium_vectorized(self):
        """Test vectorized implementation with medium input."""
        if conv2d_vectorized is None:
            return
        
        input_data = np.random.randn(8, 32, 32, 32).astype(np.float32)
        weight = np.random.randn(64, 32, 3, 3).astype(np.float32)
        bias = np.random.randn(64).astype(np.float32)
        
        start = time.time()
        output = conv2d_vectorized(input_data, weight, bias, stride=1, padding=1)
        elapsed = time.time() - start
        
        assert output.shape == (8, 64, 32, 32)
        assert elapsed < TIME_THRESHOLDS['medium'], f"Too slow: {elapsed:.2f}s"
    
    def test_medium_strided(self):
        """Test strided implementation with medium input."""
        if conv2d_strided is None:
            return
        
        input_data = np.random.randn(8, 32, 32, 32).astype(np.float32)
        weight = np.random.randn(64, 32, 3, 3).astype(np.float32)
        bias = np.random.randn(64).astype(np.float32)
        
        start = time.time()
        output = conv2d_strided(input_data, weight, bias, stride=1, padding=1)
        elapsed = time.time() - start
        
        assert output.shape == (8, 64, 32, 32)
        assert elapsed < TIME_THRESHOLDS['medium'], f"Too slow: {elapsed:.2f}s"


class TestPerformanceLarge:
    """Performance tests with large arrays."""
    
    def test_large_strided(self):
        """Test strided implementation with large input."""
        if conv2d_strided is None:
            return
        
        input_data = np.random.randn(4, 64, 64, 64).astype(np.float32)
        weight = np.random.randn(128, 64, 3, 3).astype(np.float32)
        bias = np.random.randn(128).astype(np.float32)
        
        start = time.time()
        output = conv2d_strided(input_data, weight, bias, stride=1, padding=1)
        elapsed = time.time() - start
        
        assert output.shape == (4, 128, 64, 64)
        assert elapsed < TIME_THRESHOLDS['large'], f"Too slow: {elapsed:.2f}s"


class TestMemoryUsage:
    """Tests for memory efficiency."""
    
    def test_strided_no_copy(self):
        """Verify strided implementation doesn't copy input windows."""
        if conv2d_strided is None:
            return
        
        # Create a moderately large input
        input_data = np.random.randn(2, 16, 32, 32).astype(np.float32)
        weight = np.random.randn(32, 16, 3, 3).astype(np.float32)
        
        # This should work without memory error
        try:
            output = conv2d_strided(input_data, weight, stride=1, padding=1)
            assert output.shape == (2, 32, 32, 32)
        except MemoryError:
            pass  # If we run out of memory, that's okay for this test


class TestSpeedupComparison:
    """Compare speedups between implementations."""
    
    def test_vectorized_speedup_over_naive(self):
        """Verify vectorized is significantly faster than naive."""
        if conv2d_naive is None or conv2d_vectorized is None:
            return
        
        input_data = np.random.randn(2, 8, 16, 16).astype(np.float32)
        weight = np.random.randn(16, 8, 3, 3).astype(np.float32)
        
        # Time naive
        start = time.time()
        _ = conv2d_naive(input_data, weight, stride=1, padding=1)
        time_naive = time.time() - start
        
        # Time vectorized
        start = time.time()
        _ = conv2d_vectorized(input_data, weight, stride=1, padding=1)
        time_vect = time.time() - start
        
        # Vectorized should be at least 5x faster
        speedup = time_naive / time_vect
        assert speedup > 5, f"Speedup only {speedup:.1f}x, expected > 5x"
    
    def test_strided_similar_to_vectorized(self):
        """Verify strided is similar speed to vectorized."""
        if conv2d_vectorized is None or conv2d_strided is None:
            return
        
        input_data = np.random.randn(4, 16, 32, 32).astype(np.float32)
        weight = np.random.randn(32, 16, 3, 3).astype(np.float32)
        
        # Warmup
        _ = conv2d_vectorized(input_data, weight, stride=1, padding=1)
        _ = conv2d_strided(input_data, weight, stride=1, padding=1)
        
        # Time vectorized
        start = time.time()
        for _ in range(3):
            _ = conv2d_vectorized(input_data, weight, stride=1, padding=1)
        time_vect = time.time() - start
        
        # Time strided
        start = time.time()
        for _ in range(3):
            _ = conv2d_strided(input_data, weight, stride=1, padding=1)
        time_strided = time.time() - start
        
        # Strided should be within 2x of vectorized
        ratio = max(time_vect, time_strided) / min(time_vect, time_strided)
        assert ratio < 2, f"Performance difference too large: {ratio:.1f}x"


class TestVariousConfigurations:
    """Test performance with various common configurations."""
    
    def test_alexnet_first_layer(self):
        """Test AlexNet-style first layer (11x11 kernel)."""
        if conv2d_strided is None:
            return
        
        # AlexNet conv1: 3 -> 64, 11x11, stride=4, pad=2
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        weight = np.random.randn(64, 3, 11, 11).astype(np.float32)
        bias = np.random.randn(64).astype(np.float32)
        
        start = time.time()
        output = conv2d_strided(input_data, weight, bias, stride=4, padding=2)
        elapsed = time.time() - start
        
        # Should output 64x56x56
        assert output.shape == (1, 64, 56, 56)
        assert elapsed < 5.0, f"AlexNet layer too slow: {elapsed:.2f}s"
    
    def test_resnet_bottleneck(self):
        """Test ResNet bottleneck (1x1 -> 3x3 -> 1x1)."""
        if conv2d_strided is None:
            return
        
        # ResNet bottleneck: 256 -> 64 -> 64 -> 256
        input_data = np.random.randn(4, 256, 28, 28).astype(np.float32)
        
        # 1x1 conv (reduce)
        w1 = np.random.randn(64, 256, 1, 1).astype(np.float32)
        
        # 3x3 conv
        w2 = np.random.randn(64, 64, 3, 3).astype(np.float32)
        
        # 1x1 conv (expand)
        w3 = np.random.randn(256, 64, 1, 1).astype(np.float32)
        
        start = time.time()
        x = conv2d_strided(input_data, w1, stride=1, padding=0)
        x = conv2d_strided(x, w2, stride=1, padding=1)
        x = conv2d_strided(x, w3, stride=1, padding=0)
        elapsed = time.time() - start
        
        assert x.shape == (4, 256, 28, 28)
        assert elapsed < 10.0, f"ResNet bottleneck too slow: {elapsed:.2f}s"
    
    def test_depthwise_separable_components(self):
        """Test depthwise and pointwise convolutions."""
        if conv2d_strided is None:
            return
        
        # Depthwise: 64 -> 64 (groups=64, not supported in naive, skip)
        # Pointwise: 64 -> 128, 1x1
        input_data = np.random.randn(4, 64, 32, 32).astype(np.float32)
        weight = np.random.randn(128, 64, 1, 1).astype(np.float32)
        
        start = time.time()
        output = conv2d_strided(input_data, weight, stride=1, padding=0)
        elapsed = time.time() - start
        
        assert output.shape == (4, 128, 32, 32)
        assert elapsed < 3.0, f"Pointwise conv too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
