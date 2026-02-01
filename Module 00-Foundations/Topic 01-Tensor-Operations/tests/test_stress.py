"""
Topic 01: Stress Tests

Performance and memory tests for tensor operations.
Run with: pytest tests/test_stress.py -v

Note: These tests verify performance meets curriculum constraints.
"""

import numpy as np
import pytest
import time


class TestPerformanceBatched:
    """Performance tests for batched operations."""
    
    @pytest.mark.timeout(10)
    def test_large_batch_matmul(self):
        """Large batch matmul should complete in reasonable time."""
        # Simulate a typical forward pass
        batch_size = 64
        seq_len = 128
        hidden = 256
        
        a = np.random.randn(batch_size, seq_len, hidden)
        b = np.random.randn(batch_size, hidden, hidden)
        
        start = time.perf_counter()
        result = np.matmul(a, b)
        elapsed = time.perf_counter() - start
        
        assert result.shape == (batch_size, seq_len, hidden)
        # Should complete in under 1 second
        assert elapsed < 1.0, f"Took {elapsed:.2f}s, expected < 1.0s"
    
    @pytest.mark.timeout(30)
    def test_attention_scale_computation(self):
        """Attention-style computation at scale."""
        batch_size = 16
        num_heads = 8
        seq_len = 256
        head_dim = 64
        
        query = np.random.randn(batch_size, num_heads, seq_len, head_dim)
        key = np.random.randn(batch_size, num_heads, seq_len, head_dim)
        
        start = time.perf_counter()
        # Compute attention scores
        scores = np.matmul(query, np.transpose(key, (0, 1, 3, 2))) / np.sqrt(head_dim)
        elapsed = time.perf_counter() - start
        
        assert scores.shape == (batch_size, num_heads, seq_len, seq_len)
        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"Took {elapsed:.2f}s, expected < 5.0s"


class TestMemoryEfficiency:
    """Tests for memory-efficient implementations."""
    
    def test_broadcast_to_no_copy(self):
        """broadcast_to should not allocate new memory."""
        x = np.array([1, 2, 3])
        result = np.broadcast_to(x, (1000, 3))
        
        # broadcast_to creates a view, not a copy
        assert result.base is x or result.base.base is x or np.shares_memory(x, result)
        # Memory should be same as original
        assert result.nbytes > x.nbytes  # Logical size is larger
    
    def test_transpose_is_view(self):
        """Transpose should be a view."""
        x = np.random.randn(100, 200)
        x_t = x.T
        
        assert np.shares_memory(x, x_t)
    
    def test_reshape_is_view(self):
        """Reshape of contiguous array should be view."""
        x = np.random.randn(100, 200)
        y = x.reshape(200, 100)
        
        assert np.shares_memory(x, y)
    
    def test_slice_is_view(self):
        """Slicing should be a view."""
        x = np.random.randn(100, 200)
        y = x[::2, ::2]
        
        assert np.shares_memory(x, y)


class TestStridePerformance:
    """Tests for stride trick performance."""
    
    def test_sliding_window_memory(self):
        """Sliding window should not allocate proportionally to output size."""
        from numpy.lib.stride_tricks import as_strided
        
        x = np.arange(1000)
        window_size = 100
        
        # Create sliding window view
        output_len = len(x) - window_size + 1
        new_shape = (output_len, window_size)
        new_strides = (x.strides[0], x.strides[0])
        result = as_strided(x, shape=new_shape, strides=new_strides)
        
        # Verify it's a view
        assert np.shares_memory(x, result)
        
        # Logical size is much larger than actual memory
        logical_size = result.size * result.itemsize
        actual_size = x.nbytes
        assert logical_size > actual_size * 10  # At least 10x larger logically
    
    def test_im2col_style_operation(self):
        """Im2col-style operation for convolution."""
        # Simulate a 2D convolution setup
        H, W = 32, 32
        kH, kW = 3, 3
        
        image = np.random.randn(H, W)
        
        from numpy.lib.stride_tricks import as_strided
        
        out_H = H - kH + 1
        out_W = W - kW + 1
        
        shape = (out_H, out_W, kH, kW)
        strides = (image.strides[0], image.strides[1], 
                   image.strides[0], image.strides[1])
        
        patches = as_strided(image, shape=shape, strides=strides)
        
        assert patches.shape == (out_H, out_W, kH, kW)
        assert np.shares_memory(image, patches)


class TestBroadcastingPerformance:
    """Performance tests for broadcasting."""
    
    @pytest.mark.timeout(5)
    def test_large_broadcast_add(self):
        """Large broadcast should be efficient."""
        a = np.random.randn(1000, 1, 100)
        b = np.random.randn(1, 1000, 1)
        
        start = time.perf_counter()
        result = a + b
        elapsed = time.perf_counter() - start
        
        assert result.shape == (1000, 1000, 100)
        # Should complete quickly
        assert elapsed < 2.0, f"Took {elapsed:.2f}s"
    
    @pytest.mark.timeout(5)
    def test_batch_normalization_scale(self):
        """Batch norm at scale should be efficient."""
        x = np.random.randn(256, 1024)  # Large batch, many features
        
        start = time.perf_counter()
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        normalized = (x - mean) / (std + 1e-5)
        elapsed = time.perf_counter() - start
        
        assert normalized.shape == x.shape
        assert elapsed < 1.0, f"Took {elapsed:.2f}s"


class TestReductionPerformance:
    """Performance tests for reduction operations."""
    
    @pytest.mark.timeout(5)
    def test_large_sum(self):
        """Large sum should be efficient."""
        x = np.random.randn(10000, 1000)
        
        start = time.perf_counter()
        result = np.sum(x, axis=1)
        elapsed = time.perf_counter() - start
        
        assert result.shape == (10000,)
        assert elapsed < 1.0, f"Took {elapsed:.2f}s"
    
    @pytest.mark.timeout(5)
    def test_softmax_scale(self):
        """Softmax at scale."""
        x = np.random.randn(1000, 10000)
        
        def softmax(x, axis=-1):
            x_shifted = x - np.max(x, axis=axis, keepdims=True)
            exp_x = np.exp(x_shifted)
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
        start = time.perf_counter()
        result = softmax(x, axis=1)
        elapsed = time.perf_counter() - start
        
        assert result.shape == x.shape
        # Verify correctness
        np.testing.assert_allclose(result.sum(axis=1), np.ones(1000), rtol=1e-5)
        assert elapsed < 2.0, f"Took {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=60"])
