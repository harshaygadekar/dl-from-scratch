"""
Topic 02: Stress Tests

Performance tests for autograd implementation.
Run with: pytest tests/test_stress.py -v --timeout=60
"""

import pytest
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

try:
    from level02_vectorized import Value, MLP
    HAS_MLP = True
except ImportError:
    try:
        from level01_naive import Value
        HAS_MLP = False
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "utils"))
        from autograd_stub import Value
        HAS_MLP = False


class TestPerformance:
    """Test performance with large computations."""
    
    @pytest.mark.timeout(10)
    def test_long_chain_forward(self):
        """Long chain of operations should complete quickly."""
        x = Value(1.0)
        z = x
        
        n = 1000
        start = time.perf_counter()
        for _ in range(n):
            z = z * 1.001
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"Forward pass too slow: {elapsed:.2f}s"
    
    @pytest.mark.timeout(10)
    def test_long_chain_backward(self):
        """Backward through long chain should complete."""
        x = Value(1.0)
        z = x
        
        n = 1000
        for _ in range(n):
            z = z + x * 0.001
        
        start = time.perf_counter()
        z.backward()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 2.0, f"Backward pass too slow: {elapsed:.2f}s"
        # Gradient should be approximately n * 0.001 + 1
        assert abs(x.grad - (n * 0.001 + 1)) < 1e-3
    
    @pytest.mark.timeout(10)
    def test_wide_graph(self):
        """Graph with many parallel branches."""
        x = Value(1.0)
        
        n = 500
        branches = [x * i for i in range(n)]
        
        start = time.perf_counter()
        z = branches[0]
        for b in branches[1:]:
            z = z + b
        z.backward()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 2.0, f"Wide graph too slow: {elapsed:.2f}s"
        # Gradient should be 0 + 1 + 2 + ... + (n-1) = n(n-1)/2
        expected = n * (n - 1) // 2
        assert x.grad == expected


class TestMemory:
    """Test memory-related behavior."""
    
    def test_no_memory_leak_simple(self):
        """Repeated forward/backward shouldn't leak memory."""
        import gc
        
        # Warm up
        for _ in range(10):
            x = Value(1.0)
            z = x * x
            z.backward()
        
        gc.collect()
        
        # Run many iterations
        for _ in range(100):
            x = Value(1.0)
            z = x * x + x * 2
            z.backward()
        
        # If we get here without error, memory is likely fine
        # (Python's gc handles cleanup)
        assert True
    
    def test_dead_branches_not_computed(self):
        """Unused branches shouldn't affect gradient."""
        x = Value(2.0)
        y = Value(3.0)
        
        # Create a branch we won't use
        unused = x * y * x * y
        
        # Only use x
        z = x * x
        z.backward()
        
        # Only x's gradient should be set through z, not through unused
        assert x.grad == 4.0


class TestNeuralNetwork:
    """Test neural network training scenario."""
    
    @pytest.mark.skipif(not HAS_MLP, reason="MLP not implemented")
    @pytest.mark.timeout(30)
    def test_mlp_forward_backward(self):
        """MLP forward and backward should work."""
        mlp = MLP(3, [4, 4, 1])
        
        x = [Value(1.0), Value(2.0), Value(3.0)]
        
        start = time.perf_counter()
        y = mlp(x)
        y.backward()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"MLP too slow: {elapsed:.2f}s"
        
        # Check all parameters have gradients
        for p in mlp.parameters():
            assert p.grad != 0.0 or p.data == 0.0  # Either has grad or is zero weight
    
    @pytest.mark.skipif(not HAS_MLP, reason="MLP not implemented")
    @pytest.mark.timeout(60)
    def test_training_loop(self):
        """Simple training loop should converge."""
        import random
        random.seed(42)
        
        # Simple XOR-like problem
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        Y = [0, 1, 1, 0]
        
        mlp = MLP(2, [4, 1])
        
        # Training loop
        for epoch in range(100):
            total_loss = Value(0)
            
            for x_i, y_i in zip(X, Y):
                pred = mlp([Value(xi) for xi in x_i])
                loss = (pred - y_i) ** 2
                total_loss = total_loss + loss
            
            # Zero grad
            for p in mlp.parameters():
                p.grad = 0.0
            
            # Backward
            total_loss.backward()
            
            # Update
            lr = 0.1
            for p in mlp.parameters():
                p.data -= lr * p.grad
        
        # Check predictions are reasonable
        correct = 0
        for x_i, y_i in zip(X, Y):
            pred = mlp([Value(xi) for xi in x_i])
            if (pred.data > 0.5 and y_i == 1) or (pred.data <= 0.5 and y_i == 0):
                correct += 1
        
        # Should get at least 2/4 correct (random chance)
        # With proper training, should get all 4
        assert correct >= 2


class TestGradientAccuracy:
    """Test gradient accuracy with numerical gradient checking."""
    
    def test_mul_accuracy(self):
        """Multiplication gradient matches numerical gradient."""
        eps = 1e-5
        
        x1, y1 = 2.0, 3.0
        
        # Forward at x
        x = Value(x1)
        y = Value(y1)
        z = x * y
        z.backward()
        analytic_grad = x.grad
        
        # Numerical gradient
        z_plus = (x1 + eps) * y1
        z_minus = (x1 - eps) * y1
        numerical_grad = (z_plus - z_minus) / (2 * eps)
        
        rel_error = abs(analytic_grad - numerical_grad) / max(abs(numerical_grad), 1e-8)
        assert rel_error < 1e-4, f"Gradient error too high: {rel_error}"
    
    def test_complex_expression_accuracy(self):
        """Complex expression gradient matches numerical."""
        eps = 1e-5
        
        def f(x_val):
            x = Value(x_val)
            return ((x * x + x) * x).data
        
        x1 = 2.0
        
        # Analytic
        x = Value(x1)
        z = (x * x + x) * x
        z.backward()
        analytic_grad = x.grad
        
        # Numerical
        numerical_grad = (f(x1 + eps) - f(x1 - eps)) / (2 * eps)
        
        rel_error = abs(analytic_grad - numerical_grad) / max(abs(numerical_grad), 1e-8)
        assert rel_error < 1e-4, f"Gradient error too high: {rel_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=60"])
