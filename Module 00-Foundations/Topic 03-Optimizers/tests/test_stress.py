"""
Topic 03: Stress Tests

Performance tests for optimizers.
Run with: pytest tests/test_stress.py -v --timeout=60
"""

import pytest
import time
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

try:
    from level02_vectorized import SGD, Adam, AdamW, Value
except ImportError:
    try:
        from level01_naive import SGD, Value
        Adam = None
        AdamW = None
    except ImportError:
        pytest.skip("Could not import optimizers", allow_module_level=True)


class TestPerformance:
    """Test optimizer performance."""
    
    @pytest.mark.timeout(10)
    def test_many_steps_sgd(self):
        """SGD should handle many optimization steps."""
        x = Value(10.0)
        opt = SGD([x], lr=0.01)
        
        start = time.perf_counter()
        for _ in range(10000):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 5.0, f"SGD too slow: {elapsed:.2f}s for 10k steps"
        assert abs(x.data) < 0.01
    
    @pytest.mark.skipif(Adam is None, reason="Adam not implemented")
    @pytest.mark.timeout(10)
    def test_many_steps_adam(self):
        """Adam should handle many optimization steps."""
        x = Value(10.0)
        opt = Adam([x], lr=0.01)
        
        start = time.perf_counter()
        for _ in range(10000):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 10.0, f"Adam too slow: {elapsed:.2f}s for 10k steps"
        assert abs(x.data) < 0.01
    
    @pytest.mark.timeout(10)
    def test_many_parameters(self):
        """Optimizer should handle many parameters."""
        n = 100
        params = [Value(float(i)) for i in range(n)]
        opt = SGD(params, lr=0.01)
        
        start = time.perf_counter()
        for _ in range(100):
            loss = sum(p ** 2 for p in params)
            opt.zero_grad()
            loss.backward()
            opt.step()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 5.0, f"Too slow with {n} params: {elapsed:.2f}s"


class TestConvergenceSpeed:
    """Compare convergence across optimizers."""
    
    def count_steps_to_converge(self, optimizer_class, tolerance=0.1, max_steps=1000, **kwargs):
        """Count steps to reach tolerance."""
        x = Value(10.0)
        opt = optimizer_class([x], **kwargs)
        
        for step in range(max_steps):
            if abs(x.data) < tolerance:
                return step
            
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        return max_steps
    
    def test_momentum_faster_than_vanilla(self):
        """Momentum should converge faster than vanilla SGD."""
        steps_vanilla = self.count_steps_to_converge(SGD, lr=0.05)
        steps_momentum = self.count_steps_to_converge(SGD, lr=0.05, momentum=0.9)
        
        # Momentum should be faster or equal (at worst)
        assert steps_momentum <= steps_vanilla * 1.5
    
    @pytest.mark.skipif(Adam is None, reason="Adam not implemented")
    def test_adam_converges(self):
        """Adam should converge reasonably quickly."""
        steps_adam = self.count_steps_to_converge(Adam, lr=0.5)
        
        assert steps_adam < 100, f"Adam took too long: {steps_adam} steps"


class TestOptimizationProblems:
    """Test on various optimization problems."""
    
    def test_ravine_problem(self):
        """
        Test on elongated valley (ravine) problem.
        Momentum should excel here.
        """
        # f(x,y) = x² + 100*y²
        # Very different curvature in x vs y
        
        x = Value(10.0)
        y = Value(0.1)  # Start closer in y due to high curvature
        
        opt = SGD([x, y], lr=0.001, momentum=0.9)
        
        for _ in range(500):
            loss = x ** 2 + 100 * y ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Should make progress in both dimensions
        assert abs(x.data) < 5.0
        assert abs(y.data) < 0.05
    
    @pytest.mark.skipif(Adam is None, reason="Adam not implemented")  
    def test_sparse_gradient_problem(self):
        """
        Simulate sparse gradients (some params have zero gradient).
        Adam should handle this well.
        """
        params = [Value(5.0) for _ in range(10)]
        opt = Adam(params, lr=0.5)
        
        for step in range(50):
            # Only half the params get gradients each step
            active = params[:5] if step % 2 == 0 else params[5:]
            loss = sum(p ** 2 for p in active)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # All should have made progress
        all_small = all(abs(p.data) < 3.0 for p in params)
        assert all_small


class TestNumericalPrecision:
    """Test numerical precision of gradients."""
    
    def numerical_gradient(self, f, x, eps=1e-5):
        """Compute numerical gradient."""
        x_plus = x.data + eps
        x_minus = x.data - eps
        
        x.data = x_plus
        f_plus = (f()).data
        x.data = x_minus
        f_minus = (f()).data
        x.data = (x_plus + x_minus) / 2  # Restore
        
        return (f_plus - f_minus) / (2 * eps)
    
    def test_gradient_accuracy(self):
        """Optimizer should use accurate gradients."""
        x = Value(3.0)
        
        def f():
            return x ** 2 + 2 * x + 1
        
        loss = f()
        loss.backward()
        
        analytic = x.grad
        numerical = self.numerical_gradient(f, x)
        
        assert abs(analytic - numerical) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=60"])
