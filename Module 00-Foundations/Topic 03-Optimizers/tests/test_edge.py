"""
Topic 03: Edge Case Tests

Tests unusual inputs and boundary conditions.
Run with: pytest tests/test_edge.py -v
"""

import pytest
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


class TestNumericalStability:
    """Test numerical edge cases."""
    
    def test_very_small_gradient(self):
        """Handle very small gradients."""
        x = Value(1e-10)
        opt = SGD([x], lr=0.1)
        
        loss = x ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        assert math.isfinite(x.data)
    
    def test_very_large_gradient(self):
        """Handle very large gradients."""
        x = Value(1e10)
        opt = SGD([x], lr=1e-12)
        
        loss = x ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        assert math.isfinite(x.data)
    
    def test_negative_values(self):
        """Optimize with negative starting values."""
        x = Value(-5.0)
        opt = SGD([x], lr=0.1)
        
        for _ in range(100):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        assert abs(x.data) < 0.1


@pytest.mark.skipif(Adam is None, reason="Adam not implemented")
class TestAdamEdgeCases:
    """Adam-specific edge cases."""
    
    def test_first_step(self):
        """Adam should work correctly on first step (bias correction)."""
        x = Value(5.0)
        opt = Adam([x], lr=1.0)
        
        loss = x ** 2
        opt.zero_grad()
        loss.backward()
        
        initial = x.data
        opt.step()
        
        # Should make reasonable progress, not tiny or huge
        delta = abs(x.data - initial)
        assert 0.1 < delta < 5.0
    
    def test_consistent_gradient(self):
        """Adam should adapt to consistent gradients."""
        x = Value(10.0)
        opt = Adam([x], lr=0.1)
        
        # Many steps with same sign gradient
        for _ in range(50):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        assert x.data < 1.0  # Should converge
    
    def test_oscillating_gradient(self):
        """Adam should handle oscillating gradients."""
        x = Value(0.0)
        opt = Adam([x], lr=0.1)
        
        for i in range(20):
            # Alternating target
            target = 1.0 if i % 2 == 0 else -1.0
            loss = (x - target) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Should stay bounded
        assert abs(x.data) < 2.0


class TestLearningRates:
    """Test various learning rate scenarios."""
    
    def test_zero_lr(self):
        """Zero learning rate should not change parameter."""
        x = Value(5.0)
        opt = SGD([x], lr=0.0)
        
        loss = x ** 2
        opt.zero_grad()
        loss.backward()
        
        initial = x.data
        opt.step()
        
        assert x.data == initial
    
    def test_very_small_lr(self):
        """Very small LR should still make progress."""
        x = Value(5.0)
        opt = SGD([x], lr=1e-6)
        
        for _ in range(1000):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Should make at least some progress
        assert x.data < 5.0


class TestStateManagement:
    """Test optimizer state."""
    
    def test_state_persistence(self):
        """Optimizer state should persist across steps."""
        x = Value(5.0)
        opt = SGD([x], lr=0.1, momentum=0.9)
        
        for _ in range(5):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        velocity_after_5 = opt.velocity[0]
        
        for _ in range(5):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Velocity should have continued to evolve
        assert opt.velocity[0] != velocity_after_5
    
    @pytest.mark.skipif(Adam is None, reason="Adam not implemented")
    def test_adam_timestep(self):
        """Adam timestep should increment."""
        x = Value(5.0)
        opt = Adam([x], lr=0.1)
        
        assert opt.t == 0
        
        loss = x ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        assert opt.t == 1
        
        loss = x ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        assert opt.t == 2


class TestMultipleParameters:
    """Test with multiple parameters."""
    
    def test_independent_updates(self):
        """Parameters should update independently."""
        x = Value(5.0)
        y = Value(0.0)  # y is already at optimum
        
        opt = SGD([x, y], lr=0.1)
        
        for _ in range(10):
            loss = x ** 2 + y ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # x should move, y should stay near 0
        assert abs(x.data) < 5.0
        assert abs(y.data) < 0.01
    
    def test_different_gradients(self):
        """Parameters with different gradients should update differently."""
        x = Value(1.0)
        y = Value(10.0)
        
        opt = SGD([x, y], lr=0.01)
        
        loss = x ** 2 + y ** 2
        opt.zero_grad()
        loss.backward()
        
        grad_x = x.grad
        grad_y = y.grad
        
        # y has larger gradient
        assert abs(grad_y) > abs(grad_x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
