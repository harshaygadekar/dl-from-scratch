"""
Topic 03: Basic Tests

Tests core optimizer functionality.
Run with: pytest tests/test_basic.py -v
"""

import pytest
import math
import sys
from pathlib import Path

# Add solutions to path
sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

try:
    from level02_vectorized import SGD, Adam, AdamW, RMSprop, Value
except ImportError:
    try:
        from level01_naive import SGD, Value
        Adam = None
        AdamW = None
        RMSprop = None
    except ImportError:
        pytest.skip("Could not import optimizers", allow_module_level=True)


class TestSGD:
    """Test SGD optimizer."""
    
    def test_basic_update(self):
        """SGD should decrease parameter toward minimum."""
        x = Value(5.0)
        opt = SGD([x], lr=0.1)
        
        for _ in range(10):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        assert abs(x.data) < 1.0  # Should be moving toward 0
    
    def test_convergence(self):
        """SGD should converge to minimum."""
        x = Value(5.0)
        opt = SGD([x], lr=0.1)
        
        for _ in range(100):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        assert abs(x.data) < 0.01
    
    def test_multiple_params(self):
        """SGD should update multiple parameters."""
        x = Value(3.0)
        y = Value(4.0)
        opt = SGD([x, y], lr=0.1)
        
        for _ in range(100):
            loss = x ** 2 + y ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        assert abs(x.data) < 0.1
        assert abs(y.data) < 0.1
    
    def test_zero_grad(self):
        """zero_grad should reset gradients."""
        x = Value(3.0)
        opt = SGD([x], lr=0.1)
        
        loss = x ** 2
        loss.backward()
        assert x.grad != 0
        
        opt.zero_grad()
        assert x.grad == 0


@pytest.mark.skipif(Adam is None, reason="Adam not implemented")
class TestAdam:
    """Test Adam optimizer."""
    
    def test_basic_update(self):
        """Adam should decrease parameter toward minimum."""
        x = Value(5.0)
        opt = Adam([x], lr=0.5)
        
        for _ in range(10):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        assert abs(x.data) < 3.0
    
    def test_convergence(self):
        """Adam should converge to minimum."""
        x = Value(5.0)
        opt = Adam([x], lr=0.5)
        
        for _ in range(100):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        assert abs(x.data) < 0.01
    
    def test_bias_correction(self):
        """Adam should work correctly from step 1."""
        x = Value(5.0)
        opt = Adam([x], lr=0.5)
        
        # First step should already make progress
        loss = x ** 2
        opt.zero_grad()
        loss.backward()
        
        initial = x.data
        opt.step()
        
        assert x.data < initial  # Should move toward 0
        assert abs(x.data - initial) > 0.1  # Should make meaningful progress


@pytest.mark.skipif(AdamW is None, reason="AdamW not implemented")
class TestAdamW:
    """Test AdamW optimizer."""
    
    def test_weight_decay(self):
        """AdamW should shrink parameters with weight decay."""
        x = Value(5.0)
        opt = AdamW([x], lr=0.1, weight_decay=0.5)
        
        # With strong weight decay, x should decrease even without loss gradient
        x.grad = 0.0
        initial = x.data
        opt.step()
        
        # Weight decay alone should shrink x
        assert x.data < initial
    
    def test_convergence_with_decay(self):
        """AdamW should still converge with weight decay."""
        x = Value(5.0)
        opt = AdamW([x], lr=0.5, weight_decay=0.01)
        
        for _ in range(100):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        assert abs(x.data) < 0.1


class TestMomentum:
    """Test momentum in SGD."""
    
    def test_momentum_accumulates(self):
        """Momentum should accumulate velocity."""
        x = Value(5.0)
        opt = SGD([x], lr=0.1, momentum=0.9)
        
        # Run a few steps
        for _ in range(5):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Momentum should have built up velocity
        assert hasattr(opt, 'velocity')
        assert opt.velocity[0] != 0
    
    def test_faster_than_vanilla(self):
        """Momentum should converge faster than vanilla SGD."""
        x1 = Value(5.0)
        x2 = Value(5.0)
        
        vanilla = SGD([x1], lr=0.1, momentum=0.0)
        momentum = SGD([x2], lr=0.1, momentum=0.9)
        
        for _ in range(20):
            loss1 = x1 ** 2
            vanilla.zero_grad()
            loss1.backward()
            vanilla.step()
            
            loss2 = x2 ** 2
            momentum.zero_grad()
            loss2.backward()
            momentum.step()
        
        # Momentum should converge faster
        assert abs(x2.data) < abs(x1.data)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_zero_gradient(self):
        """Optimizer should handle zero gradients."""
        x = Value(0.0)
        opt = SGD([x], lr=0.1)
        
        # At x=0, gradient of x² is 0
        loss = x ** 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        assert x.data == 0.0  # Should not change
    
    def test_empty_params(self):
        """Optimizer should handle empty parameter list."""
        opt = SGD([], lr=0.1)
        opt.zero_grad()
        opt.step()  # Should not error
    
    def test_large_learning_rate(self):
        """Large LR might oscillate but shouldn't crash."""
        x = Value(1.0)
        opt = SGD([x], lr=10.0)
        
        for _ in range(5):
            loss = x ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Should not be NaN or inf
        assert math.isfinite(x.data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
