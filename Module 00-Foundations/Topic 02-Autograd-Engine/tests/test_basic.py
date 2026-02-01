"""
Topic 02: Basic Tests

Tests core autograd functionality.
Run with: pytest tests/test_basic.py -v
"""

import pytest
import math
import sys
from pathlib import Path

# Add solutions to path
sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

try:
    from level02_vectorized import Value
except ImportError:
    try:
        from level01_naive import Value
    except ImportError:
        # Use the utils escape hatch
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "utils"))
        from autograd_stub import Value


class TestForwardPass:
    """Test forward computation is correct."""
    
    def test_addition(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x + y
        assert z.data == 5.0
    
    def test_multiplication(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x * y
        assert z.data == 6.0
    
    def test_subtraction(self):
        x = Value(5.0)
        y = Value(3.0)
        z = x - y
        assert z.data == 2.0
    
    def test_negation(self):
        x = Value(3.0)
        z = -x
        assert z.data == -3.0
    
    def test_combined(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x * y + x
        assert z.data == 8.0


class TestBasicGradients:
    """Test gradient computation for basic operations."""
    
    def test_add_grad(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x + y
        z.backward()
        
        assert x.grad == 1.0
        assert y.grad == 1.0
    
    def test_mul_grad(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x * y
        z.backward()
        
        assert x.grad == 3.0  # dz/dx = y
        assert y.grad == 2.0  # dz/dy = x
    
    def test_combined_grad(self):
        # z = x * y + x = x * (y + 1)
        # dz/dx = y + 1 = 4
        # dz/dy = x = 2
        x = Value(2.0)
        y = Value(3.0)
        z = x * y + x
        z.backward()
        
        assert x.grad == 4.0
        assert y.grad == 2.0
    
    def test_squared_grad(self):
        # z = x * x = x^2
        # dz/dx = 2x = 6
        x = Value(3.0)
        z = x * x
        z.backward()
        
        assert x.grad == 6.0


class TestGradientAccumulation:
    """Test that gradients accumulate correctly when a value is used multiple times."""
    
    def test_same_variable_twice(self):
        x = Value(3.0)
        z = x + x  # z = 2x
        z.backward()
        # dz/dx = 2 (from both paths)
        assert x.grad == 2.0
    
    def test_diamond_graph(self):
        # Diamond: z uses a and b, both of which use x
        #      z
        #     / \
        #    a   b
        #     \ /
        #      x
        x = Value(2.0)
        a = x + 1  # a = 3
        b = x * 2  # b = 4
        z = a * b  # z = 12
        z.backward()
        
        # dz/dx = dz/da * da/dx + dz/db * db/dx
        #       = b * 1 + a * 2
        #       = 4 * 1 + 3 * 2
        #       = 4 + 6 = 10
        assert x.grad == 10.0


class TestChainRule:
    """Test proper chain rule application."""
    
    def test_nested_operations(self):
        # z = (x + y) * (x + y) = (x + y)^2
        x = Value(1.0)
        y = Value(2.0)
        a = x + y  # 3
        z = a * a  # 9
        z.backward()
        
        # dz/dx = 2 * (x + y) * 1 = 2 * 3 = 6
        # dz/dy = 2 * (x + y) * 1 = 2 * 3 = 6
        assert x.grad == 6.0
        assert y.grad == 6.0
    
    def test_three_level_chain(self):
        x = Value(2.0)
        a = x * 2   # 4
        b = a * 3   # 12
        c = b * 4   # 48
        c.backward()
        
        # dc/dx = 4 * 3 * 2 = 24
        assert x.grad == 24.0


class TestPower:
    """Test power operator if implemented."""
    
    def test_square(self):
        try:
            x = Value(3.0)
            z = x ** 2
            z.backward()
            # dz/dx = 2x = 6
            assert abs(x.grad - 6.0) < 1e-5
        except (TypeError, AttributeError):
            pytest.skip("Power operator not implemented")
    
    def test_sqrt(self):
        try:
            x = Value(4.0)
            z = x ** 0.5
            z.backward()
            # dz/dx = 0.5 * x^(-0.5) = 0.5 / 2 = 0.25
            assert abs(x.grad - 0.25) < 1e-5
        except (TypeError, AttributeError):
            pytest.skip("Power operator not implemented")


class TestActivationFunctions:
    """Test activation functions if implemented."""
    
    def test_relu_positive(self):
        try:
            x = Value(3.0)
            z = x.relu()
            z.backward()
            assert z.data == 3.0
            assert x.grad == 1.0
        except AttributeError:
            pytest.skip("ReLU not implemented")
    
    def test_relu_negative(self):
        try:
            x = Value(-3.0)
            z = x.relu()
            z.backward()
            assert z.data == 0.0
            assert x.grad == 0.0
        except AttributeError:
            pytest.skip("ReLU not implemented")
    
    def test_tanh(self):
        try:
            x = Value(0.0)
            z = x.tanh()
            z.backward()
            assert z.data == 0.0
            assert abs(x.grad - 1.0) < 1e-5  # tanh'(0) = 1
        except AttributeError:
            pytest.skip("tanh not implemented")
    
    def test_sigmoid(self):
        try:
            x = Value(0.0)
            z = x.sigmoid()
            z.backward()
            assert z.data == 0.5
            assert abs(x.grad - 0.25) < 1e-5  # sigmoid'(0) = 0.25
        except AttributeError:
            pytest.skip("sigmoid not implemented")


class TestZeroGrad:
    """Test gradient reset functionality."""
    
    def test_zero_grad(self):
        x = Value(2.0)
        z = x * x
        z.backward()
        assert x.grad == 4.0
        
        x.zero_grad()
        assert x.grad == 0.0
    
    def test_multiple_backward(self):
        x = Value(2.0)
        
        # First backward
        z1 = x * x
        z1.backward()
        grad1 = x.grad
        
        # Reset
        x.zero_grad()
        
        # Second backward (different expression)
        z2 = x + x
        z2.backward()
        
        assert x.grad == 2.0  # Should be fresh gradient


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
