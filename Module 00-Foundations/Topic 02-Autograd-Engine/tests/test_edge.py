"""
Topic 02: Edge Case Tests

Tests unusual inputs and boundary conditions for autograd.
Run with: pytest tests/test_edge.py -v
"""

import pytest
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions"))

try:
    from level02_vectorized import Value
except ImportError:
    try:
        from level01_naive import Value
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "utils"))
        from autograd_stub import Value


class TestScalarWrapping:
    """Test that raw numbers are handled correctly."""
    
    def test_add_int(self):
        x = Value(2.0)
        z = x + 3
        assert z.data == 5.0
        z.backward()
        assert x.grad == 1.0
    
    def test_radd_int(self):
        x = Value(2.0)
        z = 3 + x
        assert z.data == 5.0
        z.backward()
        assert x.grad == 1.0
    
    def test_mul_int(self):
        x = Value(2.0)
        z = x * 3
        assert z.data == 6.0
        z.backward()
        assert x.grad == 3.0
    
    def test_rmul_int(self):
        x = Value(2.0)
        z = 3 * x
        assert z.data == 6.0
        z.backward()
        assert x.grad == 3.0


class TestZeroValues:
    """Test operations with zero."""
    
    def test_zero_input(self):
        x = Value(0.0)
        y = Value(3.0)
        z = x * y
        z.backward()
        assert x.grad == 3.0
        assert y.grad == 0.0
    
    def test_add_zero(self):
        x = Value(5.0)
        z = x + 0
        assert z.data == 5.0
        z.backward()
        assert x.grad == 1.0
    
    def test_mul_zero(self):
        x = Value(5.0)
        z = x * 0
        assert z.data == 0.0
        z.backward()
        assert x.grad == 0.0


class TestNegativeValues:
    """Test operations with negative numbers."""
    
    def test_negative_input(self):
        x = Value(-3.0)
        y = Value(2.0)
        z = x * y
        z.backward()
        assert x.grad == 2.0
        assert y.grad == -3.0
    
    def test_negative_result(self):
        x = Value(3.0)
        z = -x * x  # -9
        z.backward()
        # d(-x^2)/dx = -2x = -6
        assert x.grad == -6.0


class TestComplexExpressions:
    """Test more complex expression patterns."""
    
    def test_nested_parentheses(self):
        x = Value(2.0)
        z = ((x + 1) * (x + 2)) * (x + 3)
        # z = (3)(4)(5) = 60
        assert z.data == 60.0
    
    def test_many_operations(self):
        x = Value(1.0)
        z = x + x + x + x + x  # 5x
        z.backward()
        assert x.grad == 5.0
    
    def test_alternating_ops(self):
        x = Value(2.0)
        z = x * 3 + x * 4  # 7x
        z.backward()
        assert x.grad == 7.0


class TestLargeGraphs:
    """Test with larger computational graphs."""
    
    def test_deep_chain(self):
        x = Value(1.0)
        z = x
        for _ in range(100):
            z = z * 2
        # z = 2^100 * x
        z.backward()
        # dz/dx = 2^100
        expected = 2**100
        assert abs(x.grad - expected) < expected * 1e-10
    
    def test_wide_graph(self):
        x = Value(1.0)
        outputs = [x * i for i in range(1, 11)]
        z = outputs[0]
        for o in outputs[1:]:
            z = z + o
        # z = x * (1 + 2 + ... + 10) = 55x
        z.backward()
        assert x.grad == 55.0


class TestNumericalStability:
    """Test edge cases for numerical stability."""
    
    def test_very_small_values(self):
        x = Value(1e-10)
        y = Value(1e-10)
        z = x * y
        z.backward()
        assert x.grad == 1e-10
    
    def test_very_large_values(self):
        x = Value(1e10)
        y = Value(2.0)
        z = x * y
        z.backward()
        assert x.grad == 2.0


class TestReluEdgeCases:
    """Test ReLU at boundaries."""
    
    def test_relu_exactly_zero(self):
        try:
            x = Value(0.0)
            z = x.relu()
            z.backward()
            assert z.data == 0.0
            # Gradient at 0 is typically 0
            assert x.grad == 0.0
        except AttributeError:
            pytest.skip("ReLU not implemented")
    
    def test_relu_tiny_positive(self):
        try:
            x = Value(1e-10)
            z = x.relu()
            z.backward()
            assert z.data == 1e-10
            assert x.grad == 1.0
        except AttributeError:
            pytest.skip("ReLU not implemented")
    
    def test_relu_tiny_negative(self):
        try:
            x = Value(-1e-10)
            z = x.relu()
            z.backward()
            assert z.data == 0.0
            assert x.grad == 0.0
        except AttributeError:
            pytest.skip("ReLU not implemented")


class TestExpLogEdgeCases:
    """Test exp and log edge cases."""
    
    def test_exp_large(self):
        try:
            x = Value(10.0)
            z = x.exp()
            z.backward()
            expected = math.exp(10.0)
            assert abs(z.data - expected) < expected * 1e-10
            assert abs(x.grad - expected) < expected * 1e-10
        except AttributeError:
            pytest.skip("exp not implemented")
    
    def test_log_one(self):
        try:
            x = Value(1.0)
            z = x.log()
            z.backward()
            assert z.data == 0.0
            assert x.grad == 1.0
        except AttributeError:
            pytest.skip("log not implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
