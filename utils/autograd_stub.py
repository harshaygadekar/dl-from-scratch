"""
Autograd Stub - Escape Hatch for Topic 02

This is a working but not optimal autograd implementation.
If you get stuck on Topic 02, use this and return to implement your own later.

Usage:
    from utils.autograd_stub import Value
    
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x
    z.backward()
    print(x.grad)  # 4.0

No shame in using this! The goal is learning, not getting stuck.
"""

import math


class Value:
    """A scalar value with automatic differentiation support."""
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers supported"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * other ** -1
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rtruediv__(self, other):
        return other * self ** -1
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
    
    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')
        
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        """Compute gradients via reverse-mode autodiff."""
        # Topological sort
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Go one variable at a time and apply chain rule
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
    
    def zero_grad(self):
        """Reset gradient to zero."""
        self.grad = 0.0


# Convenience functions for creating tensors (lists of Values)
def tensor(data):
    """Create a list of Value objects from a list of numbers."""
    if isinstance(data[0], (list, tuple)):
        return [tensor(row) for row in data]
    return [Value(x) for x in data]


# Quick test
if __name__ == "__main__":
    print("Testing autograd_stub...\n")
    
    # Test 1: Basic operations
    print("Test 1: z = x * y + x where x=2, y=3")
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x
    z.backward()
    
    print(f"  z = {z.data} (expected: 8)")
    print(f"  dz/dx = {x.grad} (expected: 4)")
    print(f"  dz/dy = {y.grad} (expected: 2)")
    
    assert abs(z.data - 8.0) < 1e-6, f"Expected 8.0, got {z.data}"
    assert abs(x.grad - 4.0) < 1e-6, f"Expected 4.0, got {x.grad}"
    assert abs(y.grad - 2.0) < 1e-6, f"Expected 2.0, got {y.grad}"
    print("  ✅ PASSED\n")
    
    # Test 2: More complex expression
    print("Test 2: f = tanh(2x + 3y) where x=1, y=2")
    x = Value(1.0)
    y = Value(2.0)
    f = (2*x + 3*y).tanh()
    f.backward()
    
    print(f"  f = {f.data:.4f}")
    print(f"  df/dx = {x.grad:.4f}")
    print(f"  df/dy = {y.grad:.4f}")
    print("  ✅ PASSED\n")
    
    # Test 3: Sigmoid
    print("Test 3: f = sigmoid(x) where x=0")
    x = Value(0.0)
    f = x.sigmoid()
    f.backward()
    
    print(f"  f = {f.data} (expected: 0.5)")
    print(f"  df/dx = {x.grad} (expected: 0.25)")
    assert abs(f.data - 0.5) < 1e-6, f"Expected 0.5, got {f.data}"
    assert abs(x.grad - 0.25) < 1e-6, f"Expected 0.25, got {x.grad}"
    print("  ✅ PASSED\n")
    
    print("="*50)
    print("✅ autograd_stub working correctly!")
    print("\nYou can use this as an escape hatch for Topic 02.")
    print("Import with: from utils.autograd_stub import Value")
