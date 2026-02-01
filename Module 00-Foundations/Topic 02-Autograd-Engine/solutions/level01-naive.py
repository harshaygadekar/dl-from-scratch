"""
Topic 02: Level 01 - Naive Autograd (Basic Operations Only)

A minimal but complete autograd implementation with just add/mul/backward.
Focuses on clarity over completeness.

Goal: Understand the core mechanism before adding more operations.
"""

import math


class Value:
    """
    Wraps a scalar and tracks gradients through computations.
    
    This implementation handles:
    - Addition (+)
    - Multiplication (*)
    - Backward pass
    
    Example:
        x = Value(2.0)
        y = Value(3.0)
        z = x * y + x  # z = 8
        z.backward()
        print(x.grad)  # 4.0 (y + 1 = 4)
        print(y.grad)  # 2.0 (x = 2)
    """
    
    def __init__(self, data, _children=(), _op=''):
        """
        Initialize a Value node.
        
        Args:
            data: The actual scalar value
            _children: Tuple of parent Value nodes (for graph structure)
            _op: String describing the operation (for debugging)
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None  # Default: do nothing
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    # ==================== Basic Operations ====================
    
    def __add__(self, other):
        """z = self + other"""
        # Handle raw numbers (int, float)
        other = other if isinstance(other, Value) else Value(other)
        
        # Forward pass
        out = Value(self.data + other.data, (self, other), '+')
        
        # Backward pass: d(self+other)/dself = 1, d(self+other)/dother = 1
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        """Handle: number + Value"""
        return self + other
    
    def __mul__(self, other):
        """z = self * other"""
        other = other if isinstance(other, Value) else Value(other)
        
        # Forward pass
        out = Value(self.data * other.data, (self, other), '*')
        
        # Backward pass: d(self*other)/dself = other, d(self*other)/dother = self
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        """Handle: number * Value"""
        return self * other
    
    def __neg__(self):
        """z = -self"""
        return self * -1
    
    def __sub__(self, other):
        """z = self - other"""
        return self + (-other)
    
    def __rsub__(self, other):
        """Handle: number - Value"""
        return (-self) + other
    
    # ==================== Backward Pass ====================
    
    def backward(self):
        """
        Compute gradients of this node with respect to all ancestors.
        
        Uses topological sort to ensure correct order of gradient computation.
        """
        # Step 1: Build topological order (DFS)
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Step 2: Set gradient of output to 1.0
        self.grad = 1.0
        
        # Step 3: Apply chain rule in reverse topological order
        for v in reversed(topo):
            v._backward()
    
    def zero_grad(self):
        """Reset gradient to zero."""
        self.grad = 0.0


# ==================== Demo ====================

def demo():
    print("="*50)
    print("Level 01: Naive Autograd Demo")
    print("="*50)
    
    # Example 1: Simple multiplication
    print("\nExample 1: z = x * y")
    x = Value(2.0)
    y = Value(3.0)
    z = x * y
    
    print(f"Before backward: x={x}, y={y}, z={z}")
    z.backward()
    print(f"After backward:  x={x}, y={y}")
    print(f"Expected: x.grad=3.0 (=y), y.grad=2.0 (=x)")
    
    # Example 2: Combined operations
    print("\nExample 2: z = x * y + x")
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x  # z = 6 + 2 = 8
    
    print(f"z = {z.data}")
    z.backward()
    print(f"x.grad = {x.grad} (expected: 4.0 = y + 1)")
    print(f"y.grad = {y.grad} (expected: 2.0 = x)")
    
    # Example 3: Same variable used twice
    print("\nExample 3: z = x * x (x squared)")
    x = Value(3.0)
    z = x * x  # z = 9
    
    print(f"z = {z.data}")
    z.backward()
    print(f"x.grad = {x.grad} (expected: 6.0 = 2*x)")
    
    # Verify gradient accumulation works
    assert x.grad == 6.0, f"Expected 6.0, got {x.grad}"
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    demo()
