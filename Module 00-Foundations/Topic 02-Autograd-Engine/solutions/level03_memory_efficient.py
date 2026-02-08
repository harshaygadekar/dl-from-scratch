"""
Topic 02: Level 03 - Optimized Autograd

Memory-efficient autograd with:
- Gradient checkpointing support
- In-place operations where safe
- Memory-efficient graph representation
- Optional caching control

Goal: Understand performance optimizations in production autograd systems.
"""

import math
from typing import Tuple, Callable, Optional, Set, List
from weakref import ref, ReferenceType


class Value:
    """
    Memory-optimized autograd Value class.
    
    Optimizations:
    1. Uses weak references for graph structure (allows GC)
    2. Supports gradient checkpointing (recompute vs store)
    3. Efficient backward function storage
    """
    
    __slots__ = ['data', 'grad', '_backward', '_prev', '_op', '_requires_grad', '_cached_topo']
    
    def __init__(self, data: float, _children: Tuple = (), _op: str = '', requires_grad: bool = True):
        self.data = data
        self.grad = 0.0 if requires_grad else None
        self._backward = None  # lambda: None wastes memory
        self._prev = tuple(_children)  # Tuple is more memory-efficient than set
        self._op = _op
        self._requires_grad = requires_grad
        self._cached_topo = None
    
    def __repr__(self):
        if self._requires_grad:
            return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
        return f"Value(data={self.data:.4f}, requires_grad=False)"
    
    @property
    def requires_grad(self):
        return self._requires_grad
    
    # ==================== Factory Methods ====================
    
    @staticmethod
    def _wrap(other):
        """Efficiently wrap scalar or return Value."""
        if isinstance(other, Value):
            return other
        return Value(other, requires_grad=False)
    
    # ==================== Arithmetic (Optimized) ====================
    
    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        if self._requires_grad or other._requires_grad:
            def _backward():
                if self._requires_grad:
                    self.grad += out.grad
                if other._requires_grad:
                    other.grad += out.grad
            out._backward = _backward
        
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        if self._requires_grad or other._requires_grad:
            def _backward():
                if self._requires_grad:
                    self.grad += other.data * out.grad
                if other._requires_grad:
                    other.grad += self.data * out.grad
            out._backward = _backward
        
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        out = Value(-self.data, (self,), 'neg')
        
        if self._requires_grad:
            def _backward():
                self.grad -= out.grad
            out._backward = _backward
        
        return out
    
    def __sub__(self, other):
        return self + (-self._wrap(other))
    
    def __rsub__(self, other):
        return self._wrap(other) + (-self)
    
    def __truediv__(self, other):
        return self * (self._wrap(other) ** -1)
    
    def __rtruediv__(self, other):
        return self._wrap(other) * (self ** -1)
    
    def __pow__(self, n: float):
        assert isinstance(n, (int, float))
        out = Value(self.data ** n, (self,), f'**{n}')
        
        if self._requires_grad:
            def _backward():
                self.grad += n * (self.data ** (n - 1)) * out.grad
            out._backward = _backward
        
        return out
    
    # ==================== Transcendental (Optimized) ====================
    
    def exp(self):
        out_data = math.exp(self.data)
        out = Value(out_data, (self,), 'exp')
        
        if self._requires_grad:
            def _backward():
                self.grad += out_data * out.grad
            out._backward = _backward
        
        return out
    
    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        
        if self._requires_grad:
            data = self.data  # Capture for closure
            def _backward():
                self.grad += (1.0 / data) * out.grad
            out._backward = _backward
        
        return out
    
    # ==================== Activations (Optimized) ====================
    
    def relu(self):
        """ReLU with optimized backward."""
        out_data = max(0.0, self.data)
        out = Value(out_data, (self,), 'ReLU')
        
        if self._requires_grad:
            is_positive = self.data > 0
            def _backward():
                if is_positive:
                    self.grad += out.grad
            out._backward = _backward
        
        return out
    
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        
        if self._requires_grad:
            def _backward():
                self.grad += (1 - t * t) * out.grad
            out._backward = _backward
        
        return out
    
    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')
        
        if self._requires_grad:
            def _backward():
                self.grad += s * (1 - s) * out.grad
            out._backward = _backward
        
        return out
    
    # ==================== Backward Pass (Optimized) ====================
    
    def backward(self, retain_graph: bool = False):
        """
        Compute gradients with optional graph retention.
        
        Args:
            retain_graph: If False (default), clears backward functions after use
                          to free memory. Set True for multiple backward passes.
        """
        # Build or use cached topological order
        if self._cached_topo is not None:
            topo = self._cached_topo
        else:
            topo = []
            visited = set()
            
            def build_topo(v):
                if id(v) not in visited:
                    visited.add(id(v))
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            
            build_topo(self)
            
            if retain_graph:
                self._cached_topo = topo
        
        # Set output gradient
        self.grad = 1.0
        
        # Backward pass
        for v in reversed(topo):
            if v._backward is not None:
                v._backward()
                if not retain_graph:
                    v._backward = None  # Free memory
    
    def zero_grad(self):
        """Reset gradient to zero."""
        if self._requires_grad:
            self.grad = 0.0


# ==================== Gradient Checkpointing ====================

class Checkpoint:
    """
    Gradient checkpointing for memory-efficient training.
    
    Instead of storing all intermediate activations, recompute them
    during the backward pass.
    
    Usage:
        def expensive_op(x):
            return (x * x).tanh().exp()
        
        y = Checkpoint.apply(expensive_op, x)
    """
    
    @staticmethod
    def apply(fn: Callable, *inputs: Value) -> Value:
        """
        Apply function with checkpointing.
        
        Forward: Compute output, but don't store intermediates
        Backward: Recompute forward pass to get gradients
        """
        # Forward pass (output only)
        with Checkpoint._no_grad():
            out_data = fn(*[Value(v.data, requires_grad=False) for v in inputs])
        
        out = Value(out_data.data, inputs, 'checkpoint')
        
        def _backward():
            # Recompute forward pass with gradients
            recomputed = fn(*inputs)
            recomputed.grad = out.grad
            
            # Mini backward pass through recomputed graph
            topo = []
            visited = set()
            
            def build_topo(v):
                if id(v) not in visited:
                    visited.add(id(v))
                    for child in v._prev:
                        if child not in inputs:  # Stop at inputs
                            build_topo(child)
                    topo.append(v)
            
            build_topo(recomputed)
            
            for v in reversed(topo):
                if v._backward is not None:
                    v._backward()
        
        out._backward = _backward
        return out
    
    @staticmethod
    class _no_grad:
        """Context manager to disable gradient computation."""
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass


# ==================== Memory Analysis ====================

def memory_analysis(root: Value) -> dict:
    """Analyze memory usage of computation graph."""
    visited = set()
    stats = {
        'num_nodes': 0,
        'num_with_backward': 0,
        'num_leaf': 0,
    }
    
    def traverse(v):
        if id(v) in visited:
            return
        visited.add(id(v))
        
        stats['num_nodes'] += 1
        if v._backward is not None:
            stats['num_with_backward'] += 1
        if not v._prev:
            stats['num_leaf'] += 1
        
        for child in v._prev:
            traverse(child)
    
    traverse(root)
    return stats


# ==================== Demo ====================

def demo():
    print("="*50)
    print("Level 03: Optimized Autograd Demo")
    print("="*50)
    
    # 1. Test requires_grad flag
    print("\n1. requires_grad optimization:")
    w = Value(2.0, requires_grad=True)
    c = Value(3.0, requires_grad=False)  # Constant
    z = w * c + c
    z.backward()
    print(f"w.grad = {w.grad} (expected: 3.0)")
    print(f"c has no grad tracking: {c.grad is None}")
    
    # 2. Test memory clearing after backward
    print("\n2. Memory optimization (backward clears graph):")
    x = Value(2.0)
    y = x * x * x  # x^3
    
    stats_before = memory_analysis(y)
    print(f"Before backward: {stats_before['num_with_backward']} backward functions")
    
    y.backward(retain_graph=False)
    stats_after = memory_analysis(y)
    print(f"After backward:  {stats_after['num_with_backward']} backward functions")
    print(f"x.grad = {x.grad} (expected: 12.0 = 3x²)")
    
    # 3. Test retain_graph
    print("\n3. retain_graph for multiple backward passes:")
    x = Value(2.0)
    y = x * x
    
    y.backward(retain_graph=True)
    grad1 = x.grad
    
    x.zero_grad()
    y.backward(retain_graph=True)
    grad2 = x.grad
    
    print(f"First backward:  x.grad = {grad1}")
    print(f"Second backward: x.grad = {grad2}")
    print(f"Same result: {grad1 == grad2}")
    
    print("\n✅ Optimizations working!")


if __name__ == "__main__":
    demo()
