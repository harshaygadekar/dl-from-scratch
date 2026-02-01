# Hint 1: Class Structure

## The Problem You're Facing
You're unsure how to structure the `Value` class.

## The Key Insight

**A `Value` is both data and a graph node.**

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        # The actual number
        self.data = data
        
        # For backward pass
        self.grad = 0.0                # Accumulated gradient
        self._backward = lambda: None  # Local backward function
        
        # Graph structure
        self._prev = set(_children)    # Parents in the graph
        self._op = _op                 # For debugging
```

## Design Pattern

Each operation returns a new `Value` and sets up backward:

```python
def __add__(self, other):
    # Handle raw numbers
    other = other if isinstance(other, Value) else Value(other)
    
    # Create output node with parents
    out = Value(self.data + other.data, (self, other), '+')
    
    # Define how gradients flow backward
    def _backward():
        self.grad += 1.0 * out.grad
        other.grad += 1.0 * out.grad
    
    out._backward = _backward
    return out
```

## Key Points

1. **Always wrap raw numbers**: `other if isinstance(other, Value) else Value(other)`
2. **Store parents**: `_children=(self, other)`
3. **Use closures**: `_backward` captures `self`, `other`, `out`
4. **Accumulate gradients**: Use `+=`, not `=`

## Still Stuck?
→ Check hint-2 for implementing the full backward pass
