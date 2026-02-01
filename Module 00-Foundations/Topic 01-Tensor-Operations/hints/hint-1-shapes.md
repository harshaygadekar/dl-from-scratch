# Hint 1: Shape Fundamentals

## The Problem You're Facing
You're getting shape mismatch errors or unexpected output shapes.

## The Key Insight

**Always work through shapes step-by-step before writing code.**

For any operation, write down:
1. Input shapes
2. Expected output shape
3. How each dimension transforms

### Shape Debugging Template

```python
def my_operation(a, b):
    print(f"a.shape: {a.shape}")
    print(f"b.shape: {b.shape}")
    
    # Your operation
    result = ...
    
    print(f"result.shape: {result.shape}")
    return result
```

## Common Shape Rules

### Matrix Multiplication (@ or np.matmul)
```
(m, k) @ (k, n) → (m, n)
(batch, m, k) @ (batch, k, n) → (batch, m, n)
```

### Element-wise Operations (*, +, -, /)
Shapes must be broadcastable. Result shape is the maximum along each dimension.

### Reduction Operations (sum, mean, max)
The reduced axis disappears (unless `keepdims=True`).

## Quick Fixes

1. **Dimension mismatch**: Use `np.expand_dims(x, axis=0)` or `x[None, :]`
2. **Wrong transpose**: Double-check `a.T` vs `np.transpose(a, axes=...)`
3. **Broadcasting confusion**: Use explicit reshapes to be safe

## Still Stuck?
→ Check hint-2 for broadcasting details
