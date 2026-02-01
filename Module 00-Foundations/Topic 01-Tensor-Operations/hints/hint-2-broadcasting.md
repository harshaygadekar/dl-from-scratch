# Hint 2: Broadcasting Rules

## The Problem You're Facing
You need to combine arrays of different shapes without explicit loops.

## The Key Insight

**Broadcasting aligns shapes from the RIGHT and stretches dimensions of size 1.**

### The Three Rules

1. **Align from right**: Compare dimensions starting from the rightmost
2. **Size 1 stretches**: A dimension of 1 can match any size
3. **Missing = size 1**: Prepend dimensions of size 1 to shorter arrays

### Visual Example

```
Array A: (32, 1, 64)
Array B:     (10, 1)

Step 1: Align from right
A: (32,  1, 64)
B: ( 1, 10,  1)  ← Prepended with 1

Step 2: Check compatibility
     32 vs 1   → OK (1 stretches to 32)
      1 vs 10  → OK (1 stretches to 10)
     64 vs 1   → OK (1 stretches to 64)

Result: (32, 10, 64)
```

## Common Broadcasting Patterns

### Subtract mean from each row
```python
data = np.random.randn(100, 64)  # (samples, features)
mean = data.mean(axis=0)          # (64,)
centered = data - mean            # (100, 64) - (64,) broadcasts!
```

### Add bias to batch
```python
x = np.random.randn(32, 128)  # (batch, features)
bias = np.random.randn(128)   # (features,)
y = x + bias                   # Bias broadcasts across batch
```

### Outer product
```python
a = np.array([1, 2, 3])       # (3,)
b = np.array([10, 20])        # (2,)
outer = a[:, None] * b[None, :]  # (3, 1) * (1, 2) → (3, 2)
```

## Debugging Tips

```python
# Check if arrays broadcast
np.broadcast_shapes(a.shape, b.shape)  # Raises if incompatible

# Manually expand dimensions
a_expanded = a[:, None, :]  # Add dimension in middle
b_expanded = np.expand_dims(b, axis=0)  # Add dimension at start
```

## Still Stuck?
→ Check hint-3 for memory-efficient implementations
