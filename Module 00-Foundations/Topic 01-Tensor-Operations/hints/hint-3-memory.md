# Hint 3: Memory Optimization

## The Problem You're Facing
Your implementation works but is slow or uses too much memory.

## The Key Insight

**Views are free, copies are expensive. Use stride tricks to avoid allocation.**

### Views vs Copies

| Operation | Type | Memory |
|-----------|------|--------|
| `a[::2]` | View | Free |
| `a.reshape(...)` | View | Free |
| `a.T` | View | Free |
| `a.copy()` | Copy | Expensive |
| `a[[0, 2, 4]]` | Copy | Expensive |
| `np.concatenate([a, b])` | Copy | Very expensive |

## Memory-Efficient Patterns

### Pattern 1: Avoid Intermediate Arrays
```python
# Bad: Creates temporary arrays
result = ((x - mean) / std) ** 2

# Better: In-place operations
result = x - mean
result /= std
result **= 2
```

### Pattern 2: Use broadcast_to for Repetition
```python
# Bad: Actually copies data
repeated = np.tile(x, (100, 1))  # Allocates 100x memory!

# Good: View with zero extra memory
repeated = np.broadcast_to(x, (100,) + x.shape)  # Read-only view
```

### Pattern 3: Stride Tricks for Sliding Windows
```python
from numpy.lib.stride_tricks import as_strided

def sliding_window(x, window_size):
    shape = (len(x) - window_size + 1, window_size)
    strides = (x.strides[0], x.strides[0])  # Same stride twice!
    return as_strided(x, shape=shape, strides=strides)
```

### Pattern 4: Pre-allocate Output
```python
# Bad: Repeated allocation in loop
results = []
for batch in batches:
    results.append(process(batch))
result = np.vstack(results)  # Copies everything again!

# Good: Pre-allocate
result = np.empty((n_batches, batch_size, features))
for i, batch in enumerate(batches):
    result[i] = process(batch)  # Write directly
```

## Memory Debugging

```python
# Check if view or copy
print(result.base is source)  # True if view
print(result.flags['OWNDATA'])  # False if view

# Check memory sharing
print(np.shares_memory(a, b))

# Check strides
print(x.strides)  # Bytes to step in each dimension
```

## Performance Tips

1. **Contiguous is faster**: `np.ascontiguousarray(x)` before heavy computation
2. **Avoid Python loops**: Each loop iteration has overhead
3. **Batch operations**: Process many examples at once
4. **Use float32**: Half the memory of float64, often sufficient

## Still Stuck?
→ Look at `solutions/level03_memory_efficient.py` for complete examples
