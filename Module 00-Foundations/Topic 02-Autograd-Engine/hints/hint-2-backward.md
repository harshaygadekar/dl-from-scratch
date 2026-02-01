# Hint 2: Implementing Backward

## The Problem You're Facing
You're unsure how to implement the `backward()` method.

## The Key Insight

**Backward does two things**:
1. Build a topologically sorted list of nodes
2. Call `_backward()` on each node in reverse order

## The Algorithm

```python
def backward(self):
    # Step 1: Topological sort (output → inputs)
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    
    build_topo(self)
    
    # Step 2: Initialize output gradient
    self.grad = 1.0  # dL/dL = 1
    
    # Step 3: Propagate gradients backward
    for v in reversed(topo):
        v._backward()
```

## Why This Order?

Consider: `z = x * y`

```
Forward:  x, y → (computed first) → z
Backward: z → (processed first) → x, y
```

Topological sort puts z last, reversing gives z first.

## Common Mistake

**Wrong**: Not reversing the topological order
```python
# WRONG - this goes forward!
for v in topo:
    v._backward()
```

**Right**: Reverse it
```python
# RIGHT - backward from output to inputs
for v in reversed(topo):
    v._backward()
```

## Testing Your Implementation

```python
x = Value(2.0)
y = Value(3.0)
z = x * y  # z = 6

z.backward()

assert x.grad == 3.0, f"Expected 3.0, got {x.grad}"
assert y.grad == 2.0, f"Expected 2.0, got {y.grad}"
print("✅ Backward pass working!")
```

## Still Stuck?
→ Check hint-3 for topological sort details
