# Hint 3: Topological Sort

## The Problem You're Facing
Your gradients are wrong because nodes are processed in the wrong order.

## The Key Insight

**Topological sort ensures parents come after children.**

For backward pass, we process children before parents:
- Output (z) is processed first
- Inputs (x, y) are processed last

## Why It Matters

Consider `z = (x + y) * x`:

```
     z
    / \
   a   x   ← x appears twice!
  / \
 x   y
```

If we process `x` before `z`, we'll miss the gradient from `z`.

## The Algorithm (DFS-based)

```python
def topological_sort(root):
    """
    Returns nodes in order: inputs → output
    (Reverse this for backward pass!)
    """
    topo = []
    visited = set()
    
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        
        # Visit all parents first (DFS)
        for parent in node._prev:
            dfs(parent)
        
        # Add self after parents
        topo.append(node)
    
    dfs(root)
    return topo
```

## Example

For `z = (x + y) * x`:

1. Start at z, visit left child (a)
2. At a, visit left child (x)
3. x has no children, add x: `[x]`
4. Back at a, visit right child (y)
5. y has no children, add y: `[x, y]`
6. a done, add a: `[x, y, a]`
7. Back at z, visit right child (x)
8. x already visited, skip
9. z done, add z: `[x, y, a, z]`

**Result**: `[x, y, a, z]`
**Reversed for backward**: `[z, a, y, x]`

## Gradient Accumulation

x appears twice in the graph. Both paths must contribute:

```
dz/dx = dz/da × da/dx + dz/dx_direct
      = (dz/da × 1) + (a)
      = gradient through a + gradient through multiplication
```

This is why `+=` is crucial:
```python
self.grad += ...  # Add from first path
self.grad += ...  # Add from second path
```

## Debug Tip

Print the order to verify:

```python
for i, v in enumerate(reversed(topo)):
    print(f"{i}: {v._op or 'input'} = {v.data}")
```

Expected output for `z = (x + y) * x`:
```
0: * = 10  (z)
1: + = 5   (a)
2: input = 3  (y)
3: input = 2  (x)
```

## Still Stuck?
→ Look at `solutions/level01-naive.py` for the complete implementation
