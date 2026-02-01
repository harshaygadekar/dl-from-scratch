# Hint 1: SGD Implementation

## The Problem You're Facing
You're unsure how to structure the basic SGD optimizer.

## The Key Insight

**SGD is literally just: θ = θ - lr × gradient**

```python
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)  # Store as list
        self.lr = lr
    
    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0
```

## Key Points

1. **Store parameters as a list**: You'll need to iterate over them
2. **Modify .data, not the Value itself**: `p.data -= ...` not `p -= ...`
3. **Always implement zero_grad()**: Essential for training loops

## Testing

```python
# Minimize f(x) = x²
from solutions.level01_naive import Value

x = Value(5.0)
optimizer = SGD([x], lr=0.1)

for _ in range(50):
    loss = x ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Final x: {x.data}")  # Should be near 0
```

## Still Stuck?
→ Check hint-2 for adding momentum
