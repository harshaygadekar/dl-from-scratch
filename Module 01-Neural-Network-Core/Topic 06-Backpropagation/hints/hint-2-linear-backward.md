# Hint 2: Linear Layer Backward

Deriving gradients for the linear layer.

---

## Forward Pass

```python
def forward(self, x):
    self.input = x  # Cache for backward
    return x @ self.W + self.b
```

---

## Backward Pass Derivation

Given: y = Wx + b, and ∂L/∂y from the next layer

### Weight Gradient
```
∂L/∂W = xᵀ · ∂L/∂y
```

### Bias Gradient
```
∂L/∂b = sum(∂L/∂y, axis=0)  # Sum over batch
```

### Input Gradient (to pass backward)
```
∂L/∂x = ∂L/∂y · Wᵀ
```

---

## Implementation

```python
def backward(self, grad_output):
    # grad_output shape: (batch, out_features)
    # self.input shape: (batch, in_features)
    
    self.grad_W = self.input.T @ grad_output
    self.grad_b = grad_output.sum(axis=0)
    grad_input = grad_output @ self.W.T
    
    return grad_input
```

---

## Shape Verification

```
input:       (B, in)
grad_output: (B, out)
W:           (in, out)

grad_W = input.T @ grad_output
       = (in, B) @ (B, out) = (in, out) ✓

grad_input = grad_output @ W.T
           = (B, out) @ (out, in) = (B, in) ✓
```

---

*Next: [Hint 3 - Full Network](hint-3-full-network.md)*
