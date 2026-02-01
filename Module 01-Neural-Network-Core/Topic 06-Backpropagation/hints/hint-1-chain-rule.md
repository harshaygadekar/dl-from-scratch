# Hint 1: The Chain Rule

Understanding the foundation of backpropagation.

---

## The Chain Rule Basics

For a composition of functions:
```
y = f(g(x))
```

The derivative is:
```
dy/dx = df/dg × dg/dx
```

---

## Neural Network Application

Each layer is a function. For a 2-layer network:
```
x → [Layer 1] → h → [Layer 2] → y → [Loss] → L
      g(x)          f(h)
```

To get ∂L/∂x:
```
∂L/∂x = ∂L/∂y × ∂y/∂h × ∂h/∂x
```

---

## Key Insight

Each layer only needs to compute **local gradients**:
- How does my output change w.r.t. my input?
- How does my output change w.r.t. my parameters?

Then multiply by the **upstream gradient** (∂L/∂output).

---

## Template

```python
class Layer:
    def forward(self, x):
        # Compute output
        # CACHE anything needed for backward
        self.input = x
        return output
    
    def backward(self, grad_output):
        # grad_output = ∂L/∂output
        # Compute gradients for parameters
        # Return ∂L/∂input for previous layer
        return grad_input
```

---

*Next: [Hint 2 - Linear Backward](hint-2-linear-backward.md)*
