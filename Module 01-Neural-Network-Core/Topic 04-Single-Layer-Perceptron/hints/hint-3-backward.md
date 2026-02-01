# Hint 3: Backward Pass

Computing gradients manually for the perceptron.

---

## The Key Insight

For BCE loss with sigmoid activation, the gradient simplifies to:
```
∂L/∂z = ŷ - y
```

This is beautifully simple! No need to compute σ'(z) separately.

---

## Gradient Formulas

```
∂L/∂w = (ŷ - y) · x
∂L/∂b = (ŷ - y)
```

Where:
- ŷ is the prediction
- y is the true label
- x is the input

---

## Why Does It Simplify?

The full derivation:
```
∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂w
```

Breaking it down:
- ∂L/∂ŷ = (ŷ - y) / [ŷ(1-ŷ)]  (from BCE)
- ∂ŷ/∂z = ŷ(1-ŷ)               (sigmoid derivative)
- ∂z/∂w = x                     (from z = w·x + b)

Result:
```
∂L/∂w = (ŷ - y) / [ŷ(1-ŷ)] · ŷ(1-ŷ) · x = (ŷ - y) · x
```

The ŷ(1-ŷ) terms cancel!

---

## Implementation Template

```python
def backward(self, x, y_true, y_pred):
    """
    Compute gradients for weights and bias.
    
    Args:
        x: Input vector
        y_true: True label
        y_pred: Predicted probability
    """
    # Compute error signal
    error = # YOUR CODE HERE
    
    # Gradient for weights
    self.dw = # YOUR CODE HERE
    
    # Gradient for bias
    self.db = # YOUR CODE HERE
```

---

## Update Step

After computing gradients:
```python
def update(self, lr):
    self.w -= lr * self.dw
    self.b -= lr * self.db
```

---

## Gradient Checking

Verify your gradients with finite differences:
```python
def check_gradient(model, x, y, eps=1e-5):
    # Numerical gradient for weight 0
    model.w[0] += eps
    loss_plus = model.compute_loss(model.forward(x), y)
    model.w[0] -= 2*eps
    loss_minus = model.compute_loss(model.forward(x), y)
    model.w[0] += eps  # Reset
    
    numerical = (loss_plus - loss_minus) / (2*eps)
    
    # Analytical gradient
    y_pred = model.forward(x)
    model.backward(x, y, y_pred)
    analytical = model.dw[0]
    
    print(f"Numerical: {numerical:.6f}")
    print(f"Analytical: {analytical:.6f}")
    print(f"Difference: {abs(numerical - analytical):.10f}")
```

---

*You now have all the pieces to implement a perceptron!*
