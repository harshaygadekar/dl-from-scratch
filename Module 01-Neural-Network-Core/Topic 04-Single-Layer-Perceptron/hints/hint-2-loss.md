# Hint 2: Loss Computation

Computing binary cross-entropy loss correctly.

---

## The Formula

Binary cross-entropy for a single sample:
```
L = -[y · log(ŷ) + (1-y) · log(1-ŷ)]
```

Where:
- y is the true label (0 or 1)
- ŷ is the predicted probability

---

## Why This Formula?

When y = 1:
```
L = -log(ŷ)  # Lower loss when ŷ is close to 1
```

When y = 0:
```
L = -log(1-ŷ)  # Lower loss when ŷ is close to 0
```

---

## Numerical Stability

**Problem**: log(0) = -∞

**Solution**: Clip predictions away from 0 and 1

```python
def compute_loss(y_pred, y_true):
    eps = 1e-15  # Small constant
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

---

## Implementation Template

```python
def compute_loss(self, y_pred, y_true):
    """
    Compute binary cross-entropy loss.
    
    Args:
        y_pred: Predicted probability (0, 1)
        y_true: True label (0 or 1)
    
    Returns:
        Scalar loss value
    """
    eps = 1e-15
    # Clip for numerical stability
    y_pred_safe = # YOUR CODE HERE
    
    # Compute BCE
    loss = # YOUR CODE HERE
    
    return loss
```

---

## Expected Values

| y_true | y_pred | Loss |
|--------|--------|------|
| 1 | 0.99 | 0.01 |
| 1 | 0.5 | 0.69 |
| 1 | 0.01 | 4.61 |
| 0 | 0.01 | 0.01 |
| 0 | 0.5 | 0.69 |
| 0 | 0.99 | 4.61 |

---

*Next: [Hint 3 - Backward Pass](hint-3-backward.md)*
