# Hint 1: Forward Pass

The forward pass computes the prediction.

---

## Step-by-Step

### 1. Linear Combination
Compute the weighted sum of inputs:
```python
z = np.dot(self.w, x) + self.b
```

### 2. Apply Sigmoid
Squash to probability range:
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

### 3. Numerical Stability
Clip z to avoid overflow:
```python
z = np.clip(z, -500, 500)
```

---

## Implementation Template

```python
def forward(self, x):
    """
    Compute forward pass.
    
    Args:
        x: Input vector of shape (input_dim,)
    
    Returns:
        Probability output in (0, 1)
    """
    # Step 1: Linear combination
    z = # YOUR CODE HERE
    
    # Step 2: Sigmoid activation
    y_pred = # YOUR CODE HERE
    
    return y_pred
```

---

## Sanity Checks

1. Output should always be between 0 and 1
2. If all weights are 0, output should be σ(b)
3. For large positive z, output → 1
4. For large negative z, output → 0

---

*Next: [Hint 2 - Loss Computation](hint-2-loss.md)*
