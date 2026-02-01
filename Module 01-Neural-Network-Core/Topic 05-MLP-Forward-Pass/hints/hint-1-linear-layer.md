# Hint 1: Linear Layer

Building the fundamental building block.

---

## The Linear Layer

A linear layer computes: `y = Wx + b`

### Implementation Template

```python
class Linear:
    def __init__(self, in_features, out_features):
        # Initialize weights and bias
        self.W = # YOUR CODE HERE - shape (in_features, out_features)
        self.b = # YOUR CODE HERE - shape (out_features,)
    
    def forward(self, x):
        # Compute output
        # x shape: (batch_size, in_features) or (in_features,)
        return # YOUR CODE HERE
```

---

## Shape Reminder

```
Input:  x of shape (batch, n_in) or (n_in,)
Weight: W of shape (n_in, n_out)
Bias:   b of shape (n_out,)
Output: y of shape (batch, n_out) or (n_out,)
```

---

## Key Points

1. Use matrix multiplication: `np.dot(x, W)` or `x @ W`
2. Broadcasting handles bias addition automatically
3. Handle both single samples and batches

---

*Next: [Hint 2 - Initialization](hint-2-initialization.md)*
