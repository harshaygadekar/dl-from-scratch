# Hint 3: Softmax

---

## Numerical Stability

Always subtract max before exp:

```python
def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # Subtract max!
    return exp_x / exp_x.sum(axis=axis, keepdims=True)
```

---

## Softmax Gradient (Jacobian)

The full Jacobian is:
```
∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ - pⱼ)
```

But in practice, we combine with cross-entropy loss.

---

## Softmax + Cross-Entropy

**This is key!** Always combine them:

```python
class SoftmaxCrossEntropy:
    def forward(self, logits, y_true):
        self.probs = softmax(logits)
        self.y_true = y_true
        eps = 1e-15
        loss = -np.sum(y_true * np.log(self.probs + eps)) / len(y_true)
        return loss
    
    def backward(self):
        # The beautiful result!
        return (self.probs - self.y_true) / len(self.y_true)
```

The gradient is simply: **probs - labels**

---

*You now have all activations!*
