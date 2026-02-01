# Hint 3: Binary Cross-Entropy

Binary classification loss.

---

## Standard BCE (with probabilities)

```python
class BCELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    def backward(self):
        eps = 1e-15
        y_pred = np.clip(self.y_pred, eps, 1 - eps)
        n = self.y_pred.size
        return (y_pred - self.y_true) / (y_pred * (1 - y_pred) * n)
```

---

## BCE with Logits (Stable!)

More stable when working with raw logits:

```python
class BCEWithLogitsLoss:
    def forward(self, logits, y_true):
        self.logits = logits
        self.y_true = y_true
        
        # Stable formula: max(z, 0) - z*y + log(1 + exp(-|z|))
        loss = np.maximum(logits, 0) - logits * y_true + \
               np.log(1 + np.exp(-np.abs(logits)))
        return np.mean(loss)
    
    def backward(self):
        # Gradient is sigmoid(logits) - y_true
        sigmoid = 1 / (1 + np.exp(-self.logits))
        return (sigmoid - self.y_true) / self.logits.size
```

---

*You now have all loss functions!*
