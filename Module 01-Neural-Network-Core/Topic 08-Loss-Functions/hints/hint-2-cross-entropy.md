# Hint 2: Cross-Entropy

Multi-class classification loss.

---

## Standard Cross-Entropy

With softmax probabilities and one-hot labels:

```python
class CrossEntropyLoss:
    def forward(self, probs, y_true):
        self.probs = probs
        self.y_true = y_true
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        loss = -np.sum(y_true * np.log(probs)) / len(y_true)
        return loss
    
    def backward(self):
        eps = 1e-15
        return -self.y_true / (self.probs + eps) / len(self.y_true)
```

---

## Softmax + Cross-Entropy Combined

For numerical stability, compute from logits:

```python
class SoftmaxCrossEntropyLoss:
    def forward(self, logits, y_true):
        # Stable softmax
        x_max = logits.max(axis=-1, keepdims=True)
        exp_x = np.exp(logits - x_max)
        self.probs = exp_x / exp_x.sum(axis=-1, keepdims=True)
        self.y_true = y_true
        
        eps = 1e-15
        loss = -np.sum(y_true * np.log(self.probs + eps)) / len(y_true)
        return loss
    
    def backward(self):
        # The elegant gradient!
        return (self.probs - self.y_true) / len(self.y_true)
```

---

*Next: [Hint 3 - Binary CE](hint-3-binary-ce.md)*
