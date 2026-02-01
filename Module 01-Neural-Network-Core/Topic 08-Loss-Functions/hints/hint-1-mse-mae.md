# Hint 1: MSE and MAE

Implementing regression losses.

---

## MSE (Mean Squared Error)

```python
class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.diff = y_pred - y_true
        return np.mean(self.diff ** 2)
    
    def backward(self):
        n = self.y_pred.size
        return 2 * self.diff / n
```

---

## MAE (Mean Absolute Error)

```python
class MAELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean(np.abs(y_pred - y_true))
    
    def backward(self):
        n = self.y_pred.size
        return np.sign(self.y_pred - self.y_true) / n
```

---

## Huber Loss (combines both)

```python
class HuberLoss:
    def __init__(self, delta=1.0):
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        self.diff = y_pred - y_true
        abs_diff = np.abs(self.diff)
        quadratic = 0.5 * self.diff ** 2
        linear = self.delta * abs_diff - 0.5 * self.delta ** 2
        return np.mean(np.where(abs_diff <= self.delta, quadratic, linear))
```

---

*Next: [Hint 2 - Cross-Entropy](hint-2-cross-entropy.md)*
