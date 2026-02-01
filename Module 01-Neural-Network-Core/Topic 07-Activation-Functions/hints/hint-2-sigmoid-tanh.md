# Hint 2: Sigmoid and Tanh

---

## Sigmoid

```python
class Sigmoid:
    def forward(self, x):
        x = np.clip(x, -500, 500)  # Numerical stability
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, grad_output):
        # σ'(x) = σ(x)(1 - σ(x))
        return grad_output * self.output * (1 - self.output)
```

---

## Tanh

```python
class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        # tanh'(x) = 1 - tanh²(x)
        return grad_output * (1 - self.output ** 2)
```

---

## Key Insight

Both compute gradient using the **forward output**:
- Sigmoid: y(1-y)
- Tanh: 1-y²

This is why we cache `self.output` in forward.

---

*Next: [Hint 3 - Softmax](hint-3-softmax.md)*
