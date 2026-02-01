# Hint 1: ReLU Family

Implementing ReLU and its variants.

---

## ReLU

```python
class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * self.mask
```

---

## LeakyReLU

```python
class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, grad_output):
        return np.where(self.x > 0, grad_output, self.alpha * grad_output)
```

---

## ELU

```python
class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self, x):
        self.x = x
        self.output = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        return self.output
    
    def backward(self, grad_output):
        return np.where(self.x > 0, grad_output, 
                       grad_output * (self.output + self.alpha))
```

---

*Next: [Hint 2 - Sigmoid Tanh](hint-2-sigmoid-tanh.md)*
