# Hint 2: Dropout

---

## Inverted Dropout

```python
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
    
    def forward(self, x):
        if not self.training:
            return x
        
        self.mask = (np.random.rand(*x.shape) > self.p)
        return x * self.mask / (1 - self.p)
    
    def backward(self, grad_output):
        if not self.training:
            return grad_output
        return grad_output * self.mask / (1 - self.p)
```

---

## Key Points

- Scale by 1/(1-p) during training
- No scaling needed at inference
- Same mask for forward and backward

---

*Next: [Hint 3 - Batch Norm](hint-3-batch-norm.md)*
