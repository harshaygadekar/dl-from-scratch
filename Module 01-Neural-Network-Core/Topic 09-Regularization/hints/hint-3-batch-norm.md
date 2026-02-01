# Hint 3: Batch Normalization

---

## Forward Pass

```python
class BatchNorm1d:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.momentum = momentum
        self.eps = eps
        self.training = True
    
    def forward(self, x):
        if self.training:
            self.mean = x.mean(axis=0)
            self.var = x.var(axis=0)
            
            # Update running stats
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * self.var
        else:
            self.mean = self.running_mean
            self.var = self.running_var
        
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_norm + self.beta
```

---

## Backward Pass (simplified)

```python
def backward(self, grad_output):
    N = grad_output.shape[0]
    
    self.grad_gamma = (grad_output * self.x_norm).sum(axis=0)
    self.grad_beta = grad_output.sum(axis=0)
    
    dx_norm = grad_output * self.gamma
    dvar = (dx_norm * (self.x - self.mean) * (-0.5) * (self.var + self.eps)**(-1.5)).sum(axis=0)
    dmean = (dx_norm * (-1/np.sqrt(self.var + self.eps))).sum(axis=0)
    
    dx = dx_norm / np.sqrt(self.var + self.eps) + dvar * 2 * (self.x - self.mean) / N + dmean / N
    return dx
```

---

*You now have all regularization techniques!*
