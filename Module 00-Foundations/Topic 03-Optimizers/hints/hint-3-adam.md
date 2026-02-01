# Hint 3: Adam Implementation

## The Problem You're Facing
You want to implement Adam but the formula looks complex.

## The Key Insight

**Adam combines momentum (first moment) with RMSprop (second moment).**

```python
import math

class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        # Initialize moments to zero
        self.m = [0.0 for _ in self.parameters]  # First moment
        self.v = [0.0 for _ in self.parameters]  # Second moment
        self.t = 0  # Timestep (for bias correction)
    
    def step(self):
        self.t += 1  # Increment timestep
        
        for i, p in enumerate(self.parameters):
            g = p.grad
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            # Compute bias-corrected first moment
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameter
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0
```

## Breaking Down the Steps

1. **First moment (m)**: Running average of gradients → like momentum
   ```
   m = 0.9 * m + 0.1 * gradient
   ```

2. **Second moment (v)**: Running average of squared gradients → scales learning
   ```
   v = 0.999 * v + 0.001 * gradient²
   ```

3. **Bias correction**: Compensates for zero initialization
   ```
   m_hat = m / (1 - 0.9^t)   # At t=1: m_hat = m / 0.1 = 10m
   v_hat = v / (1 - 0.999^t) # At t=1: v_hat = v / 0.001 = 1000v
   ```

4. **Update**: Adaptive step size
   ```
   θ -= lr * (smooth_gradient / smooth_scale)
   ```

## AdamW Variant

For proper weight decay:
```python
# After the Adam update:
p.data -= self.lr * self.weight_decay * p.data
```

## Common Mistakes

1. **Forgetting to increment t**: Bias correction needs timestep!
2. **Using eps inside sqrt**: It should be `sqrt(v) + eps`, not `sqrt(v + eps)`
3. **Not using math.sqrt**: NumPy's sqrt might not work on scalars

## Still Stuck?
→ Look at `solutions/level02-vectorized.py` for the complete implementation
