# Hint 2: Adding Momentum

## The Problem You're Facing
You want to add momentum to SGD but aren't sure how to track velocity.

## The Key Insight

**Momentum maintains a "velocity" for each parameter.**

```python
class SGDMomentum:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        
        # Initialize velocity to zero for each parameter
        self.velocity = [0.0 for _ in self.parameters]
    
    def step(self):
        for i, p in enumerate(self.parameters):
            # Update velocity: v = β*v + g
            self.velocity[i] = self.momentum * self.velocity[i] + p.grad
            
            # Update parameter: θ = θ - lr*v
            p.data -= self.lr * self.velocity[i]
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0
```

## Variant: Nesterov Momentum

```python
def step_nesterov(self):
    for i, p in enumerate(self.parameters):
        # Save old velocity
        v_old = self.velocity[i]
        
        # Update velocity
        self.velocity[i] = self.momentum * v_old + p.grad
        
        # Nesterov update: use "lookahead" velocity
        p.data -= self.lr * (self.momentum * self.velocity[i] + p.grad)
```

## Key Insight

Momentum accumulates gradients over time:
```
Step 1: v = 0 + g₁ = g₁
Step 2: v = 0.9*g₁ + g₂
Step 3: v = 0.81*g₁ + 0.9*g₂ + g₃
...
```

Consistent gradients get amplified, oscillations cancel out!

## Still Stuck?
→ Check hint-3 for Adam implementation
