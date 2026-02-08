# Topic 03: Optimizers From Scratch

> **Goal**: Implement gradient-based optimization algorithms.
> **Time**: 2-3 hours | **Difficulty**: Moderate

---

## 🎯 Learning Objectives

By the end of this topic, you will:
1. Understand how gradient descent updates parameters
2. Implement SGD, SGD with Momentum, and Adam
3. Know when to use each optimizer and why
4. Tune learning rates and hyperparameters effectively

---

## 📋 The Problem

Implement a family of optimizers that update parameters to minimize a loss function.

### Required Optimizer Interface

```python
class Optimizer:
    """Base optimizer class."""
    
    def __init__(self, parameters, lr=0.01):
        """
        Args:
            parameters: List of Values to optimize
            lr: Learning rate
        """
        self.parameters = parameters
        self.lr = lr
    
    def step(self):
        """Update all parameters using their gradients."""
        raise NotImplementedError
    
    def zero_grad(self):
        """Reset all gradients to zero."""
        for p in self.parameters:
            p.grad = 0.0
```

### Optimizers to Implement

1. **SGD**: Vanilla gradient descent
2. **SGDMomentum**: SGD with momentum
3. **Adam**: Adaptive moment estimation

---

## 🧠 Key Concepts to Master

### 1. Gradient Descent
The fundamental update rule:
```
θ = θ - lr × ∂L/∂θ
```

### 2. Momentum
Add velocity to escape local minima:
```
v = β × v - lr × ∂L/∂θ
θ = θ + v
```

### 3. Adam
Adaptive learning rates per parameter:
```
m = β₁ × m + (1 - β₁) × g        # First moment
v = β₂ × v + (1 - β₂) × g²       # Second moment
m̂ = m / (1 - β₁ᵗ)                 # Bias correction
v̂ = v / (1 - β₂ᵗ)                 # Bias correction
θ = θ - lr × m̂ / (√v̂ + ε)
```

---

## 📁 File Structure

```
Topic 03-Optimizers/
├── README.md             # This file
├── questions.md          # Interview-style questions
├── intuition.md          # Why these algorithms work
├── math-refresh.md       # Calculus of optimization
├── hints/
│   ├── hint-1-sgd.md         # Basic SGD
│   ├── hint-2-momentum.md    # Adding momentum
│   └── hint-3-adam.md        # Adam algorithm
├── solutions/
│   ├── level01_naive.py          # Basic SGD only
│   ├── level02_vectorized.py     # All optimizers
│   ├── level03_memory_efficient.py  # Optimized
│   └── level04_pytorch_reference.py  # Verification
├── tests/
│   ├── test_basic.py     # Core functionality
│   ├── test_edge.py      # Edge cases
│   └── test_stress.py    # Performance tests
└── visualization.py      # Loss landscape visualizer
```

---

## 🎮 How to Use This Topic

### Step 1: Understand the Math
1. Read `math-refresh.md` for optimization basics
2. Study `intuition.md` for the "why" behind each algorithm
3. Work through `questions.md` for interview preparation

### Step 2: Implement Step by Step
Start with the simplest optimizer:

1. **SGD**: Just `θ -= lr * grad`
2. **Add momentum**: Maintain velocity state
3. **Adam**: Track both moments with bias correction

### Step 3: Test on Toy Problems
```python
# Minimize f(x) = x²
x = Value(5.0)
optimizer = SGD([x], lr=0.1)

for step in range(50):
    loss = x ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Step {step}: x={x.data:.4f}, loss={loss.data:.4f}")
```

### Step 4: Compare Convergence
Visualize how different optimizers converge on the same problem.

---

## 🏆 Success Criteria

| Level | Requirement |
|-------|-------------|
| Level 1 | SGD converges on x² |
| Level 2 | Momentum escapes saddle points faster |
| Level 3 | Adam matches PyTorch optimizer behavior |
| Level 4 | Can train a neural network with your optimizer |

---

## 🔍 Common Interview Questions

1. **Why use momentum?**
   - Accelerates SGD in relevant direction
   - Dampens oscillations
   - Helps escape local minima

2. **Why is Adam popular?**
   - Adaptive learning rates per parameter
   - Works well with sparse gradients
   - Less sensitive to hyperparameter choices

3. **When might SGD beat Adam?**
   - Better generalization in some cases
   - Simpler, fewer hyperparameters
   - Sometimes converges to sharper minima

---

## 🔗 Related Topics

- **Topic 02**: Autograd Engine (provides the gradients)
- **Topic 07**: Training Loop (optimizers in context)
- **Topic 08**: Learning Rate Schedulers (dynamic LR)

---

## 💡 Key Insight

> **Optimizers are the engine of learning.**
> 
> A neural network with good architecture but bad optimization may never converge.
> Understanding optimizers gives you the power to debug training issues.

---

*"The difference between a neural network that works and one that doesn't is often just the optimizer settings."*
