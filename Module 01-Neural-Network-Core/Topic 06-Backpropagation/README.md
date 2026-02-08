# Topic 06: Backpropagation

> **Goal**: Master the chain rule and implement backward pass.
> **Time**: 3-4 hours | **Difficulty**: Hard

---

## 🎯 Learning Objectives

By the end of this topic, you will:
1. Derive gradients using the chain rule
2. Implement backward pass for all layer types
3. Understand gradient flow through networks
4. Debug gradient issues (vanishing/exploding)

---

## 📋 The Problem

Compute gradients of the loss with respect to all parameters.

### The Chain Rule

For a composition f(g(x)):
```
∂f/∂x = ∂f/∂g × ∂g/∂x
```

### Neural Network Backward Pass

```
Forward:  x → h₁ → h₂ → y → L (loss)
Backward: ∂L/∂x ← ∂L/∂h₁ ← ∂L/∂h₂ ← ∂L/∂y ← 1
```

### Required Implementation

```python
class Linear:
    def forward(self, x):
        self.input = x  # Cache for backward
        return x @ self.W + self.b
    
    def backward(self, grad_output):
        # grad_output = ∂L/∂output
        self.grad_W = self.input.T @ grad_output
        self.grad_b = grad_output.sum(axis=0)
        grad_input = grad_output @ self.W.T
        return grad_input

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * self.mask
```

---

## 🧠 Key Equations

### Linear Layer
```
Forward:  y = Wx + b
Backward: ∂L/∂W = x^T · ∂L/∂y
          ∂L/∂b = sum(∂L/∂y)
          ∂L/∂x = ∂L/∂y · W^T
```

### ReLU
```
Forward:  y = max(0, x)
Backward: ∂L/∂x = ∂L/∂y · 1_{x>0}
```

### Softmax + Cross-Entropy
```
Forward:  p = softmax(z), L = -Σy_true log(p)
Backward: ∂L/∂z = p - y_true
```

---

## 📁 File Structure

```
Topic 06-Backpropagation/
├── README.md
├── questions.md
├── intuition.md
├── math-refresh.md
├── hints/
│   ├── hint-1-chain-rule.md
│   ├── hint-2-linear-backward.md
│   └── hint-3-full-network.md
├── solutions/
│   ├── level01_naive.py
│   ├── level02_vectorized.py
│   ├── level03_memory_efficient.py
│   └── level04_pytorch_reference.py
├── tests/
│   ├── test_basic.py
│   ├── test_edge.py
│   └── test_stress.py
└── visualization.py
```

---

## 🎮 How to Use

### Train a Network
```python
mlp = MLP([784, 256, 10])
for epoch in range(epochs):
    # Forward
    output = mlp.forward(X)
    loss = cross_entropy(output, y)
    
    # Backward
    grad = mlp.backward(y)
    
    # Update
    mlp.update(lr=0.01)
```

---

## 🏆 Success Criteria

| Level | Requirement |
|-------|-------------|
| Level 1 | Single layer backward works |
| Level 2 | Multi-layer backward works |
| Level 3 | Gradients match numerical check |
| Level 4 | Matches PyTorch autograd |

---

## 🔗 Related Topics

- **Topic 02**: Autograd Engine (computational graphs)
- **Topic 05**: MLP Forward Pass (forward direction)
- **Topic 03**: Optimizers (use gradients for updates)

---

*"Backpropagation is just the chain rule applied efficiently."*
