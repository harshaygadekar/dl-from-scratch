# Topic 04: Single Layer Perceptron

> **Goal**: Implement a single-layer neural network for binary classification.
> **Time**: 2-3 hours | **Difficulty**: Medium

---

## 🎯 Learning Objectives

By the end of this topic, you will:
1. Understand the perceptron as the simplest neural network
2. Derive gradients manually (no autograd)
3. Implement forward and backward passes
4. Train on synthetic binary classification data

---

## 📋 The Problem

Implement a single-layer perceptron that learns to classify linearly separable data.

### Mathematical Model

```
Input:  x ∈ ℝⁿ
Weights: w ∈ ℝⁿ, bias b ∈ ℝ
Output: ŷ = σ(w·x + b)

where σ is the sigmoid function: σ(z) = 1/(1 + e^(-z))
```

### Required Implementation

```python
class Perceptron:
    def __init__(self, input_dim):
        self.w = np.random.randn(input_dim) * 0.01
        self.b = 0.0
    
    def forward(self, x):
        """Compute prediction."""
        pass
    
    def backward(self, x, y, y_pred):
        """Compute gradients manually."""
        pass
    
    def update(self, lr):
        """Apply gradient descent."""
        pass
```

---

## 🧠 Key Concepts

### 1. The Perceptron
```
         w₁
    x₁ ──────┐
              │
         w₂   │    ┌────────┐
    x₂ ──────┼───▶│ Σ + b  │───▶ σ(z) ───▶ ŷ
              │    └────────┘
         w₃   │
    x₃ ──────┘
```

### 2. Binary Cross-Entropy Loss
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

### 3. Gradient Derivation
```
∂L/∂w = (ŷ - y) · x
∂L/∂b = (ŷ - y)
```

---

## 📁 File Structure

```
Topic 04-Single-Layer-Perceptron/
├── README.md
├── questions.md
├── intuition.md
├── math-refresh.md
├── hints/
│   ├── hint-1-forward.md
│   ├── hint-2-loss.md
│   └── hint-3-backward.md
├── solutions/
│   ├── level01-naive.py
│   ├── level02-vectorized.py
│   ├── level03-memory-efficient.py
│   └── level04-pytorch-reference.py
├── tests/
│   ├── test_basic.py
│   ├── test_edge.py
│   └── test_stress.py
└── visualization.py
```

---

## 🎮 How to Use

### Step 1: Generate Synthetic Data
```python
# Linearly separable data
np.random.seed(42)
X_pos = np.random.randn(100, 2) + np.array([2, 2])
X_neg = np.random.randn(100, 2) + np.array([-2, -2])
X = np.vstack([X_pos, X_neg])
y = np.array([1]*100 + [0]*100)
```

### Step 2: Train the Perceptron
```python
model = Perceptron(input_dim=2)
for epoch in range(100):
    total_loss = 0
    for i in range(len(X)):
        y_pred = model.forward(X[i])
        loss = -y[i]*np.log(y_pred) - (1-y[i])*np.log(1-y_pred)
        model.backward(X[i], y[i], y_pred)
        model.update(lr=0.1)
        total_loss += loss
    print(f"Epoch {epoch}: Loss = {total_loss/len(X):.4f}")
```

---

## 🏆 Success Criteria

| Level | Requirement |
|-------|-------------|
| Level 1 | Forward pass computes correct sigmoid output |
| Level 2 | Backward pass computes correct gradients |
| Level 3 | Training converges on linearly separable data |
| Level 4 | Achieves 95%+ accuracy on test set |

---

## 🔍 Why No Autograd?

This topic deliberately avoids autograd to:
1. **Cement understanding** of gradient computation
2. **Appreciate** what autograd does for us
3. **Debug** when autograd behaves unexpectedly
4. **Interview prep**: Hand-deriving gradients is common

---

## 🔗 Related Topics

- **Topic 02**: Autograd (what we're NOT using here on purpose)
- **Topic 05**: MLP Forward Pass (extends to multiple layers)
- **Topic 06**: Backpropagation (generalized gradient computation)

---

*"Before you can use the magic, you must understand the spell."*
