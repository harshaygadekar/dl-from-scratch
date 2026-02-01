# Topic 04: Math Refresh

The mathematics behind single-layer neural networks.

---

## Core Equations

### Forward Pass
```
z = w · x + b              # Linear combination
ŷ = σ(z) = 1/(1 + e^(-z))  # Sigmoid activation
```

### Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))

Properties:
- Domain: (-∞, ∞)
- Range: (0, 1)
- σ(0) = 0.5
- σ(-z) = 1 - σ(z)
```

### Binary Cross-Entropy Loss
```
L(y, ŷ) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

For a batch of N samples:
L = -(1/N) Σᵢ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
```

---

## Gradient Derivation

### Step 1: Sigmoid Derivative

```
σ(z) = (1 + e^(-z))^(-1)

dσ/dz = -1 · (1 + e^(-z))^(-2) · (-e^(-z))
      = e^(-z) / (1 + e^(-z))²
```

Simplify using σ(z) = 1/(1 + e^(-z)):
```
e^(-z) = (1 - σ)/σ · 1 = 1/σ - 1

dσ/dz = σ² · (1/σ - 1)
      = σ - σ²
      = σ(1 - σ)
```

**Result**: `dσ/dz = σ(z) · (1 - σ(z))`

---

### Step 2: Loss Gradient w.r.t. ŷ

```
L = -y·log(ŷ) - (1-y)·log(1-ŷ)

∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)
```

Combine fractions:
```
∂L/∂ŷ = [-y(1-ŷ) + (1-y)ŷ] / [ŷ(1-ŷ)]
      = [-y + yŷ + ŷ - yŷ] / [ŷ(1-ŷ)]
      = (ŷ - y) / [ŷ(1-ŷ)]
```

---

### Step 3: Chain Rule for ∂L/∂z

```
∂L/∂z = ∂L/∂ŷ · ∂ŷ/∂z
      = (ŷ - y) / [ŷ(1-ŷ)] · ŷ(1-ŷ)
      = ŷ - y
```

**Key Result**: `∂L/∂z = ŷ - y`

---

### Step 4: Gradients for Weights and Bias

Using z = w·x + b:
```
∂z/∂w = x
∂z/∂b = 1
```

Chain rule:
```
∂L/∂w = ∂L/∂z · ∂z/∂w = (ŷ - y) · x
∂L/∂b = ∂L/∂z · ∂z/∂b = (ŷ - y)
```

---

## Summary of Gradients

| Quantity | Gradient |
|----------|----------|
| ∂L/∂ŷ | (ŷ - y) / [ŷ(1-ŷ)] |
| ∂L/∂z | ŷ - y |
| ∂L/∂w | (ŷ - y) · x |
| ∂L/∂b | ŷ - y |

---

## Update Rules

Gradient descent:
```
w ← w - η · ∂L/∂w = w - η(ŷ - y)x
b ← b - η · ∂L/∂b = b - η(ŷ - y)
```

Where η is the learning rate.

---

## Numerical Stability

### Sigmoid Overflow
```python
# BAD - can overflow for large negative z
def sigmoid_bad(z):
    return 1 / (1 + np.exp(-z))

# GOOD - numerically stable
def sigmoid_good(z):
    z = np.clip(z, -500, 500)
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))
```

### Log of Small Numbers
```python
# BAD - log(0) = -inf
loss = -y * np.log(y_pred)

# GOOD - clip predictions
eps = 1e-15
y_pred = np.clip(y_pred, eps, 1 - eps)
loss = -y * np.log(y_pred)
```

---

## Decision Boundary

The decision boundary is where ŷ = 0.5, which means z = 0:
```
w₁x₁ + w₂x₂ + ... + wₙxₙ + b = 0
```

This is a hyperplane in n-dimensional space.

**Perpendicular to weight vector**: The normal to the hyperplane is w.

**Distance from origin**: d = |b| / ||w||

---

## Why BCE + Sigmoid Work Together

Cross-entropy is the negative log-likelihood for a Bernoulli distribution:
```
P(y|x) = ŷʸ · (1-ŷ)^(1-y)

log P(y|x) = y·log(ŷ) + (1-y)·log(1-ŷ)

-log P(y|x) = BCE loss
```

Maximizing likelihood = Minimizing cross-entropy

---

## Quick Reference

```
Forward:   ŷ = σ(w·x + b)
Loss:      L = -y·log(ŷ) - (1-y)·log(1-ŷ)
Gradient:  ∂L/∂w = (ŷ - y)·x,  ∂L/∂b = (ŷ - y)
Update:    w -= η·∂L/∂w,  b -= η·∂L/∂b
```

---

*"Simple math, profound implications."*
