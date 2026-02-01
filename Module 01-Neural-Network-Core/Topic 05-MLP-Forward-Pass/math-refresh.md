# Topic 05: Math Refresh

The mathematics behind MLPs and initialization.

---

## Core Equations

### Linear Layer
```
y = Wx + b

Where:
- x ∈ ℝⁿ (input)
- W ∈ ℝᵐˣⁿ (weight matrix)
- b ∈ ℝᵐ (bias)
- y ∈ ℝᵐ (output)
```

### Multi-Layer Forward Pass
```
h₁ = σ(W₁x + b₁)
h₂ = σ(W₂h₁ + b₂)
...
y = W_L h_{L-1} + b_L
```

---

## Activation Functions

### ReLU
```
ReLU(x) = max(0, x)

Derivative:
ReLU'(x) = 1 if x > 0 else 0
```

### Sigmoid
```
σ(x) = 1 / (1 + e^(-x))

Derivative:
σ'(x) = σ(x)(1 - σ(x))
```

### Tanh
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Derivative:
tanh'(x) = 1 - tanh²(x)
```

---

## Initialization Derivation

### Goal: Preserve Variance

For layer y = Wx:
```
Var(yᵢ) = Var(Σⱼ Wᵢⱼ xⱼ)
        = Σⱼ Var(Wᵢⱼ xⱼ)       (independence)
        = Σⱼ E[Wᵢⱼ²] E[xⱼ²]    (independence, zero mean)
        = n_in × Var(W) × Var(x)
```

For Var(y) = Var(x):
```
n_in × Var(W) = 1
Var(W) = 1/n_in
```

### Xavier Initialization

Considering both forward and backward:
```
Forward:  Var(W) = 1/n_in
Backward: Var(W) = 1/n_out

Compromise: Var(W) = 2/(n_in + n_out)

W ~ N(0, 2/(n_in + n_out))
```

### Kaiming Initialization

For ReLU, variance is halved (E[ReLU(x)²] = Var(x)/2):
```
Var(W) = 2/n_in

W ~ N(0, 2/n_in)
```

---

## Batch vs Single Sample

### Single sample: x ∈ ℝⁿ
```
y = Wx + b

W: (m, n)
x: (n,)
y: (m,)
```

### Batch: X ∈ ℝᴮˣⁿ
```
Y = XWᵀ + b

X: (B, n)
W: (m, n)  →  Wᵀ: (n, m)
Y: (B, m)
```

Or using row-major weights:
```
Y = XW + b

X: (B, n)
W: (n, m)
Y: (B, m)
```

---

## Initialization Formulas Summary

| Method | Variance | Use Case |
|--------|----------|----------|
| LeCun | 1/n_in | General (older) |
| Xavier (Glorot) | 2/(n_in + n_out) | Tanh, Sigmoid |
| Kaiming (He) | 2/n_in | ReLU, LeakyReLU |

---

## Forward Pass Pseudocode

```
function forward(X, layers):
    h = X
    for i = 1 to L-1:
        z = h @ W[i] + b[i]
        h = activation(z)
    
    output = h @ W[L] + b[L]  # No activation on output
    return output
```

---

## Variance Propagation

For L layers with ReLU:
```
Var(h_L) = (n × Var(W) × 0.5)^L × Var(x)
```

With Kaiming (Var(W) = 2/n):
```
Var(h_L) = (n × 2/n × 0.5)^L × Var(x)
         = (1)^L × Var(x)
         = Var(x)  ✓ Stable!
```

---

## Quick Reference

```
Linear:    y = Wx + b
ReLU:      y = max(0, x)
Xavier:    W ~ N(0, sqrt(2/(n_in + n_out)))
Kaiming:   W ~ N(0, sqrt(2/n_in))
Forward:   Loop through layers, apply activation
```

---

*"The math serves the intuition: keep signals stable as they flow."*
