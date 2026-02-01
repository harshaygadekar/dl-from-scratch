# Topic 06: Math Refresh

The mathematics behind backpropagation.

---

## The Chain Rule

### Single Variable
```
If y = f(g(x)), then:
dy/dx = dy/dg × dg/dx
```

### Multivariable
```
If z = f(x, y) and x = g(t), y = h(t), then:
dz/dt = ∂z/∂x × dx/dt + ∂z/∂y × dy/dt
```

---

## Linear Layer Gradients

### Forward
```
y = Wx + b

Where:
- x ∈ ℝⁿ (input)
- W ∈ ℝᵐˣⁿ (weights)
- b ∈ ℝᵐ (bias)
- y ∈ ℝᵐ (output)
```

### Backward
Given ∂L/∂y (gradient from next layer):

```
∂L/∂W = xᵀ · ∂L/∂y

∂L/∂b = ∂L/∂y  (summed over batch)

∂L/∂x = ∂L/∂y · Wᵀ
```

### Batch Version
For X ∈ ℝᴮˣⁿ (batch of B samples):

```
∂L/∂W = Xᵀ · ∂L/∂Y    [shape: (n, m)]
∂L/∂b = sum(∂L/∂Y, axis=0)  [shape: (m,)]
∂L/∂X = ∂L/∂Y · Wᵀ    [shape: (B, n)]
```

---

## Activation Gradients

### ReLU
```
Forward:  y = max(0, x)
Backward: ∂L/∂x = ∂L/∂y × 𝟙{x > 0}
```

### Sigmoid
```
Forward:  y = σ(x) = 1/(1 + e⁻ˣ)
Backward: ∂L/∂x = ∂L/∂y × σ(x)(1 - σ(x))
                = ∂L/∂y × y(1 - y)
```

### Tanh
```
Forward:  y = tanh(x)
Backward: ∂L/∂x = ∂L/∂y × (1 - tanh²(x))
                = ∂L/∂y × (1 - y²)
```

---

## Softmax + Cross-Entropy

### Softmax
```
p_i = exp(z_i) / Σⱼ exp(z_j)
```

### Cross-Entropy Loss
```
L = -Σᵢ yᵢ log(pᵢ)
```

### Combined Gradient (elegant!)
```
∂L/∂z = p - y
```

### Derivation
```
∂L/∂z_i = Σⱼ (∂L/∂p_j)(∂p_j/∂z_i)

∂L/∂p_j = -y_j/p_j

∂p_j/∂z_i = p_j(δ_ij - p_i)  [Jacobian of softmax]

After algebra: ∂L/∂z_i = p_i - y_i
```

---

## MSE Loss

### Forward
```
L = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

### Backward
```
∂L/∂ŷ = (2/n)(ŷ - y)
```

---

## Full Network Backward Pass

For network: x → L₁ → ReLU → L₂ → Softmax → Loss

```
1. Forward pass, cache activations:
   z₁ = W₁x + b₁
   a₁ = ReLU(z₁)
   z₂ = W₂a₁ + b₂
   p = softmax(z₂)
   L = CrossEntropy(p, y)

2. Backward pass:
   ∂L/∂z₂ = p - y                    [softmax+CE gradient]
   ∂L/∂W₂ = a₁ᵀ · ∂L/∂z₂
   ∂L/∂b₂ = sum(∂L/∂z₂)
   ∂L/∂a₁ = ∂L/∂z₂ · W₂ᵀ
   ∂L/∂z₁ = ∂L/∂a₁ × 𝟙{z₁ > 0}       [ReLU gradient]
   ∂L/∂W₁ = xᵀ · ∂L/∂z₁
   ∂L/∂b₁ = sum(∂L/∂z₁)
```

---

## Numerical Gradient Check

```
∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)
```

Use ε ≈ 1e-5 for best accuracy.

---

## Quick Reference

| Layer | Forward | Backward (∂L/∂input) |
|-------|---------|---------------------|
| Linear | y = Wx + b | ∂L/∂y · Wᵀ |
| ReLU | max(0, x) | ∂L/∂y × (x > 0) |
| Sigmoid | σ(x) | ∂L/∂y × y(1-y) |
| Tanh | tanh(x) | ∂L/∂y × (1-y²) |
| Softmax+CE | p, L | p - y |
| MSE | (y-ŷ)² | 2(ŷ-y)/n |

---

*"Backpropagation is calculus on steroids."*
