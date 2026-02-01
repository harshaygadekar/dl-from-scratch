# Topic 09: Math Refresh

---

## L2 Regularization

```
Loss = L_data + λ × (1/2) Σᵢ wᵢ²

Gradient: ∂Loss/∂w = ∂L_data/∂w + λw
```

In practice, implemented as weight decay:
```python
w = w - lr * (grad + lambda * w)
```

---

## Dropout

Forward (training):
```
mask ~ Bernoulli(1 - p)
y = x × mask / (1 - p)
```

Backward:
```
∂L/∂x = ∂L/∂y × mask / (1 - p)
```

---

## Batch Normalization

Forward:
```
μ = (1/m) Σᵢ xᵢ
σ² = (1/m) Σᵢ (xᵢ - μ)²
x̂ = (x - μ) / √(σ² + ε)
y = γx̂ + β
```

Backward:
```
∂L/∂γ = Σᵢ ∂L/∂yᵢ × x̂ᵢ
∂L/∂β = Σᵢ ∂L/∂yᵢ
∂L/∂x̂ = ∂L/∂y × γ
∂L/∂σ² = Σᵢ ∂L/∂x̂ᵢ × (xᵢ - μ) × (-1/2)(σ² + ε)^(-3/2)
∂L/∂μ = Σᵢ ∂L/∂x̂ᵢ × (-1/√(σ² + ε)) + ∂L/∂σ² × (-2/m) Σᵢ(xᵢ - μ)
∂L/∂x = ∂L/∂x̂ × 1/√(σ² + ε) + ∂L/∂σ² × 2(x - μ)/m + ∂L/∂μ × 1/m
```

Running statistics:
```
μ_running = momentum × μ_running + (1 - momentum) × μ_batch
σ²_running = momentum × σ²_running + (1 - momentum) × σ²_batch
```

---

*"The math is complex, but the intuition is simple: normalize, then rescale."*
