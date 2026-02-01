# Topic 08: Math Refresh

The mathematics of loss functions.

---

## MSE (Mean Squared Error)

```
L = (1/n) Σᵢ (yᵢ - ŷᵢ)²

Gradient:
∂L/∂ŷᵢ = (2/n)(ŷᵢ - yᵢ)
```

---

## MAE (Mean Absolute Error)

```
L = (1/n) Σᵢ |yᵢ - ŷᵢ|

Gradient:
∂L/∂ŷᵢ = (1/n) × sign(ŷᵢ - yᵢ)
```

Note: Not differentiable at ŷ = y.

---

## Cross-Entropy

```
L = -(1/n) Σᵢ Σⱼ yᵢⱼ log(pᵢⱼ)

Where yᵢⱼ is one-hot, pᵢⱼ is softmax output.
```

### With Softmax (combined)

```
pⱼ = exp(zⱼ) / Σₖ exp(zₖ)

L = -log(p_correct) = -z_correct + log(Σₖ exp(zₖ))

Gradient (elegant!):
∂L/∂zⱼ = pⱼ - yⱼ
```

---

## Binary Cross-Entropy

```
L = -(1/n) Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]

Gradient:
∂L/∂pᵢ = (1/n) × (pᵢ - yᵢ) / (pᵢ(1-pᵢ))
```

### With Logits (stable)

```
L = (1/n) Σᵢ [max(zᵢ, 0) - zᵢyᵢ + log(1 + exp(-|zᵢ|))]
```

---

## Huber Loss

Combines MSE and MAE:
```
L = 0.5 × (y - ŷ)²           if |y - ŷ| ≤ δ
    δ × |y - ŷ| - 0.5 × δ²   otherwise
```

Smooth like MSE near 0, robust like MAE far from 0.

---

## Numerical Stability

### Log-Sum-Exp Trick
```
log(Σ exp(xᵢ)) = max(x) + log(Σ exp(xᵢ - max(x)))
```

### Clipping Probabilities
```python
eps = 1e-15
p = np.clip(p, eps, 1 - eps)
loss = -y * np.log(p)  # Safe now
```

---

## Summary

| Loss | Forward | Backward |
|------|---------|----------|
| MSE | (y-ŷ)²/n | 2(ŷ-y)/n |
| MAE | \|y-ŷ\|/n | sign(ŷ-y)/n |
| CE | -Σy log(p) | (p-y)/n |
| BCE | -y log(p)-(1-y)log(1-p) | (p-y)/(p(1-p)n) |

---

*"Master the math, master the loss."*
