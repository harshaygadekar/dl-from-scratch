# Topic 07: Math Refresh

The mathematics of activation functions.

---

## ReLU Family

### ReLU
```
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```

### LeakyReLU
```
f(x) = x if x > 0, else αx
f'(x) = 1 if x > 0, else α
```
Typical α = 0.01

### ELU
```
f(x) = x if x > 0, else α(eˣ - 1)
f'(x) = 1 if x > 0, else f(x) + α
```

---

## Sigmoid

```
σ(x) = 1 / (1 + e^(-x))

Derivative:
σ'(x) = σ(x)(1 - σ(x))
```

**Properties**:
- Range: (0, 1)
- σ(0) = 0.5
- σ(-x) = 1 - σ(x)
- Max gradient: 0.25 at x=0

---

## Tanh

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        = 2σ(2x) - 1

Derivative:
tanh'(x) = 1 - tanh²(x)
```

**Properties**:
- Range: (-1, 1)
- Zero-centered
- tanh(0) = 0
- Max gradient: 1 at x=0

---

## Softmax

```
softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```

**Jacobian** (∂pᵢ/∂zⱼ):
```
∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ - pⱼ)

where δᵢⱼ = 1 if i=j, else 0
```

**Matrix form**:
```
J = diag(p) - ppᵀ
```

---

## GELU

```
GELU(x) = x × Φ(x)

where Φ(x) = CDF of standard Gaussian

Approximation:
GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

---

## Numerical Stability

### Sigmoid
```python
# Unstable for large negative x
def sigmoid_unstable(x):
    return 1 / (1 + np.exp(-x))  # exp(-(-1000)) = inf!

# Stable version
def sigmoid_stable(x):
    positive = x >= 0
    result = np.empty_like(x)
    result[positive] = 1 / (1 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1 + exp_x)
    return result
```

### Softmax
```python
# Stable softmax (subtract max)
def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)
```

---

## Gradient Summary

| Activation | Forward | Backward |
|------------|---------|----------|
| ReLU | max(0, x) | (x > 0) |
| LeakyReLU | max(αx, x) | 1 if x>0 else α |
| Sigmoid | σ(x) | y(1-y) |
| Tanh | tanh(x) | 1 - y² |
| Softmax+CE | p, L | p - y |

Note: For Sigmoid and Tanh, y = forward output.

---

*"Master the derivatives—they're the backbone of learning."*
