# Topic 02: Math Refresh

Calculus fundamentals for implementing autograd.

---

## Derivatives Basics

### Definition
The derivative of f(x) at point a:
```
f'(a) = lim[h→0] (f(a+h) - f(a)) / h
```

**Intuition**: The slope of the tangent line at point a.

---

## Common Derivatives

| Function | Derivative |
|----------|------------|
| c (constant) | 0 |
| x | 1 |
| x² | 2x |
| xⁿ | n·xⁿ⁻¹ |
| eˣ | eˣ |
| ln(x) | 1/x |
| sin(x) | cos(x) |
| cos(x) | -sin(x) |

---

## Derivative Rules

### Sum Rule
```
d/dx [f(x) + g(x)] = f'(x) + g'(x)
```

### Product Rule
```
d/dx [f(x) · g(x)] = f'(x)·g(x) + f(x)·g'(x)
```

### Quotient Rule
```
d/dx [f(x) / g(x)] = [f'(x)·g(x) - f(x)·g'(x)] / g(x)²
```

### Chain Rule (THE KEY!)
```
d/dx [f(g(x))] = f'(g(x)) · g'(x)
```

---

## Chain Rule Examples

### Example 1: Compound function
```
y = (3x + 2)²

Let u = 3x + 2, then y = u²

dy/dx = dy/du × du/dx
      = (2u) × (3)
      = 2(3x + 2) × 3
      = 6(3x + 2)
```

### Example 2: Multiple layers
```
z = sin(x²)

Let u = x², then z = sin(u)

dz/dx = dz/du × du/dx
      = cos(u) × 2x
      = cos(x²) × 2x
      = 2x·cos(x²)
```

---

## Partial Derivatives

For functions of multiple variables:

```
f(x, y) = x²y + 3xy

∂f/∂x = 2xy + 3y   (treat y as constant)
∂f/∂y = x² + 3x    (treat x as constant)
```

---

## Gradient

The gradient is a vector of all partial derivatives:

```
f(x, y, z) = x² + y² + z²

∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]
   = [2x, 2y, 2z]
```

**Intuition**: Points in the direction of steepest ascent.

---

## Activation Function Derivatives

### Sigmoid
```
σ(x) = 1 / (1 + e^(-x))

σ'(x) = σ(x) · (1 - σ(x))
```

**Derivation**:
```
σ(x) = (1 + e^(-x))^(-1)

Using chain rule:
σ'(x) = -1 · (1 + e^(-x))^(-2) · (-e^(-x))
      = e^(-x) / (1 + e^(-x))²
      = [1/(1+e^(-x))] · [e^(-x)/(1+e^(-x))]
      = σ(x) · [1 - σ(x)]
```

### Tanh
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

tanh'(x) = 1 - tanh²(x)
```

### ReLU
```
ReLU(x) = max(0, x)

ReLU'(x) = { 1  if x > 0
           { 0  if x < 0
           { undefined at x = 0 (typically set to 0)
```

### Leaky ReLU
```
LeakyReLU(x) = max(αx, x)  where α is small (e.g., 0.01)

LeakyReLU'(x) = { 1  if x > 0
                { α  if x ≤ 0
```

---

## Softmax Derivative

Softmax converts logits to probabilities:
```
softmax(xᵢ) = e^(xᵢ) / Σⱼ e^(xⱼ)
```

Let pᵢ = softmax(xᵢ), then:
```
∂pᵢ/∂xⱼ = pᵢ(δᵢⱼ - pⱼ)

where δᵢⱼ = 1 if i=j, else 0
```

In matrix form (Jacobian):
```
∂p/∂x = diag(p) - p·pᵀ
```

---

## Cross-Entropy Loss Derivative

For classification with softmax:
```
L = -Σᵢ yᵢ log(pᵢ)

where y is one-hot, p = softmax(x)
```

The gradient is beautifully simple:
```
∂L/∂x = p - y
```

**This is why softmax + cross-entropy is so common** - simple gradient!

---

## Matrix Calculus Basics

### Linear Layer
```
y = Wx + b

∂L/∂W = (∂L/∂y) · xᵀ
∂L/∂x = Wᵀ · (∂L/∂y)
∂L/∂b = ∂L/∂y
```

### Key identities
```
∂(xᵀAx)/∂x = (A + Aᵀ)x

∂(trace(AB))/∂A = Bᵀ

∂||x||²/∂x = 2x
```

---

## Numerical Gradient Checking

Verify analytic gradients with finite differences:

```python
def numerical_gradient(f, x, eps=1e-5):
    """
    Central difference formula:
    f'(x) ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
    """
    return (f(x + eps) - f(x - eps)) / (2 * eps)
```

**Why central difference?**
- More accurate than forward difference
- Error is O(eps²) instead of O(eps)

**Testing protocol**:
```python
# Compute analytic gradient
analytic = your_gradient_function(x)

# Compute numerical gradient
numerical = numerical_gradient(f, x)

# Compare
relative_error = abs(analytic - numerical) / max(abs(analytic), abs(numerical), 1e-8)
assert relative_error < 1e-5, f"Gradient check failed: {relative_error}"
```

---

## Quick Reference

### For Autograd Implementation

| Operation | Forward | Backward |
|-----------|---------|----------|
| z = x + y | z = x + y | x.grad += z.grad, y.grad += z.grad |
| z = x * y | z = x * y | x.grad += y * z.grad, y.grad += x * z.grad |
| z = x - y | z = x - y | x.grad += z.grad, y.grad -= z.grad |
| z = x / y | z = x / y | x.grad += z.grad/y, y.grad -= x*z.grad/y² |
| z = x ** n | z = pow(x, n) | x.grad += n * x^(n-1) * z.grad |
| z = exp(x) | z = e^x | x.grad += z * z.grad |
| z = log(x) | z = ln(x) | x.grad += z.grad / x |
| z = relu(x) | z = max(0,x) | x.grad += (z>0) * z.grad |
| z = tanh(x) | z = tanh(x) | x.grad += (1 - z²) * z.grad |
| z = sigmoid(x) | z = σ(x) | x.grad += z * (1-z) * z.grad |

---

## Resources

1. **3Blue1Brown - Calculus**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
2. **Matrix Calculus for Deep Learning**: [Paper](https://explained.ai/matrix-calculus/)
3. **The Matrix Cookbook**: [PDF](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

---

*"Calculus is the language that describes change. In deep learning, we're always asking: how does the loss change with respect to the weights?"*
