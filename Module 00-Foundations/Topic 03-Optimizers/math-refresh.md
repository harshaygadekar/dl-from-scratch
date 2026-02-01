# Topic 03: Math Refresh

Mathematics of optimization for deep learning.

---

## Optimization Basics

### The Goal
Find θ* that minimizes loss L(θ):
```
θ* = argmin_θ L(θ)
```

### Gradient Descent
Move in the opposite direction of the gradient:
```
θ_{t+1} = θ_t - η ∇L(θ_t)
```

Where:
- η (eta) = learning rate
- ∇L = gradient of loss

---

## Update Equations

### Vanilla SGD
```
θ = θ - lr × g

where g = ∂L/∂θ
```

### SGD with Momentum
```
v = β × v + g           (update velocity)
θ = θ - lr × v          (update parameter)

or equivalently:
v = β × v + lr × g
θ = θ - v
```

**Note**: There are two conventions for momentum! The first is "Sutskever style" (gradient goes directly into velocity), the second adds lr into velocity. PyTorch uses a hybrid.

### Nesterov Momentum
```
v = β × v + g(θ - β×v)    ("look ahead" gradient)
θ = θ - lr × v
```

### Adagrad
```
s = s + g²                 (accumulate squared gradients)
θ = θ - lr × g / (√s + ε)
```

Problem: Learning rate goes to zero as s grows!

### RMSprop
```
s = β × s + (1-β) × g²    (exponential moving average)
θ = θ - lr × g / (√s + ε)
```

### Adam (Adaptive Moment Estimation)
```
m = β₁ × m + (1-β₁) × g      (first moment)
v = β₂ × v + (1-β₂) × g²     (second moment)

# Bias correction
m̂ = m / (1 - β₁^t)
v̂ = v / (1 - β₂^t)

θ = θ - lr × m̂ / (√v̂ + ε)
```

Default hyperparameters:
- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8
- lr = 0.001

### AdamW (Adam with Decoupled Weight Decay)
```
m, v, m̂, v̂ = ... (same as Adam)

θ = θ - lr × m̂ / (√v̂ + ε) - lr × λ × θ
                            ↑ weight decay
```

---

## Convergence Analysis

### For Convex Functions
With proper learning rate, SGD converges to global minimum.

Convergence rate:
- Vanilla SGD: O(1/√T)
- SGD + Momentum: O(1/T)
- Adam: O(1/√T) with adaptive bounds

### Learning Rate Bounds
For convex L-smooth functions:
```
lr < 2/L

where L = max eigenvalue of Hessian
```

Too large lr → divergence
Too small lr → slow convergence

---

## Key Mathematical Concepts

### Gradient
Vector of partial derivatives:
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

Points in the direction of steepest ASCENT.

### Hessian
Matrix of second derivatives:
```
H = [∂²f/∂xᵢ∂xⱼ]
```

Describes local curvature.

### Condition Number
```
κ = λ_max / λ_min
```

High condition number → elongated valleys → SGD oscillates

### Exponential Moving Average
```
EMA_t = β × EMA_{t-1} + (1-β) × x_t
```

Effective window ≈ 1/(1-β)
- β = 0.9 → window ≈ 10
- β = 0.99 → window ≈ 100
- β = 0.999 → window ≈ 1000

---

## Loss Landscape Geometry

### Saddle Points
```
f(x,y) = x² - y²


Points where ∇f = 0 but not a minimum.
In high dimensions, most critical points are saddles!

Momentum and adaptive methods help escape saddles.

### Local Minima
```
In deep learning, local minima are often "good enough."
Loss landscape is highly non-convex but well-behaved.
```

### Sharp vs Flat Minima
```
Sharp:  Low training loss, poor generalization
Flat:   Similar training loss, better generalization
```

SGD tends to find flatter minima than Adam (one theory for why it generalizes better).

---

## Regularization via Optimization

### Weight Decay / L2 Regularization
```
L_total = L_original + (λ/2) × ||θ||²

Gradient contribution: λθ
```

Prevents weights from growing too large.

### Gradient Clipping
```
if ||g|| > max_norm:
    g = g × (max_norm / ||g||)
```

Prevents exploding gradients.

---

## Important Derivations

### Why Momentum Reduces Oscillations

Consider loss: L(x,y) = (1/2)ax² + (1/2)by² where a >> b

Without momentum:
- Large gradient in x direction
- Oscillates in x, slow progress in y

With momentum (β = 0.9):
- x gradients alternate sign, cancel out
- y gradients consistent, accumulate
- Net effect: faster progress toward minimum

### Adam Bias Correction Derivation

At step t:
```
m_t = (1-β₁) Σᵢ₌₁ᵗ β₁^(t-i) gᵢ
E[m_t] = E[g] × (1-β₁) Σᵢ₌₁ᵗ β₁^(t-i)
       = E[g] × (1 - β₁ᵗ)
```

To get unbiased estimate:
```
m̂_t = m_t / (1 - β₁ᵗ) → E[m̂_t] = E[g]  ✓
```

---

## Practical Formulas

### Learning Rate Scaling
For batch size B:
```
lr_effective ≈ lr_base × √(B / B_base)

or (linear scaling):
lr_effective ≈ lr_base × (B / B_base)
```

### Warmup Schedule
```
lr(t) = lr_target × min(1, t / warmup_steps)
```

### Cosine Annealing
```
lr(t) = lr_min + (lr_max - lr_min) × (1 + cos(πt/T)) / 2
```

---

## Quick Reference

| Optimizer | Memory | Hyperparameters | Best For |
|-----------|--------|-----------------|----------|
| SGD | O(1) | lr | Research, CV |
| Momentum | O(n) | lr, β | General use |
| RMSprop | O(n) | lr, β | RNNs |
| Adam | O(2n) | lr, β₁, β₂, ε | General, Transformers |
| AdamW | O(2n) | lr, β₁, β₂, ε, λ | SOTA best practice |

---

## Resources

1. **Adam paper**: [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
2. **AdamW paper**: [arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
3. **Why Momentum Really Works**: [distill.pub/2017/momentum/](https://distill.pub/2017/momentum/)

---

*"Understanding the math of optimization gives you superpowers for debugging training."*
