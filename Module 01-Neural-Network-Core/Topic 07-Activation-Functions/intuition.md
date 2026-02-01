# Topic 07: Intuition Guide

Understanding the "why" behind activation functions.

---

## 🧠 The Big Picture

> **Without activations, a deep network collapses to a single linear transformation.**

```
Linear₁ × Linear₂ × Linear₃ = One Big Linear
```

Activations add the non-linearity that lets networks model complex functions.

---

## Mental Model: The Gate Keeper 🚪

Each activation is a "gate" that decides what information passes through:

**ReLU**: "Pass if positive, block if negative"
```
Input:  [-2, -1, 0, 1, 2]
Output: [ 0,  0, 0, 1, 2]
         ↑   ↑
       blocked
```

**Sigmoid**: "Smooth gate from 0 to 1"
```
Input:  [-5, -2,  0,  2,  5]
Output: [.01, .1, .5, .9, .99]
         ↑            ↑
     almost closed  almost open
```

---

## The ReLU Family

```
ReLU:      ────────┐
                   │
          ─────────┘
                   0

LeakyReLU: ─────────┐
           \        │
            ────────┘
                    0

ELU:       ─────────┐
           ╮        │
            ────────┘
                    0
```

**ReLU**: Hard cutoff, can "die"
**LeakyReLU**: Small slope for negative (prevents dying)
**ELU**: Smooth, mean-centered negative outputs

---

## Sigmoid vs Tanh

```
Sigmoid: Output in (0, 1)     Tanh: Output in (-1, 1)
         1 ────────────              1 ────────────
          │         ╱               │         ╱
         .5│       ╱                0│───────╱
          │   ╱                     │   ╱
         0 ────────────             -1 ────────────
```

**Sigmoid**: Good for probabilities (output layer)
**Tanh**: Zero-centered (better for hidden layers)

---

## Softmax: The Probability Maker

Turns logits into probabilities:

```
Input logits:  [2.0, 1.0, 0.1]
After exp:     [7.4, 2.7, 1.1]
After /sum:    [0.66, 0.24, 0.10]  ← Probabilities!
                        ↑
                   Sums to 1.0
```

**Key insight**: Softmax is invariant to adding a constant:
```
softmax([x, y, z]) = softmax([x+c, y+c, z+c])
```
We use this for numerical stability (subtract max).

---

## Why Gradients Matter

The gradient determines how well learning happens:

```
Sigmoid gradient:
        0.25 ┬─────╮
             │     ╱╲
             │   ╱    ╲
          0 ─┴─────────────
                 0
        
Peak is only 0.25! Gradients are always ≤ 0.25.
In 10 layers: 0.25^10 ≈ 0.000001 → VANISHING!
```

```
ReLU gradient:
           1 ┬───────────
             │
           0 ┼───────────
                 0

Gradient is 1 for positive → no vanishing!
```

---

## When to Use What

| Situation | Use |
|-----------|-----|
| Hidden layers (default) | ReLU or LeakyReLU |
| Binary classification (output) | Sigmoid |
| Multi-class classification (output) | Softmax |
| RNN hidden states | Tanh |
| Transformers | GELU |
| Worried about dying ReLU | LeakyReLU, ELU |

---

## Intuition Checkpoints ✅

1. **Why do we need non-linearity?**
   <details><summary>Answer</summary>Without it, stacking linear layers is equivalent to a single linear layer. We need non-linearity to model complex functions.</details>

2. **Why does ReLU help with vanishing gradients?**
   <details><summary>Answer</summary>ReLU has gradient = 1 for positive inputs, so gradients don't shrink as they propagate back.</details>

3. **When should you NOT use ReLU?**
   <details><summary>Answer</summary>Output layer (use sigmoid/softmax), or when you need negative outputs (use tanh), or in RNNs where tanh is more stable.</details>

---

*"The activation function is a simple idea with profound consequences."*
