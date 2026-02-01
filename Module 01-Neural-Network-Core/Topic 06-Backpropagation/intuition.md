# Topic 06: Intuition Guide

Understanding how gradients flow backward through a neural network.

---

## 🧠 The Mental Model

> **Backpropagation is blame assignment.** Each layer asks: "How much did I contribute to the error?"

---

## Mental Model 1: The Bucket Brigade 🪣

Imagine a line of people passing water buckets:

```
[Source] → Person A → Person B → Person C → [Fire]
              ↓           ↓           ↓
           Forward pass: passing water to fire

[Source] ← Person A ← Person B ← Person C ← [Fire]
              ↓           ↓           ↓
           Backward pass: feedback on how well each did
```

In backprop:
- **Forward**: Data flows through layers to produce output
- **Backward**: Error signal flows back, each layer learns its contribution

---

## Mental Model 2: The Chain of Responsibility ⛓️

Each layer in a network is like a link in a chain:

```
Input → [Linear] → [ReLU] → [Linear] → [Softmax] → Loss
           ↓          ↓         ↓          ↓
          L1         A1        L2         S        (forward)

Input ← [Linear] ← [ReLU] ← [Linear] ← [Softmax] ← Loss
          dL1       dA1       dL2        dS        (backward)
```

**Chain rule in action**:
```
∂Loss/∂L1 = ∂Loss/∂S × ∂S/∂L2 × ∂L2/∂A1 × ∂A1/∂L1
```

---

## The Key Insight: Local Gradients

Each layer only needs to know:
1. **Its local gradient**: How does its output change w.r.t. its input?
2. **Upstream gradient**: How does the loss change w.r.t. its output?

**Multiply them together** = gradient for this layer's input

```
         ┌─────────────┐
   x ──→ │    Layer    │ ──→ y
         └─────────────┘
              
         ┌─────────────┐
 ∂L/∂x ←─│  ∂y/∂x ×   │ ←── ∂L/∂y
         │  ∂L/∂y      │
         └─────────────┘
```

---

## Visualizing Gradient Flow

### Linear Layer: y = Wx + b

```
Forward:
x ─────┬─────────→ W₁₁ ────┐
       ├─────────→ W₁₂ ────┼──→ y₁
       │                   │
       ├─────────→ W₂₁ ────┤
       └─────────→ W₂₂ ────┼──→ y₂
                           
Backward (gradients flow back):
∂L/∂x ←──┬────── W₁₁ᵀ ←────┐
         ├────── W₁₂ᵀ ←────┼── ∂L/∂y₁
         │                 │
         ├────── W₂₁ᵀ ←────┤
         └────── W₂₂ᵀ ←────┼── ∂L/∂y₂
```

### ReLU: y = max(0, x)

```
Forward:          Backward:
x > 0:            gradient passes through
x = 3 → y = 3    ∂L/∂y = 1 → ∂L/∂x = 1

x ≤ 0:            gradient is blocked
x = -2 → y = 0   ∂L/∂y = 1 → ∂L/∂x = 0
```

ReLU is like a **gate**: open for positive, closed for negative.

---

## Why Caching Matters

During forward pass, we cache values needed for backward:

```python
class Linear:
    def forward(self, x):
        self.input = x  # CACHE! Needed for ∂L/∂W
        return x @ self.W + self.b
    
    def backward(self, grad_output):
        # Use cached input to compute weight gradient
        self.grad_W = self.input.T @ grad_output
        ...
```

**Without caching**: Would need to recompute forward pass during backward
**With caching**: One forward, one backward (efficient)

---

## Gradient Accumulation

For batches, gradients accumulate:

```
Batch of 32 samples, each gives a gradient
Final gradient = average of all 32 gradients

∂L/∂W = (1/32) × Σᵢ ∂Lᵢ/∂W
```

---

## Intuition Checkpoints ✅

Before moving on, understand:

1. **What does backprop compute?**
   <details><summary>Answer</summary>The gradient of the loss with respect to all parameters (∂L/∂W, ∂L/∂b for each layer).</details>

2. **Why do we need the chain rule?**
   <details><summary>Answer</summary>Because the loss depends on parameters through many intermediate computations. The chain rule lets us break this into local gradients.</details>

3. **What does each layer pass backward?**
   <details><summary>Answer</summary>The gradient of the loss with respect to its input (∂L/∂x), so the previous layer can continue the chain.</details>

4. **Why cache the forward pass inputs?**
   <details><summary>Answer</summary>We need the input x to compute ∂L/∂W = xᵀ · ∂L/∂y. Without caching, we'd need to recompute.</details>

---

*"Forward pass is computation. Backward pass is learning."*
