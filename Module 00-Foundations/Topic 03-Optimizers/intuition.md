# Topic 03: Intuition Guide

Building intuition for why these optimization algorithms work.

---

## 🧠 The Core Problem

> **We want to find the lowest point in a landscape we can only feel, not see.**

Imagine you're blindfolded on a hilly terrain, and you can only feel the slope under your feet. How do you find the valley?

---

## Mental Model 1: The Ball Rolling Downhill ⚽

**Gradient Descent** is like a ball rolling downhill:
- The ball moves in the direction of steepest descent
- The learning rate is like the ball's mass - bigger = faster but less control
- The ball might get stuck in small valleys (local minima)

**Adding Momentum** is like a heavier ball:
- It builds up speed on long descents
- It can roll through small bumps
- It oscillates less on narrow valleys

---

## Mental Model 2: The Skeptical Navigator 🧭

Think of **Adam** as a skeptical navigator:

1. **First moment (momentum)**: "What direction have we been going?"
2. **Second moment (RMSprop)**: "How confident are we in this direction?"
3. **Adaptive step**: In uncertain directions → smaller steps

```
Confident direction (consistent gradients)  → BIG steps
Uncertain direction (noisy gradients)       → small steps
```

---

## Mental Model 3: The Learning Rate Story 📖

**SGD**: One learning rate for everyone
```
"Everyone moves at the same speed,
 whether they're on a highway or a mountain path."
```

**Adam**: Personalized learning rates
```
"Parameters on bumpy terrain walk slowly.
 Parameters on smooth terrain can run."
```

---

## Why Momentum Works

### The Oscillation Problem

Without momentum on an elongated valley:

```
        Goal
         ↓
    ←  →  ←  →  ←  →  ← →↓
    oscillating toward goal
```

The gradient points toward the walls, not toward the goal!

### Momentum Averages Out

```
Step 1: gradient = [+1, -0.5]
Step 2: gradient = [-1, -0.5]  
Step 3: gradient = [+1, -0.5]
...
Average: [0, -0.5]  ← Points toward goal!
```

Momentum accumulates consistent components while canceling oscillations.

---

## Why Adam Works

### The Feature Learning Problem

In neural networks, different parameters learn at different rates:
- Frequently-used features have large gradients
- Rarely-used features have small gradients

### Adam's Solution

```
If gradients are consistently large:
    Second moment v is large
    Step size = lr / √v is SMALL
    → Prevents overshooting

If gradients are consistently small:
    Second moment v is small
    Step size = lr / √v is LARGE
    → Accelerates learning
```

This is called **adaptive learning rates**.

---

## The Bias Correction Mystery

At initialization:
```
m₀ = 0
v₀ = 0
```

After one step (β₁ = 0.9):
```
m₁ = 0.9 × 0 + 0.1 × g = 0.1g
```

But we want `m` to estimate the true gradient `g`, not `0.1g`!

**Bias correction** fixes this:
```
m̂₁ = m₁ / (1 - 0.9¹) = 0.1g / 0.1 = g  ✓
```

As training progresses, (1 - β₁ᵗ) → 1, so correction fades away.

---

## When to Use What

### Use SGD + Momentum when:
- Training computer vision models
- Want maximum generalization
- Have time to tune learning rate schedule
- Research/benchmarking (it's still SOTA for some tasks!)

### Use Adam when:
- Training transformers or RNNs
- Starting a new project (faster to get working)
- Dealing with sparse data
- Limited hyperparameter tuning budget

### Use AdamW when:
- You need weight decay (most cases!)
- Training large language models
- Modern best practice

---

## Hyperparameter Intuitions

### Learning Rate
```
Too small: Training will take forever
Too large: Training will explode (NaN)
Just right: Smooth but fast convergence
```

**Rule of thumb**: Start with 3e-4 for Adam, 0.1 for SGD

### Momentum (β₁ for Adam, momentum for SGD)
```
0.9:  Standard, works for most cases
0.99: Smoother, slower to respond to changes
0.0:  No momentum, equivalent to vanilla SGD
```

### Second Moment (β₂ for Adam)
```
0.999: Standard, tracks long-term gradient scale
0.99:  Adapts faster, more responsive
0.9999: Very stable, slow to adapt
```

### Epsilon (ε)
```
1e-8: Standard, prevents division by zero
1e-4: Can help with very small gradients
```

---

## Common Mistakes

### 1. Not zeroing gradients
```python
# WRONG - gradients accumulate!
for step in range(1000):
    loss.backward()
    optimizer.step()

# RIGHT
for step in range(1000):
    optimizer.zero_grad()  # ← Essential!
    loss.backward()
    optimizer.step()
```

### 2. Wrong learning rate for optimizer
```python
# For Adam, start small
Adam(params, lr=1e-3)  # Good
Adam(params, lr=0.1)    # Too high!

# For SGD, can go higher
SGD(params, lr=0.1)     # Often fine
SGD(params, lr=1e-4)    # Too low for SGD
```

### 3. Forgetting weight decay with Adam
```python
# Use AdamW, not Adam + L2
Adam(params, lr=1e-3, weight_decay=0.01)     # Works, but not optimal
AdamW(params, lr=1e-3, weight_decay=0.01)    # Better!
```

---

## Intuition Checkpoints ✅

Before moving on, make sure you understand:

1. **Why does momentum help?**
   <details><summary>Answer</summary>It accumulates consistent gradient directions while canceling oscillations.</details>

2. **What makes Adam "adaptive"?**
   <details><summary>Answer</summary>It maintains per-parameter learning rates based on gradient history.</details>

3. **Why do we need bias correction?**
   <details><summary>Answer</summary>The exponential moving averages are biased toward zero at initialization.</details>

4. **When might SGD beat Adam?**
   <details><summary>Answer</summary>SGD sometimes generalizes better because it finds wider minima.</details>

---

*"The optimizer is the workhorse of training. Choose wisely, and tune carefully."*
