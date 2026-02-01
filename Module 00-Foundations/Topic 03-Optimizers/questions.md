# Topic 03: Interview Questions

Optimizer questions are common in ML systems interviews!

---

## Q1: Explain Gradient Descent Variants (Entry Level)

**Difficulty**: Easy | **Time**: 5 min

What's the difference between Batch, Mini-batch, and Stochastic Gradient Descent?

<details>
<summary>Answer</summary>

**Batch Gradient Descent**:
- Uses ALL training samples to compute one gradient
- Very stable but slow
- Memory-intensive for large datasets

**Stochastic Gradient Descent (SGD)**:
- Uses ONE sample to compute gradient
- Very noisy but fast
- Good for online learning

**Mini-batch Gradient Descent**:
- Uses a BATCH of samples (e.g., 32, 64, 256)
- Balances stability and speed
- Enables GPU parallelism
- **This is what everyone actually uses!**

```
Update frequency:
Batch:      1 update per epoch
Mini-batch: N/batch_size updates per epoch
SGD:        N updates per epoch (N = dataset size)
```

</details>

---

## Q2: Implement SGD with Momentum (Common)

**Difficulty**: Medium | **Time**: 5 min

Write the update equations for SGD with momentum.

```python
class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        # Your code here
        pass
    
    def step(self):
        # Your code here
        pass
```

<details>
<summary>Answer</summary>

```python
class SGDMomentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        # Initialize velocity for each parameter
        self.velocity = [0.0 for _ in params]
    
    def step(self):
        for i, p in enumerate(self.params):
            # Update velocity: v = β*v - lr*grad
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * p.grad
            # Update parameter: θ = θ + v
            p.data += self.velocity[i]
```

**Key insight**: Velocity accumulates past gradients, creating "momentum" in consistent directions while canceling oscillations.

**With momentum = 0.9**:
- Each gradient contributes 10% immediately
- ...and 9% × 10% = 0.9% the next step
- ...and 0.81% × 10% = 0.081% two steps later
- Effectively averages over ~10 recent gradients

</details>

---

## Q3: Why Does Adam Work So Well? (Google, OpenAI)

**Difficulty**: Medium | **Time**: 5 min

Explain the components of Adam and why it's the default choice.

<details>
<summary>Answer</summary>

**Adam = Momentum + RMSprop + Bias Correction**

**Component 1: First Moment (Momentum)**
```
m = β₁ × m + (1 - β₁) × g
```
- Exponential moving average of gradients
- Smooths gradient estimates
- β₁ = 0.9 typical

**Component 2: Second Moment (RMSprop)**
```
v = β₂ × v + (1 - β₂) × g²
```
- Exponential moving average of squared gradients
- Tracks gradient magnitude per parameter
- β₂ = 0.999 typical

**Component 3: Bias Correction**
```
m̂ = m / (1 - β₁ᵗ)
v̂ = v / (1 - β₂ᵗ)
```
- Corrects for initialization bias toward zero
- Critical in early training

**Why it works well**:
1. **Adaptive**: Different learning rates per parameter
2. **Robust**: Works with sparse gradients
3. **Efficient**: Low memory overhead
4. **Practical**: Less hyperparameter tuning needed

**When Adam might fail**:
- May not generalize as well as SGD for some tasks
- Can get stuck with very large/small learning rates

</details>

---

## Q4: Learning Rate Warm-up (Systems Interview)

**Difficulty**: Medium | **Time**: 5 min

Why do we sometimes warm-up the learning rate at the start of training?

<details>
<summary>Answer</summary>

**The problem**: At initialization, the model's output is essentially random. Large gradients + large learning rate = unstable updates.

**The solution**: Start with a small learning rate and gradually increase it.

```python
def warmup_lr(step, warmup_steps, target_lr):
    if step < warmup_steps:
        return target_lr * (step / warmup_steps)
    return target_lr
```

**Why it helps**:
1. **Stable early gradients**: Random initialization → poor gradients
2. **Better momentum initialization**: Allows Adam/momentum to build up meaningful statistics
3. **Prevents explosion**: Especially important with LayerNorm and large batch sizes

**When to use**:
- Large batch training
- Transformer models
- When training on new tasks

**Typical warmup**: 1-5% of total training steps

</details>

---

## Q5: Weight Decay vs L2 Regularization (Tricky!)

**Difficulty**: Hard | **Time**: 10 min

Are weight decay and L2 regularization the same thing?

<details>
<summary>Answer</summary>

**Short answer**: They're equivalent for vanilla SGD, but DIFFERENT for Adam!

**L2 Regularization** (add to loss):
```
L_total = L_original + (λ/2) × ||θ||²

Gradient: ∂L/∂θ = ∂L_orig/∂θ + λθ
```

**Weight Decay** (add to update):
```
θ = θ - lr × (∂L/∂θ + λθ)
  = θ - lr × ∂L/∂θ - lr×λ × θ
```

**For SGD**: Mathematically equivalent!

**For Adam**: DIFFERENT!

With L2 in Adam:
```
m = β₁m + (1-β₁)(g + λθ)  # λθ goes into momentum
v = β₂v + (1-β₂)(g + λθ)² # λθ affects second moment
θ = θ - lr × m̂/√v̂
```

With decoupled weight decay (AdamW):
```
m = β₁m + (1-β₁)g  # Clean gradient
v = β₂v + (1-β₂)g²
θ = θ - lr × m̂/√v̂ - lr×λ × θ  # Separate decay
```

**AdamW is preferred** because weight decay is applied consistently, regardless of gradient magnitude.

</details>

---

## Q6: Gradient Clipping (Practical)

**Difficulty**: Medium | **Time**: 5 min

When and how do you clip gradients?

<details>
<summary>Answer</summary>

**When to use**: 
- Exploding gradients (loss goes to NaN)
- RNNs and Transformers
- Large learning rates

**Two approaches**:

**1. Clip by Value** (per-element):
```python
grad = np.clip(grad, -max_val, max_val)
```
- Simple but changes gradient direction

**2. Clip by Norm** (preserve direction):
```python
grad_norm = np.linalg.norm(all_grads)
if grad_norm > max_norm:
    scale = max_norm / grad_norm
    for g in all_grads:
        g *= scale
```
- Preserves gradient direction
- More commonly used

**In PyTorch**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Typical values**: max_norm = 1.0 or 0.5

**Implementation note**: Always clip BEFORE optimizer.step()!

</details>

---

## Q7: Compare Optimizer Convergence (Whiteboard)

**Difficulty**: Medium | **Time**: 10 min

On a whiteboard, sketch how SGD, Momentum, and Adam converge on a saddle point.

<details>
<summary>Answer</summary>

**Saddle Point Scenario**:
```
Loss landscape: f(x, y) = x² - y²

            y
            │
        ────┼────
            │
───x────────●───── x
            │
        ────┼────
            │
            y

At origin: gradient = 0, but it's not a minimum!
```

**SGD**:
- Gets stuck or very slow near saddle
- No momentum to push through flat regions
- Path: slow, oscillating

**Momentum**:
- Builds up speed approaching saddle
- Can "roll through" the flat region
- Path: faster, less oscillation

**Adam**:
- Adapts step size based on gradient history
- In flat region: larger effective learning rate
- Path: efficient, adaptive

**Visualization**:
```
SGD:       ............●.........→ (slow progress)

Momentum:  ....●====→→→→→→→ (accelerates through)

Adam:      ..●═══→→→ (adaptive, efficient)
```

</details>

---

## 🎯 Interview Tips

1. **Know the math**: Be ready to write update equations
2. **Know the defaults**: Adam β₁=0.9, β₂=0.999, ε=1e-8
3. **Know when to use what**: Adam for most cases, SGD for research
4. **Mention learning rate scheduling**: Shows depth of knowledge
5. **Discuss AdamW vs Adam**: Shows you're up-to-date

---

*"The optimizer interview question is a test of whether you've actually trained neural networks, not just read about them."*
