# Topic 07: Interview Questions

Activation functions are fundamental interview topics!

---

## Q1: Why ReLU Over Sigmoid? (Common)

**Difficulty**: Easy | **Time**: 3 min

<details>
<summary>Answer</summary>

**Three key reasons**:

1. **No vanishing gradient**: ReLU gradient is 1 for positive inputs
   - Sigmoid: gradient → 0 for large |x|
   - ReLU: gradient = 1 for x > 0

2. **Computational efficiency**:
   - Sigmoid: requires exp(), division
   - ReLU: just max(0, x)

3. **Sparsity**: ReLU outputs zero for negative inputs
   - Creates sparse representations
   - Acts as implicit regularization

**Caveat**: ReLU can "die" (output always 0 if weights go too negative).

</details>

---

## Q2: Dying ReLU Problem (Google)

**Difficulty**: Medium | **Time**: 5 min

<details>
<summary>Answer</summary>

**The problem**: If weights push a neuron's pre-activation permanently negative:
- ReLU always outputs 0
- Gradient is always 0
- Neuron never updates → "dead"

**Causes**:
- Large learning rate
- Bad initialization
- Large negative bias

**Solutions**:
1. **LeakyReLU**: Small gradient for x < 0
   ```python
   def leaky_relu(x, alpha=0.01):
       return np.where(x > 0, x, alpha * x)
   ```

2. **ELU**: Smooth negative region
   ```python
   def elu(x, alpha=1.0):
       return np.where(x > 0, x, alpha * (np.exp(x) - 1))
   ```

3. **PReLU**: Learnable α parameter

4. **Careful initialization**: Kaiming init

</details>

---

## Q3: Softmax Temperature (Amazon)

**Difficulty**: Medium | **Time**: 5 min

What does temperature do in softmax?

<details>
<summary>Answer</summary>

**Standard softmax**: p_i = exp(z_i) / Σ exp(z_j)

**With temperature T**: p_i = exp(z_i/T) / Σ exp(z_j/T)

**Effects**:
- **T → 0**: Distribution becomes one-hot (argmax)
- **T = 1**: Standard softmax
- **T → ∞**: Distribution becomes uniform

**Use cases**:
- **Knowledge distillation**: Higher T for "soft" targets
- **Sampling**: Control randomness in generation
- **Exploration**: Higher T = more exploration

```python
def softmax_with_temperature(x, T=1.0):
    exp_x = np.exp(x / T)
    return exp_x / exp_x.sum()
```

</details>

---

## Q4: GELU vs ReLU (Meta)

**Difficulty**: Medium | **Time**: 5 min

<details>
<summary>Answer</summary>

**GELU** (Gaussian Error Linear Unit):
```
GELU(x) = x × Φ(x)
```
where Φ(x) is the standard Gaussian CDF.

**Approximation**:
```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

**Why GELU in Transformers**:
1. **Smooth**: Differentiable everywhere (ReLU has a kink)
2. **Non-monotonic**: Has a slight "bump" near 0
3. **Empirically better**: Works well in attention mechanisms

**Comparison**:
- ReLU: Sharp cutoff at 0
- GELU: Smooth, probabilistic transition

</details>

---

## Q5: Softmax Gradient (Whiteboard)

**Difficulty**: Hard | **Time**: 8 min

<details>
<summary>Answer</summary>

**Softmax**: p_i = exp(z_i) / Σ exp(z_j)

**Jacobian** (∂p_i/∂z_j):
- If i = j: p_i(1 - p_i)
- If i ≠ j: -p_i × p_j

**Matrix form**: J = diag(p) - p × pᵀ

**With cross-entropy loss**:
L = -Σ y_i log(p_i)

**Combined gradient** (the elegant result):
```
∂L/∂z = p - y
```

This is why we always combine softmax + cross-entropy!

</details>

---

## Q6: Implement All Activations (Whiteboard)

**Difficulty**: Medium | **Time**: 10 min

<details>
<summary>Answer</summary>

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_grad(y):  # Takes sigmoid output
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def tanh_grad(y):  # Takes tanh output
    return 1 - y**2

def softmax(x, axis=-1):
    exp_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
```

</details>

---

## 🎯 Interview Tips

1. Know forward AND backward for all activations
2. Understand why ReLU dominates hidden layers
3. Know when to use sigmoid (binary) vs softmax (multi-class)
4. Be able to derive softmax gradient
5. Know GELU for transformer interviews

---

*"Choose your activation wisely—it defines how your network thinks."*
