# Topic 05: Interview Questions

MLP and initialization questions are FAANG favorites!

---

## Q1: Why Xavier Initialization? (Google, Meta)

**Difficulty**: Medium | **Time**: 5 min

Explain Xavier initialization and why it's important.

<details>
<summary>Answer</summary>

**The Problem**: With random initialization, activations can explode or vanish.

If we initialize weights too large:
- Activations grow exponentially through layers
- Gradients explode during backprop

If we initialize weights too small:
- Activations shrink to zero
- Gradients vanish

**Xavier Solution**: 
```
W ~ N(0, 2/(n_in + n_out))
```

This keeps variance of activations constant across layers by balancing:
- Forward pass: Var(output) ≈ Var(input)
- Backward pass: Var(grad) ≈ Var(next_grad)

**Derivation sketch**:
For y = Wx, Var(y) = n_in × Var(W) × Var(x)
To keep Var(y) = Var(x), we need Var(W) = 1/n_in
Considering backward pass too: Var(W) = 2/(n_in + n_out)

</details>

---

## Q2: Xavier vs Kaiming (Amazon)

**Difficulty**: Medium | **Time**: 5 min

When should you use Kaiming initialization instead of Xavier?

<details>
<summary>Answer</summary>

**Use Kaiming (He) for ReLU networks**.

Xavier assumes activation is linear (or tanh). But ReLU zeros out half the outputs:
```
ReLU(x) = max(0, x)
E[ReLU(x)] ≈ 0.5 × E[x]  (for symmetric x around 0)
Var[ReLU(x)] ≈ 0.5 × Var[x]
```

This halving compounds through layers, causing vanishing activations.

**Kaiming fix**: Multiply variance by 2 to compensate:
```
W ~ N(0, 2/n_in)
```

**Rule of thumb**:
- Tanh/Sigmoid → Xavier
- ReLU/LeakyReLU → Kaiming

```python
# Xavier
W = np.random.randn(n_in, n_out) * np.sqrt(2 / (n_in + n_out))

# Kaiming (fan_in mode)
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```

</details>

---

## Q3: Why Bias Initialized to Zero? (Basic)

**Difficulty**: Easy | **Time**: 2 min

Why initialize biases to zero while weights are random?

<details>
<summary>Answer</summary>

**Symmetry breaking is the goal**.

If all weights AND biases were zero:
- All neurons compute the same thing
- All gradients are identical
- Network can never learn different features

Random weights break this symmetry.

**Biases don't need randomness** because:
1. Different random weights already ensure neurons behave differently
2. Zero bias is a reasonable starting point (no prior assumption)
3. Biases will be learned during training

**Exception**: Sometimes biases are initialized to small positive values (e.g., 0.01) for ReLU to ensure neurons are "alive" initially.

</details>

---

## Q4: Dying ReLU Problem (Common)

**Difficulty**: Medium | **Time**: 5 min

What is the dying ReLU problem and how can you prevent it?

<details>
<summary>Answer</summary>

**The Problem**: ReLU neurons can "die" and never activate again.

If a neuron's output becomes negative for all training examples:
- ReLU outputs 0 for all inputs
- Gradient is 0 (no learning happens)
- Neuron is permanently dead

**Causes**:
1. Large learning rate causes weights to overshoot
2. Poor initialization (too many negative pre-activations)
3. Bad luck during training

**Solutions**:

1. **Use LeakyReLU**: Small gradient for negative inputs
   ```python
   def leaky_relu(x, alpha=0.01):
       return np.where(x > 0, x, alpha * x)
   ```

2. **Use Kaiming initialization**: Proper variance prevents early death

3. **Lower learning rate**: Prevents overshooting

4. **Use batch normalization**: Keeps activations in reasonable range

5. **Use ELU**: Smooth and has negative values
   ```python
   def elu(x, alpha=1.0):
       return np.where(x > 0, x, alpha * (np.exp(x) - 1))
   ```

</details>

---

## Q5: Implement Forward Pass (Whiteboard)

**Difficulty**: Medium | **Time**: 10 min

Implement a forward pass through a 3-layer MLP.

<details>
<summary>Answer</summary>

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def kaiming_init(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)

class MLP:
    def __init__(self, sizes):  # e.g., [784, 256, 128, 10]
        self.weights = []
        self.biases = []
        
        for i in range(len(sizes) - 1):
            W = kaiming_init(sizes[i], sizes[i+1])
            b = np.zeros(sizes[i+1])
            self.weights.append(W)
            self.biases.append(b)
    
    def forward(self, x):
        """
        Forward pass through all layers.
        
        Args:
            x: Input of shape (batch_size, input_dim) or (input_dim,)
        
        Returns:
            Output logits
        """
        # Handle single sample
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        h = x
        # All layers except last use ReLU
        for i in range(len(self.weights) - 1):
            z = np.dot(h, self.weights[i]) + self.biases[i]
            h = relu(z)
        
        # Last layer: no activation (logits)
        output = np.dot(h, self.weights[-1]) + self.biases[-1]
        return output

# Example usage
mlp = MLP([784, 256, 128, 10])
x = np.random.randn(32, 784)  # Batch of 32
logits = mlp.forward(x)  # Shape: (32, 10)
```

</details>

---

## Q6: Effect of Initialization Scale (Deep)

**Difficulty**: Hard | **Time**: 8 min

What happens if you scale weights by 0.001 vs 1.0? Show mathematically.

<details>
<summary>Answer</summary>

Consider L layers, each with weight variance σ².

**Forward pass variance propagation**:
```
Var(h_L) = (n × σ²)^L × Var(x)
```

where n is layer width (assuming all layers same size).

**With σ² = 0.001 (too small)**:
If n = 256: n × σ² = 0.256
After 10 layers: 0.256^10 ≈ 1.2e-6

Activations shrink to effectively zero! This is **vanishing activations**.

**With σ² = 1.0 (too large)**:
n × σ² = 256
After 10 layers: 256^10 ≈ 1.2e24

Activations explode! Neural network outputs become NaN.

**With Xavier (σ² = 1/n)**:
n × σ² = 1
After 10 layers: 1^10 = 1

Activations stay stable!

**Practical test**:
```python
def test_initialization(scale, depth=10, width=256):
    x = np.random.randn(100)
    for _ in range(depth):
        W = np.random.randn(width, width) * scale
        x = np.maximum(0, np.dot(x, W))  # ReLU
    return np.mean(x), np.std(x)
```

</details>

---

## 🎯 Interview Tips

1. **Know the formulas**: Xavier = sqrt(2/(in+out)), Kaiming = sqrt(2/in)
2. **Understand when to use each**: Activation function determines init
3. **Be ready to derive**: Show why variance is preserved
4. **Know the failure modes**: Vanishing/exploding activations

---

*"Initialization is not just a detail—it determines if your network can learn at all."*
