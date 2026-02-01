# Topic 09: Interview Questions

Regularization is a favorite interview topic!

---

## Q1: L2 vs L1 Regularization (Common)

**Difficulty**: Easy | **Time**: 3 min

<details>
<summary>Answer</summary>

**L2 (Ridge/Weight Decay)**:
- Penalty: λ × Σ wᵢ²
- Effect: Shrinks weights toward zero
- Keeps all weights small but non-zero
- Smooth, differentiable everywhere

**L1 (Lasso)**:
- Penalty: λ × Σ |wᵢ|
- Effect: Drives weights exactly to zero
- Creates sparse models (feature selection)
- Not differentiable at zero

**When to use**:
- L2: Default choice, when you need all features
- L1: When you want automatic feature selection

</details>

---

## Q2: How Does Dropout Work? (Google)

**Difficulty**: Medium | **Time**: 5 min

<details>
<summary>Answer</summary>

**During Training**:
1. For each neuron, randomly set output to 0 with probability p
2. Scale remaining outputs by 1/(1-p) (inverted dropout)

```python
def dropout_train(x, p=0.5):
    mask = np.random.rand(*x.shape) > p
    return x * mask / (1 - p)
```

**During Inference**:
- No dropout applied (all neurons active)
- Scaling during training means no adjustment needed

**Why it works**:
- Prevents co-adaptation of neurons
- Creates implicit ensemble of sub-networks
- Acts as regularization

</details>

---

## Q3: Batch Normalization Explained (Meta)

**Difficulty**: Medium | **Time**: 5 min

<details>
<summary>Answer</summary>

**Batch Norm**:
```
y = γ × (x - μ_batch) / √(σ²_batch + ε) + β
```

**During Training**:
- Use batch statistics (μ_batch, σ²_batch)
- Update running averages for inference

**During Inference**:
- Use running averages (μ_running, σ²_running)

**Why it works**:
1. Reduces internal covariate shift
2. Allows higher learning rates
3. Acts as regularization
4. Makes optimization landscape smoother

```python
class BatchNorm:
    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            # Update running stats
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            mean, var = self.running_mean, self.running_var
        
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        return self.gamma * x_norm + self.beta
```

</details>

---

## Q4: BN vs Layer Norm (Transformer interviews)

**Difficulty**: Medium | **Time**: 5 min

<details>
<summary>Answer</summary>

**Batch Norm**:
- Normalizes across batch dimension
- μ, σ per feature, computed over batch
- Needs batches of reasonable size
- Used in: CNNs, MLPs

**Layer Norm**:
- Normalizes across feature dimension
- μ, σ per sample, computed over features
- Works with batch size 1
- Used in: Transformers, RNNs

**Why Layer Norm for Transformers**:
- Variable sequence lengths
- Batch size can be small
- No dependency on other samples in batch

</details>

---

## Q5: Implement Dropout Backward (Whiteboard)

**Difficulty**: Medium | **Time**: 5 min

<details>
<summary>Answer</summary>

```python
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
    
    def forward(self, x):
        if self.training:
            self.mask = np.random.rand(*x.shape) > self.p
            return x * self.mask / (1 - self.p)
        return x
    
    def backward(self, grad_output):
        if self.training:
            return grad_output * self.mask / (1 - self.p)
        return grad_output
```

**Key insight**: Same mask used in forward and backward!

</details>

---

## 🎯 Interview Tips

1. Know L1 vs L2 tradeoffs cold
2. Understand inverted dropout scaling
3. Know BatchNorm train vs eval difference
4. Understand why LayerNorm for transformers
5. Be able to implement all three on whiteboard

---

*"Regularization prevents your model from memorizing—it forces generalization."*
