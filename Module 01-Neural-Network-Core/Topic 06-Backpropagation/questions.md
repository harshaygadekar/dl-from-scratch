# Topic 06: Interview Questions

Backpropagation is THE most asked topic in ML interviews!

---

## Q1: Explain Backpropagation (Google, Meta, Amazon)

**Difficulty**: Medium | **Time**: 5 min

Explain how backpropagation works in simple terms.

<details>
<summary>Answer</summary>

**Backpropagation = Chain rule applied systematically through a network**

**Three steps**:
1. **Forward pass**: Compute output and loss
2. **Backward pass**: Compute gradients of loss w.r.t. all parameters
3. **Update**: Adjust parameters using gradients

**The chain rule**:
For L = loss(f(g(x))), we compute:
```
∂L/∂x = ∂L/∂f × ∂f/∂g × ∂g/∂x
```

**In a neural network**:
```
x → Linear → ReLU → Linear → Softmax → Loss
         ←   ←   ←    ←    ←    ←    ← gradients flow backward
```

Each layer receives ∂L/∂output and computes:
- ∂L/∂input (to pass to previous layer)
- ∂L/∂weights (to update this layer's weights)
- ∂L/∂bias (to update this layer's bias)

</details>

---

## Q2: Derive Linear Layer Backward (Whiteboard Classic)

**Difficulty**: Hard | **Time**: 10 min

Given forward pass y = Wx + b, derive ∂L/∂W, ∂L/∂b, ∂L/∂x.

<details>
<summary>Answer</summary>

**Given**: y = Wx + b, and we receive ∂L/∂y from the next layer.

**Bias gradient** (easiest):
```
∂L/∂b = ∂L/∂y × ∂y/∂b = ∂L/∂y × 1 = ∂L/∂y
```
For batches: sum over batch dimension.

**Weight gradient**:
```
y_i = Σⱼ W_ij × x_j + b_i

∂y_i/∂W_ij = x_j

∂L/∂W_ij = Σᵢ (∂L/∂y_i) × (∂y_i/∂W_ij) = (∂L/∂y_i) × x_j

In matrix form: ∂L/∂W = x^T · ∂L/∂y
```

**Input gradient** (to pass backward):
```
∂L/∂x_j = Σᵢ (∂L/∂y_i) × (∂y_i/∂x_j) = Σᵢ (∂L/∂y_i) × W_ij

In matrix form: ∂L/∂x = ∂L/∂y · W^T
```

**Summary**:
```python
def backward(self, grad_output):  # grad_output = ∂L/∂y
    self.grad_W = self.input.T @ grad_output
    self.grad_b = grad_output.sum(axis=0)
    grad_input = grad_output @ self.W.T
    return grad_input
```

</details>

---

## Q3: ReLU Backward (Common)

**Difficulty**: Easy | **Time**: 3 min

Derive the gradient of ReLU.

<details>
<summary>Answer</summary>

**Forward**: 
```
y = max(0, x) = x if x > 0 else 0
```

**Derivative**:
```
∂y/∂x = 1 if x > 0 else 0
```

**Backward** (applying chain rule):
```
∂L/∂x = ∂L/∂y × ∂y/∂x = ∂L/∂y × 1_{x>0}
```

**Implementation**:
```python
class ReLU:
    def forward(self, x):
        self.mask = (x > 0)  # Cache the mask
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * self.mask
```

**Note**: At x = 0, ReLU is technically not differentiable. In practice, we set the gradient to 0 at x = 0.

</details>

---

## Q4: Vanishing Gradient Problem (Conceptual)

**Difficulty**: Medium | **Time**: 5 min

What causes vanishing gradients and how do you fix it?

<details>
<summary>Answer</summary>

**Cause**: When gradients are < 1 at each layer, they multiply together:
```
∂L/∂x = ∂L/∂y_L × ∂y_L/∂y_{L-1} × ... × ∂y_1/∂x

If each term ≈ 0.1, after 10 layers: 0.1^10 = 10^-10 (vanished!)
```

**Common culprits**:
1. **Sigmoid/Tanh saturation**: σ'(x) → 0 for large |x|
2. **Deep networks**: More multiplications
3. **Poor initialization**: Wrong weight variance

**Solutions**:
1. **Use ReLU**: Gradient is 1 for positive inputs (no saturation)
2. **Skip connections**: ResNets pass gradients directly
3. **Batch normalization**: Keeps activations in good range
4. **Proper initialization**: Xavier/Kaiming
5. **LSTM/GRU for RNNs**: Gated mechanisms for gradient flow

</details>

---

## Q5: Softmax + Cross-Entropy Gradient (Google)

**Difficulty**: Hard | **Time**: 8 min

Derive the gradient of softmax cross-entropy.

<details>
<summary>Answer</summary>

**Softmax**: p_i = exp(z_i) / Σⱼ exp(z_j)

**Cross-entropy**: L = -Σᵢ y_i log(p_i)

**Combined gradient** (elegant result):
```
∂L/∂z_i = p_i - y_i
```

**Derivation sketch**:
```
∂L/∂z_i = Σⱼ (∂L/∂p_j) × (∂p_j/∂z_i)

∂L/∂p_j = -y_j/p_j

∂p_j/∂z_i = p_j(δ_ij - p_i)  [Jacobian of softmax]

Combining and simplifying:
∂L/∂z_i = p_i - y_i
```

**Why it's beautiful**: The gradient is just prediction - target!

```python
def softmax_cross_entropy_backward(logits, y_true):
    probs = softmax(logits)
    return probs - y_true  # That's it!
```

</details>

---

## Q6: Numerical Gradient Check (Practical)

**Difficulty**: Medium | **Time**: 5 min

How do you verify your gradients are correct?

<details>
<summary>Answer</summary>

**Numerical gradient** using finite differences:
```
∂L/∂w ≈ (L(w + ε) - L(w - ε)) / (2ε)
```

**Implementation**:
```python
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += eps
        x_minus = x.copy()
        x_minus.flat[i] -= eps
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad

def gradient_check(analytical, numerical, rtol=1e-4):
    diff = np.abs(analytical - numerical)
    rel_error = diff / (np.abs(analytical) + np.abs(numerical) + 1e-8)
    return rel_error.max() < rtol
```

**Common issues**:
- Use ε ≈ 1e-5 (not too small due to floating point)
- Check relative error, not absolute
- Be careful with discontinuities (ReLU at 0)

</details>

---

## 🎯 Interview Tips

1. **Know the chain rule cold**: It's the foundation
2. **Be able to derive gradients**: Especially for Linear, ReLU, Softmax+CE
3. **Understand gradient flow**: Through a full network
4. **Know the failure modes**: Vanishing/exploding gradients
5. **Practice whiteboard derivations**: Clean matrix notation

---

*"Backpropagation is just calculus. The magic is in applying it efficiently."*
