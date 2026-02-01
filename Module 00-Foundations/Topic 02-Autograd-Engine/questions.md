# Topic 02: Interview Questions

Autograd is a favorite interview topic at ML research labs!

---

## Q1: Explain Backpropagation (Google Brain, OpenAI)

**Difficulty**: Medium | **Time**: 5 min

Explain backpropagation to someone who knows calculus but not machine learning.

<details>
<summary>Answer</summary>

**One-sentence summary**: Backpropagation is an efficient algorithm for computing gradients of a function by working backward from the output.

**Key points to cover**:

1. **Forward pass**: Compute the output by flowing data through operations
2. **Backward pass**: Apply chain rule in reverse order
3. **Why backward?**: Reuses intermediate values, O(n) instead of O(n²)

**Example**:
```
f(x, y) = (x + y) * x

Forward:
  a = x + y = 2 + 3 = 5
  b = a * x = 5 * 2 = 10

Backward (starting with db/df = 1):
  da/df = db/da = x = 2
  dx/df = db/dx + da/dx × da/df = a + 1 × 2 = 5 + 2 = 7
  dy/df = da/dy × da/df = 1 × 2 = 2
```

**Why not forward mode?**: With many inputs and one output (typical in ML), reverse mode is much more efficient.

</details>

---

## Q2: Implement Multiplication Backward (Meta, DeepMind)

**Difficulty**: Easy | **Time**: 3 min

Given the forward pass `z = x * y`, write the backward pass.

```python
class Value:
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            # Your code here
            pass
        
        out._backward = _backward
        return out
```

<details>
<summary>Answer</summary>

```python
def _backward():
    self.grad += other.data * out.grad
    other.grad += self.data * out.grad
```

**Why `+=` instead of `=`?**

A value might be used multiple times in an expression:
```python
x = Value(2.0)
z = x * x  # x is used twice!
```

Here, `dz/dx = 2x = 4`, but if we used `=`, we'd only get half the gradient.

**Derivation**:
```
z = x * y
∂z/∂x = y
∂z/∂y = x

With chain rule:
∂L/∂x = ∂L/∂z × ∂z/∂x = out.grad × y
∂L/∂y = ∂L/∂z × ∂z/∂y = out.grad × x
```

</details>

---

## Q3: Why Topological Sort? (Apple, NVIDIA)

**Difficulty**: Medium | **Time**: 5 min

Why does backpropagation require topological ordering?

<details>
<summary>Answer</summary>

**The problem**: We need to process nodes in the correct order so that `out.grad` is complete before we use it.

**Example of wrong order**:
```
x → a → b → L
      ↘ /
```

If we compute `∂L/∂a` before `∂L/∂b`, we'll miss part of the gradient that flows through `b`.

**Topological sort guarantees**:
1. A node is processed only after all its consumers are processed
2. `out.grad` accumulates all contributions before being used

**Implementation pattern**:
```python
def backward(self):
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    
    build_topo(self)
    
    self.grad = 1.0
    for v in reversed(topo):  # Reverse = output to input
        v._backward()
```

</details>

---

## Q4: Gradient Accumulation Bug (Common Interview)

**Difficulty**: Medium | **Time**: 5 min

What's wrong with this code?

```python
x = Value(2.0)
y = Value(3.0)

# First forward/backward
z1 = x * y
z1.backward()
print(x.grad)  # 3.0 ✓

# Second forward/backward
z2 = x + y
z2.backward()
print(x.grad)  # What does this print?
```

<details>
<summary>Answer</summary>

**The bug**: `x.grad` accumulates across multiple backward passes!

After the second backward:
```python
print(x.grad)  # 4.0, NOT 1.0!
```

**Why?**: Gradients use `+=`, so the second backward adds to the first.

**The fix**: Zero gradients before each backward pass!

```python
x = Value(2.0)
y = Value(3.0)

z1 = x * y
z1.backward()
print(x.grad)  # 3.0

# Zero gradients!
x.grad = 0.0
y.grad = 0.0

z2 = x + y
z2.backward()
print(x.grad)  # 1.0 ✓
```

**In PyTorch**: This is why you always call `optimizer.zero_grad()` before each training step!

</details>

---

## Q5: Implement ReLU Backward (Entry Level)

**Difficulty**: Easy | **Time**: 2 min

Implement the backward pass for ReLU.

```python
def relu(self):
    out = Value(max(0, self.data), (self,), 'ReLU')
    
    def _backward():
        # Your code here
        pass
    
    out._backward = _backward
    return out
```

<details>
<summary>Answer</summary>

```python
def _backward():
    self.grad += (out.data > 0) * out.grad
```

**Explanation**:
- ReLU(x) = max(0, x)
- d/dx ReLU(x) = 1 if x > 0, else 0
- We use `out.data > 0` because that's where we stored the forward result

**Common mistake**: Using `self.data > 0`:
```python
# Wrong if self.data was 0.1 but out.data is 0.1 (correct)
# But what if numerical precision issues?
```

**Best practice**: Check the output, not the input, for threshold boundary cases.

</details>

---

## Q6: Gradient Through Non-Differentiable Point (Advanced)

**Difficulty**: Hard | **Time**: 10 min

ReLU is not differentiable at x=0. How do autograd systems handle this?

<details>
<summary>Answer</summary>

**Short answer**: They use a **subgradient** (typically 0 at the kink).

**Practical implementations**:

1. **PyTorch**: d/dx ReLU(0) = 0
2. **TensorFlow**: d/dx ReLU(0) = 0
3. **JAX**: d/dx ReLU(0) = 0

**Why this works in practice**:
1. The probability of landing exactly at x=0 (with float precision) is ~0
2. Even if we hit 0, a single wrong gradient doesn't break training
3. The model averages over many samples, washing out individual errors

**Alternative: Leaky ReLU**:
```
LeakyReLU(x) = x if x > 0 else 0.01x
```
This is differentiable everywhere!

**Interview follow-up**: What about Heaviside step function (output = 0 if x < 0, else 1)?
- This has zero gradient almost everywhere, making it useless for gradient-based learning
- One solution: Straight-Through Estimator (use surrogate gradient)

</details>

---

## Q7: Memory vs Compute Tradeoff (Systems Interview)

**Difficulty**: Hard | **Time**: 10 min

In your autograd implementation, what's stored during the forward pass?

<details>
<summary>Answer</summary>

**What we store**:
1. **Output values** (`out.data`)
2. **Parent references** (`_prev`)
3. **Backward functions** (`_backward` closures)

**Memory cost per operation**: O(1) for the reference, but the closure captures:
- References to input nodes
- The output node itself
- Any intermediate values needed for the backward

**Example for matmul y = x @ W**:
```python
def _backward():
    self.grad += out.grad @ W.T  # Need W
    W.grad += x.T @ out.grad     # Need x
```

We need to store both `x` and `W` to compute gradients!

**Tradeoff**: Gradient checkpointing
- Don't store intermediate activations
- Recompute them during backward
- Trades compute for memory (useful for very deep networks)

**PyTorch equivalent**: `torch.utils.checkpoint`

</details>

---

## 🎯 Interview Tips

1. **Draw the graph**: Always sketch the computational graph
2. **Start simple**: Implement add/mul before complex ops
3. **Test with numbers**: Hand-compute simple cases
4. **Know the chain rule**: It's the foundation of everything
5. **Mention zero_grad()**: Shows you understand real-world usage

---

*"If you can explain backprop on a whiteboard with a simple example, you'll ace any ML interview question about gradients."*
