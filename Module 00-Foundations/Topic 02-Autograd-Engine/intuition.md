# Topic 02: Intuition Guide

Understanding autograd is about building the right mental models.

---

## 🧠 The Core Insight

> **Every computation is a graph. Gradients flow backward through that graph.**

This single idea powers all of modern deep learning.

---

## Mental Model 1: The River of Gradients 🌊

Think of a computational graph as a river system:

```
Inputs (Source) → Operations → Output (Ocean)
```

**Forward pass**: Water flows downstream (inputs → output)
**Backward pass**: Gradients flow upstream (output → inputs)

Just as you'd trace a river back to its sources, we trace gradients back to the original variables.

---

## Mental Model 2: The Chain of Responsibility ⛓️

Each operation is a link in a chain:

```
x → [+3] → [×2] → [relu] → [sum] → loss
```

During backward:
1. Loss says "I need gradient 1.0"
2. Sum says "I'll pass 1.0 to each input"
3. ReLU says "I'll pass gradient through if positive"
4. ×2 says "I'll multiply gradient by 2"
5. +3 says "I'll pass gradient unchanged"

Each operation only needs to know its **local derivative**!

---

## Mental Model 3: Computational Graphs 📊

Every expression has a graph structure:

```python
z = (x + y) * x

# Graph:
#       z (*)
#      / \
#     a   x
#    (+)
#   / \
#  x   y
```

**Node**: A value (x, y, a, z)
**Edge**: Data dependency (who needs whom)

**Forward**: Bottom-up (evaluate leaves first)
**Backward**: Top-down (start from root)

---

## Mental Model 4: The Chain Rule Machine ⚙️

The chain rule is your only tool:

```
If z = f(y) and y = g(x), then:

dz/dx = dz/dy × dy/dx
```

For a computation path:
```
x → y → z → L

dL/dx = dL/dz × dz/dy × dy/dx
```

Each operation contributes one factor!

---

## Mental Model 5: Gradient as "Sensitivity" 📈

What does a gradient actually mean?

```python
x = Value(2.0)
z = x * x  # z = 4
z.backward()
print(x.grad)  # 4.0
```

**Interpretation**: "If I change x by a tiny amount ε, z changes by 4ε"

At x=2: z = x² = 4
At x=2.001: z = 4.004001 ≈ 4 + 4×0.001

The gradient 4.0 tells us the **sensitivity** of output to input.

---

## The Backward Pass Algorithm

Here's the intuition for implementing `backward()`:

### Step 1: Build the graph during forward pass

```python
x = Value(2)
y = Value(3)
z = x * y  # z remembers it came from x, y via *
```

### Step 2: Start at the output

```python
z.grad = 1.0  # "How much does z affect z? 100%"
```

### Step 3: Work backward, applying chain rule

```python
# For z = x * y:
x.grad += z.grad * y.data  # dz/dx = y = 3
y.grad += z.grad * x.data  # dz/dy = x = 2
```

### Step 4: Handle graph structure correctly

**Problem**: What if a value is used multiple times?

```python
x = Value(2)
z = x + x  # x appears twice!
```

**Solution**: Accumulate gradients with `+=`

```python
x.grad += 1.0 * z.grad  # From first x
x.grad += 1.0 * z.grad  # From second x
# Total: 2.0, which is correct!
```

### Step 5: Process in topological order

We must process nodes so that a node's `grad` is complete before we use it.

```
z → y → x  (process in this order for backward)
```

---

## Common Pitfalls 🚫

### 1. Forgetting to accumulate

```python
# Wrong:
self.grad = out.grad * local_derivative

# Right:
self.grad += out.grad * local_derivative
```

### 2. Processing in wrong order

```python
# Wrong: arbitrary order
for node in all_nodes:
    node._backward()

# Right: topological order (reversed)
for node in reversed(topo_sorted_nodes):
    node._backward()
```

### 3. Not handling scalars vs Values

```python
# Need to wrap scalars
x = Value(2.0)
y = x * 3  # 3 needs to become Value(3)
```

### 4. Forgetting to initialize output gradient

```python
# Wrong: forgot to set initial gradient
self._backward()

# Right:
self.grad = 1.0  # Start with 1
for v in reversed(topo):
    v._backward()
```

---

## Why Reverse Mode?

**Question**: Why do we go backward instead of forward?

**Answer**: Efficiency!

For `f: R^n → R^m`:
- **Forward mode**: O(n) passes for n gradients
- **Reverse mode**: O(m) passes for all gradients

Neural networks: millions of inputs, one output (loss).
**Reverse mode wins by a factor of millions!**

---

## Visualization

```
Forward pass (compute values):

  x=2     y=3
    \     /
     \   /
      (+)→a=5
       \
        \   x=2
         \ /
         (*)→z=10


Backward pass (compute gradients):

  x.grad=7  y.grad=2
     ↑        ↑
     |        |
   dz/da=2   |
      ↖     /
       (+)←a.grad=2
        ↖
         \  da/dx=1
          \ 
         (*)←z.grad=1
          ↑
    dz/dz = 1 (start here)
```

---

## Intuition Checkpoints ✅

Before moving on, make sure you can answer:

1. **Why `+=` for gradient accumulation?**
   <details><summary>Answer</summary>A value may be used multiple times, and we need to sum all gradient paths.</details>

2. **What's the initial gradient for backward?**
   <details><summary>Answer</summary>1.0 for the output node (dL/dL = 1)</details>

3. **Why topological order?**
   <details><summary>Answer</summary>To ensure a node's gradient is complete before we propagate through it.</details>

4. **What's the derivative of ReLU?**
   <details><summary>Answer</summary>1 if x > 0, else 0 (step function)</details>

---

*"Once you truly understand how gradients flow backward, no PyTorch bug will ever confuse you again."*
