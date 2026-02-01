# Topic 02: Autograd Engine From Scratch

> **Goal**: Implement a complete automatic differentiation engine.
> **Time**: 3-4 hours | **Difficulty**: Challenging

---

## ⚠️ Escape Hatch Available

This topic is challenging. If you get stuck after 3+ hours:

```python
# Use the provided working autograd
from utils.autograd_stub import Value

# Continue with the curriculum and return to implement your own later
```

**No shame in using this!** The goal is learning, not getting stuck.

---

## 🎯 Learning Objectives

By the end of this topic, you will:
1. Understand computational graphs and how they represent operations
2. Implement reverse-mode automatic differentiation (backpropagation)
3. Build a `Value` class that tracks gradients through any computation
4. Debug gradient issues using numerical gradient checking

---

## 📋 The Problem

Implement a complete autograd engine:

### Required Class

```python
class Value:
    """
    Wraps a scalar and tracks gradient through computations.
    
    Example usage:
        x = Value(2.0)
        y = Value(3.0)
        z = x * y + x  # z = 8
        z.backward()   # Compute gradients
        print(x.grad)  # dz/dx = 4 (y + 1 = 4)
        print(y.grad)  # dz/dy = 2 (x = 2)
    """
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data       # The actual scalar value
        self.grad = 0.0        # Accumulated gradient
        self._backward = lambda: None  # Function to compute local gradients
        self._prev = set(_children)    # Parent nodes in the graph
        self._op = _op                 # Operation that created this node
    
    def __add__(self, other): ...
    def __mul__(self, other): ...
    def __pow__(self, other): ...
    def relu(self): ...
    def tanh(self): ...
    def backward(self): ...  # The key method!
```

---

## 🧠 Key Concepts to Master

### 1. Computational Graphs
Every mathematical expression is a graph:
```
z = x * y + x

     +  ← z
    / \
   *   x
  / \
 x   y
```

### 2. Forward vs Backward
- **Forward**: Compute output from inputs
- **Backward**: Compute gradients from output back to inputs

### 3. The Chain Rule
```
If z = f(y) and y = g(x), then:
dz/dx = dz/dy × dy/dx
```

### 4. Local Gradients
Each operation knows its own derivative:
- `z = x + y` → `dz/dx = 1`, `dz/dy = 1`
- `z = x * y` → `dz/dx = y`, `dz/dy = x`
- `z = relu(x)` → `dz/dx = 1 if x > 0 else 0`

---

## 📁 File Structure

```
Topic 02-Autograd-Engine/
├── README.md             # This file
├── questions.md          # Interview-style questions
├── intuition.md          # Conceptual explanations
├── math-refresh.md       # Calculus refresher
├── hints/
│   ├── hint-1-structure.md   # Class design
│   ├── hint-2-backward.md    # Backprop implementation
│   └── hint-3-topological.md # Graph traversal
├── solutions/
│   ├── level01-naive.py          # Basic ops only
│   ├── level02-vectorized.py     # More operations
│   ├── level03-memory-efficient.py  # Optimized
│   └── level04-pytorch-reference.py  # Verification
├── tests/
│   ├── test_basic.py     # Core functionality
│   ├── test_edge.py      # Edge cases
│   └── test_stress.py    # Performance tests
└── visualization.py      # Computational graph visualizer
```

---

## 🎮 How to Use This Topic

### Step 1: Understand the Concept
1. Read `intuition.md` to understand computational graphs
2. Review `math-refresh.md` for chain rule practice
3. Look at `questions.md` for interview preparation

### Step 2: Implement Step by Step
Start with the simplest operations:

1. **Addition and multiplication** (most fundamental)
2. **Backward pass** for these ops
3. **Topological sort** for proper ordering
4. **More operations** (relu, tanh, pow, etc.)

### Step 3: Test Extensively
```bash
# Compare against numerical gradients
python -c "
from utils.finite_difference_checker import check_gradients
# Your tests here
"

# Run official tests
pytest tests/test_basic.py -v
```

### Step 4: Verify Against PyTorch
```bash
python solutions/level04-pytorch-reference.py
```

---

## 🏆 Success Criteria

| Level | Requirement |
|-------|-------------|
| Level 1 | `+`, `*`, `backward()` work correctly |
| Level 2 | All ops (relu, tanh, pow, exp) work |
| Level 3 | Matches PyTorch gradients within 1e-5 |
| Level 4 | Can train a simple neural network |

---

## 🆘 When to Use the Escape Hatch

Use `utils/autograd_stub.py` if:
- You've been stuck for 3+ hours
- You want to progress to later topics
- You'll return to implement it properly later

```python
from utils.autograd_stub import Value

# This works exactly like your implementation should
x = Value(2.0)
y = Value(3.0)
z = x * y + x
z.backward()
print(x.grad)  # 4.0
```

---

## 🔗 Related Topics

- **Topic 01**: Tensor Operations (tensors are just multi-dim Values)
- **Topic 06**: Backpropagation (autograd applied to neural networks)
- **Topic 25**: Self-Attention (uses autograd for gradients)

---

## 💡 Key Insight

> **Autograd is the foundation of ALL modern deep learning.**
> 
> Once you understand how gradients flow backward through a computation graph,
> you'll never be confused by PyTorch's `.backward()` again.

---

*"The engineer who can implement autograd from scratch has truly understood deep learning."*
