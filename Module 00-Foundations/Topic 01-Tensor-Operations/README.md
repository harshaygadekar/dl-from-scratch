# Topic 01: Tensor Operations & Broadcasting

> **Goal**: Master NumPy memory layouts, stride manipulation, and broadcasting rules.
> **Time**: 2-3 hours | **Difficulty**: Foundation

---

## 🎯 Learning Objectives

By the end of this topic, you will:
1. Understand how arrays are stored in memory (row-major vs column-major)
2. Master broadcasting rules and avoid common shape mismatch bugs
3. Implement efficient element-wise operations using stride tricks
4. Debug shape-related issues in neural network code

---

## 📋 The Problem

Implement the following tensor operations from scratch:

### Required Functions

```python
def batched_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Batched matrix multiplication.
    
    Args:
        a: Shape (batch, m, k)
        b: Shape (batch, k, n)
    
    Returns:
        c: Shape (batch, m, n)
    """
    pass

def broadcast_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Add two arrays with broadcasting (like NumPy's + operator).
    
    The smaller array is "stretched" to match the larger one.
    """
    pass

def efficient_repeat(x: np.ndarray, repeats: int, axis: int) -> np.ndarray:
    """
    Repeat array along axis WITHOUT copying memory.
    
    Use np.broadcast_to or stride tricks.
    """
    pass
```

---

## 🧠 Key Concepts to Master

### 1. Memory Layouts
- **Row-major (C-order)**: Last axis changes fastest
- **Column-major (Fortran-order)**: First axis changes fastest
- **Strides**: Bytes to step in each dimension

### 2. Broadcasting Rules
1. Align shapes from the right
2. Dimensions of size 1 stretch to match
3. Missing dimensions are prepended with 1

### 3. Views vs Copies
- **Views**: Same memory, different interpretation
- **Copies**: New memory allocation (expensive)

---

## 📁 File Structure

```
Topic 01-Tensor-Operations/
├── README.md             # This file
├── questions.md          # Interview-style questions
├── intuition.md          # Conceptual explanations
├── math-refresh.md       # Linear algebra refresher
├── hints/
│   ├── hint-1-shapes.md  # Basic shape rules
│   ├── hint-2-broadcasting.md  # Broadcasting details
│   └── hint-3-memory.md  # Memory optimization
├── solutions/
│   ├── level01-naive.py          # For-loops, readable
│   ├── level02-vectorized.py     # NumPy broadcast ops
│   ├── level03-memory-efficient.py  # Stride tricks
│   └── level04-pytorch-reference.py  # Verification
├── tests/
│   ├── test_basic.py     # Core functionality
│   ├── test_edge.py      # Edge cases
│   └── test_stress.py    # Performance tests
├── visualization.py      # Memory layout visualizer
└── assets/
    └── numpy_stride_visualization.html
```

---

## 🎮 How to Use This Topic

### Step 1: Read Background Material
1. Review `math-refresh.md` if your linear algebra is rusty
2. Read `intuition.md` for the conceptual foundation
3. Look at `questions.md` for interview-style problems

### Step 2: Implement Your Solution
Start with Level 1 (naive implementation):
```bash
# Create your solution file
cp solutions/level01-naive.py my_solution.py

# Test it
pytest tests/test_basic.py -v
```

### Step 3: Level Up
Once tests pass, optimize:
- **Level 2**: Replace loops with broadcasting
- **Level 3**: Use stride tricks for zero-copy operations

### Step 4: Verify Against PyTorch
```bash
python solutions/level04-pytorch-reference.py
```

---

## 🏆 Success Criteria

| Level | Requirement |
|-------|-------------|
| Level 1 | All `test_basic.py` pass |
| Level 2 | All tests pass, no explicit loops |
| Level 3 | Memory usage matches PyTorch |
| Level 4 | Numerical equality with PyTorch |

---

## 🆘 Escape Hatch

Totally stuck after 3+ hours? It's okay to:

1. ✅ Study `solutions/level02-vectorized.py` for broadcast patterns
2. ✅ Move to Topic 02 and return later
3. ❌ Copy-paste without understanding (defeats the purpose)

---

## 🔗 Related Topics

- **Topic 02**: Autograd Engine (uses tensors extensively)
- **Topic 05**: MLP Forward Pass (matmul in practice)
- **Topic 11**: Conv2D (advanced stride tricks)

---

*"The difference between a senior ML engineer and a junior one often comes down to understanding how tensors move through memory."*
