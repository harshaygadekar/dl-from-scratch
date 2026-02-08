# Topic 11: Conv2D Sliding Window

> **Goal**: Implement 2D convolution from scratch using sliding window approach.
> **Time**: 2-3 hours | **Difficulty**: Medium

---

## 🎯 Learning Objectives

By the end of this topic, you will:
1. Understand how convolutional layers work at the pixel level
2. Implement sliding window convolution with proper padding
3. Handle multiple input/output channels
4. Calculate output dimensions given input, kernel, stride, and padding

---

## 📋 The Problem

Implement 2D convolution (the core operation of CNNs) from scratch:

### Required Functions

```python
def conv2d_naive(input, weight, bias=None, stride=1, padding=0):
    """
    2D convolution using nested loops (naive implementation).
    
    Args:
        input: Shape (N, C_in, H, W) - batch of images
        weight: Shape (C_out, C_in, K, K) - convolution kernels
        bias: Shape (C_out,) - optional bias terms
        stride: Step size for sliding window
        padding: Number of zeros to pad on each side
    
    Returns:
        output: Shape (N, C_out, H_out, W_out)
    """
    pass

def calculate_output_size(H, K, P, S):
    """
    Calculate output spatial dimension.
    
    Formula: H_out = floor((H - K + 2*P) / S) + 1
    """
    pass

def pad_array(x, padding):
    """
    Pad array with zeros on spatial dimensions.
    
    Args:
        x: Shape (N, C, H, W)
        padding: Number of zeros to add on each side
    
    Returns:
        Padded array of shape (N, C, H+2*P, W+2*P)
    """
    pass
```

---

## 🧠 Key Concepts to Master

### 1. The Convolution Operation

```
Input Image (H×W)          Kernel (K×K)              Output (H_out×W_out)
┌───┬───┬───┬───┐         ┌───┬───┐                ┌───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │    ⊗    │ a │ b │      =         │ y₁│ y₂│ y₃│
├───┼───┼───┼───┤         ├───┼───┤                ├───┼───┼───┤
│ 5 │ 6 │ 7 │ 8 │         │ c │ d │                │ y₄│ y₅│ y₆│
├───┼───┼───┼───┤         └───┴───┘                └───┴───┴───┘
│ 9 │10 │11 │12 │
└───┴───┴───┴───┘

Where y₁ = 1*a + 2*b + 5*c + 6*d (element-wise multiply and sum)
```

### 2. Multi-Channel Convolution

Each output channel is computed by:
1. Taking a weighted sum across ALL input channels
2. Using a different kernel per output channel
3. Adding bias (optional)

```
Output[c_out] = Σ_{c_in} Conv2D(Input[c_in], Weight[c_out, c_in]) + Bias[c_out]
```

### 3. Padding Strategies

- **Valid**: No padding, output is smaller than input
- **Same**: Pad so output spatial size equals input (when stride=1)
  - For kernel K: pad = (K-1) // 2 on each side

### 4. Stride

Stride > 1 skips pixels, reducing output size:
- Stride 1: Slide window by 1 pixel (default)
- Stride 2: Slide window by 2 pixels (halves spatial dimensions)

---

## 📁 File Structure

```
Topic 11-Conv2D-Sliding-Window/
├── README.md              # This file
├── questions.md           # Interview-style questions
├── intuition.md           # Conceptual explanations
├── math-refresh.md        # Formulas and derivations
├── hints/
│   ├── hint-1-kernel-sliding.md
│   ├── hint-2-padding-logic.md
│   └── hint-3-loop-structure.md
├── solutions/
│   ├── level01_naive.py          # 6-nested-loop implementation
│   ├── level02_vectorized.py     # NumPy optimized
│   ├── level03_memory_efficient.py  # Stride tricks
│   └── level04_pytorch_reference.py # Ground truth
├── tests/
│   ├── test_basic.py      # Core functionality
│   ├── test_edge.py       # Edge cases
│   └── test_stress.py     # Performance tests
├── visualization.py       # Interactive convolution visualizer
└── assets/
    └── conv_visualization.html  # Interactive demo
```

---

## 🎮 How to Use This Topic

### Step 1: Understand the Math
1. Read `math-refresh.md` for formulas
2. Work through the dimension calculations by hand
3. Review `intuition.md` for analogies

### Step 2: Implement Level 1 (Naive)
```bash
cd "Module 02-CNNs/Topic 11-Conv2D-Sliding-Window"
python solutions/level01_naive.py
```

Start with explicit loops over:
- Batch samples (n)
- Output channels (c_out)
- Output height (h_out)
- Output width (w_out)
- Input channels (c_in)
- Kernel height (k_h)
- Kernel width (k_w)

### Step 3: Test Your Implementation
```bash
# Test correctness
python -m pytest tests/test_basic.py -v

# Test edge cases
python -m pytest tests/test_edge.py -v
```

### Step 4: Level Up
- **Level 2**: Replace inner loops with `np.sum(input_window * kernel)`
- **Level 3**: Use `np.lib.stride_tricks.as_strided` for zero-copy window extraction
- **Level 4**: Verify against PyTorch

---

## 🏆 Success Criteria

| Level | Requirement |
|-------|-------------|
| Level 1 | All `test_basic.py` pass (correctness) |
| Level 2 | No explicit kernel loops (vectorized) |
| Level 3 | Passes stress tests (speed + memory) |
| Level 4 | Numerical match with PyTorch (1e-5 tolerance) |

---

## 🆘 Escape Hatch

If stuck after 3+ hours:
1. Study `hints/hint-2-padding-logic.md` for pseudocode
2. Look at `solutions/level02_vectorized.py` for structure
3. Use PyTorch `nn.Conv2d` to verify your math understanding

**Do NOT copy-paste without understanding!**

---

## 🔗 Related Topics

- **Topic 12**: Im2Col Vectorization (speed up this implementation)
- **Topic 13**: Pooling & Strides (uses similar sliding logic)
- **Topic 14**: ResNet (stacks conv layers with skip connections)

---

*"Convolution is just a weighted average with a learned weight template."*
