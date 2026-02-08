# Math Refresh: Convolutional Operations

## Prerequisites

Before this topic, you should understand:
- Matrix multiplication (from Topic 01)
- Summation notation (Σ)
- Basic geometric series

---

## Key Equations

### 1. 2D Convolution (Single Channel)

Given input $X \in \mathbb{R}^{H \times W}$ and kernel $K \in \mathbb{R}^{k \times k}$:

$$(X * K)[i, j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X[i \cdot s + m, j \cdot s + n] \cdot K[m, n]$$

Where:
- $s$ = stride
- $i \in [0, H_{out})$, $j \in [0, W_{out})$
- Only valid where $i \cdot s + m < H$ and $j \cdot s + n < W$

### 2. Output Dimension Calculation

With padding $p$ and stride $s$:

$$H_{out} = \left\lfloor \frac{H_{in} - k + 2p}{s} \right\rfloor + 1$$

**Derivation**:
1. Padded input size: $H_{in} + 2p$
2. Valid positions for kernel: $(H_{in} + 2p) - k + 1$
3. With stride $s$: divide by $s$, take floor, add 1

### 3. Multi-Channel Convolution

For input $X \in \mathbb{R}^{C_{in} \times H \times W}$ and weights $W \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$:

$$Y[c_{out}, i, j] = \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X[c_{in}, i \cdot s + m, j \cdot s + n] \cdot W[c_{out}, c_{in}, m, n] + b[c_{out}]$$

### 4. Parameter Count

A Conv2D layer with $C_{in}$ input channels, $C_{out}$ output channels, and kernel size $k \times k$:

$$\text{Parameters} = (C_{in} \times k \times k + 1) \times C_{out}$$

(+1 for bias per output channel)

### 5. FLOPs (Floating Point Operations)

Per output element: $C_{in} \times k \times k$ multiplies + $(C_{in} \times k \times k - 1)$ adds

Total FLOPs per forward pass:
$$\text{FLOPs} \approx 2 \times N \times C_{out} \times H_{out} \times W_{out} \times C_{in} \times k^2$$

(Factor of 2 for multiply-add)

---

## Worked Examples

### Example 1: Simple Convolution (No Padding, Stride 1)

**Input**: $X = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{bmatrix}$ (4×4)

**Kernel**: $K = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$ (2×2)

**Calculate output**:

$H_{out} = (4 - 2 + 0) / 1 + 1 = 3$

$W_{out} = (4 - 2 + 0) / 1 + 1 = 3$

Output is 3×3:

$$\text{Output}[0,0] = 1 \cdot 1 + 2 \cdot 0 + 5 \cdot 0 + 6 \cdot (-1) = 1 - 6 = -5$$

$$\text{Output}[0,1] = 2 \cdot 1 + 3 \cdot 0 + 6 \cdot 0 + 7 \cdot (-1) = 2 - 7 = -5$$

$$\text{Output}[1,0] = 5 \cdot 1 + 6 \cdot 0 + 9 \cdot 0 + 10 \cdot (-1) = 5 - 10 = -5$$

Continuing this pattern:

$$\text{Output} = \begin{bmatrix} -5 & -5 & -5 \\ -5 & -5 & -5 \\ -5 & -5 & -5 \end{bmatrix}$$

This kernel computes the difference between top-left and bottom-right of each 2×2 window.

### Example 2: With Padding and Stride

**Input**: 7×7 image
**Kernel**: 3×3
**Padding**: 1 (add 1 pixel border of zeros)
**Stride**: 2

**Output size calculation**:

$$H_{out} = \lfloor (7 - 3 + 2 \cdot 1) / 2 \rfloor + 1 = \lfloor 7/2 \rfloor + 1 = 3 + 1 = 4$$

Output is 4×4.

**Valid positions** (stride=2, starting at 0):
- Height: positions 0, 2, 4, 6 → 4 positions
- Width: positions 0, 2, 4, 6 → 4 positions

### Example 3: Multi-Channel Forward Pass

**Input**: $X \in \mathbb{R}^{2 \times 4 \times 4}$ (2 channels, 4×4 spatial)

**Weights**: $W \in \mathbb{R}^{3 \times 2 \times 3 \times 3}$ (3 output channels, 2 input channels, 3×3 kernel)

**Bias**: $b \in \mathbb{R}^3$

**Padding**: 1, **Stride**: 1

**Output size**: $3 \times 4 \times 4$

**Computation for output[0, 0, 0]** (first output channel, top-left):

$$\text{Output}[0, 0, 0] = \sum_{c_{in}=0}^{1} \sum_{m=0}^{2} \sum_{n=0}^{2} X[c_{in}, m, n] \cdot W[0, c_{in}, m, n] + b[0]$$

This is the dot product between:
- The 2×3×3 input region (2 channels, 3×3 window)
- The first output channel's weights (2×3×3)

### Example 4: Parameter Count

Conv2D layer from AlexNet first layer:
- Input: 3 channels (RGB)
- Output: 96 channels
- Kernel: 11×11

$$\text{Parameters} = (3 \times 11 \times 11 + 1) \times 96 = (363 + 1) \times 96 = 34,944$$

Compare to fully connected: $3 \times 224 \times 224 \times 96 = 14,450,688$ parameters!

---

## Dimension Analysis

### Shape Transformation Table

| Component | Shape | Notes |
|-----------|-------|-------|
| Input | (N, C_in, H, W) | Batch, channels, height, width |
| Weights | (C_out, C_in, K, K) | Output channels, input channels, kernel |
| Bias | (C_out,) | One per output channel |
| Output | (N, C_out, H_out, W_out) | Computed from formula |

### Common Conv2D Configurations

| Name | Kernel | Padding | Stride | Effect |
|------|--------|---------|--------|--------|
| Same conv | 3×3 | 1 | 1 | Preserves spatial size |
| Valid conv | 3×3 | 0 | 1 | Reduces by 2 in each dim |
| Downsampling | 3×3 | 1 | 2 | Halves spatial size |
| Large receptive | 7×7 | 3 | 1 | Sees 7×7 region |
| 1×1 conv | 1×1 | 0 | 1 | Channel mixing only |

---

## Common Mistakes

### Mistake 1: Forgetting Batch Dimension

```python
# WRONG - Missing batch loop
for c_out in range(C_out):
    for h in range(H_out):
        ...

# CORRECT
for n in range(N):  # ← Don't forget!
    for c_out in range(C_out):
        for h in range(H_out):
            ...
```

### Mistake 2: Wrong Indexing

```python
# WRONG - Using output indices for input
input[h, w]  # Should use h*stride + m, w*stride + n

# CORRECT
input[h * stride + m, w * stride + n]
```

### Mistake 3: Off-by-One in Output Size

```python
# WRONG
H_out = (H_in - K + 2*P) / S  # Missing +1!

# CORRECT
H_out = (H_in - K + 2*P) // S + 1
```

### Mistake 4: Channel Index Confusion

```python
# WRONG - Flipped channel order
weight[c_in, c_out, k1, k2]

# CORRECT - Output channel first
weight[c_out, c_in, k1, k2]
```

---

## Memory Layout Note

NumPy uses **row-major** (C-style) order:
- Stride in memory: element at [n, c, h, w] is at offset: `((n * C + c) * H + h) * W + w`
- Last dimension (W) changes fastest

This is why we use shape `(N, C, H, W)` (channels-first) matching PyTorch convention.

---

## Computational Complexity

Time complexity for single Conv2D forward pass:

$$O(N \times C_{out} \times H_{out} \times W_{out} \times C_{in} \times K^2)$$

For a typical layer:
- N = 32 (batch size)
- C_out = 128 (output channels)
- H_out = W_out = 32 (spatial)
- C_in = 64 (input channels)
- K = 3

Operations: $32 \times 128 \times 32 \times 32 \times 64 \times 9 \approx 2.4$ billion multiply-adds!

**This is why we need vectorization (Level 2) and im2col (Topic 12).**
