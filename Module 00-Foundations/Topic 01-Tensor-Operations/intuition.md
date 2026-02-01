# Topic 01: Intuition Guide

Understanding tensors isn't about memorizing APIs—it's about building mental models.

---

## 🧠 The Core Insight

> **A tensor is just a fancy word for "multi-dimensional array with consistent data type."**

But the magic isn't in the definition—it's in how computers store and manipulate them.

---

## Mental Model 1: Tensors as Cuboids 📦

Think of tensors as nested rectangular containers:

| Rank | Name | Example | Shape |
|------|------|---------|-------|
| 0 | Scalar | 5 | () |
| 1 | Vector | [1, 2, 3] | (3,) |
| 2 | Matrix | [[1,2], [3,4]] | (2, 2) |
| 3 | 3D Tensor | Images | (batch, height, width) |
| 4 | 4D Tensor | Color images | (batch, height, width, channels) |

**Key insight**: The shape tells you how many drawers you need to open to reach a single number.

---

## Mental Model 2: Memory is Flat 🔢

Computers don't have multi-dimensional memory. Everything is a 1D sequence of bytes.

```
Logical view:              Memory view:
[[1, 2, 3],                [1, 2, 3, 4, 5, 6]
 [4, 5, 6]]                 ↑  ↑  ↑  ↑  ↑  ↑
                            0  1  2  3  4  5 (byte offsets)
```

**Strides** tell NumPy how to translate logical indices to memory positions:
- For shape (2, 3) with 8-byte integers: strides = (24, 8)
- To access `[1, 2]`: skip 24 bytes (row) + 16 bytes (2 columns)

---

## Mental Model 3: Broadcasting as Stretching 🔄

Broadcasting isn't magic—it's lazy repetition.

```
Adding:     [1, 2, 3]  (shape: 3)
        +   [[10],     (shape: 2, 1)
             [20]]

Step 1: Prepend 1s to align shapes
        [1, 2, 3]     → [[1, 2, 3]]      (shape: 1, 3)

Step 2: Stretch dimensions of size 1
        [[1, 2, 3]]   → [[1, 2, 3],      (virtual shape: 2, 3)
                         [1, 2, 3]]
        
        [[10],        → [[10, 10, 10],   (virtual shape: 2, 3)
         [20]]           [20, 20, 20]]

Result: [[11, 12, 13],
         [21, 22, 23]]
```

**The data isn't actually copied!** NumPy uses stride tricks internally.

---

## Mental Model 4: Views as Windows 🪟

A view is like looking at the same data through a different window.

```python
a = np.array([1, 2, 3, 4, 5, 6])
b = a.reshape(2, 3)  # Same memory, different shape!

print(a)  # [1 2 3 4 5 6]
print(b)  # [[1 2 3]
          #  [4 5 6]]

b[0, 0] = 999
print(a)  # [999, 2, 3, 4, 5, 6]  <-- a changed too!
```

**Views are powerful** because they avoid memory allocation. But they can also cause subtle bugs.

---

## Mental Model 5: Axes as "Collapse Directions" ⬇️

When you sum/mean along an axis, you're "collapsing" that dimension:

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)

# Sum along axis 0 (collapse rows → 1 row)
np.sum(a, axis=0)  # [5, 7, 9]  Shape: (3,)

# Sum along axis 1 (collapse columns → 1 column)
np.sum(a, axis=1)  # [6, 15]   Shape: (2,)

# Sum all (collapse everything)
np.sum(a)          # 21        Shape: ()
```

**Mnemonic**: The axis you specify **disappears** from the output shape.

---

## Why This Matters for Deep Learning 🧬

### 1. Batch Processing
Modern ML processes many examples at once:
```
Single image: (height, width, channels) = (224, 224, 3)
Batch of 32:  (batch, height, width, channels) = (32, 224, 224, 3)
```

Every operation must handle the batch dimension correctly.

### 2. Matrix Multiplication Everywhere
Neural network layers are matrix multiplications:
```
Forward pass:  output = input @ weights + bias
Backward pass: gradients flow backward through the same @
```

Understanding shapes prevents 90% of neural network bugs.

### 3. Memory Efficiency
The difference between a 10-minute training run and a 10-hour one often comes down to:
- Using views instead of copies
- Batching operations
- Memory-aligned access patterns

---

## Common Pitfalls 🚫

### 1. Shape Mismatch
```python
a = np.ones((3, 4))
b = np.ones((4, 3))
a + b  # Error! (3,4) + (4,3) doesn't broadcast
```

### 2. Unintentional Copies
```python
a = np.arange(10)
b = a[[2, 5, 7]]  # Fancy indexing = COPY
c = a[2:6]        # Slicing = VIEW
```

### 3. Axis Confusion
```python
# Common bug: using wrong axis
batch_means = data.mean(axis=0)  # Mean of each feature
# vs
example_means = data.mean(axis=1)  # Mean of each example
```

### 4. In-Place Modification of Views
```python
a = np.arange(10)
b = a[::2]  # View: [0, 2, 4, 6, 8]
b *= 2      # Modifies a! a = [0, 1, 4, 3, 8, 5, 12, 7, 16, 9]
```

---

## Intuition Checkpoints ✅

Before moving on, make sure you can answer:

1. **What's the shape of `a + b` if `a.shape = (32, 1, 10)` and `b.shape = (5, 10)`?**
   <details><summary>Answer</summary>(32, 5, 10) — broadcasting stretches the 1 and prepends 32</details>

2. **If a 2D array has strides (80, 8), what's the data type likely to be?**
   <details><summary>Answer</summary>Float64 (8 bytes), with 10 columns (80/8 = 10)</details>

3. **Does `a.T` create a copy or a view?**
   <details><summary>Answer</summary>View! The strides are reversed, but the memory is shared.</details>

4. **What axis do you reduce over for batch normalization?**
   <details><summary>Answer</summary>axis=0 (the batch axis) — you compute statistics per feature</details>

---

*"If you truly understand how tensors are stored and manipulated, you'll never fear a shape error again."*
