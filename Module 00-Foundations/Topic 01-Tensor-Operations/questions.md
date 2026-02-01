# Topic 01: Interview Questions

Practice these before your next ML interview!

---

## Q1: Broadcasting Bug (Google, Meta)

**Difficulty**: Medium | **Time**: 5 min

You have a batch of images and want to subtract the mean RGB values:

```python
images = np.random.randn(32, 224, 224, 3)  # (batch, height, width, channels)
mean = np.array([0.485, 0.456, 0.406])     # RGB means

# Your colleague wrote:
normalized = images - mean  # Does this work?
```

**Questions**:
1. Does this code work? If not, why?
2. What's the correct way to do this?
3. What if `mean` was shape `(3, 1, 1)`?

<details>
<summary>Answer</summary>

1. **Yes, it works!** Broadcasting aligns from the right:
   - `images`: (32, 224, 224, 3)
   - `mean`: (3,) → broadcast to (1, 1, 1, 3) → then to (32, 224, 224, 3)

2. The code is correct as written. The mean broadcasts across all pixels and batches.

3. If `mean` was `(3, 1, 1)`:
   - It would NOT broadcast correctly
   - Shape alignment: (32, 224, 224, 3) vs (3, 1, 1) = FAIL
   - You'd need `mean.reshape(1, 1, 1, 3)` or use `mean[None, None, None, :]`

**Key insight**: NumPy prepends 1s, not appends. The rightmost dimensions must match.

</details>

---

## Q2: Memory Layout (Apple, NVIDIA)

**Difficulty**: Hard | **Time**: 10 min

```python
a = np.zeros((1000, 1000))

# Which is faster?
def row_sum_v1(a):
    return np.sum(a, axis=1)  # Sum along rows

def row_sum_v2(a):
    return np.sum(a, axis=0)  # Sum along columns
```

**Questions**:
1. Which is faster and why?
2. How would you verify this?
3. What if `a = np.zeros((1000, 1000), order='F')`?

<details>
<summary>Answer</summary>

1. **`row_sum_v1` (sum along rows) is faster** in C-order (default):
   - C-order stores rows contiguously in memory
   - Summing axis=1 means accessing memory sequentially
   - Cache lines get fully utilized (fewer cache misses)

2. Verification:
```python
import timeit
a = np.zeros((1000, 1000))
print(timeit.timeit(lambda: np.sum(a, axis=1), number=100))
print(timeit.timeit(lambda: np.sum(a, axis=0), number=100))
# axis=1 is typically 1.5-2x faster
```

3. With Fortran order (`order='F'`):
   - Columns are stored contiguously
   - `axis=0` (column sum) would be faster
   - This is why some numerical libraries (BLAS) expect column-major

**Production tip**: Always check array order when interfacing with C/Fortran libraries!

</details>

---

## Q3: View vs Copy (Amazon, Microsoft)

**Difficulty**: Medium | **Time**: 5 min

```python
a = np.arange(12).reshape(3, 4)
b = a[::2, :]   # Every other row
c = a[:2, :2]   # Top-left 2x2

b[0, 0] = 999
```

**Questions**:
1. What is `a[0, 0]` after this code?
2. Is `b` a view or a copy?
3. How do you check if two arrays share memory?

<details>
<summary>Answer</summary>

1. `a[0, 0] = 999` — Both `b` and `a` are modified!

2. **`b` is a view**:
   - Slicing with steps creates views in NumPy
   - No new memory is allocated
   - Changes to `b` affect `a`

3. Check with:
```python
np.shares_memory(a, b)  # Returns True
b.base is a             # Returns True for views
b.flags['OWNDATA']      # False for views
```

**Subtlety**: Some operations that seem like they'd create views actually copy:
```python
d = a[[0, 2], :]  # Fancy indexing = COPY
d.base is a       # False!
```

</details>

---

## Q4: Stride Tricks (Meta, Google Brain)

**Difficulty**: Expert | **Time**: 15 min

Implement a sliding window view without copying memory:

```python
def sliding_window_view(x, window_size):
    """
    Create a view of sliding windows.
    
    x: 1D array of shape (n,)
    Returns: shape (n - window_size + 1, window_size)
    
    Example:
        x = [1, 2, 3, 4, 5], window_size = 3
        result = [[1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 5]]
    """
    pass
```

<details>
<summary>Answer</summary>

```python
import numpy as np
from numpy.lib.stride_tricks import as_strided

def sliding_window_view(x, window_size):
    n = x.shape[0]
    output_len = n - window_size + 1
    
    # Key insight: use the SAME stride for both dimensions
    new_shape = (output_len, window_size)
    new_strides = (x.strides[0], x.strides[0])  # Same stride!
    
    return as_strided(x, shape=new_shape, strides=new_strides)

# Example
x = np.array([1, 2, 3, 4, 5])
result = sliding_window_view(x, 3)
# [[1, 2, 3],
#  [2, 3, 4],
#  [3, 4, 5]]
```

**Why this works**:
- Original stride: 8 bytes (for int64)
- Row stride in output: 8 bytes (move by 1 element)
- Column stride in output: 8 bytes (also move by 1 element)
- The data overlaps! No copying needed.

**Interview follow-up**: What happens if you write to this view?
- It modifies the original! Multiple elements in the view may update.

</details>

---

## Q5: Einsum Magic (DeepMind, OpenAI)

**Difficulty**: Medium | **Time**: 5 min

Explain what each einsum does:

```python
# 1
result = np.einsum('ij,jk->ik', A, B)

# 2
result = np.einsum('bij,bjk->bik', A, B)

# 3
result = np.einsum('ij->', A)

# 4
result = np.einsum('ij->ji', A)
```

<details>
<summary>Answer</summary>

1. **Matrix multiplication**: `A @ B`
   - `ij,jk->ik` = sum over `j` (the shared dimension)

2. **Batched matrix multiplication**: `A @ B` for each batch
   - `bij,bjk->bik` = `b` is preserved, sum over `j`
   - Equivalent to: `np.matmul(A, B)` or `A @ B` with batch dim

3. **Sum all elements**: `np.sum(A)`
   - `ij->` means no output indices = reduce everything

4. **Transpose**: `A.T`
   - `ij->ji` swaps dimensions

**Pro tip**: einsum is slower for simple ops but more readable for complex contractions (like attention).

</details>

---

## 🎯 Interview Tips

1. **Always clarify shapes** before writing code
2. **Draw the broadcasting** if unsure
3. **Mention memory implications** (views, cache, strides)
4. **Know when einsum helps** (multi-dim contractions)
5. **Test with small examples** first

---

*"In ML interviews, shape errors are the #1 debugging topic. Master broadcasting and you're ahead of 80% of candidates."*
