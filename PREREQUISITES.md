# Prerequisites Self-Assessment

Before starting Topic 01, complete these 5 challenges. If you struggle with any, review the linked resources first.

---

## Challenge 1: Matrix Multiplication (5 min)

Compute by hand (no calculator):

```
A = [[1, 2],     B = [[5, 6],
     [3, 4]]          [7, 8]]

A @ B = ?
```

<details>
<summary>Answer</summary>

```
[[1*5+2*7, 1*6+2*8],   [[19, 22],
 [3*5+4*7, 3*6+4*8]] =  [43, 50]]
```

**How it works:**
- Row 1 of A · Column 1 of B = 1×5 + 2×7 = 19
- Row 1 of A · Column 2 of B = 1×6 + 2×8 = 22
- Row 2 of A · Column 1 of B = 3×5 + 4×7 = 43
- Row 2 of A · Column 2 of B = 3×6 + 4×8 = 50

</details>

---

## Challenge 2: Derivative of Sigmoid (5 min)

Given: σ(x) = 1 / (1 + e^(-x))

Derive: dσ/dx = ?

<details>
<summary>Answer</summary>

**dσ/dx = σ(x) × (1 - σ(x))**

**Derivation:**
1. σ(x) = (1 + e^(-x))^(-1)
2. Using chain rule: dσ/dx = -1 × (1 + e^(-x))^(-2) × (-e^(-x))
3. = e^(-x) / (1 + e^(-x))²
4. = [1/(1+e^(-x))] × [e^(-x)/(1+e^(-x))]
5. = σ(x) × [1 - σ(x)]

**Why this matters:** This compact form makes backprop efficient - you only need the forward pass output!

</details>

---

## Challenge 3: Broadcasting Quiz (3 min)

What is the shape of the result?

```python
import numpy as np
a = np.ones((3, 1))   # Shape: (3, 1)
b = np.ones((1, 4))   # Shape: (1, 4)
c = a + b             # Shape: ?
```

<details>
<summary>Answer</summary>

**Shape: (3, 4)**

**Broadcasting rules:**
1. Align shapes from the right: `(3, 1)` and `(1, 4)`
2. Dimensions of size 1 stretch to match the other dimension
3. Result: `(3, 4)` - a broadcasts its columns, b broadcasts its rows

```python
# Visual:
# a (3,1):     b (1,4):        a + b (3,4):
# [[1],       [[1, 1, 1, 1]]   [[2, 2, 2, 2],
#  [1],   +                =    [2, 2, 2, 2],
#  [1]]                         [2, 2, 2, 2]]
```

</details>

---

## Challenge 4: Chain Rule Application (5 min)

Given:
- L = f(y)
- y = g(x)
- dL/dy = 3
- dy/dx = 2

What is dL/dx?

<details>
<summary>Answer</summary>

**dL/dx = dL/dy × dy/dx = 3 × 2 = 6**

This is the **chain rule**: the foundation of backpropagation!

In neural networks:
- L is the loss
- y is the output of a layer
- x is the input to that layer
- We chain these derivatives backward through all layers

</details>

---

## Challenge 5: NumPy Speed Test (5 min)

Run this code. What's the speedup?

```python
import numpy as np
import time

n = 1000000

# Python lists
start = time.time()
a = list(range(n))
b = list(range(n))
c = [a[i] + b[i] for i in range(n)]
list_time = time.time() - start

# NumPy arrays
start = time.time()
a = np.arange(n)
b = np.arange(n)
c = a + b
numpy_time = time.time() - start

print(f"Python lists: {list_time:.4f}s")
print(f"NumPy arrays: {numpy_time:.4f}s")
print(f"Speedup: {list_time/numpy_time:.1f}x")
```

<details>
<summary>Expected Result</summary>

NumPy should be **50-100x faster** depending on your machine.

**Why NumPy is faster:**
1. **Vectorized operations**: No Python loop overhead
2. **Contiguous memory**: Better cache utilization
3. **C implementation**: Compiled code for arithmetic
4. **SIMD instructions**: Processes multiple elements per CPU cycle

This speedup is why we implement deep learning with NumPy, not pure Python!

</details>

---

## 📊 Self-Assessment

| Challenge | Passed? | If Not, Review |
|-----------|---------|----------------|
| Matrix Multiplication | ☐ | [3Blue1Brown Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) |
| Sigmoid Derivative | ☐ | [3Blue1Brown Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) |
| Broadcasting | ☐ | [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) |
| Chain Rule | ☐ | [Calculus Refresher](https://www.khanacademy.org/math/calculus-1/cs1-derivatives-chain-rule-and-other-advanced-topics) |
| NumPy Speed | ☐ | [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html) |

---

## Ready?

✅ **All 5 passed?** → Proceed to [SETUP.md](SETUP.md)

⚠️ **Struggled with any?** → Review the linked resources first. These concepts are used daily in this curriculum.

---

*Time invested in prerequisites = time saved debugging later.*
