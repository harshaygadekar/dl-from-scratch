# Topic 01: Math Refresh

Quick linear algebra review for tensor operations.

---

## Vectors and Matrices

### Vectors
A vector is an ordered list of numbers:
```
v = [v₁, v₂, ..., vₙ]ᵀ  (column vector, n × 1)
```

**Operations**:
- **Scalar multiplication**: `c·v = [c·v₁, c·v₂, ..., c·vₙ]`
- **Addition**: `u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]`
- **Dot product**: `u·v = Σᵢ uᵢvᵢ` (scalar result)

### Matrices
A matrix is a 2D array of numbers:
```
A = [[a₁₁, a₁₂, ..., a₁ₙ],
     [a₂₁, a₂₂, ..., a₂ₙ],
     ...
     [aₘ₁, aₘ₂, ..., aₘₙ]]  (m × n)
```

---

## Matrix Multiplication

Given A (m × k) and B (k × n), the product C = AB is (m × n):

```
Cᵢⱼ = Σₗ Aᵢₗ · Bₗⱼ
```

**In words**: Element (i, j) of C is the dot product of row i of A with column j of B.

### Example
```
A = [[1, 2],      B = [[5, 6],
     [3, 4]]           [7, 8]]

C = A @ B = [[1·5+2·7, 1·6+2·8],    [[19, 22],
             [3·5+4·7, 3·6+4·8]] =   [43, 50]]
```

### Key Properties
1. **Not commutative**: AB ≠ BA (usually)
2. **Associative**: (AB)C = A(BC)
3. **Distributive**: A(B + C) = AB + AC

---

## Transpose

The transpose swaps rows and columns:

```
A = [[1, 2, 3],       Aᵀ = [[1, 4],
     [4, 5, 6]]             [2, 5],
                            [3, 6]]
```

**Properties**:
- `(Aᵀ)ᵀ = A`
- `(AB)ᵀ = BᵀAᵀ`
- `(A + B)ᵀ = Aᵀ + Bᵀ`

---

## Element-wise Operations

Also called Hadamard product (⊙):

```
A ⊙ B = [[a₁₁·b₁₁, a₁₂·b₁₂],
         [a₂₁·b₂₁, a₂₂·b₂₂]]
```

**In NumPy**: `A * B` (element-wise), `A @ B` (matrix multiply)

---

## Broadcasting Math

Broadcasting extends scalars/vectors across dimensions:

### Outer Product via Broadcasting
```python
u = [1, 2, 3]     # Shape (3,)
v = [10, 20]      # Shape (2,)

# Outer product: every combination
u[:, None] * v[None, :]
# = [[1*10, 1*20],   [[10, 20],
#    [2*10, 2*20],    [20, 40],
#    [3*10, 3*20]] =  [30, 60]]
```

**Mathematical notation**: `uvᵀ` produces an m×n matrix from m and n vectors.

---

## Norms

Norms measure the "size" of vectors/matrices:

### L1 Norm (Manhattan)
```
||x||₁ = Σᵢ |xᵢ|
```

### L2 Norm (Euclidean)
```
||x||₂ = √(Σᵢ xᵢ²)
```

### Frobenius Norm (for matrices)
```
||A||_F = √(Σᵢⱼ aᵢⱼ²)
```

**In NumPy**:
```python
np.linalg.norm(x, ord=1)   # L1
np.linalg.norm(x, ord=2)   # L2 (default)
np.linalg.norm(A, ord='fro')  # Frobenius
```

---

## Einsum Notation

Einstein summation: repeated indices are summed over.

### Examples

| Operation | Einsum | NumPy Equivalent |
|-----------|--------|------------------|
| Dot product | `i,i->` | `np.dot(a, b)` |
| Outer product | `i,j->ij` | `np.outer(a, b)` |
| Matrix multiply | `ij,jk->ik` | `a @ b` |
| Transpose | `ij->ji` | `a.T` |
| Trace | `ii->` | `np.trace(a)` |
| Batch matmul | `bij,bjk->bik` | `a @ b` (with batch) |

---

## Derivatives for Backprop

### Vector by Scalar
```
∂/∂x [f(x)] = [∂f₁/∂x, ∂f₂/∂x, ..., ∂fₙ/∂x]
```

### Scalar by Vector (Gradient)
```
∇ₓf = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
```

### Chain Rule for Matrices
If L = g(Y) and Y = f(X):
```
∂L/∂X = (∂L/∂Y)(∂Y/∂X)
```

For matrix operations:
```
If Y = XW:
∂L/∂X = (∂L/∂Y) @ Wᵀ
∂L/∂W = Xᵀ @ (∂L/∂Y)
```

---

## Quick Reference

| Operation | Notation | NumPy |
|-----------|----------|-------|
| Matrix multiply | AB | `A @ B` |
| Element-wise multiply | A ⊙ B | `A * B` |
| Transpose | Aᵀ | `A.T` |
| Inverse | A⁻¹ | `np.linalg.inv(A)` |
| Dot product | u·v | `np.dot(u, v)` |
| Outer product | uvᵀ | `np.outer(u, v)` |
| Trace | tr(A) | `np.trace(A)` |
| Determinant | det(A) | `np.linalg.det(A)` |

---

## Resources for Deeper Learning

1. **3Blue1Brown - Essence of Linear Algebra**: [YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
2. **Gilbert Strang - MIT OCW 18.06**: [Course Page](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
3. **Matrix Cookbook**: [PDF](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

---

*"Linear algebra is the language of deep learning. Fluency here accelerates everything that follows."*
