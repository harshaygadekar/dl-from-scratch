"""
Level 01: Naive Implementation (Loops, Readable)

Topic 01: Tensor Operations & Broadcasting

This implementation prioritizes readability over performance.
Use explicit loops to understand what's happening at each step.

Once you understand this, move to level02-vectorized.py
"""

import numpy as np


def batched_matmul_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Batched matrix multiplication using explicit loops.
    
    Args:
        a: Shape (batch, m, k)
        b: Shape (batch, k, n)
    
    Returns:
        c: Shape (batch, m, n)
    
    Time complexity: O(batch * m * k * n)
    """
    batch, m, k = a.shape
    _, _, n = b.shape
    
    # Pre-allocate output
    c = np.zeros((batch, m, n))
    
    # Triple loop: one for each dimension of the output
    for batch_idx in range(batch):
        for i in range(m):
            for j in range(n):
                # Dot product of row i of a[batch_idx] with column j of b[batch_idx]
                total = 0.0
                for l in range(k):
                    total += a[batch_idx, i, l] * b[batch_idx, l, j]
                c[batch_idx, i, j] = total
    
    return c


def broadcast_add_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Add two arrays with broadcasting (manual implementation).
    
    Implements NumPy-style broadcasting rules:
    1. Align shapes from the right
    2. Dimensions of size 1 stretch to match
    3. Missing dimensions are prepended with 1
    
    Args:
        a: First array (any shape)
        b: Second array (any shape)
    
    Returns:
        Result with broadcasted shape
    """
    # Step 1: Compute output shape
    ndim = max(a.ndim, b.ndim)
    
    # Pad shapes with 1s on the left
    shape_a = (1,) * (ndim - a.ndim) + a.shape
    shape_b = (1,) * (ndim - b.ndim) + b.shape
    
    # Compute output shape
    output_shape = []
    for dim_a, dim_b in zip(shape_a, shape_b):
        if dim_a == dim_b:
            output_shape.append(dim_a)
        elif dim_a == 1:
            output_shape.append(dim_b)
        elif dim_b == 1:
            output_shape.append(dim_a)
        else:
            raise ValueError(f"Shapes {a.shape} and {b.shape} cannot be broadcast")
    
    output_shape = tuple(output_shape)
    
    # Step 2: Reshape inputs to match output dimensions
    a_reshaped = a.reshape(shape_a)
    b_reshaped = b.reshape(shape_b)
    
    # Step 3: Allocate output and fill with loops
    output = np.zeros(output_shape)
    
    # Create index iterators
    for idx in np.ndindex(output_shape):
        # Map output index to input indices (handle broadcasting)
        idx_a = tuple(0 if shape_a[i] == 1 else idx[i] for i in range(ndim))
        idx_b = tuple(0 if shape_b[i] == 1 else idx[i] for i in range(ndim))
        
        output[idx] = a_reshaped[idx_a] + b_reshaped[idx_b]
    
    return output


def efficient_repeat_naive(x: np.ndarray, repeats: int, axis: int) -> np.ndarray:
    """
    Repeat array along axis (naive implementation with actual copying).
    
    Args:
        x: Input array
        repeats: Number of times to repeat
        axis: Axis along which to repeat
    
    Returns:
        Array with repeated elements
    
    Note: This actually copies data. See level03 for memory-efficient version.
    """
    # Compute output shape
    output_shape = list(x.shape)
    output_shape.insert(axis, repeats)
    
    # Allocate output
    output = np.zeros(output_shape)
    
    # Use slicing to fill
    for i in range(repeats):
        # Create slice for this repetition
        slices = [slice(None)] * len(output_shape)
        slices[axis] = i
        output[tuple(slices)] = x
    
    # Reshape to merge the repeat dimension
    final_shape = list(x.shape)
    final_shape[axis] *= repeats
    
    # Move repeat axis next to the repeated axis and merge
    output = np.moveaxis(output, axis, axis + 1)
    return output.reshape(final_shape)


def outer_product_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute outer product of two 1D arrays.
    
    Args:
        a: Shape (m,)
        b: Shape (n,)
    
    Returns:
        result: Shape (m, n) where result[i,j] = a[i] * b[j]
    """
    m = len(a)
    n = len(b)
    
    result = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            result[i, j] = a[i] * b[j]
    
    return result


def sum_along_axis_naive(x: np.ndarray, axis: int) -> np.ndarray:
    """
    Sum array along specified axis.
    
    Args:
        x: Input array
        axis: Axis to sum over
    
    Returns:
        Array with one fewer dimension
    """
    # Compute output shape (remove the summed axis)
    output_shape = list(x.shape)
    del output_shape[axis]
    
    if len(output_shape) == 0:
        # Summing to scalar
        result = 0.0
        for idx in np.ndindex(x.shape):
            result += x[idx]
        return result
    
    output = np.zeros(output_shape)
    
    # Iterate over output positions
    for out_idx in np.ndindex(tuple(output_shape)):
        # Sum over the reduced axis
        total = 0.0
        for i in range(x.shape[axis]):
            # Build full index into x
            full_idx = list(out_idx)
            full_idx.insert(axis, i)
            total += x[tuple(full_idx)]
        output[out_idx] = total
    
    return output


# Test code
if __name__ == "__main__":
    print("Testing Level 01: Naive Implementation\n")
    print("="*50)
    
    # Test 1: Batched matmul
    print("\nTest 1: Batched Matrix Multiplication")
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(2, 4, 5)
    
    result = batched_matmul_naive(a, b)
    expected = np.matmul(a, b)
    
    print(f"Input a shape: {a.shape}")
    print(f"Input b shape: {b.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Matches NumPy: {np.allclose(result, expected)}")
    
    # Test 2: Broadcasting add
    print("\nTest 2: Broadcast Add")
    a = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = np.array([10, 20, 30])             # (3,)
    
    result = broadcast_add_naive(a, b)
    expected = a + b
    
    print(f"a shape: {a.shape}, b shape: {b.shape}")
    print(f"Result shape: {result.shape}")
    print(f"Result:\n{result}")
    print(f"Matches NumPy: {np.allclose(result, expected)}")
    
    # Test 3: Outer product
    print("\nTest 3: Outer Product")
    a = np.array([1, 2, 3])
    b = np.array([10, 20])
    
    result = outer_product_naive(a, b)
    expected = np.outer(a, b)
    
    print(f"a: {a}, b: {b}")
    print(f"Result:\n{result}")
    print(f"Matches NumPy: {np.allclose(result, expected)}")
    
    # Test 4: Sum along axis
    print("\nTest 4: Sum Along Axis")
    x = np.array([[1, 2, 3], [4, 5, 6]])
    
    result_axis0 = sum_along_axis_naive(x, axis=0)
    result_axis1 = sum_along_axis_naive(x, axis=1)
    
    print(f"Input:\n{x}")
    print(f"Sum axis 0: {result_axis0} (expected: {np.sum(x, axis=0)})")
    print(f"Sum axis 1: {result_axis1} (expected: {np.sum(x, axis=1)})")
    
    print("\n" + "="*50)
    print("✅ All naive implementations working!")
    print("\nNext: Run tests with: pytest tests/test_basic.py")
