"""
Level 02: Vectorized Implementation (NumPy Broadcasting)

Topic 01: Tensor Operations & Broadcasting

This implementation uses NumPy's built-in broadcasting.
No explicit loops - operations are vectorized for speed.

This is typically 10-100x faster than level01.
"""

import numpy as np


def batched_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Batched matrix multiplication using NumPy.
    
    Args:
        a: Shape (batch, m, k)
        b: Shape (batch, k, n)
    
    Returns:
        c: Shape (batch, m, n)
    
    NumPy's @ operator handles batched matmul automatically!
    """
    return a @ b  # or np.matmul(a, b)


def broadcast_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Add two arrays with broadcasting.
    
    NumPy handles all the broadcasting logic automatically.
    """
    return a + b


def efficient_repeat(x: np.ndarray, repeats: int, axis: int) -> np.ndarray:
    """
    Repeat array along axis.
    
    Uses np.repeat which is optimized but still copies data.
    See level03 for zero-copy version.
    """
    return np.repeat(np.expand_dims(x, axis), repeats, axis=axis)


def outer_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute outer product using broadcasting.
    
    Key insight: a[:, None] makes a column vector (m, 1)
                 b[None, :] makes a row vector (1, n)
                 Multiplication broadcasts to (m, n)
    """
    return a[:, None] * b[None, :]  # or np.outer(a, b)


def batched_outer_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Batched outer product.
    
    Args:
        a: Shape (batch, m)
        b: Shape (batch, n)
    
    Returns:
        result: Shape (batch, m, n)
    """
    # a[:, :, None] is (batch, m, 1)
    # b[:, None, :] is (batch, 1, n)
    return a[:, :, None] * b[:, None, :]


def normalize_vectors(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Normalize vectors to unit length along specified axis.
    
    Args:
        x: Input array
        axis: Axis along which to normalize
    
    Returns:
        Normalized array (same shape)
    """
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return x / norms


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax along specified axis.
    
    Uses the numerically stable version: softmax(x) = softmax(x - max(x))
    
    Args:
        x: Input logits
        axis: Axis along which to compute softmax
    
    Returns:
        Softmax probabilities (same shape, sums to 1 along axis)
    """
    # Subtract max for numerical stability (doesn't change result)
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def batch_normalize(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Batch normalization (simplified, no learnable params).
    
    Normalizes across the batch dimension (axis 0).
    
    Args:
        x: Shape (batch, features)
        eps: Small constant for numerical stability
    
    Returns:
        Normalized x with mean=0, std=1 per feature
    """
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + eps)


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between two sets of vectors.
    
    Args:
        a: Shape (n, d) - n vectors of dimension d
        b: Shape (m, d) - m vectors of dimension d
    
    Returns:
        distances: Shape (n, m) - distances[i, j] = ||a[i] - b[j]||
    
    Uses the identity: ||a-b||² = ||a||² + ||b||² - 2*a·b
    This is more efficient than computing differences directly.
    """
    # ||a||² for each row of a, shape (n, 1)
    a_sq = np.sum(a ** 2, axis=1, keepdims=True)
    
    # ||b||² for each row of b, shape (1, m)
    b_sq = np.sum(b ** 2, axis=1, keepdims=True).T
    
    # a @ b.T gives dot products, shape (n, m)
    ab = a @ b.T
    
    # ||a-b||² = ||a||² + ||b||² - 2*a·b
    sq_distances = a_sq + b_sq - 2 * ab
    
    # Clamp to avoid numerical issues with sqrt of small negative numbers
    sq_distances = np.maximum(sq_distances, 0)
    
    return np.sqrt(sq_distances)


def attention_scores(query: np.ndarray, key: np.ndarray, 
                     scale: bool = True) -> np.ndarray:
    """
    Compute (scaled) dot-product attention scores.
    
    Args:
        query: Shape (batch, seq_q, d_k)
        key: Shape (batch, seq_k, d_k)
        scale: Whether to scale by sqrt(d_k)
    
    Returns:
        scores: Shape (batch, seq_q, seq_k)
    
    This is the core of the Transformer's attention mechanism!
    """
    # Transpose key: (batch, d_k, seq_k)
    key_T = np.transpose(key, (0, 2, 1))
    
    # Batched matmul: (batch, seq_q, seq_k)
    scores = query @ key_T
    
    if scale:
        d_k = query.shape[-1]
        scores = scores / np.sqrt(d_k)
    
    return scores


# Test code
if __name__ == "__main__":
    print("Testing Level 02: Vectorized Implementation\n")
    print("="*50)
    
    # Test 1: Batched matmul
    print("\nTest 1: Batched Matrix Multiplication")
    a = np.random.randn(2, 3, 4)
    b = np.random.randn(2, 4, 5)
    result = batched_matmul(a, b)
    print(f"Result shape: {result.shape}")
    print(f"Matches np.matmul: {np.allclose(result, np.matmul(a, b))}")
    
    # Test 2: Outer product
    print("\nTest 2: Outer Product")
    a = np.array([1, 2, 3], dtype=float)
    b = np.array([10, 20], dtype=float)
    result = outer_product(a, b)
    print(f"Result:\n{result}")
    print(f"Matches np.outer: {np.allclose(result, np.outer(a, b))}")
    
    # Test 3: Softmax
    print("\nTest 3: Softmax")
    logits = np.array([[1, 2, 3], [1, 1, 1]], dtype=float)
    probs = softmax(logits, axis=1)
    print(f"Logits:\n{logits}")
    print(f"Probabilities:\n{probs}")
    print(f"Row sums: {probs.sum(axis=1)}")  # Should be [1, 1]
    
    # Test 4: Pairwise distances
    print("\nTest 4: Pairwise Distances")
    a = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    b = np.array([[1, 1], [2, 2]], dtype=float)
    distances = pairwise_distances(a, b)
    print(f"Points a:\n{a}")
    print(f"Points b:\n{b}")
    print(f"Distances:\n{distances}")
    
    # Test 5: Attention scores
    print("\nTest 5: Attention Scores")
    batch_size, seq_len, d_k = 2, 4, 8
    query = np.random.randn(batch_size, seq_len, d_k)
    key = np.random.randn(batch_size, seq_len, d_k)
    scores = attention_scores(query, key)
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Scores shape: {scores.shape}")
    
    print("\n" + "="*50)
    print("✅ All vectorized implementations working!")
    print("\nNext: Optimize with stride tricks in level03_memory_efficient.py")
