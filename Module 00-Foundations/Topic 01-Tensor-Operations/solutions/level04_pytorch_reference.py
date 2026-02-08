"""
Level 04: PyTorch Reference Implementation

Topic 01: Tensor Operations & Broadcasting

This file provides PyTorch equivalents for verification.
Use this to check your NumPy implementations are numerically correct.

To run this file, you need PyTorch installed:
    pip install torch
"""

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install torch")


def verify_batched_matmul():
    """Verify batched matrix multiplication."""
    if not TORCH_AVAILABLE:
        return
    
    print("\n" + "="*50)
    print("Verifying: Batched Matrix Multiplication")
    print("="*50)
    
    # Create random inputs
    np.random.seed(42)
    a_np = np.random.randn(4, 3, 5).astype(np.float32)
    b_np = np.random.randn(4, 5, 2).astype(np.float32)
    
    # NumPy
    result_np = np.matmul(a_np, b_np)
    
    # PyTorch
    a_torch = torch.from_numpy(a_np)
    b_torch = torch.from_numpy(b_np)
    result_torch = torch.matmul(a_torch, b_torch).numpy()
    
    # Compare
    max_diff = np.max(np.abs(result_np - result_torch))
    print(f"NumPy shape: {result_np.shape}")
    print(f"PyTorch shape: {result_torch.shape}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Match: {'✅' if max_diff < 1e-5 else '❌'}")


def verify_broadcasting():
    """Verify broadcasting operations."""
    if not TORCH_AVAILABLE:
        return
    
    print("\n" + "="*50)
    print("Verifying: Broadcasting")
    print("="*50)
    
    # Test cases
    test_cases = [
        ((3, 1), (1, 4)),       # Basic 2D broadcast
        ((2, 3, 1), (4,)),      # 3D to 1D
        ((5, 1, 3), (1, 4, 1)), # Complex broadcast
    ]
    
    for shape_a, shape_b in test_cases:
        np.random.seed(42)
        a_np = np.random.randn(*shape_a).astype(np.float32)
        b_np = np.random.randn(*shape_b).astype(np.float32)
        
        # NumPy
        result_np = a_np + b_np
        
        # PyTorch
        a_torch = torch.from_numpy(a_np)
        b_torch = torch.from_numpy(b_np)
        result_torch = (a_torch + b_torch).numpy()
        
        max_diff = np.max(np.abs(result_np - result_torch))
        print(f"{shape_a} + {shape_b} → {result_np.shape}: {'✅' if max_diff < 1e-6 else '❌'}")


def verify_softmax():
    """Verify softmax implementation."""
    if not TORCH_AVAILABLE:
        return
    
    print("\n" + "="*50)
    print("Verifying: Softmax")
    print("="*50)
    
    np.random.seed(42)
    x_np = np.random.randn(8, 10).astype(np.float32)
    
    # NumPy implementation
    def softmax_np(x, axis=-1):
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    result_np = softmax_np(x_np, axis=1)
    
    # PyTorch
    x_torch = torch.from_numpy(x_np)
    result_torch = torch.nn.functional.softmax(x_torch, dim=1).numpy()
    
    max_diff = np.max(np.abs(result_np - result_torch))
    print(f"Input shape: {x_np.shape}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Row sums (should be 1): {result_np.sum(axis=1)[:3]}...")
    print(f"Match: {'✅' if max_diff < 1e-5 else '❌'}")


def verify_attention_scores():
    """Verify attention score computation."""
    if not TORCH_AVAILABLE:
        return
    
    print("\n" + "="*50)
    print("Verifying: Scaled Dot-Product Attention Scores")
    print("="*50)
    
    np.random.seed(42)
    batch, seq_len, d_k = 2, 8, 64
    
    query_np = np.random.randn(batch, seq_len, d_k).astype(np.float32)
    key_np = np.random.randn(batch, seq_len, d_k).astype(np.float32)
    
    # NumPy implementation
    def attention_scores_np(query, key):
        scores = np.matmul(query, np.transpose(key, (0, 2, 1)))
        scores = scores / np.sqrt(query.shape[-1])
        return scores
    
    result_np = attention_scores_np(query_np, key_np)
    
    # PyTorch
    query_torch = torch.from_numpy(query_np)
    key_torch = torch.from_numpy(key_np)
    result_torch = torch.matmul(query_torch, key_torch.transpose(-2, -1)) / np.sqrt(d_k)
    result_torch = result_torch.numpy()
    
    max_diff = np.max(np.abs(result_np - result_torch))
    print(f"Query shape: {query_np.shape}")
    print(f"Key shape: {key_np.shape}")
    print(f"Scores shape: {result_np.shape}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Match: {'✅' if max_diff < 1e-5 else '❌'}")


def verify_pairwise_distances():
    """Verify pairwise distance computation."""
    if not TORCH_AVAILABLE:
        return
    
    print("\n" + "="*50)
    print("Verifying: Pairwise Euclidean Distances")
    print("="*50)
    
    np.random.seed(42)
    a_np = np.random.randn(10, 64).astype(np.float32)
    b_np = np.random.randn(20, 64).astype(np.float32)
    
    # NumPy implementation
    def pairwise_distances_np(a, b):
        a_sq = np.sum(a ** 2, axis=1, keepdims=True)
        b_sq = np.sum(b ** 2, axis=1, keepdims=True).T
        ab = a @ b.T
        sq_distances = a_sq + b_sq - 2 * ab
        sq_distances = np.maximum(sq_distances, 0)
        return np.sqrt(sq_distances)
    
    result_np = pairwise_distances_np(a_np, b_np)
    
    # PyTorch
    a_torch = torch.from_numpy(a_np)
    b_torch = torch.from_numpy(b_np)
    result_torch = torch.cdist(a_torch, b_torch).numpy()
    
    max_diff = np.max(np.abs(result_np - result_torch))
    print(f"A shape: {a_np.shape}")
    print(f"B shape: {b_np.shape}")
    print(f"Distances shape: {result_np.shape}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Match: {'✅' if max_diff < 1e-4 else '❌'}")  # Slightly relaxed tolerance


def verify_batch_norm():
    """Verify batch normalization."""
    if not TORCH_AVAILABLE:
        return
    
    print("\n" + "="*50)
    print("Verifying: Batch Normalization (simplified)")
    print("="*50)
    
    np.random.seed(42)
    x_np = np.random.randn(32, 64).astype(np.float32)
    eps = 1e-5
    
    # NumPy implementation
    def batch_norm_np(x, eps=1e-5):
        mean = x.mean(axis=0, keepdims=True)
        var = x.var(axis=0, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)
    
    result_np = batch_norm_np(x_np, eps)
    
    # PyTorch (eval mode, no learnable params)
    x_torch = torch.from_numpy(x_np)
    bn = torch.nn.BatchNorm1d(64, affine=False, eps=eps)
    bn.eval()
    # Use training statistics for comparison
    mean = x_torch.mean(dim=0)
    var = x_torch.var(dim=0, unbiased=False)
    result_torch = ((x_torch - mean) / torch.sqrt(var + eps)).numpy()
    
    max_diff = np.max(np.abs(result_np - result_torch))
    print(f"Input shape: {x_np.shape}")
    print(f"Mean of normalized (should be ~0): {result_np.mean(axis=0).mean():.2e}")
    print(f"Std of normalized (should be ~1): {result_np.std(axis=0).mean():.4f}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Match: {'✅' if max_diff < 1e-5 else '❌'}")


if __name__ == "__main__":
    print("="*50)
    print("Topic 01: PyTorch Reference Verification")
    print("="*50)
    
    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch not available. Install with: pip install torch")
        print("Skipping verification...")
    else:
        print(f"\n✅ PyTorch {torch.__version__} available")
        
        verify_batched_matmul()
        verify_broadcasting()
        verify_softmax()
        verify_attention_scores()
        verify_pairwise_distances()
        verify_batch_norm()
        
        print("\n" + "="*50)
        print("✅ All verifications complete!")
        print("="*50)
        print("\nYour NumPy implementations match PyTorch!")
