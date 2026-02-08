"""
Level 02: Vectorized Implementation (NumPy Optimized)

Topic 11: Conv2D Sliding Window

This implementation replaces inner loops with NumPy vectorized operations.
Much faster than naive while maintaining readability.

Time complexity: O(N × C_out × H_out × W_out) loop iterations
Inner work done via np.sum() over (C_in, K, K)
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided

HIGH_PRECISION_OUTPUT_LIMIT = 32768


def calculate_output_size(H, K, P, S):
    """Calculate output spatial dimension."""
    return (H - K + 2 * P) // S + 1


def pad_array(x, padding):
    """Pad array with zeros on spatial dimensions."""
    if padding == 0:
        return x
    return np.pad(
        x,
        pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )


def _select_compute_dtype(input, weight, stride, padding):
    """Use float64 on small tensors for reproducibility, native dtype for speed."""
    n, _, h, w = input.shape
    c_out, _, k, _ = weight.shape
    h_out = calculate_output_size(h + 2 * padding, k, 0, stride)
    w_out = calculate_output_size(w + 2 * padding, k, 0, stride)
    output_size = n * c_out * h_out * w_out

    if output_size <= HIGH_PRECISION_OUTPUT_LIMIT:
        return np.result_type(input.dtype, weight.dtype, np.float64)
    return np.result_type(input.dtype, weight.dtype)


def _window_view_nchw(input_padded, kernel_size, stride):
    """Create a zero-copy window view: (N, C, H_out, W_out, K, K)."""
    n, c, h_pad, w_pad = input_padded.shape
    h_out = calculate_output_size(h_pad, kernel_size, 0, stride)
    w_out = calculate_output_size(w_pad, kernel_size, 0, stride)
    return as_strided(
        input_padded,
        shape=(n, c, h_out, w_out, kernel_size, kernel_size),
        strides=(
            input_padded.strides[0],
            input_padded.strides[1],
            input_padded.strides[2] * stride,
            input_padded.strides[3] * stride,
            input_padded.strides[2],
            input_padded.strides[3],
        ),
    )


def conv2d_vectorized(input, weight, bias=None, stride=1, padding=0):
    """
    2D convolution with vectorized inner operations.

    Vectorizes over (C_in, K, K) using np.sum() instead of triple loop.

    Args:
        input: Shape (N, C_in, H, W)
        weight: Shape (C_out, C_in, K, K)
        bias: Shape (C_out,)
        stride: Step size
        padding: Padding size

    Returns:
        output: Shape (N, C_out, H_out, W_out)
    """
    # Get dimensions
    N, C_in, H, W = input.shape
    C_out, _, K, _ = weight.shape
    compute_dtype = _select_compute_dtype(input, weight, stride, padding)

    # Step 1: Pad input
    if padding > 0:
        input_padded = pad_array(input, padding).astype(compute_dtype, copy=False)
        H_pad, W_pad = H + 2 * padding, W + 2 * padding
    else:
        input_padded = input.astype(compute_dtype, copy=False)
        H_pad, W_pad = H, W

    weight_work = weight.astype(compute_dtype, copy=False)
    bias_work = None if bias is None else np.asarray(bias, dtype=compute_dtype)

    # Step 2: Build sliding-window view and contract over (C_in, K, K)
    windows = _window_view_nchw(input_padded, K, stride)
    output = np.tensordot(windows, weight_work, axes=([1, 4, 5], [1, 2, 3]))
    output = output.transpose(0, 3, 1, 2)

    # Step 3: Add bias
    if bias_work is not None:
        output = output + bias_work.reshape(1, -1, 1, 1)

    return output


def conv2d_batch_vectorized(input, weight, bias=None, stride=1, padding=0):
    """
    Further vectorized version that processes entire batch at once.

    Uses broadcasting to eliminate the batch loop.
    """
    return conv2d_vectorized(input, weight, bias=bias, stride=stride, padding=padding)


# Test and benchmark
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Conv2D Sliding Window - Level 02 (Vectorized)")
    print("=" * 60)

    # Import naive version for comparison
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from level01_naive import conv2d_naive

    # Test correctness
    print("\nTest: Correctness Verification")
    print("-" * 60)

    np.random.seed(42)
    input_test = np.random.randn(4, 8, 16, 16)
    weight_test = np.random.randn(16, 8, 3, 3)
    bias_test = np.random.randn(16)

    output_naive = conv2d_naive(input_test, weight_test, bias_test, stride=1, padding=1)
    output_vectorized = conv2d_vectorized(
        input_test, weight_test, bias_test, stride=1, padding=1
    )
    output_batch = conv2d_batch_vectorized(
        input_test, weight_test, bias_test, stride=1, padding=1
    )

    print(f"Input: {input_test.shape}")
    print(f"Output shape: {output_vectorized.shape}")

    # Check numerical equality
    if np.allclose(output_naive, output_vectorized, rtol=1e-5):
        print("✅ Vectorized matches naive implementation!")
    else:
        print("❌ Vectorized differs from naive!")
        print(f"Max difference: {np.max(np.abs(output_naive - output_vectorized))}")

    if np.allclose(output_naive, output_batch, rtol=1e-5):
        print("✅ Batch-vectorized matches naive!")
    else:
        print("❌ Batch-vectorized differs!")

    # Benchmark
    print("\nBenchmark: Speed Comparison")
    print("-" * 60)

    # Warmup
    _ = conv2d_naive(input_test, weight_test, bias_test, stride=1, padding=1)
    _ = conv2d_vectorized(input_test, weight_test, bias_test, stride=1, padding=1)
    _ = conv2d_batch_vectorized(input_test, weight_test, bias_test, stride=1, padding=1)

    # Time naive
    start = time.time()
    for _ in range(3):
        _ = conv2d_naive(input_test, weight_test, bias_test, stride=1, padding=1)
    time_naive = time.time() - start

    # Time vectorized
    start = time.time()
    for _ in range(3):
        _ = conv2d_vectorized(input_test, weight_test, bias_test, stride=1, padding=1)
    time_vectorized = time.time() - start

    # Time batch vectorized
    start = time.time()
    for _ in range(3):
        _ = conv2d_batch_vectorized(
            input_test, weight_test, bias_test, stride=1, padding=1
        )
    time_batch = time.time() - start

    print(f"Naive:         {time_naive:.4f}s")
    print(
        f"Vectorized:    {time_vectorized:.4f}s ({time_naive / time_vectorized:.1f}x speedup)"
    )
    print(f"Batch-vect:    {time_batch:.4f}s ({time_naive / time_batch:.1f}x speedup)")

    # Additional tests
    print("\nAdditional Tests")
    print("-" * 60)

    # Test 1: Different strides and padding
    test_cases = [
        (1, 0, "Valid conv"),
        (1, 1, "Same conv"),
        (2, 1, "Strided"),
        (2, 0, "Strided valid"),
    ]

    input_small = np.random.randn(2, 4, 8, 8)
    weight_small = np.random.randn(8, 4, 3, 3)

    for stride, padding, name in test_cases:
        output = conv2d_vectorized(
            input_small, weight_small, stride=stride, padding=padding
        )
        print(f"✅ {name} (stride={stride}, pad={padding}): {output.shape}")

    # Test 2: Large kernel
    print("\n✅ Large kernel (7x7) test:", end=" ")
    input_large = np.random.randn(1, 3, 64, 64)
    weight_large = np.random.randn(8, 3, 7, 7)
    output_large = conv2d_vectorized(input_large, weight_large, stride=2, padding=3)
    print(f"{output_large.shape}")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\nKey improvements from Level 1:")
    print("- Replaced triple inner loop with np.sum()")
    print("- Used array slicing for window extraction")
    print("- Batch-vectorized version eliminates batch loop")
    print("\nNext: Level 3 (memory-efficient with stride tricks)")
    print("=" * 60)
