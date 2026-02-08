"""
Level 03: Memory-Efficient Implementation (Stride Tricks)

Topic 11: Conv2D Sliding Window

Uses np.lib.stride_tricks.as_strided to create window views without copying.
This is the most efficient pure NumPy implementation.

Key insight: Instead of extracting windows (which copies), create strided views
that share memory with the original array.
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


def conv2d_strided(input, weight, bias=None, stride=1, padding=0):
    """
    2D convolution using stride tricks for memory efficiency.

    Creates strided views of input windows, avoiding memory copies.

    Args:
        input: Shape (N, C_in, H, W)
        weight: Shape (C_out, C_in, K, K)
        bias: Shape (C_out,)
        stride: Step size
        padding: Padding size

    Returns:
        output: Shape (N, C_out, H_out, W_out)
    """
    N, C_in, H, W = input.shape
    C_out, _, K, _ = weight.shape
    compute_dtype = _select_compute_dtype(input, weight, stride, padding)
    input_work = input.astype(compute_dtype, copy=False)
    weight_work = weight.astype(compute_dtype, copy=False)
    bias_work = None if bias is None else np.asarray(bias, dtype=compute_dtype)

    # Pad input
    if padding > 0:
        input_padded = pad_array(input_work, padding)
        H_pad, W_pad = H + 2 * padding, W + 2 * padding
    else:
        input_padded = input_work
        H_pad, W_pad = H, W

    # Add trailing padding when stride does not tile exactly (ceil-style tail window).
    extra_h = (stride - ((H_pad - K) % stride)) % stride
    extra_w = (stride - ((W_pad - K) % stride)) % stride
    if extra_h != 0 or extra_w != 0:
        input_padded = np.pad(
            input_padded,
            ((0, 0), (0, 0), (0, extra_h), (0, extra_w)),
            mode="constant",
            constant_values=0,
        )
        H_pad += extra_h
        W_pad += extra_w

    # Calculate output size
    H_out = calculate_output_size(H_pad, K, 0, stride)
    W_out = calculate_output_size(W_pad, K, 0, stride)

    # Create strided view of input windows
    # Shape: (N, C_in, H_out, W_out, K, K)
    # Each element [n, c, h, w, :, :] is a K×K window starting at (h, w)
    windows = as_strided(
        input_padded,
        shape=(N, C_in, H_out, W_out, K, K),
        strides=(
            input_padded.strides[0],  # Batch stride
            input_padded.strides[1],  # Channel stride
            input_padded.strides[2] * stride,  # H stride with step
            input_padded.strides[3] * stride,  # W stride with step
            input_padded.strides[2],  # Window H stride
            input_padded.strides[3],  # Window W stride
        ),
    )

    # Compute convolution using tensordot
    # windows: (N, C_in, H_out, W_out, K, K)
    # weight:  (C_out, C_in, K, K)
    # We want: (N, C_out, H_out, W_out)

    # tensordot over axes (1, 4, 5) of windows with (1, 2, 3) of weight
    # This contracts (C_in, K, K) dimensions
    output = np.tensordot(windows, weight_work, axes=([1, 4, 5], [1, 2, 3]))

    # tensordot puts contracted dims last, so transpose to (N, C_out, H_out, W_out)
    output = output.transpose(0, 3, 1, 2)

    # Add bias
    if bias_work is not None:
        output = output + bias_work.reshape(1, -1, 1, 1)

    return output


def conv2d_chunked(input, weight, bias=None, stride=1, padding=0, chunk_size=4):
    """
    Memory-efficient version that processes batch in chunks.

    Useful when input is too large to fit in memory with strided view.
    """
    N = input.shape[0]
    outputs = []

    for i in range(0, N, chunk_size):
        chunk = input[i : i + chunk_size]
        output_chunk = conv2d_strided(chunk, weight, bias, stride, padding)
        outputs.append(output_chunk)

    return np.concatenate(outputs, axis=0)


def verify_strided_correctness():
    """Verify strided implementation matches standard approach."""
    print("Verifying strided implementation...")

    # Import vectorized version for comparison
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from level02_vectorized import conv2d_vectorized

    test_cases = [
        # (N, C_in, H, W, C_out, K, stride, padding)
        (2, 3, 8, 8, 4, 3, 1, 0),
        (2, 3, 8, 8, 4, 3, 1, 1),
        (2, 3, 8, 8, 4, 3, 2, 1),
        (4, 16, 32, 32, 32, 3, 1, 1),
        (1, 1, 5, 5, 1, 3, 1, 0),
    ]

    np.random.seed(42)

    for N, C_in, H, W, C_out, K, stride, padding in test_cases:
        input_data = np.random.randn(N, C_in, H, W).astype(np.float32)
        weight = np.random.randn(C_out, C_in, K, K).astype(np.float32)
        bias = np.random.randn(C_out).astype(np.float32)

        # Compute both ways
        expected = conv2d_vectorized(input_data, weight, bias, stride, padding)
        result = conv2d_strided(input_data, weight, bias, stride, padding)

        # Check shapes match
        assert expected.shape == result.shape, (
            f"Shape mismatch: {expected.shape} vs {result.shape}"
        )

        # Check values match
        max_diff = np.max(np.abs(expected - result))
        assert max_diff < 1e-4, f"Values differ: max diff = {max_diff}"

        print(
            f"  ✅ (N={N}, C_in={C_in}, H={H}, K={K}, S={stride}, P={padding}): max_diff={max_diff:.2e}"
        )

    print("✅ All strided correctness tests passed!")


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Conv2D Sliding Window - Level 03 (Memory-Efficient)")
    print("=" * 60)

    # Verify correctness first
    print("\nCorrectness Verification")
    print("-" * 60)
    verify_strided_correctness()

    # Benchmark
    print("\n\nPerformance Benchmark")
    print("-" * 60)

    # Import for comparison
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from level01_naive import conv2d_naive
    from level02_vectorized import conv2d_vectorized

    # Test configurations
    benchmarks = [
        ("Small", (4, 8, 16, 16), 16, 3),
        ("Medium", (8, 32, 32, 32), 64, 3),
        ("Large", (4, 64, 64, 64), 128, 3),
    ]

    for name, input_shape, C_out, K in benchmarks:
        print(f"\n{name}: input={input_shape}, C_out={C_out}, K={K}x{K}")

        N, C_in, H, W = input_shape
        input_data = np.random.randn(*input_shape).astype(np.float32)
        weight = np.random.randn(C_out, C_in, K, K).astype(np.float32)
        bias = np.random.randn(C_out).astype(np.float32)

        # Warmup
        _ = conv2d_naive(input_data, weight, bias, stride=1, padding=1)
        _ = conv2d_vectorized(input_data, weight, bias, stride=1, padding=1)
        _ = conv2d_strided(input_data, weight, bias, stride=1, padding=1)

        # Benchmark naive (skip for large)
        if name != "Large":
            start = time.time()
            for _ in range(3):
                _ = conv2d_naive(input_data, weight, bias, stride=1, padding=1)
            time_naive = time.time() - start
            print(f"  Naive:      {time_naive:.4f}s")
        else:
            time_naive = None
            print(f"  Naive:      (skipped - too slow)")

        # Benchmark vectorized
        start = time.time()
        for _ in range(3):
            _ = conv2d_vectorized(input_data, weight, bias, stride=1, padding=1)
        time_vect = time.time() - start
        print(f"  Vectorized: {time_vect:.4f}s", end="")
        if time_naive:
            print(f" ({time_naive / time_vect:.1f}x vs naive)")
        else:
            print()

        # Benchmark strided
        start = time.time()
        for _ in range(3):
            _ = conv2d_strided(input_data, weight, bias, stride=1, padding=1)
        time_strided = time.time() - start
        print(f"  Strided:    {time_strided:.4f}s", end="")
        if time_naive:
            print(f" ({time_naive / time_strided:.1f}x vs naive)")
        else:
            print(f" ({time_vect / time_strided:.1f}x vs vectorized)")

    # Test chunked processing
    print("\n\nChunked Processing Test")
    print("-" * 60)

    large_input = np.random.randn(16, 32, 64, 64).astype(np.float32)
    weight_large = np.random.randn(64, 32, 3, 3).astype(np.float32)
    bias_large = np.random.randn(64).astype(np.float32)

    print(f"Large input: {large_input.shape}")
    print("Processing in chunks of 4...")

    start = time.time()
    output_chunked = conv2d_chunked(
        large_input, weight_large, bias_large, stride=1, padding=1, chunk_size=4
    )
    time_chunked = time.time() - start

    # Compare to full processing
    start = time.time()
    output_full = conv2d_strided(
        large_input, weight_large, bias_large, stride=1, padding=1
    )
    time_full = time.time() - start

    print(f"Full:   {time_full:.4f}s")
    print(f"Chunked: {time_chunked:.4f}s")
    print(f"Results match: {np.allclose(output_chunked, output_full)}")

    print("\n" + "=" * 60)
    print("✅ Level 03 complete!")
    print("\nKey techniques:")
    print("- as_strided(): Zero-copy window extraction")
    print("- tensordot(): Efficient contraction over (C,K,K)")
    print("- Chunked processing: Memory-constrained scenarios")
    print("\nNext: Level 04 (PyTorch verification)")
    print("=" * 60)
