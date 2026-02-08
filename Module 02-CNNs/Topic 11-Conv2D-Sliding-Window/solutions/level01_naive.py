"""
Level 01: Naive Implementation (Explicit Loops)

Topic 11: Conv2D Sliding Window

This implementation uses 6 nested loops to make every operation explicit.
Priority: Readability and correctness over speed.

Time complexity: O(N × C_out × H_out × W_out × C_in × K²)
Space complexity: O(N × C_out × H_out × W_out) for output
"""

import numpy as np


def calculate_output_size(H, K, P, S):
    """
    Calculate output spatial dimension.

    Formula: H_out = floor((H - K + 2*P) / S) + 1

    Args:
        H: Input height (or width)
        K: Kernel size
        P: Padding
        S: Stride

    Returns:
        Output height (or width)
    """
    return (H - K + 2 * P) // S + 1


def pad_array(x, padding):
    """
    Pad array with zeros on spatial dimensions (H, W).

    Args:
        x: Input array of shape (N, C, H, W)
        padding: Number of zeros to add on each side

    Returns:
        Padded array of shape (N, C, H+2*padding, W+2*padding)
    """
    if padding == 0:
        return x

    # Pad dimensions 2 and 3 (H and W)
    return np.pad(
        x,
        pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0,
    )


def conv2d_naive(input, weight, bias=None, stride=1, padding=0):
    """
    2D convolution using explicit nested loops.

    Args:
        input: Shape (N, C_in, H, W) - batch of images
        weight: Shape (C_out, C_in, K, K) - convolution kernels
        bias: Shape (C_out,) - optional bias terms
        stride: Step size for sliding window (default: 1)
        padding: Number of zeros to pad on each side (default: 0)

    Returns:
        output: Shape (N, C_out, H_out, W_out)

    Example:
        >>> input = np.random.randn(2, 3, 32, 32)  # 2 images, 3 channels, 32x32
        >>> weight = np.random.randn(16, 3, 3, 3)  # 16 filters, 3x3 kernel
        >>> output = conv2d_naive(input, weight, stride=1, padding=1)
        >>> print(output.shape)
        (2, 16, 32, 32)
    """
    # Get dimensions
    N, C_in, H, W = input.shape
    C_out, _, K, _ = weight.shape
    compute_dtype = np.result_type(input.dtype, weight.dtype, np.float64)

    # Step 1: Pad input if needed
    if padding > 0:
        input_padded = pad_array(input, padding).astype(compute_dtype, copy=False)
        H_pad, W_pad = H + 2 * padding, W + 2 * padding
    else:
        input_padded = input.astype(compute_dtype, copy=False)
        H_pad, W_pad = H, W

    weight_work = weight.astype(compute_dtype, copy=False)
    bias_work = None if bias is None else np.asarray(bias, dtype=compute_dtype)

    # Step 2: Calculate output dimensions
    H_out = calculate_output_size(H_pad, K, 0, stride)
    W_out = calculate_output_size(W_pad, K, 0, stride)

    # Step 3: Initialize output array
    output = np.zeros((N, C_out, H_out, W_out), dtype=compute_dtype)

    # Step 4: Six nested loops for convolution
    for n in range(N):  # Iterate over batch samples
        for c_out in range(C_out):  # Iterate over output channels
            for h_out in range(H_out):  # Iterate over output height
                for w_out in range(W_out):  # Iterate over output width
                    # Compute one output pixel
                    accum = compute_dtype.type(0.0)

                    for c_in in range(C_in):  # Iterate over input channels
                        for kh in range(K):  # Iterate over kernel height
                            for kw in range(K):  # Iterate over kernel width
                                # Calculate input position
                                h_in = h_out * stride + kh
                                w_in = w_out * stride + kw

                                # Accumulate: input * weight
                                accum += (
                                    input_padded[n, c_in, h_in, w_in]
                                    * weight_work[c_out, c_in, kh, kw]
                                )

                    # Add bias if provided
                    if bias_work is not None:
                        accum += bias_work[c_out]

                    # Store result
                    output[n, c_out, h_out, w_out] = accum

    return output


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("Conv2D Sliding Window - Level 01 (Naive)")
    print("=" * 60)

    # Test 1: Basic shape test
    print("\nTest 1: Basic Shape Verification")
    print("-" * 60)

    input1 = np.random.randn(2, 3, 5, 5)  # 2 samples, 3 channels, 5x5
    weight1 = np.random.randn(4, 3, 3, 3)  # 4 filters, 3x3 kernel
    bias1 = np.random.randn(4)

    output1 = conv2d_naive(input1, weight1, bias1, stride=1, padding=0)

    print(f"Input shape: {input1.shape}")
    print(f"Weight shape: {weight1.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Expected output shape: (2, 4, 3, 3)")
    assert output1.shape == (2, 4, 3, 3), f"Shape mismatch: {output1.shape}"
    print("✅ Shape test passed!")

    # Test 2: Padding test (same convolution)
    print("\nTest 2: 'Same' Convolution (stride=1, padding=1)")
    print("-" * 60)

    input2 = np.random.randn(1, 1, 5, 5)
    weight2 = np.random.randn(2, 1, 3, 3)

    output2 = conv2d_naive(input2, weight2, stride=1, padding=1)

    print(f"Input: {input2.shape}")
    print(f"With padding=1, kernel=3x3, stride=1")
    print(f"Output: {output2.shape}")
    print(f"Expected: (1, 2, 5, 5) (same spatial size)")
    assert output2.shape == (1, 2, 5, 5), f"Shape mismatch: {output2.shape}"
    print("✅ Padding test passed!")

    # Test 3: Stride test
    print("\nTest 3: Strided Convolution (stride=2)")
    print("-" * 60)

    input3 = np.random.randn(1, 1, 7, 7)
    weight3 = np.random.randn(1, 1, 3, 3)

    output3 = conv2d_naive(input3, weight3, stride=2, padding=1)

    print(f"Input: {input3.shape}")
    print(f"With stride=2, padding=1, kernel=3x3")
    print(f"Output: {output3.shape}")
    # H_out = (7 + 2*1 - 3) // 2 + 1 = 7 // 2 + 1 = 3 + 1 = 4
    print(f"Expected: (1, 1, 4, 4)")
    assert output3.shape == (1, 1, 4, 4), f"Shape mismatch: {output3.shape}"
    print("✅ Stride test passed!")

    # Test 4: Known values
    print("\nTest 4: Known Values Verification")
    print("-" * 60)

    # Simple 3x3 input with 2x2 kernel
    input4 = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])  # Shape: (1, 1, 3, 3)

    weight4 = np.array([[[[1, 0], [0, -1]]]])  # Shape: (1, 1, 2, 2)

    output4 = conv2d_naive(input4, weight4, stride=1, padding=0)

    # Manual calculation:
    # [0,0]: 1*1 + 2*0 + 4*0 + 5*(-1) = 1 - 5 = -4
    # [0,1]: 2*1 + 3*0 + 5*0 + 6*(-1) = 2 - 6 = -4
    # [1,0]: 4*1 + 5*0 + 7*0 + 8*(-1) = 4 - 8 = -4
    # [1,1]: 5*1 + 6*0 + 8*0 + 9*(-1) = 5 - 9 = -4
    expected4 = np.array([[[[-4, -4], [-4, -4]]]])

    print(f"Input:\n{input4[0, 0]}")
    print(f"\nKernel:\n{weight4[0, 0]}")
    print(f"\nOutput:\n{output4[0, 0]}")
    print(f"\nExpected:\n{expected4[0, 0]}")

    np.testing.assert_array_equal(output4, expected4)
    print("✅ Known values test passed!")

    # Test 5: Multi-channel input
    print("\nTest 5: Multi-Channel Convolution")
    print("-" * 60)

    input5 = np.random.randn(1, 3, 4, 4)  # 3 input channels
    weight5 = np.random.randn(2, 3, 3, 3)  # 2 output channels, 3 input channels
    bias5 = np.array([0.5, -0.5])

    output5 = conv2d_naive(input5, weight5, bias5, stride=1, padding=0)

    print(f"Input: {input5.shape} (3 channels)")
    print(f"Weight: {weight5.shape} (2 output channels)")
    print(f"Bias: {bias5}")
    print(f"Output: {output5.shape}")
    # (4 - 3) // 1 + 1 = 2
    assert output5.shape == (1, 2, 2, 2), f"Shape mismatch: {output5.shape}"
    print("✅ Multi-channel test passed!")

    # Test 6: Identity kernel
    print("\nTest 6: Identity Kernel Test")
    print("-" * 60)

    input6 = np.random.randn(1, 1, 5, 5)
    # 1x1 identity kernel (just multiplies by 1)
    weight6 = np.ones((1, 1, 1, 1))

    output6 = conv2d_naive(input6, weight6, stride=1, padding=0)

    print(f"1x1 conv with weight=1 should preserve input")
    print(f"Input[0,0]: {input6[0, 0, 0, 0]:.4f}")
    print(f"Output[0,0]: {output6[0, 0, 0, 0]:.4f}")
    np.testing.assert_allclose(output6, input6, rtol=1e-5)
    print("✅ Identity kernel test passed!")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\nNext steps:")
    print("1. Study hints for optimization ideas")
    print("2. Implement Level 2 (vectorized)")
    print("3. Run: pytest tests/test_basic.py")
    print("=" * 60)
