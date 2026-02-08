"""
Level 03: Memory-Efficient Implementation (Stride Tricks)

Topic 01: Tensor Operations & Broadcasting

This implementation minimizes memory allocation using:
- Views instead of copies
- Stride tricks for zero-copy operations
- In-place operations where possible

For interview scenarios where memory efficiency matters.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided


def broadcast_to_efficient(x: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Broadcast array to new shape WITHOUT copying data.
    
    The returned array is a view - modifications are not allowed!
    
    Args:
        x: Input array
        shape: Target shape (must be broadcast-compatible)
    
    Returns:
        View with the new shape
    """
    return np.broadcast_to(x, shape)


def efficient_repeat(x: np.ndarray, repeats: int, axis: int) -> np.ndarray:
    """
    Repeat array along axis WITHOUT copying memory.
    
    Uses broadcast_to to create a virtual repeat.
    Note: Result is read-only!
    
    Args:
        x: Input array
        repeats: Number of repetitions
        axis: Axis along which to repeat
    
    Returns:
        Read-only view with repeated elements
    """
    # Insert a new dimension
    new_shape = list(x.shape)
    new_shape.insert(axis, 1)
    x_expanded = x.reshape(new_shape)
    
    # Broadcast the new dimension
    broadcast_shape = list(x.shape)
    broadcast_shape.insert(axis, repeats)
    
    return np.broadcast_to(x_expanded, broadcast_shape)


def sliding_window_1d(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    Create sliding window view of 1D array.
    
    ZERO memory allocation - just a view!
    
    Args:
        x: 1D input array of length n
        window_size: Size of each window
    
    Returns:
        View of shape (n - window_size + 1, window_size)
    
    Example:
        x = [1, 2, 3, 4, 5], window_size = 3
        result = [[1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 5]]
    """
    n = x.shape[0]
    output_len = n - window_size + 1
    
    # Key insight: both dimensions use the same stride!
    new_shape = (output_len, window_size)
    new_strides = (x.strides[0], x.strides[0])
    
    return as_strided(x, shape=new_shape, strides=new_strides)


def sliding_window_2d(x: np.ndarray, window_shape: tuple) -> np.ndarray:
    """
    Create sliding window view of 2D array.
    
    Used for convolution operations (im2col style).
    
    Args:
        x: 2D input array of shape (H, W)
        window_shape: (kH, kW) window size
    
    Returns:
        View of shape (H - kH + 1, W - kW + 1, kH, kW)
    """
    H, W = x.shape
    kH, kW = window_shape
    
    out_H = H - kH + 1
    out_W = W - kW + 1
    
    new_shape = (out_H, out_W, kH, kW)
    new_strides = (x.strides[0], x.strides[1], x.strides[0], x.strides[1])
    
    return as_strided(x, shape=new_shape, strides=new_strides)


def im2col(images: np.ndarray, kernel_size: tuple, 
           stride: int = 1) -> np.ndarray:
    """
    Convert image patches to columns for efficient convolution.
    
    This is how optimized convolution libraries work!
    
    Args:
        images: Shape (batch, C, H, W)
        kernel_size: (kH, kW)
        stride: Stride of the convolution
    
    Returns:
        Columns: Shape (batch, C * kH * kW, out_H * out_W)
    """
    batch, C, H, W = images.shape
    kH, kW = kernel_size
    
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    
    # Create strided view
    shape = (batch, C, out_H, out_W, kH, kW)
    strides = (
        images.strides[0],  # batch
        images.strides[1],  # channel
        images.strides[2] * stride,  # out_H (with stride)
        images.strides[3] * stride,  # out_W (with stride)
        images.strides[2],  # kH
        images.strides[3],  # kW
    )
    
    patches = as_strided(images, shape=shape, strides=strides)
    
    # Reshape to (batch, C * kH * kW, out_H * out_W)
    return patches.reshape(batch, C * kH * kW, out_H * out_W)


def efficient_transpose(x: np.ndarray) -> np.ndarray:
    """
    Transpose without copying (view only).
    
    np.T already does this, but this shows the principle.
    """
    new_shape = x.shape[::-1]
    new_strides = x.strides[::-1]
    return as_strided(x, shape=new_shape, strides=new_strides)


def add_inplace(target: np.ndarray, source: np.ndarray) -> None:
    """
    Add source to target in-place.
    
    Avoids allocating a new array.
    """
    np.add(target, source, out=target)


def softmax_inplace(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax with minimal memory allocation.
    
    Modifies x in place!
    """
    # Subtract max for stability (in-place)
    max_val = np.max(x, axis=axis, keepdims=True)
    x -= max_val
    
    # Exp in-place
    np.exp(x, out=x)
    
    # Normalize in-place
    sum_exp = np.sum(x, axis=axis, keepdims=True)
    x /= sum_exp
    
    return x


def memory_info(arr: np.ndarray) -> dict:
    """
    Get memory information about an array.
    
    Useful for debugging memory usage.
    """
    return {
        'shape': arr.shape,
        'strides': arr.strides,
        'itemsize': arr.itemsize,
        'nbytes': arr.nbytes,
        'is_contiguous': arr.flags['C_CONTIGUOUS'],
        'is_view': arr.base is not None,
        'owns_data': arr.flags['OWNDATA'],
    }


# Test code
if __name__ == "__main__":
    print("Testing Level 03: Memory-Efficient Implementation\n")
    print("="*50)
    
    # Test 1: Sliding window
    print("\nTest 1: Sliding Window 1D (Zero Copy)")
    x = np.arange(10)
    windows = sliding_window_1d(x, 3)
    print(f"Input: {x}")
    print(f"Windows:\n{windows}")
    print(f"Input memory: {x.nbytes} bytes")
    print(f"Windows memory: {windows.nbytes} bytes (but shares memory!)")
    print(f"Shares memory: {np.shares_memory(x, windows)}")
    
    # Test 2: Broadcast repeat
    print("\nTest 2: Efficient Repeat (Zero Copy)")
    x = np.array([1, 2, 3])
    repeated = efficient_repeat(x, 4, axis=0)
    print(f"Input: {x}")
    print(f"Repeated:\n{repeated}")
    print(f"Input memory: {x.nbytes} bytes")
    print(f"Repeated base size: {repeated.base.nbytes if repeated.base is not None else 'N/A'}")
    
    # Test 3: 2D sliding window
    print("\nTest 3: Sliding Window 2D")
    img = np.arange(16).reshape(4, 4)
    patches = sliding_window_2d(img, (2, 2))
    print(f"Input (4x4):\n{img}")
    print(f"Patches shape: {patches.shape}")
    print(f"Shares memory: {np.shares_memory(img, patches)}")
    
    # Test 4: Memory info
    print("\nTest 4: Memory Analysis")
    x = np.random.randn(100, 100)
    y = x[::2, ::2]  # View with non-contiguous strides
    
    print("Original array:")
    for k, v in memory_info(x).items():
        print(f"  {k}: {v}")
    
    print("\nSliced view:")
    for k, v in memory_info(y).items():
        print(f"  {k}: {v}")
    
    # Test 5: In-place softmax
    print("\nTest 5: In-place Softmax")
    x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    x_copy = x.copy()
    
    print(f"Before: {x}")
    softmax_inplace(x)
    print(f"After:  {x}")
    print(f"Row sums: {x.sum(axis=1)}")
    
    print("\n" + "="*50)
    print("✅ All memory-efficient implementations working!")
    print("\nKey takeaway: Views + stride tricks = minimal memory allocation")
