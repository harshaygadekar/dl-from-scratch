# Hint 2: Padding Logic

## Why Padding?

Without padding, convolutions **shrink** the image:

```
Input: 5×5, Kernel: 3×3, Stride: 1

Valid positions for 3×3 kernel:
Row 0: Cols 0,1,2 → 3 positions
Row 1: Cols 0,1,2 → 3 positions  
Row 2: Cols 0,1,2 → 3 positions

Output: 3×3 (lost 1 pixel border on each side!)
```

## Padding Formula

To maintain spatial size with stride=1:

```
pad = (K - 1) // 2

Examples:
- K=3 → pad=1 (add 1 row on top/bottom, 1 col on left/right)
- K=5 → pad=2
- K=7 → pad=3
```

## Padding Implementation

### Method 1: NumPy Pad (Easy)

```python
def pad_array(input, padding):
    """
    Pad spatial dimensions (H, W) with zeros.
    
    Args:
        input: Shape (N, C, H, W)
        padding: Number of zeros to add on each side
    
    Returns:
        Padded array: Shape (N, C, H+2*pad, W+2*pad)
    """
    if padding == 0:
        return input
    
    # np.pad format: ((before_0, after_0), (before_1, after_1), ...)
    # We want to pad dimensions 2 and 3 (H and W)
    return np.pad(input, 
                  pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
                  mode='constant', 
                  constant_values=0)
```

### Method 2: Manual Padding (Educational)

```python
def pad_array_manual(input, padding):
    """Manual implementation without np.pad."""
    N, C, H, W = input.shape
    
    # Create larger array filled with zeros
    padded = np.zeros((N, C, H + 2*padding, W + 2*padding))
    
    # Copy original data to center
    padded[:, :, padding:padding+H, padding:padding+W] = input
    
    return padded
```

## Visual Example

```
Before padding (1×1×3×3):
┌───┬───┬───┐
│ 1 │ 2 │ 3 │
├───┼───┼───┤
│ 4 │ 5 │ 6 │
├───┼───┼───┤
│ 7 │ 8 │ 9 │
└───┴───┴───┘

After pad=1 (1×1×5×5):
┌───┬───┬───┬───┬───┐
│ 0 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┼───┤
│ 0 │ 1 │ 2 │ 3 │ 0 │
├───┼───┼───┼───┼───┤
│ 0 │ 4 │ 5 │ 6 │ 0 │
├───┼───┼───┼───┼───┤
│ 0 │ 7 │ 8 │ 9 │ 0 │
├───┼───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │ 0 │
└───┴───┴───┴───┴───┘
```

## Complete Integration

```python
def conv2d_naive(input, weight, bias=None, stride=1, padding=0):
    N, C_in, H, W = input.shape
    C_out, _, K, _ = weight.shape
    
    # Step 1: Pad input
    if padding > 0:
        input_padded = np.pad(input, 
                             ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                             mode='constant')
        H_pad, W_pad = H + 2*padding, W + 2*padding
    else:
        input_padded = input
        H_pad, W_pad = H, W
    
    # Step 2: Calculate output size
    H_out = (H_pad - K) // stride + 1
    W_out = (W_pad - K) // stride + 1
    
    # Step 3: Initialize output
    output = np.zeros((N, C_out, H_out, W_out))
    
    # Step 4: Convolution loops (see hint-1 for structure)
    for n in range(N):
        for c_out in range(C_out):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    # Extract input window
                    h_start = h_out * stride
                    w_start = w_out * stride
                    
                    # Use the PADDED input here!
                    input_window = input_padded[n, :, h_start:h_start+K, w_start:w_start+K]
                    kernel = weight[c_out]
                    
                    # Compute dot product over C_in, K, K
                    output[n, c_out, h_out, w_out] = np.sum(input_window * kernel)
            
            # Add bias for this output channel
            if bias is not None:
                output[n, c_out] += bias[c_out]
    
    return output
```

## Padding Modes Reference

| Mode | Behavior | Use Case |
|------|----------|----------|
| `'constant'` | Pad with constant value (0) | Standard zero-padding |
| `'edge'` | Pad with edge value | Avoid edge artifacts |
| `'reflect'` | Reflect about edge | Texture synthesis |
| `'symmetric'` | Mirror about edge | Image processing |

**Most common in CNNs**: `'constant'` with value 0

## Debugging Tip

If your output size is wrong, check:

```python
# Add debug prints
print(f"Input: {input.shape}")
print(f"After padding: {input_padded.shape}")
print(f"Kernel: {K}, Stride: {stride}")
print(f"Expected output: ({N}, {C_out}, {H_out}, {W_out})")
print(f"Actual output: {output.shape}")
```

## Edge Cases

1. **padding=0**: Valid convolution (output smaller)
2. **padding > 0 with stride > 1**: May still shrink spatially
3. **Large padding**: Wastes computation on zeros

## Next Steps

Once padding works:
1. Test with different padding values
2. Verify output size formula: `(H + 2*P - K) // S + 1`
3. Move to Level 2: Vectorize the inner loops

**See hint-3-loop-structure.md for optimization tips**
