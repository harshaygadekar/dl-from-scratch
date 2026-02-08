# Hint 3: Loop Structure and Optimization

## From 6 Loops to Vectorized

### Level 1: Full Nested Loops (Slow but Clear)

```python
for n in range(N):
    for c_out in range(C_out):
        for h_out in range(H_out):
            for w_out in range(W_out):
                accum = 0
                for c_in in range(C_in):
                    for kh in range(K):
                        for kw in range(K):
                            accum += input[n, c_in, h*stride+kh, w*stride+kw] * weight[c_out, c_in, kh, kw]
                output[n, c_out, h, w] = accum
```

**Complexity**: $N \times C_{out} \times H_{out} \times W_{out} \times C_{in} \times K^2$ loop iterations

### Level 2: Vectorize Inner 3 Loops

Replace the inner triple loop with NumPy operations:

```python
for n in range(N):
    for c_out in range(C_out):
        for h_out in range(H_out):
            for w_out in range(W_out):
                # Extract window: (C_in, K, K)
                h_start = h_out * stride
                w_start = w_out * stride
                window = input[n, :, h_start:h_start+K, w_start:w_start+K]
                
                # Element-wise multiply and sum
                # (C_in, K, K) * (C_in, K, K) → sum all
                output[n, c_out, h_out, w_out] = np.sum(window * weight[c_out])
```

**Speedup**: ~10-50x from removing Python loops over kernel

### Level 3: Vectorize Over Spatial Dimensions

Use array broadcasting to compute all spatial positions at once:

```python
# Approach: Use as_strided to create windows without copying
from numpy.lib.stride_tricks import as_strided

def conv2d_strided(input, weight, stride=1, padding=0):
    N, C_in, H, W = input.shape
    C_out, _, K, _ = weight.shape
    
    if padding > 0:
        input = np.pad(input, ((0,0), (0,0), (padding, padding), (padding, padding)))
        H, W = H + 2*padding, W + 2*padding
    
    H_out = (H - K) // stride + 1
    W_out = (W - K) // stride + 1
    
    # Create strided view: (N, C_in, H_out, W_out, K, K)
    # Each position [i, j] contains the K×K window starting at (i, j)
    itemsize = input.itemsize
    windows = as_strided(
        input,
        shape=(N, C_in, H_out, W_out, K, K),
        strides=(
            input.strides[0],  # Batch stride
            input.strides[1],  # Channel stride  
            input.strides[2] * stride,  # H stride (with step)
            input.strides[3] * stride,  # W stride (with step)
            input.strides[2],  # Kernel H stride
            input.strides[3]   # Kernel W stride
        )
    )
    
    # Now compute: (N, C_in, H_out, W_out, K, K) × (C_out, C_in, K, K)
    # We want: (N, C_out, H_out, W_out)
    
    # Method: tensordot over (C_in, K, K) dimensions
    output = np.tensordot(windows, weight, axes=([1, 4, 5], [1, 2, 3]))
    
    # tensordot puts contracted dimensions last, so transpose
    output = output.transpose(0, 3, 1, 2)
    
    return output
```

**How `as_strided` works**:
- Creates a **view** (no copy!) with custom strides
- Each output element "sees" a K×K window of the input
- Changes to windows affect original array (be careful!)

## Output Size Verification

Always verify your output dimensions:

```python
def verify_output_size(input_shape, kernel_size, padding, stride):
    """Verify conv output size calculation."""
    N, C, H, W = input_shape
    K = kernel_size
    P = padding
    S = stride
    
    H_out = (H + 2*P - K) // S + 1
    W_out = (W + 2*P - K) // S + 1
    
    print(f"Input: {input_shape}")
    print(f"Kernel: {K}×{K}, Padding: {P}, Stride: {S}")
    print(f"Output: ({N}, C_out, {H_out}, {W_out})")
    
    return H_out, W_out

# Test cases
test_cases = [
    ((1, 3, 32, 32), 3, 1, 1),   # Same conv
    ((1, 3, 32, 32), 3, 0, 1),   # Valid conv
    ((1, 3, 32, 32), 3, 1, 2),   # Strided
    ((1, 64, 224, 224), 7, 3, 2), # AlexNet first layer
]

for shape, k, p, s in test_cases:
    verify_output_size(shape, k, p, s)
    print()
```

## Common Optimization Pitfalls

### Pitfall 1: Creating Copies Unnecessarily

```python
# SLOW - Creates copies in inner loop
for h in range(H_out):
    for w in range(W_out):
        window = input[:, :, h:h+K, w:w+K].copy()  # ← Copy!
        output[:, :, h, w] = np.sum(window * weight, axis=(1,2,3))

# FAST - Use views
for h in range(H_out):
    for w in range(W_out):
        window = input[:, :, h:h+K, w:w+K]  # View (no copy)
        output[:, :, h, w] = np.sum(window * weight, axis=(1,2,3))
```

### Pitfall 2: Bad Memory Access Pattern

```python
# SLOW - Jumping around in memory
for c_in in range(C_in):
    for kh in range(K):
        for kw in range(K):
            accum += input[n, c_in, h+kh, w+kw] * weight[c_out, c_in, kh, kw]

# FAST - Process channels together (contiguous memory)
window = input[n, :, h:h+K, w:w+K]  # All channels at once
output[n, c_out, h, w] = np.sum(window * weight[c_out])
```

### Pitfall 3: Python Loop Overhead

```python
# SLOW - Python loops over spatial dims
for h in range(H_out):
    for w in range(W_out):
        output[n, c_out, h, w] = compute_conv(...)

# FAST - Vectorize spatial dimensions
# Extract all windows at once, then dot product
windows = extract_all_windows(input)  # (N, C_in, H_out, W_out, K, K)
output = np.tensordot(windows, weight, axes=3)
```

## Performance Benchmarking

```python
import time

def benchmark_conv(conv_fn, input, weight, name=""):
    """Benchmark a convolution function."""
    # Warmup
    _ = conv_fn(input, weight)
    
    # Time it
    start = time.time()
    for _ in range(10):
        _ = conv_fn(input, weight)
    elapsed = time.time() - start
    
    print(f"{name}: {elapsed:.4f}s")
    return elapsed

# Compare implementations
input = np.random.randn(4, 3, 32, 32)  # Small batch
weight = np.random.randn(16, 3, 3, 3)

# Level 1: Naive
# benchmark_conv(conv2d_naive, input, weight, "Naive")

# Level 2: Vectorized
# benchmark_conv(conv2d_vectorized, input, weight, "Vectorized")

# Level 3: Strided
# benchmark_conv(conv2d_strided, input, weight, "Strided")
```

Expected speedups:
- Vectorized vs Naive: 10-50x
- Strided vs Vectorized: 2-5x
- PyTorch (cuDNN) vs Strided: 10-100x (GPU)

## Memory Usage Tips

1. **Avoid materializing full window array**: Use strided views
2. **Process in chunks if memory limited**: Batch size reduction
3. **Use float32 instead of float64**: Half the memory

```python
# Memory-efficient: Process batch in chunks
def conv2d_memory_efficient(input, weight, chunk_size=8):
    N = input.shape[0]
    outputs = []
    
    for i in range(0, N, chunk_size):
        chunk = input[i:i+chunk_size]
        output_chunk = conv2d_strided(chunk, weight)
        outputs.append(output_chunk)
    
    return np.concatenate(outputs, axis=0)
```

## Next Steps

1. **Verify correctness**: Compare with PyTorch reference
2. **Profile**: Find bottlenecks with `%timeit` or `cProfile`
3. **Optimize**: Apply techniques from this hint
4. **Move to Topic 12**: Learn im2col for even more speed

## Quick Reference

```python
# Formula cheat sheet
H_out = (H_in + 2*P - K) // S + 1
params = (C_in * K * K + 1) * C_out
flops ≈ 2 * N * C_out * H_out * W_out * C_in * K * K

# Padding for "same" conv
pad = (K - 1) // 2

# Index calculation
h_in = h_out * S + kh
w_in = w_out * S + kw
```
