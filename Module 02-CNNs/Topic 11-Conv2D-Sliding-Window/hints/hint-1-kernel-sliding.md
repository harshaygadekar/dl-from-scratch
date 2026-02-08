# Hint 1: Kernel Sliding Direction

## The Basic Idea

Think of convolution as a **flashlight** scanning an image:

```
Image (7×7):            Flashlight (3×3):
┌───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │     ┌───┬───┬───┐
├───┼───┼───┼───┼───┼───┼───┤     │ w │ w │ w │
│   │ █ │ █ │ █ │   │   │   │  ⊗  ├───┼───┼───┤
├───┼───┼───┼───┼───┼───┼───┤     │ w │ w │ w │
│   │ █ │ █ │ █ │   │   │   │     ├───┼───┼───┤
├───┼───┼───┼───┼───┼───┼───┤     │ w │ w │ w │
│   │ █ │ █ │ █ │   │   │   │     └───┴───┴───┘
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┘

Flashlight illuminates 3×3 region
Multiply element-wise with weights
Sum all 9 values = ONE output pixel
Slide right, repeat...
```

## Step-by-Step Algorithm

For each **output position** (h_out, w_out):

1. **Find input region**: 
   - Start row: `h_out * stride`
   - Start col: `w_out * stride`
   - Region: input[start:start+K, start:start+K]

2. **Compute dot product**:
   - Multiply input region with kernel (element-wise)
   - Sum all values

3. **Store result**: `output[h_out, w_out] = sum`

## Loop Structure

```python
def conv2d_naive(input, weight, bias=None, stride=1, padding=0):
    N, C_in, H, W = input.shape
    C_out, _, K, _ = weight.shape
    
    # 1. Pad input if needed
    if padding > 0:
        input = pad_array(input, padding)
        H, W = H + 2*padding, W + 2*padding
    
    # 2. Calculate output size
    H_out = (H - K) // stride + 1
    W_out = (W - K) // stride + 1
    
    # 3. Initialize output
    output = np.zeros((N, C_out, H_out, W_out))
    
    # 4. Six nested loops (naive version)
    for n in range(N):                    # Batch samples
        for c_out in range(C_out):        # Output channels
            for h_out in range(H_out):    # Output height
                for w_out in range(W_out): # Output width
                    # Compute one output pixel
                    sum_val = 0
                    for c_in in range(C_in):  # Input channels
                        for kh in range(K):   # Kernel height
                            for kw in range(K): # Kernel width
                                # Get input position
                                h_in = h_out * stride + kh
                                w_in = w_out * stride + kw
                                
                                # Accumulate
                                sum_val += input[n, c_in, h_in, w_in] * weight[c_out, c_in, kh, kw]
                    
                    # Add bias and store
                    if bias is not None:
                        sum_val += bias[c_out]
                    output[n, c_out, h_out, w_out] = sum_val
    
    return output
```

## Key Checkpoints

✅ **Loop order matters**: Batch → Output Channel → Spatial → Input Channel → Kernel

✅ **Index calculation**: `h_in = h_out * stride + kh` (not just `h_out + kh`)

✅ **Padding first**: Pad the input, then convolve

✅ **Bias addition**: Once per output pixel (not per accumulation)

## Common Bug: Index Confusion

```python
# WRONG - Forgetting stride
h_in = h_out + kh

# CORRECT  
h_in = h_out * stride + kh
```

## Next Steps

Once this works, move to Level 2:
- Replace inner 3 loops with vectorized operations
- Use `np.sum(input_window * kernel)`

**See hint-2-padding-logic.md for padding implementation details**
