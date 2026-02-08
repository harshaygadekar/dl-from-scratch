# Topic 11: Interview Questions

## Question 1: Output Size Calculation

**Question**: 
Given a 224×224 input image, what are the output dimensions after:
1. Conv2D with 7×7 kernel, padding=3, stride=2
2. Conv2D with 3×3 kernel, padding=1, stride=1
3. Conv2D with 3×3 kernel, padding=0, stride=2

**Answer**:

Using formula: $H_{out} = \lfloor (H_{in} - K + 2P) / S \rfloor + 1$

1. $(224 - 7 + 6) / 2 + 1 = 223 / 2 + 1 = 111 + 1 = 112$ → **112×112**
2. $(224 - 3 + 2) / 1 + 1 = 224$ → **224×224** ("same" conv)
3. $(224 - 3 + 0) / 2 + 1 = 221 / 2 + 1 = 110 + 1 = 111$ → **111×111**

**Key Insight**: Integer division floors automatically. Even with "same" padding, stride > 1 reduces size.

---

## Question 2: Parameter Count

**Question**: 
A Conv2D layer takes 128-channel input and produces 256-channel output with 3×3 kernels. How many parameters (including bias)?

**Answer**:

$$\text{Parameters} = (C_{in} \times K \times K + 1) \times C_{out}$$

$$= (128 \times 3 \times 3 + 1) \times 256$$
$$= (1152 + 1) \times 256$$
$$= 1153 \times 256$$
$$= 295,168$$

Compare to fully connected: $128 \times H \times W \times 256$ would be millions!

---

## Question 3: Receptive Field

**Question**: 
In a CNN with three 3×3 Conv2D layers (stride=1, padding=1), what is the receptive field of an output pixel?

**Answer**:

Each 3×3 conv sees a 3×3 region. Stacking them:

- Layer 1: 3×3
- Layer 2: 3×3 on layer 1 output = 5×5 on original image
- Layer 3: 3×3 on layer 2 output = 7×7 on original image

Formula: $RF = 1 + \sum_{l=1}^{L} (K_l - 1) \times \prod_{i=1}^{l-1} S_i$

With all K=3, S=1: $RF = 1 + 2 + 2 + 2 = 7$

**Key Insight**: Three 3×3 convs have the same receptive field as one 7×7 conv, but with fewer parameters and more non-linearities!

---

## Question 4: Stride vs Kernel Size

**Question**: 
What's the difference between using stride=2 vs using stride=1 followed by 2×2 max pooling for downsampling?

**Answer**:

**Stride=2 Conv**:
- Pros: Learned downsampling, computationally efficient (skip pixels)
- Cons: Loss of information, aliasing possible
- Used in: ResNet, EfficientNet

**Conv + MaxPool**:
- Pros: Preserves information through conv, pooling is shift-invariant
- Cons: Extra computation, two separate operations
- Used in: VGGNet, AlexNet

**Modern preference**: Strided conv is more common now (end-to-end learnable), but some architectures use both (e.g., stride=2 conv followed by maxpool in early layers).

---

## Question 5: 1×1 Convolution

**Question**: 
What does a 1×1 convolution do? Why is it useful?

**Answer**:

A 1×1 conv with $C_{in}$ input channels and $C_{out}$ output channels:

1. **Channel mixing**: Each output pixel is a linear combination of all input channels at that spatial location
2. **Dimensionality reduction**: If $C_{out} < C_{in}$, reduces channels (used in bottleneck layers)
3. **Dimensionality increase**: If $C_{out} > C_{in}$, expands channels

**Math**: Output[c_out, h, w] = Σ_{c_in} Input[c_in, h, w] × Weight[c_out, c_in]

**Use cases**:
- **Inception modules**: Reduce channels before expensive 3×3, 5×5 convs
- **ResNet bottlenecks**: 1×1 (reduce) → 3×3 → 1×1 (restore)
- **Pointwise conv**: Part of depthwise separable convolutions

**Parameters**: $(C_{in} + 1) \times C_{out}$ — very efficient!

---

## Question 6: Backward Pass Complexity

**Question**: 
For a Conv2D layer in the backward pass, how do we compute gradients w.r.t. input and weights?

**Answer**:

**Gradient w.r.t. input** ($\frac{\partial L}{\partial X}$):
- Convolve the upstream gradient (output grad) with the **flipped** kernel
- Full convolution (not valid) to match input size
- Sum over output channels

**Gradient w.r.t. weights** ($\frac{\partial L}{\partial W}$):
- For each kernel position, compute correlation between input and upstream gradient
- Essentially: conv2d(input, output_grad) with appropriate stride/padding

**Gradient w.r.t. bias**:
- Sum upstream gradient over batch, height, width dimensions
- $\frac{\partial L}{\partial b} = \sum_{n,h,w} \frac{\partial L}{\partial Y[n, :, h, w]}$

**Implementation**: Often done via im2col to reuse efficient matrix multiplication code.

---

## Question 7: Padding Modes

**Question**: 
Explain "same" vs "valid" padding. When would you use each?

**Answer**:

**"Valid" (padding=0)**:
- Kernel only placed where fully inside input
- Output smaller than input
- No edge effects from padding
- Used when: You want to avoid border artifacts, exact spatial control needed

**"Same" (padding chosen to maintain size)**:
- Pad so output spatial dims = input spatial dims (when stride=1)
- For kernel K: pad = (K-1)//2 on each side
- Preserves spatial resolution through network
- Used when: Building deep networks, need feature map alignment

**Formula for "same" padding**:
- K=3 → pad=1
- K=5 → pad=2
- K=7 → pad=3
- General: pad = floor((K-1)/2)

---

## Question 8: Grouped Convolution

**Question**: 
What is grouped convolution? How does it reduce computation?

**Answer**:

**Grouped Conv**: Split input channels into G groups, apply separate conv to each.

**Example**: 
- Input: 256 channels, split into 4 groups of 64
- Output: 256 channels, 4 groups of 64
- Each group conv: 64 input → 64 output

**Parameters comparison**:
- Standard: $(256 \times 3 \times 3 + 1) \times 256 = 590,080$
- Grouped (G=4): $4 \times (64 \times 3 \times 3 + 1) \times 64 = 147,712$ (4× reduction!)

**Extreme case (G=C_in)**: Depthwise convolution — each input channel has its own filter (used in MobileNet).

**Trade-off**: 
- Pros: Fewer params, less computation, less memory
- Cons: Reduced channel mixing (each output channel only sees subset of inputs)

**Used in**: ResNeXt, Xception, MobileNet, ShuffleNet

---

## Question 9: Dilated Convolution

**Question**: 
What is dilated (atrous) convolution? What's its benefit?

**Answer**:

**Dilated Conv**: Insert zeros between kernel elements to expand receptive field without increasing parameters.

**Example**:
- Standard 3×3 kernel: covers 3×3 region
- Dilated 3×3 with rate=2: covers 5×5 region!

```
Standard 3×3:      Dilated 3×3 (rate=2):
┌───┬───┬───┐     ┌───┬───┬───┬───┬───┐
│ ✓ │ ✓ │ ✓ │     │ ✓ │   │ ✓ │   │ ✓ │
├───┼───┼───┤     ├───┼───┼───┼───┼───┤
│ ✓ │ ✓ │ ✓ │     │   │   │   │   │   │
├───┼───┼───┤     ├───┼───┼───┼───┼───┤
│ ✓ │ ✓ │ ✓ │     │ ✓ │   │ ✓ │   │ ✓ │
└───┴───┴───┘     ├───┼───┼───┼───┼───┤
                  │   │   │   │   │   │
                  ├───┼───┼───┼───┼───┤
                  │ ✓ │   │ ✓ │   │ ✓ │
                  └───┴───┴───┴───┴───┘
```

**Benefit**: 
- Larger receptive field without more parameters
- Maintains resolution (no downsampling)
- Used in: Semantic segmentation (DeepLab), audio generation (WaveNet)

---

## Question 10: Computational Bottlenecks

**Question**: 
Your Conv2D layer is the bottleneck in training. How do you speed it up?

**Answer**:

**Algorithmic improvements**:
1. **im2col + GEMM**: Convert to matrix multiplication (cuDNN does this)
2. **Winograd convolution**: For small kernels (3×3), reduces multiplies
3. **FFT convolution**: For large kernels, use Fast Fourier Transform

**Architecture improvements**:
1. **Depthwise separable**: Factor into depthwise + pointwise (MobileNet)
2. **Reduce channels**: Use 1×1 conv bottleneck before 3×3
3. **Grouped conv**: Split channels into groups

**System improvements**:
1. **cuDNN**: Optimized CUDA kernels
2. **Mixed precision**: FP16 training
3. **Kernel fusion**: Fuse conv + batchnorm + relu into one kernel

**Rule of thumb**: 
- 3×3 conv: Use im2col or Winograd
- 1×1 conv: Already efficient (direct)
- 5×5, 7×7: Consider depthwise separable or FFT
