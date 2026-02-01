# dl-from-scratch: Core Concept Document
**For AI Agent Implementation**

## 1. The Core Idea
A 34-day structured curriculum where learners implement Deep Learning from absolute scratch using **only NumPy**. No PyTorch, no TensorFlow, no Keras. The goal is to build neural network components (from basic perceptrons to full Transformers) manually to understand the underlying logic, math, and engineering constraints.

**Target Audience**: ML Engineers preparing for FAANG interviews, researchers who want to debug internals, and developers seeking deep understanding over API usage.

**Tagline**: "If you can't implement it in NumPy, you don't understand it."

---

## 2. The "From Scratch" Rules (Hard Constraints)

### Forbidden (Black Boxes)
- PyTorch (`torch.nn`, `torch.optim`, `torch.autograd`)
- TensorFlow/Keras (`tf.keras`, `tf.GradientTape`)
- JAX, Autograd libraries, Scikit-Learn
- Pre-trained models or `torchvision.models`

### Allowed (Building Blocks Only)
- **NumPy**: `np.dot`, array ops, broadcasting (no `np.linalg.lstsq`)
- **Python stdlib**: `math`, `random`, `collections`
- **Matplotlib**: Visualization only
- **Pandas/PIL**: Data loading only (CSV reading, image loading)

### Implementation Levels (Per topic)
Every day has 4 solution tiers in `solutions/`:
1. **Level01-Naive.py**: Correct but slow (loops, readable)
2. **Level02-Vectorized.py**: NumPy-optimized (broadcasting)
3. **Level03-Memory-Efficient.py**: In-place ops, float16, cache-aware
4. **Level04-PyTorch-Reference.py**: Verification only (ground truth)

---

## 3. 34-topic Roadmap Structure

### Module 0: Foundations (Survival Module)
- **topic 01**: Tensor Operations & Broadcasting (memory layouts, stride tricks)
- **topic 02**: Autograd Engine from Scratch (computational graphs, reverse-mode AD)
  - *Escape Hatch*: Provide `utils/autograd_stub.py` for those who get stuck
- **topic 03**: Optimizers (SGD, Momentum, Adam update equations)

### Module 1: Neural Network Core (Make-or-Break)
- **topic 04**: Single Layer Perceptron
- **topic 05**: MLP Forward Pass (Xavier/Kaiming init)
- **topic 06**: Backpropagation Implementation (manual chain rule)
- **topic 07**: Activations & Initialization (ReLU variants, dead neurons)
- **topic 08**: Regularization (L2, Dropout - inverted implementation)
- **topic 09**: Batch Normalization (running stats, train vs eval mode)
- **topic 10**: End-to-End MNIST (95% accuracy, &lt;10 min CPU training)

### Module 2: CNNs (Spatial Reasoning)
- **topic 11**: Conv2D Sliding Window (naive loops)
- **topic 12**: Im2Col Vectorization (convert conv to matmul)
- **topic 13**: Pooling & Strides (max pool backward with masks)
- **topic 14**: ResNet Skip Connections (vanishing gradient fix)
- **topic 15**: Depthwise Separable Convs (MobileNet style)
- **topic 16**: Advanced Normalizations (LayerNorm, GroupNorm)
- **topic 17**: CIFAR-10 from Scratch (ResNet-18, small dataset)

### Module 3: Sequence Models (Temporal)
- **topic 18**: Vanilla RNN (BPTT - backprop through time)
- **topic 19**: LSTM Gates (forget/input/output equations)
- **topic 20**: GRU Variant
- **topic 21**: Bidirectional RNNs & Masking (variable lengths, padding)
- **topic 22**: Word2Vec Logic (negative sampling, skip-gram)
- **topic 23**: Seq2Seq Encoder-Decoder
- **topic 24**: Attention Mechanism (Bahdanau - score functions, alignment)

### Module 4: Transformers & Production (The Payoff)
- **topic 25**: Self-Attention O(n²) (causal masking)
- **topic 26**: Multi-Head Attention (split heads, concat)
- **topic 27**: Positional Encodings (Sinusoidal + RoPE)
- **topic 28**: Transformer Encoder Block (Pre-norm architecture)
- **topic 29**: Transformer Decoder Block (cross-attention)
- **topic 30**: Mini-GPT Training (TinyShakespeare, generation)
- **topic 31**: KV-Cache Optimization (for fast inference)
- **topic 32**: FlashAttention Algorithm (tiling strategy, pseudo-code OK)
- **topic 33**: Quantization (INT8, fake quantization)
- **topic 34**: Distributed Training Logic (data parallelism concept, parameter averaging)

---

## 4. Daily File Structure Template
Each `DayXX-Topic/` folder contains:
- `README.md`: Problem statement, constraints, success criteria
- `questions.md`: Specific tasks (e.g., "Implement forward pass with shape (B, T, C)")
- `intuition.md`: Why this matters, math reminders, analogies, common pitfalls
- `math-refresh.md`: Derivations needed for this day (chain rule, matrix calculus)
- `hints/`: 
  - `hint-1-shapes.md` (basic direction)
  - `hint-2-algorithm.md` (pseudocode)
  - `hint-3-optimization.md` (memory/speed tips)
- `solutions/`: Level 1-4 as defined above
- `tests/`:
  - `test_basic.py` (shape checks, value assertions)
  - `test_edge.py` (batch_size=1, empty, NaN, Inf)
  - `test_stress.py` (large arrays, performance thresholds)
- `visualization.py`: Plotting helpers (loss curves, attention heatmaps)

---

## 5. Critical Pain Points & Safeguards

### The topic 2 Cliff (Autograd)
- **Problem**: If users fail topic 2 (autograd), Days 3-34 are blocked
- **Solution**: Provide `utils/autograd_stub.py` (working but unoptimized) as escape hatch. Users can proceed and return to implement their own later.

### Silent Failures
- **Problem**: Code runs but is subtly wrong (e.g., BatchNorm running stats)
- **Solution**: Every day includes `test_edge.py` with golden master outputs (pre-computed with fixed seeds). Gradient checks (numerical vs analytic) mandatory for learnable parameters.

### Forever Training (Slow NumPy)
- **Problem**: CPU training too slow kills motivation
- **Solution**: 
  - Days 1-10: Use MNIST subset (1000 samples) for logic verification, then full
  - Max sequence length 64 for Transformers (not 512)
  - Hard constraint: &lt;10 minutes per training run on 2019 MacBook Pro

### The Math Gap
- **Problem**: Users don't know chain rule or broadcasting
- **Solution**: `PREREQUISITES.md` with 5 coding challenges (matrix mult by hand, broadcasting quiz). Must pass before topic 1. Inline math refreshers every day.

### Karpathy Overlap
- **Problem**: "Why not just watch nn-zero-to-hero?"
- **Solution**: 
  - Karpathy = Video lectures, one long project, theory-heavy
  - This = Interview-style daily problems, modular (can jump to topic 25), production edge cases (KV-cache, quantization), text-based not video

---

## 6. Testing Requirements

**Test Runner**: `utils/test_runner.py --day XX` must output:
- Shape tests: Pass/Fail per tensor
- Value tests: Match golden master within tolerance (1e-5)
- Gradient tests: Analytic vs numerical error &lt; 1e-5
- Edge cases: Empty batches, single samples, extreme values
- Performance: Must complete under time threshold (prevent O(n³) solutions)

**Golden Master**: Level04 (PyTorch) generates reference outputs. Level01-03 must match within tolerance.

---

## 7. Content Differentiation Rules

**What Makes This Unique**:
1. **No PyTorch until topic 4 verification**: Build your own autograd on topic 2, use it for Weeks 1-3
2. **Production Realities**: Days 31-34 cover inference optimization (KV-cache), quantization, distributed logic - not just training
3. **Interview-Style Constraints**: Explicit time/space complexity targets per day (e.g., "Optimize from O(n²) to O(n log n)")
4. **Progressive Disclosure**: Hints are tiered, solutions are leveled (naive → optimal)

**What NOT to Include**:
- Classical ML (SVM, Random Forests) - this is Deep Learning only
- High-level tutorials (no "how to install Python")
- Jupyter notebooks (use `.py` files for real engineering practice)
- Video content (text + code only)

---

## 8. Success Criteria (Per topic)

User must be able to:
1. Run `python utils/test_runner.py --day XX` and see all green checks
2. Understand WHY the math works (via intuition.md), not just copy code
3. Handle edge cases (batch_size=1, NaN gradients) without crashing
4. (Days 10, 17, 30): Achieve target metric (e.g., 95% MNIST accuracy, coherent text generation)

---

## 9. Repository Metadata

- **Name**: `dl-from-scratch`
- **Dependencies**: `numpy`, `matplotlib`, `requests` (for data download only)
- **Python**: 3.8+
- **Hardware**: CPU sufficient for all days. GPU optional for Days 28-34 only.
- **License**: MIT

---

## 10. Implementation Priority for AI Agent

**Phase 1 (Foundation)**:
- Create `Day01/` through `Day03/` with full structure (README, intuition, tests, all 4 solution levels)
- Implement `utils/test_runner.py` and `utils/autograd_stub.py`
- Create `PREREQUISITES.md` with 5 challenge questions

**Phase 2 (Core)**:
- Days 4-10 (get to working MNIST)
- Days 11-17 (CNNs with CIFAR-10)

**Phase 3 (Advanced)**:
- Days 18-30 (RNNs to GPT)
- Days 31-34 (Production/Advanced - marked clearly as optional)

**Quality Gates**:
- Every day must have working Level04 (PyTorch reference) that generates golden outputs
- Every day must include at least 3 hint files
- Every day must pass the "10-minute training" rule on CPU (or explicit warning if exempt)