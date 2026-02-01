# DL-From-Scratch: Complete Implementation Blueprint

**Repository Name**: `dl-from-scratch`  
**Tagline**: *"Implementing Deep Learning from the ground up. No PyTorch. No TensorFlow. Just NumPy and understanding."*  
**Target Audience**: ML Engineers preparing for FAANG interviews, CS students seeking deep understanding, researchers debugging model internals.  
**Time Commitment**: 34 days, 2-3 hours/day.  
**Prerequisites**: Python proficiency, Linear Algebra 101, Calculus I (derivatives).

---

## 1. Repository Structure example as follows create this for all modules and topics as refering to module 01 and topic 01

dl-from-scratch/
├── README.md                    # Main entry + motivation
├── PREREQUISITES.md             # Self-assessment checklist
├── SETUP.md                     # Environment + pre-flight check
├── CONTRIBUTING.md              # How to add hints/fixes
├── ROADMAP.md                   # Visual progress tracker
│
├── Module 00-Foundations/
│   ├── Topic 01-Tensor-Operations/
│   │   ├── README.md            # Problem + Intuition
│   │   ├── questions.md         # Specific tasks
│   │   ├── hints/
│   │   │   ├── hint-1-shapes.md
│   │   │   ├── hint-2-broadcasting.md
│   │   │   └── hint-3-memory.md
│   │   ├── intuition.md         # Why tensors matter
│   │   ├── math-refresh.md      # Broadcasting math
│   │   ├── solutions/           # Progressive disclosure
│   │   │   ├── level01-naive.py
│   │   │   ├── level02-vectorized.py
│   │   │   ├── level03-memory-efficient.py
│   │   │   └── level04-pytorch-reference.py
│   │   ├── tests/
│   │   │   ├── test_basic.py    # Shape checks
│   │   │   ├── test_edge.py     # Empty arrays, dims=1
│   │   │   └── test_stress.py   # Large arrays
│   │   ├── visualization.py     # Plot tensor layouts
│   │   └── assets/
│   │       └── numpy_stride_visualization.html  # Interactive memory layout viewer
│   └── ...
│
├── Module 01-Neural-Network-Core/
├── Module 02-CNNs/
├── Module 03-RNNs-Sequences/
├── Module 04-Transformers-Production/
│
├── utils/
│   ├── autograd_stub.py         # Escape hatch for Topic 02
│   ├── finite_difference_checker.py  # Numerical gradient verification
│   ├── gradient_check.py        # Analytic vs numerical gradient comparison
│   ├── test_runner.py           # Daily validation script
│   ├── timer.py                 # Performance benchmarking
│   └── progress.py              # Progress tracking CLI
│
└── .github/
├── ISSUE_TEMPLATE/
│   ├── stuck-on-day.md      # "I'm stuck" template
│   └── bug-report.md
└── workflows/
└── test-solutions.yml   # CI to verify solutions work


---

## 2. Detailed Curriculum (34-topic Breakdown)

### Module 0: Foundations (The Survival Module)
**Critical Note**: Module 0 determines retention. Topic 02 has an "Escape Hatch" (see Pain Points).

**topic 01: Tensor Operations & Broadcasting**
- **Core Task**: Implement matrix multiplication, broadcasting, reshaping without `np.matmul` (use `np.dot` or loops, understand stride tricks)
- **Why**: Memory layout is the #1 source of DL bugs
- **Test**: 3D tensor batch multiplication (2,3,4) @ (2,4,5) → (2,3,5)
- **Intuition**: Tensors are just fancy arrays with rules
- **Anti-Failure**: Includes `assets/numpy_stride_visualization.html` to see memory layout interactively

**topic 02: Autograd Engine From Scratch**
- **Core Task**: Build a micro-autograd library that tracks operations and computes gradients via reverse-mode automatic differentiation
- **Minimum Viable**: Scalar ops first (add, mul, pow), then vectorize
- **Escape Hatch**: `utils/autograd_stub.py` provided (functional but not optimal). Users can use it and return to implement later.
- **Validation**: Gradient check against finite differences via `utils/finite_difference_checker.py`
- **Intuition**: "Gradients flow backward through the computation graph like water through pipes"

**topic 03: Optimization Algorithms**
- **Core Task**: Implement SGD, SGD with Momentum, RMSprop, Adam from update equations
- **Edge Cases**: Learning rate scheduling, gradient clipping
- **Visual**: Plot optimization paths on 2D contour (beale function)

### Module 1: Neural Network Core (The Make-or-Break Module)
**Goal**: By topic 10, user trains MNIST to 95%+ without touching PyTorch.

**topic 04: Single Layer Perceptron**
- Binary classification on synthetic data
- Manual backprop derivation (no autograd yet to cement understanding)

**topic 05: MLP Forward Pass**
- Layer initialization (Xavier/Kaiming from scratch)
- Forward pass with ReLU, Sigmoid, Tanh

**topic 06: Backpropagation Implementation**
- Use topic 2 autograd OR manual backprop (user choice)
- Compute gradients for 2-layer network
- **Pain Point Mitigation**: Includes "Gradient Check" script - compares analytic vs numerical gradients

**topic 07: Activation Functions & Initialization**
- Implement ReLU, LeakyReLU, ELU, Swish
- Dead neurons detection
- Initialization strategies comparison

**topic 08: Regularization**
- L2 regularization (weight decay math)
- Dropout (inverted dropout implementation + testing inference mode)
- Early stopping logic

**topic 09: Batch Normalization (The Beast)**
- **Why Hard**: Running mean/variance tracking, train vs eval mode, batch size 1 edge case
- **Implementation**: Forward pass with momentum, backward pass (chain rule nightmare)
- **Tests**: Batch size 1, large batch, eval mode consistency

**topic 10: End-to-End MNIST**
- Data loading (numpy memmap, no PyTorch DataLoader)
- Training loop with validation
- Target: 95% accuracy in <10 minutes on CPU
- **Achievement Unlocked**: Can implement a classifier from zero

### Module 2: Convolutional Networks (Spatial Reasoning)
**Constraint**: All convs implemented with sliding windows or im2col (no FFT for simplicity).

**topic 11: Conv2D Sliding Window**
- Naive nested loop implementation (O(n²k²))
- Padding (same/valid) logic
- **Visualization**: Filter sliding over image animation

**topic 12: Im2Col Vectorization**
- Convert conv to matrix multiplication for speed
- Memory overhead discussion (im2col vs direct)

**topic 13: Pooling & Strides**
- Max pooling (forward + backward with mask tracking)
- Average pooling
- Strided convolutions

**topic 14: ResNet Skip Connections**
- Why: Vanishing gradient problem explanation
- Implement residual block with batch norm
- Identity vs projection shortcuts

**topic 15: Modern Convolutions**
- Depthwise Separable Conv (MobileNet style)
- 1x1 convolutions (bottlenecks)
- Dilated convolutions (atrous)

**topic 16: Advanced Normalizations**
- LayerNorm (compute stats over features, not batch)
- GroupNorm
- InstanceNorm (compare to BatchNorm)

**topic 17: CIFAR-10 From Scratch**
- Data augmentation (random flip, crop - implemented manually)
- Train ResNet-18 (small variant)
- Learning rate scheduling (cosine annealing)

### Module 3: Sequence Models (Temporal Reasoning)
**Challenge**: Backpropagation through time (BPTT) implementation.

**topic 18: Vanilla RNN**
- Forward pass through time
- BPTT manual implementation or autograd
- Gradient clipping (exploding gradients demo)

**topic 19: LSTM Gates**
- Forget gate, input gate, output gate equations
- Cell state vs hidden state
- Backprop through gates (painful but educational)

**topic 20: GRU Variant**
- Update gate, reset gate
- Comparison to LSTM (fewer parameters)

**topic 21: Bidirectional RNNs & Masking**
- Stacking forward + backward
- Variable length sequences (padding + masking logic)
- Packed sequences concept (pseudo-implementation)

**topic 22: Word Embeddings (Word2Vec Logic)**
- Skip-gram negative sampling implementation
- Subsampling frequent words
- Not full Word2Vec training (too slow), but the loss function and sampling

**topic 23: Seq2Seq Encoder-Decoder**
- Encoder: final hidden state as context
- Decoder: auto-regressive generation
- Teacher forcing implementation

**topic 24: Attention Mechanism (Bahdanau)**
- Score functions (dot, general, concat)
- Alignment weights (softmax)
- Context vector computation
- **Bridge to Module 4**: "This was the gateway drug to Transformers"

### Module 4: Transformers & Production (The Payoff)
**Focus**: Modern architectures and deployment realities.

**topic 25: Efficient Self-Attention**
- Naive O(n²) attention (pure numpy)
- Memory complexity analysis
- Causal (autoregressive) masking

**topic 26: Multi-Head Attention**
- Splitting heads (batch matrix tricks)
- Concatenation and projection
- Masking per head

**topic 27: Positional Encodings**
- Sinusoidal encodings (original Transformer)
- Rotary Position Embeddings (RoPE) - modern standard
- Relative positions

**topic 28: Transformer Encoder Block**
- Pre-norm vs Post-norm architecture
- FFN (two linear layers with GELU)
- Residual connections everywhere

**topic 29: Transformer Decoder Block**
- Cross-attention (encoder-decoder)
- Causal self-attention
- Output projection to vocab

**topic 30: Mini-GPT Training**
- Train on TinyShakespeare or similar
- Generation logic (sampling, temperature, top-k)
- **Milestone**: Generates coherent text

**topic 31: KV-Cache Optimization**
- Problem: Recomputing attention for all previous tokens is O(n²) per step
- Solution: Cache Key, Value tensors
- Implementation: State management for generation
- Speed comparison: With vs Without cache

**topic 32: Modern Optimizations (Theoretical/Pseudo)**
- FlashAttention algorithm (tiling strategy - paper walkthrough, not CUDA)
- Gradient checkpointing (trade compute for memory)
- Mixed precision training (FP16/FP32 casting, loss scaling)

**topic 33: Quantization & Efficiency**
- Post-training quantization (INT8)
- Quantization-aware training (fake quantization)
- Knowledge distillation logic (teacher-student)

**topic 34: Distributed Training Logic**
- Data parallelism concept (parameter averaging)
- Gradient accumulation (simulating large batches)
- **Note**: This is algorithmic understanding, not multi-GPU coding (hardware barrier)

---

## 3. Pain Point Mitigations (Detailed)

### Pain Point 1: The topic 2 Cliff (Autograd Failure)
**Symptom**: User can't build autograd, gets frustrated, abandons repo.
**Solution**:
- **Progressive Autograd**: topic 2 has 3 levels:
  - Level 1: Scalar autograd (easy, a + b * c)
  - Level 2: Vector autograd (medium)
  - Level 3: Tensor autograd (hard, full numpy)
- **Escape Hatch**: `utils/autograd_stub.py` provided (works but not optimized). Users can use stub and return to Level 3 later.
- **Validation**: Topic 02 includes `utils/finite_difference_checker.py` - if their autograd passes this, they're good to proceed.

### Pain Point 2: Silent Failures (Wrong But Running)
**Symptom**: Code runs, loss decreases, but implementation is subtly wrong (e.g., BatchNorm running stats incorrect).
**Solution**:
- **Golden Master Tests**: Pre-computed outputs for specific random seeds
- **Shape Assertions**: Every intermediate tensor shape checked
- **Gradient Checks**: For every learnable parameter, compare analytic vs numerical gradient using `utils/gradient_check.py`
- **Invariant Tests**: 
  - BatchNorm: Output should have mean ≈ 0, var ≈ 1 (train mode)
  - Softmax: Should sum to 1.0
  - Attention: Weights should sum to 1.0, all positive

### Pain Point 3: Forever Training (Slow NumPy)
**Symptom**: User waits 2 hours for MNIST epoch, thinks they suck, quits.
**Solution**:
- **Dataset Constraints**:
  - Days 1-10: Subset MNIST (1000 samples) to verify logic, then full
  - Days 11-17: CIFAR-10 downsampled to 32x32 (no resizing)
  - Days 25-30: Sequence length max 64 (not 512) for transformers
- **Timebox Guarantee**: Each day's training completes in <10 minutes on 2019 MacBook Pro (stated in README)
- **Hardware Tiers**: 
  - Tier 1 (Required): CPU, 8GB RAM
  - Tier 2 (Optional): GPU for Days 28-34 only (but CPU still possible with tiny data)

### Pain Point 4: The Math Gap
**Symptom**: User doesn't know chain rule or broadcasting rules, copies code blindly.
**Solution**:
- **Prereq Check**: `PREREQUISITES.md` includes 5 coding challenges:
  1. Matrix multiplication by hand (2x2)
  2. Derivative of sigmoid from scratch
  3. Broadcasting quiz (what shape is (3,1) + (1,3)?)
  4. Chain rule application: dL/dx given dL/dy and dy/dx
  5. Python list vs numpy array speed comparison
- **Inline Math**: Every `intuition.md` has "Math Reminder" section (not links to textbooks)
- **Notation Guide**: Standardized notation (W for weights, X for input, dW for gradients) in glossary

### Pain Point 5: Karpathy Overlap
**Symptom**: "Why not just watch Karpathy's videos?"
**Solution**:
- **Differentiation** (stated clearly in README):
  - Karpathy = Video lectures, theory-heavy, one long project
  - dl-from-scratch = Text/problem-based, interview-style, modular (can jump to topic 25), focus on production edge cases (KV-cache, quantization)
- **Complementary**: "If you get stuck on topic 6, watch Karpathy's micrograd video, then return"

### Pain Point 6: No Verification (Imposter Syndrome)
**Symptom**: User codes solution, has no idea if it's correct, loses motivation.
**Solution**:
- **Daily Test Runner**: `python utils/test_runner.py --day 05` runs all tests for that day
- **Progressive Checkmarks**: Visual progress bar in terminal (10/10 tests passed)
- **Community Validation**: Discord/Reddit channel for "Is my topic 9 BatchNorm correct?" (peer review)

### Pain Point 7: Scope Creep (CUDA/Distributed Confusion)
**Symptom**: topic 32 FlashAttention requires CUDA knowledge user doesn't have.
**Solution**:
- **Clear Labeling**: 
  - Days 1-30: "Core Track" (Pure Python/NumPy)
  - Days 31-34: "Advanced Track" (Algorithms only, pseudo-code for CUDA acceptable)
- **Hardware Reality Checks**: Explicit "You cannot actually test multi-GPU code without multiple GPUs" warnings
- **Focus on Logic**: topic 34 tests understanding of parameter averaging, not actual NCCL usage

---

## 4. File Templates (Standardized Format)

### `README.md` (Per topic)
```markdown
# topic XX: [Topic]

## Learning Objectives
- Understand [concept]
- Implement [specific algorithm]
- Handle [edge case]

## Problem Statement
[Detailed description]

## Constraints
- Time Limit: O(n²) baseline → optimize to O(n log n) or better
- Memory Limit: O(1) extra space (in-place preferred)
- Hardware: CPU only (< 10 min training time)

## Prerequisites (Check Before Starting)
- [ ] Completed topic XX-1
- [ ] Understand [specific concept from previous day]
- [ ] Passed prerequisite quiz: [link to PREREQUISITES.md section]

## Files to Create
- `solution.py`: Your implementation
- `train.py`: Training loop (if applicable)

## Success Criteria
Run `python utils/test_runner.py --day XX`
- All basic tests: PASS
- All edge tests: PASS
- Training converges (loss < X within Y steps)

## Escape Hatches
- Stuck on math? See `math-refresh.md`
- Stuck on implementation? See `hints/hint-1.md`
- Completely lost? Use `utils/stubs/dayXX_stub.py` (but come back!)

## Next topic Preview
Tomorrow you'll build [teaser for next topic]


# what to include in intuition.md file (Per topic)

# Why [Topic] Matters

## The Problem
[Before this invention, what was hard?]

## The Insight
[The key idea in 2 sentences]

## Analogy
[Real-world analogy, e.g., "BatchNorm is like a thermostat..."]

## Math Reminder
[Specific equations needed, derived step by step]

## Common Pitfalls
- Pitfall 1: [Explanation]
- Pitfall 2: [Explanation]

## In Production
[Where this is used at Google/Meta/OpenAI]

# examples of different files per topics/day:

## how solutions should look like: solutions/level01-naive.py

"""
Level 1: Naive Implementation
- Priority: Correctness over speed
- Constraints: Use only basic numpy operations
- Target: Pass functional tests only (may be slow)
"""

import numpy as np

def naive_implementation(x):
    # Step 1: [comment]
    result = ...
    
    # Step 2: [comment]
    ...
    
    return result

if __name__ == "__main__":
    # Self-test with toy data
    x = np.array([...])
    out = naive_implementation(x)
    print(f"Output: {out}, Shape: {out.shape}")
    assert out.shape == (expected_shape), "Shape mismatch!"

## solutions/level04-pytorch-reference.py

"""
PyTorch Reference Implementation
- Used only for verification, not the solution
- Run this to generate 'golden' outputs for testing
"""

import torch
import torch.nn as nn

class ReferenceImplementation(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(...)
    
    def forward(self, x):
        return self.layer(x)

if __name__ == "__main__":
    # Save outputs for test suite
    torch.manual_seed(42)
    model = ReferenceImplementation()
    x = torch.randn(32, 784)
    out = model(x)
    np.save("golden_output.npy", out.detach().numpy())

## 5. Testing Strategy
Test Categories
Shape Tests: Verify output dimensions match expected
Value Tests: Specific input → specific output (hardcoded)
Gradient Tests: Numerical gradient vs analytic gradient (relative error < 1e-5)
Edge Tests: Empty batches, batch_size=1, NaN inputs, Inf inputs
Determinism Tests: Same seed → same results (for reproducibility)
Performance Tests: Runtime < threshold (prevent O(n³) solutions where O(n²) expected)


## How Test Runner files should look like (utils/test_runner.py)

# Usage: python test_runner.py --day 05 --level 02
# Outputs:
# ✅ Shape tests: 5/5 passed
# ✅ Value tests: 10/10 passed
# ⚠️  Performance: 2.3s (threshold 2.0s) - consider optimization
# ❌ Edge tests: 2/4 passed - failed on batch_size=1

## Continuous Integration
GitHub Actions runs all tests on Python 3.8, 3.9, 3.10, 3.11
Tests verify that level04-pytorch-reference.py matches level03-memory-efficient.py within tolerance
Prevents solution drift over PRs

## 6. Community & Maintenance
Issue Templates
"I'm Stuck" Template:

**topic**: [topic number]
**Specific Problem**: [e.g., "BatchNorm backward pass shape mismatch"]
**What I've tried**: [Describe attempts]
**Error message**: [Full traceback]
**My code**: [Minimal reproducible example or link to gist]

*Maintainer will direct to appropriate hint level, not give full solution*

"Clarification" Template:
For ambiguous problem statements.
Contribution Guidelines
Hints Only: Contributors can add hints, never full solutions to existing days
New Days: Propose new days for "Bonus Track" (Days 35+) covering:
Diffusion models
State space models (Mamba)
Neural Architecture Search
Language Ports: Python is official, but JavaScript/C++ ports welcome in ports/ directory
Progress Tracking
utils/progress.py generates badge:
[###########>          ] Module 2 topic 15 (44%)
Users update ROADMAP.md checkbox via PR when completed (gamification)

# 7. Success Metrics should look like this after the blueprint gets executed
For Users:
Completion rate: Target 30% finish topic 34 (high for technical content)
Interview success: Users report solving "implement attention" questions at Google/Meta
Time to topic 10: Median 10 days (1 day per day)

#8. Anti-Waste Checklist
Before launching, verify:
[ ] topic 2 stub exists and works
[ ] topic 10 trains in <10 minutes on CPU
[ ] Every day has 3+ hints
[ ] Test runner passes for all 34 days (reference solutions)
[ ] No external dependencies beyond numpy, matplotlib, requests (for data)
[ ] PREREQUISITES.md quiz actually filters out unprepared users
[ ] "Why not Karpathy?" answered in FAQ
[ ] Escape hatches clearly marked (no shame in using stubs)
[ ] topics 31-34 labeled as "Advanced/Optional"