# 🗺️ Learning Roadmap

Track your progress through the DL-From-Scratch curriculum.

---

## Module 0: Foundations (The Survival Module) 🏔️

> **Critical**: This module determines your success. Topic 02 has an escape hatch if needed.

- [ ] **Topic 01**: Tensor Operations & Broadcasting
- [ ] **Topic 02**: Autograd Engine From Scratch ⚠️ *Has escape hatch*
- [ ] **Topic 03**: Optimization Algorithms

---

## Module 1: Neural Network Core (Make-or-Break) 💪

> **Goal**: By Topic 10, train MNIST to 95%+ without touching PyTorch.

- [ ] **Topic 04**: Single Layer Perceptron
- [ ] **Topic 05**: MLP Forward Pass
- [ ] **Topic 06**: Backpropagation Implementation
- [ ] **Topic 07**: Activation Functions & Initialization
- [ ] **Topic 08**: Loss Functions
- [ ] **Topic 09**: Regularization
- [ ] **Topic 10**: End-to-End MNIST 🎯 *Milestone: 95% accuracy*

---

## Module 2: Convolutional Networks (Spatial Reasoning) 🖼️

> **Constraint**: All convolutions implemented with sliding windows or im2col.

- [ ] **Topic 11**: Conv2D Sliding Window
- [ ] **Topic 12**: Im2Col Vectorization
- [ ] **Topic 13**: Pooling & Strides
- [ ] **Topic 14**: ResNet Skip Connections
- [ ] **Topic 15**: Modern Convolutions (Depthwise, Dilated)
- [ ] **Topic 16**: Advanced Normalizations
- [ ] **Topic 17**: CIFAR-10 From Scratch 🎯 *Milestone: Working ResNet*

---

## Module 3: Sequence Models (Temporal Reasoning) 📈

> **Challenge**: Backpropagation through time implementation.

- [ ] **Topic 18**: Vanilla RNN
- [ ] **Topic 19**: LSTM Gates
- [ ] **Topic 20**: GRU Variant
- [ ] **Topic 21**: Bidirectional RNNs & Masking
- [ ] **Topic 22**: Word Embeddings (Word2Vec Logic)
- [ ] **Topic 23**: Seq2Seq Encoder-Decoder
- [ ] **Topic 24**: Attention Mechanism (Bahdanau) 🌉 *Bridge to Transformers*

---

## Module 4: Transformers & Production (The Payoff) 🚀

> **Focus**: Modern architectures and deployment realities.

### Core Track
- [ ] **Topic 25**: Efficient Self-Attention
- [ ] **Topic 26**: Multi-Head Attention
- [ ] **Topic 27**: Positional Encodings
- [ ] **Topic 28**: Transformer Encoder Block
- [ ] **Topic 29**: Transformer Decoder Block
- [ ] **Topic 30**: Mini-GPT Training 🎯 *Milestone: Coherent text generation*

### Advanced Track (Core)
- [ ] **Topic 31**: KV-Cache Optimization
- [ ] **Topic 32**: Modern Optimizations (FlashAttention, Gradient Checkpointing)
- [ ] **Topic 33**: Quantization & Efficiency
- [ ] **Topic 34**: Distributed Training Logic

---

## Module 5: Bonus LLM Systems (Optional) 🧪

> **Optional extension**: complete after Topics 01-34.

- [ ] **Topic 35**: LoRA and QLoRA Fundamentals
- [ ] **Topic 36**: Speculative Decoding Simulation
- [ ] **Topic 37**: Mixture-of-Experts Routing
- [ ] **Topic 38**: SSM and Mamba Selective Scan

---

## 📊 Progress Tracker

Run this command to see your visual progress:

```bash
python3 utils/progress.py

# Mark completed topics in your local state
python3 utils/progress.py --mark-topic 1 2 3

# Undo a mark
python3 utils/progress.py --unmark-topic 2
```

---

## 🏆 Milestones

| Milestone | Topic | Target | Status |
|-----------|-------|--------|--------|
| First Neural Network | 10 | 95% MNIST accuracy | ⬜ |
| CNN Expert | 17 | CIFAR-10 ResNet | ⬜ |
| Sequence Master | 24 | Bahdanau Attention | ⬜ |
| Transformer Builder | 30 | Coherent text generation | ⬜ |
| Production Ready | 34 | All advanced topics | ⬜ |

Use milestone harness commands:

```bash
# Fast deterministic checks
python3 utils/milestone_eval.py --smoke

# Full milestone checks via topic tests
python3 utils/milestone_eval.py
```

---

## 💡 Tips

- **Stuck on Topic 02?** Use the escape hatch: `utils/autograd_stub.py`
- **Taking too long?** Use dataset subsets first, then full data
- **Want to skip ahead?** Each topic is modular, but dependencies exist
- **Need help?** Open an issue using the "stuck-on-day" template

---

*Use `utils/progress.py` local state commands to track your journey.*
