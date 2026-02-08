# DL-From-Scratch 🧠

> **"Implementing Deep Learning from the ground up. No PyTorch. No TensorFlow. Just NumPy and understanding."**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)

## 🎯 Who Is This For?

- **ML Engineers** preparing for FAANG interviews who need to implement attention from scratch
- **CS Students** seeking deep understanding over API usage
- **Researchers** debugging model internals and custom architectures

**Tagline**: *"If you can't implement it in NumPy, you don't understand it."*

## ⏱️ Time Commitment

**34 topics** • **2-3 hours/day** • **Prerequisites:** Python, Linear Algebra 101, Calculus I

## 📌 Current Status

- Curriculum fully scaffolded across **Modules 00 to 04** (**Topics 01 to 34**)
- Every topic includes `README.md`, `hints/hint-1..3`, `tests/test_basic.py` + `test_edge.py` + `test_stress.py`, and `solutions/level01..level04`
- Hint content is now standardized with detailed scaffolding across later modules (CNN advanced topics, RNNs, Transformers)
- Recent validation run completed successfully across all topics (`01-34`) in a clean virtual environment

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/harshaygadekar/dl-from-scratch.git
cd dl-from-scratch
pip install -r requirements.txt

# Verify your setup
python utils/test_runner.py --verify-setup

# Start learning!
cd "Module 00-Foundations/Topic 01-Tensor-Operations"
```

## 📚 Curriculum

### Module 0: Foundations (The Survival Module)
| Topic | Title | Key Concept |
|-------|-------|-------------|
| 01 | Tensor Operations & Broadcasting | Memory layouts, stride tricks |
| 02 | Autograd Engine From Scratch | Computational graphs, reverse-mode AD |
| 03 | Optimization Algorithms | SGD, Momentum, Adam from equations |

### Module 1: Neural Network Core
| Topic | Title | Key Concept |
|-------|-------|-------------|
| 04 | Single Layer Perceptron | Sigmoid, BCE loss, binary classification |
| 05 | MLP Forward Pass | Xavier/Kaiming initialization |
| 06 | Backpropagation | Chain rule, gradient computation |
| 07 | Activation Functions | ReLU, Sigmoid, Tanh, Softmax |
| 08 | Loss Functions | MSE, Cross-Entropy, Binary CE |
| 09 | Regularization | L2, Dropout, Batch Normalization |
| 10 | End-to-End MNIST | 95% accuracy target |

### Module 2: Convolutional Networks
| Topic | Title | Key Concept |
|-------|-------|-------------|
| 11-17 | Conv2D → CIFAR-10 | Im2col, ResNet, advanced norms |

### Module 3: Sequence Models
| Topic | Title | Key Concept |
|-------|-------|-------------|
| 18-24 | RNN → Attention | BPTT, LSTM, Bahdanau attention |

### Module 4: Transformers & Production
| Topic | Title | Key Concept |
|-------|-------|-------------|
| 25-30 | Self-Attention → Mini-GPT | Build a small language model |
| 31-34 | Production Optimization | KV-cache, quantization, distributed |

## 🔧 The "From Scratch" Rules

### ❌ Forbidden (Black Boxes)
- PyTorch (`torch.nn`, `torch.optim`, `torch.autograd`)
- TensorFlow/Keras (`tf.keras`, `tf.GradientTape`)
- JAX, Autograd libraries, Scikit-Learn
- Pre-trained models or `torchvision.models`

### ✅ Allowed (Building Blocks Only)
- **NumPy**: `np.dot`, array ops, broadcasting
- **Python stdlib**: `math`, `random`, `collections`
- **Matplotlib**: Visualization only
- **Pandas/PIL**: Data loading only

## 📊 Solution Levels

Every topic has 4 solution tiers:

| Level | Focus | Description |
|-------|-------|-------------|
| **Level 1** | Correctness | Naive but working (loops, readable) |
| **Level 2** | Speed | NumPy-optimized (broadcasting) |
| **Level 3** | Memory | In-place ops, float16, cache-aware |
| **Level 4** | Reference | PyTorch verification (ground truth) |

## 🆘 Stuck?

1. Check `hints/hint-1-*.md` for basic direction
2. Check `hints/hint-2-*.md` for pseudocode
3. Check `hints/hint-3-*.md` for optimization tips
4. Still stuck? Use the escape hatch (see Topic folder README)

**Topic 02 Escape Hatch**: If autograd blocks you, use `utils/autograd_stub.py` and return later.

## 🧭 Suggested Learning Flow

1. Work **module-by-module** (`00 -> 01 -> 02 -> 03 -> 04`)
2. For each topic, attempt `level01` first, use hints progressively (`hint-1 -> hint-2 -> hint-3`), then run tests
3. Move to `level02`/`level03` only after `level01` passes
4. Use `level04` as reference verification, not as first attempt
5. Keep a short implementation log: what failed, what changed, what you learned

This sequencing keeps understanding ahead of optimization and reduces confusion across dependencies.

## 📊 Track Your Progress

```bash
python utils/progress.py
# [###########>          ] Module 0 Topic 02 (6%)
```

## 🤔 Why Not Just Watch Karpathy?

We love Karpathy's content! This is **complementary**, not competitive:

| Karpathy | DL-From-Scratch |
|----------|-----------------|
| Video lectures | Text/problem-based |
| One long project | 34 modular daily problems |
| Theory-heavy | Interview-style constraints |
| General education | FAANG interview preparation |
| PyTorch from start | NumPy-only until verification |

**Complementary use**: Stuck on Topic 6? Watch Karpathy's micrograd video, then return!

## 🧪 Testing

```bash
# Run tests for a specific topic
python utils/test_runner.py --day 05

# List all topics
python utils/test_runner.py --list

# Verify your setup
python utils/test_runner.py --verify-setup
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🚦 Start Your Journey

1. **Assess yourself**: [PREREQUISITES.md](PREREQUISITES.md)
2. **Set up environment**: [SETUP.md](SETUP.md)
3. **Begin Topic 01**: [Module 00/Topic 01](Module%2000-Foundations/Topic%2001-Tensor-Operations/)

---

*Built with ❤️ for those who want to truly understand deep learning.*
