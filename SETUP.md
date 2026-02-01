# Environment Setup

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 8GB | 16GB |
| GPU | Not required | Optional for Topics 28-34 |

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/dl-from-scratch.git
cd dl-from-scratch
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python utils/test_runner.py --verify-setup
```

**Expected output:**
```
🔍 Verifying setup...

✅ Python 3.10 detected
✅ NumPy 1.24.0 installed
✅ Matplotlib 3.7.0 installed

✅ All systems go! Start with Topic 01.
```

---

## Troubleshooting

### ImportError: No module named 'numpy'

```bash
pip install numpy matplotlib pytest
```

### Permission denied

```bash
pip install --user numpy matplotlib pytest
```

### Python version too old

Use pyenv or download Python 3.10+ from [python.org](https://www.python.org/downloads/)

### Matplotlib backend issues (headless server)

```bash
# Add to your script:
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
```

---

## Hardware Tiers

| Tier | Specs | Topics Supported | Notes |
|------|-------|------------------|-------|
| **Tier 1 (Required)** | CPU, 8GB RAM | Topics 1-34 | All topics work on CPU |
| **Tier 2 (Optional)** | GPU (any CUDA) | Topics 28-34 | Faster but not required |

### Topic-Specific Training Times (Tier 1)

| Topic | Dataset | Max Training Time |
|-------|---------|-------------------|
| 10 | MNIST (subset) | < 10 minutes |
| 17 | CIFAR-10 (small) | < 10 minutes |
| 30 | TinyShakespeare | < 10 minutes |

**Constraint**: Every topic is designed to complete training in under 10 minutes on a 2019 MacBook Pro.

---

## IDE Recommendations

Any Python IDE works. Recommended:

- **VS Code** with Python extension
- **PyCharm** (Community Edition is free)
- **Vim/Neovim** with Python LSP

**Note**: We use `.py` files, not Jupyter notebooks. This is intentional for engineering practice.

---

## Ready?

After successful verification:

```bash
cd "Module 00-Foundations/Topic 01-Tensor-Operations"
cat README.md
# Start learning!
```

---

*Setup complete? Start your journey: [Topic 01 →](Module%2000-Foundations/Topic%2001-Tensor-Operations/)*
