# Topic 08: Interview Questions

Loss functions are fundamental ML interview topics!

---

## Q1: MSE vs MAE (Common)

**Difficulty**: Easy | **Time**: 3 min

When would you use MAE over MSE?

<details>
<summary>Answer</summary>

**MSE** (Mean Squared Error):
- Penalizes large errors more (quadratic)
- Sensitive to outliers
- Smooth gradient everywhere

**MAE** (Mean Absolute Error):
- Linear penalty
- Robust to outliers
- Gradient is constant (±1)

**Use MAE when**:
1. Data has many outliers
2. You want equal penalty for all error magnitudes
3. Median prediction is more meaningful than mean

**Use MSE when**:
1. You want to penalize large errors more
2. Gradient-based optimization (smoother gradients)
3. Normal error distribution

</details>

---

## Q2: Why Log in Cross-Entropy? (Google)

**Difficulty**: Medium | **Time**: 5 min

<details>
<summary>Answer</summary>

**Cross-Entropy**: L = -Σ y_i log(p_i)

**Three reasons for log**:

1. **Information theory**: Cross-entropy measures bits needed to encode distribution Q using distribution P. Logarithm measures information content.

2. **Numerical stability**: Probabilities near 0 give large loss (log(0.01) = -4.6), enforcing confident correct predictions.

3. **Nice gradient**: For softmax + CE, gradient is simply (p - y), which is clean and stable.

**Intuition**: Log transforms multiplicative errors into additive errors. Small probabilities for correct class → large loss.

</details>

---

## Q3: Cross-Entropy vs KL Divergence (Meta)

**Difficulty**: Medium | **Time**: 5 min

<details>
<summary>Answer</summary>

**Cross-Entropy**: H(P, Q) = -Σ p(x) log q(x)

**KL Divergence**: D_KL(P || Q) = Σ p(x) log(p(x)/q(x))

**Relationship**:
```
H(P, Q) = H(P) + D_KL(P || Q)
```

**In training**:
- H(P) is entropy of true distribution (constant)
- Minimizing cross-entropy = minimizing KL divergence

**Why use cross-entropy**:
1. Simpler to compute (don't need P's entropy)
2. Gradient is the same either way
3. When P is one-hot, H(P) = 0, so H(P,Q) = D_KL(P||Q)

</details>

---

## Q4: Binary CE with Logits (Practical)

**Difficulty**: Medium | **Time**: 5 min

<details>
<summary>Answer</summary>

**Standard Binary CE** (with sigmoid output p):
```
L = -[y log(p) + (1-y) log(1-p)]
```

**With logits z** (more stable):
```
L = max(z, 0) - z*y + log(1 + exp(-|z|))
```

**Why more stable**:
- Avoids computing sigmoid(z) then log(sigmoid(z))
- Handles large |z| without overflow/underflow
- This is what PyTorch's BCEWithLogitsLoss does

```python
def bce_with_logits(z, y):
    return np.maximum(z, 0) - z * y + np.log(1 + np.exp(-np.abs(z)))
```

</details>

---

## Q5: Focal Loss (Meta, CV interviews)

**Difficulty**: Hard | **Time**: 5 min

<details>
<summary>Answer</summary>

**Problem**: Class imbalance (e.g., 99% background in object detection)

**Standard CE**: L = -log(p_t)

**Focal Loss**: L = -(1 - p_t)^γ × log(p_t)

Where p_t = p if y=1, else (1-p).

**How it works**:
- Easy examples (high p_t): (1-p_t)^γ → 0, down-weighted
- Hard examples (low p_t): (1-p_t)^γ → 1, normal weight

**Typical γ = 2**:
- If p_t = 0.9 (easy): weight = (0.1)² = 0.01
- If p_t = 0.1 (hard): weight = (0.9)² = 0.81

**Use case**: Object detection (RetinaNet), heavily imbalanced data

</details>

---

## Q6: Label Smoothing (NLP/CV)

**Difficulty**: Medium | **Time**: 5 min

<details>
<summary>Answer</summary>

**Standard one-hot**: y = [0, 0, 1, 0, 0]

**Label smoothing** (ε = 0.1):
```
y_smooth = (1 - ε) × y + ε/K
         = [0.02, 0.02, 0.92, 0.02, 0.02]
```

**Benefits**:
1. **Regularization**: Prevents overconfident predictions
2. **Calibration**: Better probability estimates
3. **Generalization**: Slightly better test accuracy

**When to use**:
- Classification with many classes
- When you want well-calibrated probabilities
- NLP tasks (transformers often use it)

</details>

---

## 🎯 Interview Tips

1. Know MSE/MAE tradeoffs
2. Understand why log is in cross-entropy
3. Know numerical stability tricks (logits vs probabilities)
4. Be familiar with Focal Loss for CV roles
5. Understand label smoothing for NLP/large models

---

*"Choose the right loss, and training becomes much easier."*
