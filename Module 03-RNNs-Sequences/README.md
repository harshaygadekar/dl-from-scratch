# Module 03: RNNs & Sequences

> Process sequential data with recurrent architectures.

---

## 📋 Overview

This module covers recurrent neural networks and sequence modeling:
- RNN cells and backpropagation through time
- LSTM and GRU architectures
- Sequence-to-sequence models
- Embeddings

---

## 📚 Topics

| Topic | Name | Description | Duration |
|-------|------|-------------|----------|
| 14 | RNN Cell | Basic recurrence and BPTT | 3-4 hrs |
| 15 | LSTM | Long short-term memory | 3-4 hrs |
| 16 | GRU | Gated recurrent unit | 2-3 hrs |
| 17 | Embeddings | Word → vector representations | 2-3 hrs |
| 18 | Seq2Seq | Encoder-decoder architecture | 3-4 hrs |
| 19 | Attention Basics | Pre-transformer attention | 3-4 hrs |

---

## 🎯 Learning Objectives

After completing this module, you will:
1. Understand how RNNs maintain hidden state
2. Implement LSTM gates and cell states
3. Build sequence-to-sequence models
4. Implement basic attention mechanisms

---

## 🔧 Prerequisites

- ✅ Module 01: Neural Network Core
- ✅ Understanding of sequences and time series
- ✅ Matrix operations

---

## 📈 Difficulty Progression

```
Topic 14 (RNN)      ██████░░░░ Medium-Hard
Topic 15 (LSTM)     ████████░░ Hard
Topic 16 (GRU)      ██████░░░░ Medium-Hard
Topic 17 (Embed)    █████░░░░░ Medium
Topic 18 (Seq2Seq)  ███████░░░ Hard
Topic 19 (Attn)     ████████░░ Hard
```

---

## ⏱️ Estimated Time

**Total**: 17-22 hours

---

## 🗂️ Directory Structure

```
Module 03-RNNs-Sequences/
├── README.md           # This file
├── Topic 14-RNN-Cell/
├── Topic 15-LSTM/
├── Topic 16-GRU/
├── Topic 17-Embeddings/
├── Topic 18-Seq2Seq/
└── Topic 19-Attention-Basics/
```

---

## 🏆 Module Milestone

By the end of this module, you should be able to:

```python
# Build a character-level language model
vocab_size = 50
embed_dim = 64
hidden_dim = 128

embedding = Embedding(vocab_size, embed_dim)
lstm = LSTM(embed_dim, hidden_dim)
output = Linear(hidden_dim, vocab_size)

def forward(chars):
    x = embedding(chars)           # [seq, batch, embed]
    hidden = lstm.init_hidden()
    outputs = []
    for t in range(len(chars)):
        h, hidden = lstm(x[t], hidden)
        outputs.append(output(h))
    return outputs

# Generate text character by character!
```

---

## 🔍 Key Interview Topics

- Why do RNNs suffer from vanishing gradients?
- How do LSTM gates solve this?
- What makes GRU simpler than LSTM?
- How does attention help with long sequences?

---

*"Sequences are the language of time."*
