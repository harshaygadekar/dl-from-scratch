# Module 04: Transformers & Production

> Build transformer internals, then finish with practical inference/training systems topics.

---

## Overview

This module covers modern attention architectures and production constraints:
- Self-attention and masking
- Multi-head attention and position encoding
- Encoder/decoder block assembly
- Mini-GPT training loop
- KV-cache, quantization, and distributed-training concepts

---

## Topics

| Topic | Name | Description | Duration |
|-------|------|-------------|----------|
| 25 | Efficient Self-Attention | Q/K/V, causal masking, complexity | 2-3 hrs |
| 26 | Multi-Head Attention | Parallel head projections and merge | 2-3 hrs |
| 27 | Positional Encodings | Sinusoidal and modern variants | 2-3 hrs |
| 28 | Transformer Encoder Block | Residual, norm, FFN integration | 2-3 hrs |
| 29 | Transformer Decoder Block | Masked attention and cross-attention | 2-3 hrs |
| 30 | Mini-GPT Training | End-to-end language model baseline | 3-4 hrs |
| 31 | KV-Cache Optimization | Faster autoregressive decoding | 2-3 hrs |
| 32 | Modern Optimizations | Checkpointing/attention efficiency concepts | 2-3 hrs |
| 33 | Quantization & Efficiency | INT8 and memory/perf tradeoffs | 2-3 hrs |
| 34 | Distributed Training Logic | Data/model parallel fundamentals | 2-3 hrs |

---

## Milestone

Topic 30 is the core completion target. Topics 31-34 are advanced systems extensions.
