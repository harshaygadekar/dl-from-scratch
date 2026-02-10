# Intuition: SSM and Mamba Selective Scan

## Problem

A naive implementation can be correct but too slow or memory-heavy for modern LLM workflows.

## Insight

Use structure-aware approximations and caching to reduce redundant compute while preserving outputs within tolerance.

## Common Pitfalls

- Shape or index mistakes that still return plausible tensors.
- Optimizations that silently alter semantics.
- Missing deterministic checks before measuring speedups.
