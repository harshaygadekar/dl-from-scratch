# Topic 17 Milestone Metrics

## Milestone

CNN Expert checkpoint: CIFAR-10 training path is operational and measurable.

## Targets

- Aspirational learning target: reproducible CIFAR-10 baseline training with improving evaluation accuracy.
- Automated full gate: `utils/test_runner.py --day 17` exits with code 0.
- Automated smoke gate: synthetic softmax classifier accuracy >= 0.65.

## Commands

- Full milestone check:
  - `python3 utils/milestone_eval.py --topic 17`
- Fast smoke check:
  - `python3 utils/milestone_eval.py --topic 17 --smoke`

## Reproducibility Notes

- Smoke metric uses deterministic synthetic data to avoid download variance.
- For CIFAR runs, keep normalization and flattening settings constant across comparisons.
