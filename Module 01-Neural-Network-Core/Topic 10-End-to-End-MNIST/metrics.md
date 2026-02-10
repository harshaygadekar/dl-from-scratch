# Topic 10 Milestone Metrics

## Milestone

First Neural Network: MNIST training pipeline works end-to-end with reproducible behavior.

## Targets

- Aspirational learning target: >= 95% MNIST test accuracy (extended run).
- Automated full gate: `utils/test_runner.py --day 10` exits with code 0.
- Automated smoke gate: synthetic separable-task accuracy >= 0.90.

## Commands

- Full milestone check:
  - `python3 utils/milestone_eval.py --topic 10`
- Fast smoke check:
  - `python3 utils/milestone_eval.py --topic 10 --smoke`

## Reproducibility Notes

- Use fixed random seeds in smoke mode.
- If running extended MNIST accuracy checks manually, keep dataset normalization and batch size consistent.
