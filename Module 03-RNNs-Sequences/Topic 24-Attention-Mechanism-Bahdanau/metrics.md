# Topic 24 Milestone Metrics

## Milestone

Sequence Master checkpoint: Bahdanau attention implementation is numerically correct and mask-safe.

## Targets

- Automated full gate: `utils/test_runner.py --day 24` exits with code 0.
- Automated smoke gate:
  - maximum masked attention weight < 1e-6
  - maximum row-sum error from 1.0 < 1e-6

## Commands

- Full milestone check:
  - `python3 utils/milestone_eval.py --topic 24`
- Fast smoke check:
  - `python3 utils/milestone_eval.py --topic 24 --smoke`

## Reproducibility Notes

- Smoke metric uses deterministic random seeds and explicit mask construction.
- Ensure softmax is computed over encoder-time axis for row-sum checks.
