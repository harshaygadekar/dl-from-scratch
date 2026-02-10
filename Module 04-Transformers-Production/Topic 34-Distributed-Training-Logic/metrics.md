# Topic 34 Milestone Metrics

## Milestone

Production Ready checkpoint: distributed-training logic primitives are correct and internally consistent.

## Targets

- Automated full gate: `utils/test_runner.py --day 34` exits with code 0.
- Automated smoke gate:
  - sync update maximum numeric error < 1e-7
  - world_size=1 all-reduce overhead bytes == 0

## Commands

- Full milestone check:
  - `python3 utils/milestone_eval.py --topic 34`
- Fast smoke check:
  - `python3 utils/milestone_eval.py --topic 34 --smoke`

## Reproducibility Notes

- Keep parameter and gradient fixtures identical when comparing different sync implementations.
- Validate dictionary key alignment across worker gradients before averaging.
