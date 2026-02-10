# Topic 30 Milestone Metrics

## Milestone

Transformer Builder checkpoint: Mini-GPT data path and token prediction loop are operational.

## Targets

- Aspirational learning target: coherent text generation on a tiny corpus.
- Automated full gate: `utils/test_runner.py --day 30` exits with code 0.
- Automated smoke gate: cross-entropy on perfect logits < 1e-4.

## Commands

- Full milestone check:
  - `python3 utils/milestone_eval.py --topic 30`
- Fast smoke check:
  - `python3 utils/milestone_eval.py --topic 30 --smoke`

## Reproducibility Notes

- Smoke metric uses deterministic synthetic logits and exact target alignment.
- For manual generation checks, record temperature and max_new_tokens values for repeatability.
