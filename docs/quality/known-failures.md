# Known Failures Ledger

This file tracks historically observed full-validation failures and their resolution status.

## Snapshot: 2026-02-08 historical run

Source:
- `.validation.log`
- `.validation_failures.log`

Observed failing topics:
1. Topic 02 (`Module 00-Foundations/Topic 02-Autograd-Engine/tests/test_stress.py:45`)
2. Topic 03 (`Module 00-Foundations/Topic 03-Optimizers/tests/test_basic.py:166`)
3. Topic 11 (`Module 02-CNNs/Topic 11-Conv2D-Sliding-Window/tests/test_basic.py:228`)
4. Topic 13 (`Module 02-CNNs/Topic 13-Pooling-Strides/tests/test_basic.py:24`)

## Investigation Summary (2026-02-10)

Re-run commands:
- `.venv-validation/bin/python utils/test_runner.py --day 2`
- `.venv-validation/bin/python utils/test_runner.py --day 3`
- `.venv-validation/bin/python utils/test_runner.py --day 11`
- `.venv-validation/bin/python utils/test_runner.py --day 13`

Result:
- All four topics now pass.

Likely root cause:
- The tracked failure files represent an older snapshot and were not being refreshed automatically with a machine-readable source of truth.

Resolution:
1. Added deterministic full-run validator: `utils/validate_all.py`.
2. Added JSON summary output: `.validation_report.json`.
3. Added compatibility failure output: `.validation_failures.log`.
4. Updated CI to use the validator as the primary topic test gate.

Status:
- Resolved and confirmed by full run (`.venv-validation/bin/python utils/validate_all.py`) on 2026-02-10.
