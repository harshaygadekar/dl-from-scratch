# Phase 2 PR Draft

## PR Title
`Phase 2: reliability, depth parity, milestones, bonus track, and contributor quality gates`

## PR Body

### Summary
This PR ships the full Phase 2 upgrade (A-F) for `dl-from-scratch` with a clean commit series.

Primary outcomes:
- Reproducible curriculum-wide validation and JSON reporting
- Learner-owned progress tracking instead of scaffold-derived completion
- Depth parity across Modules 02-04 with stronger topic tests (25-34)
- Milestone evaluation harness for Topics 10/17/24/30/34
- Optional 2026 bonus module (Topics 35-38)
- Contributor governance via quality checklist, topic linting, and quality-gap issue intake

### Commit Series
1. `c6b97f4` `phase-a-b: add full validator and user-owned progress tracking`
2. `b8be66e` `phase-c: deepen modules 2-4 curriculum and strengthen topic 25-34 tests`
3. `0b9a4fa` `phase-d: add milestone evaluation harness and metric contracts`
4. `e3778f7` `phase-e: add optional bonus module (topics 35-38) and docs wiring`
5. `712d4f2` `phase-f: add contributor quality gates, linting, and execution tracking`

### Key Changes by Phase

#### A-B: Reliability + Learner Progress
- Added full validator: `utils/validate_all.py`
- Refactored progress tracking to local user state: `utils/progress.py`
- Added known failures ledger: `docs/quality/known-failures.md`
- Added local progress state placeholder: `data/progress/.gitkeep`
- Updated setup docs for `python3` and progress initialization: `SETUP.md`

#### C: Curriculum Depth Parity (Modules 02-04)
- Expanded READMEs/questions/hints for Topics 12-34
- Strengthened tests across Topics 25-34 (`test_basic.py`, `test_edge.py`, `test_stress.py`)

#### D: Milestone Harness
- Added milestone runner: `utils/milestone_eval.py`
- Added milestone metrics docs:
  - `Module 01-Neural-Network-Core/Topic 10-End-to-End-MNIST/metrics.md`
  - `Module 02-CNNs/Topic 17-CIFAR10-From-Scratch/metrics.md`
  - `Module 03-RNNs-Sequences/Topic 24-Attention-Mechanism-Bahdanau/metrics.md`
  - `Module 04-Transformers-Production/Topic 30-Mini-GPT-Training/metrics.md`
  - `Module 04-Transformers-Production/Topic 34-Distributed-Training-Logic/metrics.md`

#### E: Optional Bonus Track
- Added `Module 05-Bonus-LLM-Systems/` with Topics 35-38
- Added benchmark runner: `Module 05-Bonus-LLM-Systems/run_benchmarks.py`
- Extended runner coverage to Topics 35-38: `utils/test_runner.py`
- Updated discoverability docs: `README.md`, `ROADMAP.md`

#### F: Contributor Governance
- Added authoring checklist: `docs/authoring/topic-quality-checklist.md`
- Added topic content linter: `scripts/lint_topic_content.py`
- Added quality-gap issue template: `.github/ISSUE_TEMPLATE/curriculum-quality-gap.md`
- Updated contribution process: `CONTRIBUTING.md`
- Updated CI with content lint + milestone smoke + full validator gate: `.github/workflows/test-solutions.yml`
- Added artifact ignores: `.gitignore`
- Persisted execution status and final completion state:
  - `.omc/plans/phase-2-creative-improvements-2026-02-10.md`
  - `.omc/plans/phase-2-creative-improvements-2026-02-10.status.md`

### Verification Run (Local)
- `python3 -m py_compile utils/*.py scripts/*.py` -> pass
- `python3 scripts/lint_topic_content.py --quiet` -> pass (`38` checked, `0` failed)
- `.venv-validation/bin/python utils/validate_all.py` -> pass (`34/34`)
- `.venv-validation/bin/python utils/milestone_eval.py --smoke` -> pass
- `.venv-validation/bin/python utils/milestone_eval.py` -> pass
- `.venv-validation/bin/python utils/test_runner.py --day 35`..`38` -> pass
- `.venv-validation/bin/python Module 05-Bonus-LLM-Systems/run_benchmarks.py` -> pass

### Release Notes (Short)
- Phase 2 is now complete (A-F).
- Core path remains 34 topics; optional bonus path adds Topics 35-38.
- New commands:
  - `python3 utils/validate_all.py`
  - `python3 utils/milestone_eval.py --smoke`
  - `python3 utils/milestone_eval.py`
  - `python3 scripts/lint_topic_content.py`

### Post-merge
- Monitor real learner friction via `.github/ISSUE_TEMPLATE/curriculum-quality-gap.md`
- Collect one week of reports, then define Phase G using observed blockers.
