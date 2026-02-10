# Phase 2 Plan: Creative Intelligence Upgrade

Date: 2026-02-10  
Status: Execution complete (Phases A-F completed)
Execution Status Tracker: `.omc/plans/phase-2-creative-improvements-2026-02-10.status.md`

## 1) Objective

Evolve `dl-from-scratch` from a strong phase-1 scaffold into a high-retention, interview-relevant, and 2026-timely learning system, while preserving the "NumPy-first, from-scratch" core.

## 2) Baseline Evidence (Current State)

Project strengths:
- Full 34-topic structure is present and testable (`README.md:23`, `README.md:24`, `README.md:26`).
- CI already validates setup + utilities + per-topic tests (`.github/workflows/test-solutions.yml:30`, `.github/workflows/test-solutions.yml:42`).
- From-scratch constraints and learning sequence are clearly documented (`README.md:79`, `README.md:115`).

Key gaps to prioritize:
- Validation log shows known failing topics from a full-run snapshot: 02, 03, 11, 13 (`.validation.log:22`, `.validation.log:40`, `.validation.log:168`, `.validation.log:214`; `.validation_failures.log:1`).
- `utils/progress.py` marks progress from solution-file existence, which reports 100% on fresh clone (`utils/progress.py:40`, `utils/progress.py:71`, `utils/progress.py:83`).
- Large content-depth inconsistency across topics:
  - early topics are detailed (e.g., `Module 00-Foundations/Topic 02-Autograd-Engine/README.md:1`)
  - many later topics are scaffold-level and very short (e.g., `Module 04-Transformers-Production/Topic 30-Mini-GPT-Training/README.md:1`, `Module 02-CNNs/Topic 12-Im2Col-Vectorization/README.md:1`, `Module 03-RNNs-Sequences/Topic 24-Attention-Mechanism-Bahdanau/README.md:1`).
- Setup commands assume `python`, but some environments only expose `python3` (observed locally; also `SETUP.md:42`, `README.md:37`, `README.md:126`).

## 3) Creative Intelligence Workflow Applied

Technique set used:
- `SWOT` for strategic positioning
- `SCAMPER` for feature expansion
- `Reverse Brainstorming` for failure-prevention design
- `Diverge -> Converge` to move from broad ideas to an implementable roadmap

### SWOT (Condensed)

Strengths:
- End-to-end 34-topic architecture and strong module framing (`ROADMAP.md:7`, `ROADMAP.md:59`).
- Existing testing skeleton for every topic (`README.md:24`, `.github/workflows/test-solutions.yml:57`).

Weaknesses:
- Reliability snapshot has unresolved failing days (`.validation.log:22`, `.validation.log:40`, `.validation.log:168`, `.validation.log:214`).
- Progress tracking currently measures repository completeness, not learner completion (`utils/progress.py:40`).
- Modules 2-4 pedagogical depth uneven vs Modules 0-1.

Opportunities:
- Expand into modern LLM systems topics as optional bonus path without breaking core curriculum.
- Add benchmarking/evaluation rails to make interview prep more measurable.

Threats:
- Competing resources continue to raise expectations in quality, coverage, and practical relevance.

## 4) Research Signals (Timely + Competitive)

Competitive references (as of 2026-02-10):
- `karpathy/micrograd` (~44k stars) focuses autograd fundamentals, narrow scope.
- `rasbt/LLMs-from-scratch` (~66k stars) provides broad LLM build path.
- `d2l-ai/d2l-en` (~24k stars) has broad textbook-style adoption.
- `EleutherAI/lm-evaluation-harness` (~7.6k stars) signals demand for standardized evaluation.
- `openai/evals` (~14k stars) reinforces evaluation-first workflows.

Technical trend references:
- FlashAttention-2 (throughput and training efficiency): https://arxiv.org/abs/2307.08691
- Mamba (selective state space models): https://arxiv.org/abs/2312.00752
- QLoRA (efficient adaptation path): https://arxiv.org/abs/2305.14314
- Speculative Decoding (inference speedups): https://arxiv.org/abs/2211.17192

Learning product signal:
- Hugging Face learning tracks now include agent/tooling workflows, showing market pull toward practical systems and tooling literacy: https://huggingface.co/learn/agents-course/unit0/introduction

## 5) Converged Improvement Themes

1. Reliability and trust first
- Remove ambiguity in "does it really pass?" by introducing repeatable whole-curriculum validation and tracked fix tickets for known failing snapshots.

2. Learner truth over repository truth
- Replace solution-file-based progress with user-owned progress state and milestone evidence.

3. Depth parity across modules
- Bring Modules 2-4 docs/tests/hints to the pedagogical quality bar already visible in Modules 0-1.

4. Timely, optional advanced track
- Add practical 2026-relevant topics as optional post-34 "Bonus Track", keeping core path unchanged.

5. Measurable outcomes
- Add small benchmark/eval harness for topic milestones (10, 17, 24, 30, 34) so progress can be quantified.

## 6) Requirements Summary

Functional requirements:
1. Progress tracking must represent individual learner completion, not repository scaffold status.
2. A full-curriculum validation command must exist and produce machine-readable pass/fail summary.
3. All topics must meet minimum content quality standards (README + hints + tests depth thresholds).
4. Milestone topics must expose reproducible metrics and expected targets.
5. Optional bonus topics must be clearly labeled and isolated from the required 34-topic path.

Non-functional requirements:
1. Keep dependencies minimal and compatible with current constraints (`requirements.txt:1`).
2. Preserve CPU-friendly path for core track and avoid mandatory GPU assumptions (`SETUP.md:91`).
3. Keep contributor workflow simple and enforce consistency via templates/linting.

## 7) Acceptance Criteria (Testable)

Reliability:
1. A new full validation script runs all 34 topics and exits non-zero on any failure.
2. Validation output includes per-topic status and a final summary JSON file.
3. Two consecutive clean runs produce identical pass/fail outcomes in CI.

Progress:
4. Fresh clone shows 0% progress until user marks completion.
5. Progress state persists in a local state file and supports `--reset` and `--mark-topic`.
6. `utils/progress.py --detailed` reflects learner-marked completion accurately.

Curriculum quality:
7. Every topic README meets a minimum structural checklist (objectives, constraints, success criteria, edge cases).
8. Every topic has at least 3 topic-specific questions (not generic template text).
9. Every topic keeps 3 hints, each with topic-specific examples and one concrete failure mode.
10. Every topic test suite includes at least one deterministic numerical assertion and one shape-contract assertion.

Milestones and eval:
11. Topics 10/17/24/30/34 each have a `metrics.md` with target metrics and reproducibility notes.
12. A milestone eval script can run in isolation and report pass/fail against target thresholds.

Bonus track:
13. New bonus topics are discoverable but explicitly optional in roadmap and README.
14. Bonus topics each include at least one implementation task and one comparison benchmark vs baseline.

## 8) Implementation Plan (No Execution Yet)

## Phase A (Week 1): Reliability + Ground Truth

Primary outcomes:
- Replace ad-hoc validation logs with reproducible tooling.
- Resolve discrepancy between historical fail snapshot and current expected pass state.

Planned file work:
1. Add `utils/validate_all.py` to run Topic 01-34 and emit:
   - terminal summary
   - `.validation.log` (human)
   - `.validation_report.json` (machine)
2. Update `.github/workflows/test-solutions.yml` to call the new validator as primary gate.
3. Add `docs/quality/known-failures.md` to track root cause and resolution record for topics seen in `.validation_failures.log`.
4. Investigate and patch solution/test mismatches for:
   - `Module 00-Foundations/Topic 02-Autograd-Engine/tests/test_stress.py:45`
   - `Module 00-Foundations/Topic 03-Optimizers/tests/test_basic.py:166`
   - `Module 02-CNNs/Topic 11-Conv2D-Sliding-Window/tests/test_basic.py:228`
   - `Module 02-CNNs/Topic 13-Pooling-Strides/tests/test_basic.py:24`

## Phase B (Week 2): Learner Progress System

Primary outcomes:
- Progress becomes user-centric, not repository-centric.

Planned file work:
1. Refactor `utils/progress.py`:
   - replace `scan_completed_topics()` logic (`utils/progress.py:40`) with state-backed completion.
   - add CLI flags: `--mark-topic`, `--unmark-topic`, `--reset`, `--status-json`.
2. Add `data/progress/.gitkeep` and use `data/progress/progress_state.json` as local state (ignored in git).
3. Update usage docs:
   - `README.md:123`
   - `ROADMAP.md:79`
   - `SETUP.md` quickstart block.

## Phase C (Weeks 3-5): Curriculum Depth Parity (Modules 2-4)

Primary outcomes:
- Bring later modules to a consistent pedagogical bar.

Planned file work:
1. Expand topic READMEs for:
   - `Module 02-CNNs/Topic 12-Im2Col-Vectorization/README.md`
   - `Module 02-CNNs/Topic 13-Pooling-Strides/README.md`
   - `Module 02-CNNs/Topic 14-ResNet-Skip-Connections/README.md`
   - `Module 02-CNNs/Topic 15-Modern-Convolutions/README.md`
   - `Module 02-CNNs/Topic 16-Advanced-Normalizations/README.md`
   - `Module 02-CNNs/Topic 17-CIFAR10-From-Scratch/README.md`
   - `Module 03-RNNs-Sequences/Topic 18-Vanilla-RNN/README.md` through `Topic 24`.
   - `Module 04-Transformers-Production/Topic 25-Efficient-Self-Attention/README.md` through `Topic 34`.
2. Replace generic question prompts with topic-specific prompts in each `questions.md` for Topics 12-34.
3. Upgrade hints in Topics 12-34 so each hint includes:
   - one concrete pitfall tied to that topic
   - one numerical mini-example
   - one debugging checklist specific to that topic.
4. Expand minimal tests where currently sparse (especially Topics 25-34) to enforce:
   - deterministic seed behavior
   - numerical tolerance checks
   - explicit edge-condition checks.

## Phase D (Weeks 6-7): Milestone Evaluation Harness

Primary outcomes:
- Add measurable outcomes for key checkpoints.

Planned file work:
1. Add `utils/milestone_eval.py` with milestone selectors for topics 10, 17, 24, 30, 34.
2. Add `metrics.md` files under each milestone topic directory.
3. Add optional CI job for milestone eval smoke tests.
4. Add docs:
   - `README.md` milestone section update
   - `ROADMAP.md` milestone status guidance.

## Phase E (Weeks 8-10): Optional 2026 Bonus Track

Primary outcomes:
- Timely relevance without destabilizing core 34-topic path.

Proposed bonus topics (optional, post-34):
1. Topic 35: LoRA and QLoRA fundamentals (NumPy simulation)
2. Topic 36: Speculative Decoding simulation and speed tradeoffs
3. Topic 37: Mixture-of-Experts routing toy implementation
4. Topic 38: SSM/Mamba intuition + minimal selective scan prototype

Planned file work:
1. Add `Module 05-Bonus-LLM-Systems/` with full topic structure mirroring existing conventions.
2. Add "Optional" gating in `README.md` and `ROADMAP.md`.
3. Add benchmark scripts comparing baseline vs optimized variant per topic.

## Phase F (Week 11): Contributor Experience + Governance

Primary outcomes:
- Scale maintenance and quality with clear standards.

Planned file work:
1. Add `docs/authoring/topic-quality-checklist.md`.
2. Add `scripts/lint_topic_content.py` to enforce required sections and file presence.
3. Update `CONTRIBUTING.md` with checklist-driven PR guidance.
4. Add issue template for "curriculum quality gap" to capture learning-blocker reports.

## 9) Risks and Mitigations

Risk 1: Scope expansion delays reliability fixes.
- Mitigation: Phase A and B are mandatory gates before any bonus-track work.

Risk 2: Overly generic "modern topics" drift from from-scratch identity.
- Mitigation: Each bonus topic must include at least one NumPy implementation artifact and one measurable benchmark.

Risk 3: CI time explosion from expanded tests.
- Mitigation: Split CI into `quick` and `full` profiles; run full profile on nightly and release branches.

Risk 4: Learners confuse required vs optional topics.
- Mitigation: Label optional track in all entry docs and CLI output.

## 10) Verification Plan

Verification steps after implementation (planned):
1. Run `python3 utils/validate_all.py` locally and confirm 34/34 pass in clean venv.
2. Run `python3 utils/progress.py` on fresh clone and confirm 0% initial state.
3. Mark 3 topics complete and verify progress percentage updates correctly.
4. Run milestone eval for topics 10/17/24/30/34 and validate target reporting.
5. Run content lint script and ensure no missing required sections across all topics.
6. Confirm CI green on both quick and full workflows.

## 11) Approval Gates

Gate 1: Approve Phase A+B only (recommended start).  
Gate 2: Approve Phase C depth expansion after reliability/progress success.  
Gate 3: Approve Phase D-E milestone harness + bonus track.  
Gate 4: Approve Phase F contributor governance hardening.

Recommended execution order:
1. A+B
2. C
3. D
4. E
5. F

## 12) External Source Links

- https://github.com/karpathy/micrograd
- https://github.com/rasbt/LLMs-from-scratch
- https://github.com/d2l-ai/d2l-en
- https://github.com/EleutherAI/lm-evaluation-harness
- https://github.com/openai/evals
- https://huggingface.co/learn/agents-course/unit0/introduction
- https://arxiv.org/abs/2307.08691
- https://arxiv.org/abs/2312.00752
- https://arxiv.org/abs/2305.14314
- https://arxiv.org/abs/2211.17192
