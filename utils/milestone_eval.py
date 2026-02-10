#!/usr/bin/env python3
"""
Milestone evaluation harness for core curriculum checkpoints.

Milestones:
- Topic 10: End-to-End MNIST
- Topic 17: CIFAR-10 From Scratch
- Topic 24: Bahdanau Attention
- Topic 30: Mini-GPT Training
- Topic 34: Distributed Training Logic

Modes:
- default: run full topic tests through utils/test_runner.py
- --smoke: run fast deterministic milestone checks without pytest
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

MILESTONE_TOPICS = [10, 17, 24, 30, 34]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_topic_dir(day: int) -> Path:
    root = repo_root()
    for topic_dir in root.glob("Module*/Topic *"):
        name = topic_dir.name
        if name.startswith(f"Topic {day:02d}"):
            return topic_dir
    raise FileNotFoundError(f"Could not locate topic directory for day {day:02d}")


def load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_test_runner_day(day: int) -> Tuple[bool, float, str]:
    root = repo_root()
    cmd = [sys.executable, str(root / "utils" / "test_runner.py"), "--day", str(day)]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    combined_output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode == 0, elapsed, combined_output.strip()


def smoke_topic_10() -> Dict:
    topic = find_topic_dir(10)
    mod = load_module_from_file(
        "topic10_level01", topic / "solutions" / "level01_naive.py"
    )

    np.random.seed(2026)
    x = np.random.randn(128, 2).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
    y_oh = np.eye(2, dtype=np.float32)[y]

    model = mod.MLP([2, 16, 2])
    loss_fn = mod.SoftmaxCrossEntropy()
    optimizer = mod.SGD(
        [layer for layer in model.layers if hasattr(layer, "W")], lr=0.1
    )

    for _ in range(80):
        logits = model.forward(x)
        loss = loss_fn.forward(logits, y_oh)
        grad = loss_fn.backward()
        model.backward(grad)
        optimizer.step()

    logits = model.forward(x)
    preds = np.argmax(logits, axis=1)
    acc = float(np.mean(preds == y))

    return {
        "metric_name": "synthetic_linearly_separable_accuracy",
        "value": acc,
        "target": 0.90,
        "passed": acc >= 0.90,
    }


def smoke_topic_17() -> Dict:
    topic = find_topic_dir(17)
    mod = load_module_from_file(
        "topic17_level01", topic / "solutions" / "level01_naive.py"
    )

    np.random.seed(2027)
    n, d, c = 600, 32, 10
    W_true = np.random.randn(d, c).astype(np.float32)
    x = np.random.randn(n, d).astype(np.float32)
    y = np.argmax(x @ W_true, axis=1).astype(np.int64)

    model = mod.SoftmaxClassifier(in_features=d, num_classes=c)
    for _ in range(8):
        _ = mod.train_one_epoch(model, x, y, batch_size=128, lr=0.4)
    acc = mod.evaluate(model, x, y)

    return {
        "metric_name": "synthetic_softmax_classifier_accuracy",
        "value": float(acc),
        "target": 0.65,
        "passed": float(acc) >= 0.65,
    }


def smoke_topic_24() -> Dict:
    topic = find_topic_dir(24)
    mod = load_module_from_file(
        "topic24_level01", topic / "solutions" / "level01_naive.py"
    )

    np.random.seed(2028)
    b, t_enc, d_q, d_k, d_attn = 2, 7, 6, 6, 5
    query = np.random.randn(b, d_q).astype(np.float32)
    # Topic 24 implementation expects keys/values as (T, B, H)
    keys = np.random.randn(t_enc, b, d_k).astype(np.float32)
    values = np.random.randn(t_enc, b, d_k).astype(np.float32)
    w_q = np.random.randn(d_q, d_attn).astype(np.float32) * 0.1
    w_k = np.random.randn(d_k, d_attn).astype(np.float32) * 0.1
    v_a = np.random.randn(d_attn).astype(np.float32) * 0.1
    b_a = np.zeros(d_attn, dtype=np.float32)
    mask = np.ones((b, t_enc), dtype=np.float32)
    mask[:, -2:] = 0.0

    _, weights = mod.bahdanau_context(
        query, keys, values, w_q, w_k, v_a, b_a, mask=mask
    )
    masked_weight = float(np.max(np.abs(weights[:, -2:])))
    row_sum_error = float(np.max(np.abs(weights.sum(axis=-1) - 1.0)))
    passed = masked_weight < 1e-6 and row_sum_error < 1e-6

    return {
        "metric_name": "masked_attention_consistency",
        "value": {
            "max_masked_weight": masked_weight,
            "max_row_sum_error": row_sum_error,
        },
        "target": {
            "max_masked_weight": "<1e-6",
            "max_row_sum_error": "<1e-6",
        },
        "passed": passed,
    }


def smoke_topic_30() -> Dict:
    topic = find_topic_dir(30)
    mod = load_module_from_file(
        "topic30_level01", topic / "solutions" / "level01_naive.py"
    )

    np.random.seed(2029)
    b, t, v = 4, 6, 15
    targets = np.random.randint(0, v, size=(b, t))
    logits = np.full((b, t, v), -8.0, dtype=np.float32)
    logits[np.arange(b)[:, None], np.arange(t)[None, :], targets] = 8.0
    loss = float(mod.cross_entropy_from_logits(logits, targets))

    return {
        "metric_name": "next_token_cross_entropy_on_perfect_logits",
        "value": loss,
        "target": 1e-4,
        "passed": loss < 1e-4,
    }


def smoke_topic_34() -> Dict:
    topic = find_topic_dir(34)
    mod = load_module_from_file(
        "topic34_level02", topic / "solutions" / "level02_vectorized.py"
    )
    mod3 = load_module_from_file(
        "topic34_level03", topic / "solutions" / "level03_memory_efficient.py"
    )

    params = {"w": np.array([2.0, -1.0], dtype=np.float32)}
    grads = [
        {"w": np.array([1.0, 3.0], dtype=np.float32)},
        {"w": np.array([3.0, 1.0], dtype=np.float32)},
    ]
    out = mod.sync_sgd_step(params, grads, lr=0.5)
    expected = np.array([1.0, -2.0], dtype=np.float32)
    max_err = float(np.max(np.abs(out["w"] - expected)))
    zero_overhead = mod3.allreduce_volume_bytes(1000, 2, world_size=1)
    passed = max_err < 1e-7 and zero_overhead == 0

    return {
        "metric_name": "sync_update_and_world_size_one_overhead",
        "value": {
            "max_update_error": max_err,
            "world_size_one_overhead_bytes": int(zero_overhead),
        },
        "target": {
            "max_update_error": "<1e-7",
            "world_size_one_overhead_bytes": 0,
        },
        "passed": passed,
    }


SMOKE_FUNCS: Dict[int, Callable[[], Dict]] = {
    10: smoke_topic_10,
    17: smoke_topic_17,
    24: smoke_topic_24,
    30: smoke_topic_30,
    34: smoke_topic_34,
}


def evaluate_smoke(topics: List[int]) -> List[Dict]:
    rows = []
    for topic in topics:
        start = time.perf_counter()
        metric = SMOKE_FUNCS[topic]()
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "topic": topic,
                "mode": "smoke",
                "metric": metric["metric_name"],
                "value": metric["value"],
                "target": metric["target"],
                "passed": bool(metric["passed"]),
                "duration_seconds": round(elapsed, 3),
            }
        )
    return rows


def evaluate_full(topics: List[int]) -> List[Dict]:
    rows = []
    for topic in topics:
        passed, elapsed, output = run_test_runner_day(topic)
        rows.append(
            {
                "topic": topic,
                "mode": "full",
                "metric": "topic_test_runner_exit_code",
                "value": 0 if passed else 1,
                "target": 0,
                "passed": passed,
                "duration_seconds": round(elapsed, 3),
                "output_tail": output[-1200:],
            }
        )
    return rows


def parse_topics(raw: Optional[List[int]]) -> List[int]:
    if raw is None or len(raw) == 0:
        return list(MILESTONE_TOPICS)

    unique = sorted(set(raw))
    invalid = [t for t in unique if t not in MILESTONE_TOPICS]
    if invalid:
        raise ValueError(
            f"Invalid milestone topics: {invalid}. Allowed: {MILESTONE_TOPICS}"
        )
    return unique


def write_report(report_path: Path, payload: Dict) -> None:
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary(rows: List[Dict], json_out: Path) -> None:
    print("\nMilestone Evaluation Summary")
    print("---------------------------")
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"Topic {row['topic']:02d} [{row['mode']}] {status} "
            f"({row['duration_seconds']:.2f}s) - {row['metric']}"
        )
    failed = [row["topic"] for row in rows if not row["passed"]]
    print(f"JSON report: {json_out}")
    if failed:
        print("Failing milestones: " + ", ".join(f"{d:02d}" for d in failed))
    else:
        print("All requested milestones passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run milestone evaluations")
    parser.add_argument(
        "--topic",
        type=int,
        nargs="+",
        help=f"Subset of milestones to evaluate. Allowed: {MILESTONE_TOPICS}",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run lightweight deterministic smoke checks instead of full topic tests.",
    )
    parser.add_argument(
        "--json-out",
        default=".milestone_eval_report.json",
        help="Path for JSON output report.",
    )
    args = parser.parse_args()

    try:
        topics = parse_topics(args.topic)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    rows = evaluate_smoke(topics) if args.smoke else evaluate_full(topics)
    all_passed = all(row["passed"] for row in rows)

    payload = {
        "mode": "smoke" if args.smoke else "full",
        "topics": topics,
        "all_passed": all_passed,
        "results": rows,
    }
    json_out = Path(args.json_out)
    write_report(json_out, payload)
    print_summary(rows, json_out)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
