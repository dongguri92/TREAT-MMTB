"""Sealed fresh 50-epoch Apple-MPS convergence run for Issue #95.

Dry-run is the default.  The five-epoch health result is an approval gate only:
its checkpoint is never used to initialize this run.  A possible epoch-150
continuation is never launched here; it requires a separately reviewed runner,
attempt ID, queue specification, and exact epoch-50 checkpoint receipt.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import reproduce_teammate_l05_mps as engine
from reproduction import canonical_sha256, sha256_file, write_json_once

ISSUE_URL = "https://github.com/choco9966/TREAT-MMTB-2026/issues/95"
EPOCHS = 50
PLATEAU_WINDOW = 10
PLATEAU_RANGE_TOLERANCE = 0.002
PLATEAU_SLOPE_TOLERANCE = 0.0002
EXTENSION_COMPOSITE_SLOPE = 0.0002
SHORT_BUDGET_NOISE_FLOOR = 0.01


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("slope requires at least two finite values")
    mean_x = (len(values) - 1) / 2
    mean_y = sum(values) / len(values)
    numerator = sum(
        (index - mean_x) * (value - mean_y) for index, value in enumerate(values)
    )
    denominator = sum((index - mean_x) ** 2 for index in range(len(values)))
    return numerator / denominator


def convergence_decision(
    epochs: Sequence[Mapping[str, Any]], *, health_gate_passed: bool
) -> dict[str, Any]:
    if not epochs:
        raise ValueError("convergence decision requires epoch evidence")
    ordered = sorted(epochs, key=lambda row: int(row["epoch"]))
    if [int(row["epoch"]) for row in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("epoch evidence must be complete and contiguous")
    for row in ordered:
        for key in ("weighted_composite", "validation_loss"):
            if not math.isfinite(float(row[key])):
                raise ValueError("convergence evidence contains non-finite metrics")
    recent = ordered[-PLATEAU_WINDOW:]
    composites = [float(row["weighted_composite"]) for row in recent]
    validation_losses = [float(row["validation_loss"]) for row in recent]
    composite_slope = _slope(composites) if len(recent) > 1 else 0.0
    validation_loss_slope = _slope(validation_losses) if len(recent) > 1 else 0.0
    best = max(
        ordered, key=lambda row: (float(row["weighted_composite"]), -int(row["epoch"]))
    )
    completed_fifty = int(ordered[-1]["epoch"]) == EPOCHS
    plateaued = (
        len(recent) == PLATEAU_WINDOW
        and max(composites) - min(composites) <= PLATEAU_RANGE_TOLERANCE
        and abs(composite_slope) <= PLATEAU_SLOPE_TOLERANCE
    )
    gain = float(best["weighted_composite"]) - float(ordered[4]["weighted_composite"])
    eligible = (
        completed_fifty
        and health_gate_passed
        and not plateaued
        and composite_slope > EXTENSION_COMPOSITE_SLOPE
        and validation_loss_slope <= 0
        and gain > SHORT_BUDGET_NOISE_FLOOR
    )
    return {
        "schema_version": 1,
        "issue": 95,
        "completed_epochs": len(ordered),
        "best_epoch": int(best["epoch"]),
        "final_epoch": int(ordered[-1]["epoch"]),
        "last_10_composite_slope": composite_slope,
        "last_10_validation_loss_slope": validation_loss_slope,
        "best_delta_vs_epoch_5": gain,
        "plateaued": plateaued,
        "health_gate_passed": health_gate_passed,
        "extension_to_150_eligible": eligible,
        "next_action": "request_fresh_reviewed_continuation_to_epoch_150"
        if eligible
        else "stop_at_50",
        "auto_launch": False,
        "extension_semantics": "continue_exact_epoch_50_best_checkpoint_with_optimizer_scheduler_and_rng_state",
    }


def validate_health_approval(path: Path, expected_attempt_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "issue",
        "status",
        "review_state",
        "reviewer",
        "health_attempt_id",
        "health_run_record_path",
        "health_run_record_sha256",
        "health_artifact_index_path",
        "health_artifact_index_sha256",
        "allowed_to",
        "external_final_accessed",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("health approval receipt schema is invalid")
    if (
        payload["issue"] != 95
        or payload["status"] != "approved"
        or payload["review_state"] != "independently_reviewed"
    ):
        raise ValueError("health approval receipt is not approved")
    if not isinstance(payload["reviewer"], str) or not payload["reviewer"].strip():
        raise ValueError("health approval reviewer is missing")
    if (
        payload["allowed_to"] != expected_attempt_id
        or payload["external_final_accessed"] is not False
    ):
        raise ValueError("health approval is not bound to this attempt")
    for prefix in ("health_run_record", "health_artifact_index"):
        evidence = Path(payload[f"{prefix}_path"])
        if (
            not evidence.is_absolute()
            or evidence.is_symlink()
            or sha256_file(evidence) != payload[f"{prefix}_sha256"]
        ):
            raise ValueError(f"{prefix} bytes do not match approval")
    run_record = json.loads(
        Path(payload["health_run_record_path"]).read_text(encoding="utf-8")
    )
    index = json.loads(
        Path(payload["health_artifact_index_path"]).read_text(encoding="utf-8")
    )
    if (
        run_record.get("attempt_id") != payload["health_attempt_id"]
        or index.get("attempt_id") != payload["health_attempt_id"]
    ):
        raise ValueError("health attempt identity mismatch")
    if (
        run_record.get("status") != "completed"
        or run_record.get("phase") != "health"
        or run_record.get("completed_epochs") != 5
    ):
        raise ValueError("health run is incomplete")
    if (
        run_record.get("wandb_finished") is not True
        or index.get("status") != "completed"
    ):
        raise ValueError("health W&B/artifact completion is invalid")
    return payload


def queue_spec(args: argparse.Namespace, approval: Mapping[str, Any]) -> dict[str, Any]:
    argv = [sys.executable, str(Path(__file__).resolve()), *args.raw_argv, "--execute"]
    return {
        "schema_version": 1,
        "job_id": "task1-teammate-l05-mps-convergence-50e",
        "attempt": 1,
        "max_attempts": 1,
        "attempt_id": args.attempt_id,
        "argv": argv,
        "argv_sha256": canonical_sha256(argv),
        "executable_sha256": sha256_file(Path(sys.executable).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "cwd": str(Path(__file__).resolve().parent),
        "phase": "convergence_50e",
        "fresh_initialization": True,
        "forbidden_initialization": approval["health_artifact_index_sha256"],
        "gates": [
            {
                "type": "decision_receipt",
                "path": str(args.health_approval.resolve()),
                "sha256": sha256_file(args.health_approval),
                "expected_attempt_id": approval["health_attempt_id"],
                "allowed_to": [args.attempt_id],
            }
        ],
        "wandb": {
            "entity": engine.WANDB_ENTITY,
            "project": engine.WANDB_PROJECT,
            "id": args.attempt_id,
            "resume": "never",
            "mode": "online",
        },
        "external_final_accessed": False,
    }


def _seal_completion(
    args: argparse.Namespace, approval: Mapping[str, Any], base_index: Mapping[str, Any]
) -> dict[str, Any]:
    artifact_dir = args.artifact_root / args.attempt_id
    epochs_payload = json.loads(
        (artifact_dir / "epochs.json").read_text(encoding="utf-8")
    )
    decision = convergence_decision(epochs_payload["epochs"], health_gate_passed=True)
    write_json_once(artifact_dir / "convergence_decision.json", decision)
    handoff = {
        "schema_version": 2,
        "selection_issue": 95,
        "review_state": "pending_independent_review",
        "reviewer": None,
        "reviewed_commit": None,
        "model_id": "evax-small-multitask-l05-veto005-mps",
        "selected_budget": decision["best_epoch"],
        "selection_receipt_ref": "convergence_decision.json",
        "selection_receipt_sha256": sha256_file(
            artifact_dir / "convergence_decision.json"
        ),
        "initialization_policy": "fresh_from_reviewed_pretrained",
        "external_final_accessed": False,
        "launch_eligible": False,
    }
    write_json_once(artifact_dir / "issue93_champion_handoff.pending.json", handoff)
    final = {
        "schema_version": 1,
        "attempt_id": args.attempt_id,
        "phase": "convergence_50e",
        "status": "completed_pending_independent_review",
        "base_artifact_index_sha256": sha256_file(artifact_dir / "artifact_index.json"),
        "health_approval_sha256": sha256_file(args.health_approval),
        "convergence_decision_sha256": sha256_file(
            artifact_dir / "convergence_decision.json"
        ),
        "issue93_handoff_sha256": sha256_file(
            artifact_dir / "issue93_champion_handoff.pending.json"
        ),
        "auto_launched_150": False,
        "external_final_accessed": False,
    }
    write_json_once(artifact_dir / "convergence_index.json", final)
    return final


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--health-approval", type=Path, required=True)
    extra, _ = parser.parse_known_args(raw)
    health_index = raw.index("--health-approval")
    engine_argv = raw[:health_index] + raw[health_index + 2 :]
    args = engine.parse_args(engine_argv)
    args.health_approval = extra.health_approval
    args.raw_argv = [item for item in raw if item != "--execute"]
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    approval = validate_health_approval(args.health_approval, args.attempt_id)
    spec = queue_spec(args, approval)
    if not args.execute:
        return {
            "status": "dry_run",
            "review_required": True,
            "protocol": "fresh_50e",
            "queue_spec": spec,
        }
    engine.EPOCHS = EPOCHS
    engine.PHASE = "convergence_50e"
    engine.ISSUE_URL = ISSUE_URL
    engine.WANDB_GROUP = "task1-teammate-l05-veto-mps-convergence"
    engine.WANDB_JOB_TYPE = "reviewed-mps-convergence-50e"
    engine.MPS_ARTIFACT_NAMES = tuple(engine.MPS_ARTIFACT_NAMES)
    base = engine.run(args)
    return _seal_completion(args, approval, base)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
