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
import subprocess
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
CONTINUATION_CHECKPOINT = "continuation_epoch_50.pth"
HEALTH_ARTIFACT_NAMES = tuple(engine.MPS_ARTIFACT_NAMES)
CONVERGENCE_ARTIFACT_NAMES = (*HEALTH_ARTIFACT_NAMES, CONTINUATION_CHECKPOINT)
EXECUTION_SURFACE = (
    "datasets.py",
    "eva_x.py",
    "models.py",
    "models_evax.py",
    "reproduction.py",
    "reproduce_teammate_l05_mps.py",
    "reproduce_teammate_l05_mps_convergence.py",
    "training.py",
    "utils.py",
    "requirements-reproduction-mps.lock",
)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _safe_path(path: Path | str, label: str, *, require_absolute: bool = True) -> Path:
    lexical = Path(path)
    if require_absolute and not lexical.is_absolute():
        raise ValueError(f"{label} must be absolute")
    resolved = engine.resolve_non_external_path(lexical, label)
    if lexical.is_symlink() or any(parent.is_symlink() for parent in lexical.parents):
        raise ValueError(f"{label} must not traverse symlinks")
    return resolved


def _execution_surface() -> dict[str, str]:
    return {
        name: sha256_file(_safe_path(engine.REPO_ROOT / name, name))
        for name in EXECUTION_SURFACE
    }


def _head_source() -> dict[str, str]:
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=engine.REPO_ROOT, text=True
        ).strip(),
        "git_tree_sha1": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=engine.REPO_ROOT, text=True
        ).strip(),
    }


def _validate_source_delta(
    path: Path, expected_sha256: str, health_source: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = _safe_path(path, "source_delta")
    if resolved.is_symlink() or sha256_file(resolved) != expected_sha256:
        raise ValueError("source-delta receipt bytes do not match approval")
    delta = _read_json_object(resolved, "source-delta receipt")
    current_source = _head_source()
    if (
        set(delta)
        != {
            "schema_version",
            "status",
            "reviewer",
            "health_source",
            "convergence_source",
            "execution_surface",
            "external_final_accessed",
        }
        or delta.get("status") != "independently_reviewed"
        or not str(delta.get("reviewer", "")).strip()
        or delta.get("health_source") != dict(health_source)
        or delta.get("convergence_source") != current_source
        or delta.get("execution_surface") != _execution_surface()
        or delta.get("external_final_accessed") is not False
    ):
        raise ValueError("source-delta execution surface is not exactly approved")
    return delta


def _indexed_health_artifacts(
    index_path: Path, index: Mapping[str, Any]
) -> dict[str, Path]:
    artifacts = index.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(HEALTH_ARTIFACT_NAMES):
        raise ValueError("health artifact index does not cover the exact artifact set")
    root = _safe_path(index_path.parent, "health artifact root")
    resolved: dict[str, Path] = {}
    for name in HEALTH_ARTIFACT_NAMES:
        expected = artifacts.get(name)
        path = _safe_path(root / name, f"health artifact {name}")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or path.is_symlink()
            or not path.is_file()
            or path.parent != root
            or sha256_file(path) != expected
        ):
            raise ValueError(f"health indexed artifact bytes differ: {name}")
        resolved[name] = path
    return resolved


def _normalized_config_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    protocol = config.get("protocol")
    wandb = config.get("wandb")
    if not isinstance(protocol, dict) or not isinstance(wandb, dict):
        raise TypeError("health config lacks protocol/W&B contract")
    normalized_protocol = {
        key: value for key, value in protocol.items() if key not in {"phase", "epochs"}
    }
    return {
        "protocol": normalized_protocol,
        "num_workers": config.get("num_workers"),
        "device": config.get("device"),
        "source": config.get("source"),
        "pretrained": config.get("pretrained"),
        "dataset_scope": config.get("dataset_scope"),
        "manifest": config.get("manifest"),
        "content": config.get("content"),
        "baseline": config.get("baseline"),
        "dependency": config.get("dependency"),
        "case_identity_sha256": config.get("case_identity_sha256"),
        "train_micro_steps_per_epoch": config.get("train_micro_steps_per_epoch"),
        "train_optimizer_steps_per_epoch": config.get(
            "train_optimizer_steps_per_epoch"
        ),
        "validation_steps_per_epoch": config.get("validation_steps_per_epoch"),
        "external_final_isolation": config.get("external_final_isolation"),
        "wandb_service": {
            key: wandb.get(key) for key in ("entity", "project", "mode", "resume")
        },
    }


def _prospective_contract(args: argparse.Namespace) -> dict[str, Any]:
    source = engine.source_identity(engine.REPO_ROOT)
    protocol = engine.mps_protocol_contract()
    manifest = engine.load_canonical_manifest(args.manifest)
    content = engine.validate_canonical_content(
        manifest["identity"],
        args.train_dcm_dir,
        args.train_mask_dir,
        args.val_dcm_dir,
        args.val_mask_dir,
    )
    baseline = engine.load_pinned_baseline(
        args.baseline_score,
        args.baseline_run_record,
        manifest["identity"]["validation"],
    )
    pretrained = engine.validate_pretrained(args.pretrained)
    dependency = engine.validate_mps_runtime_dependencies(engine.MPS_LOCK_PATH)
    dependency["lock_sha256"] = sha256_file(engine.MPS_LOCK_PATH)
    config = engine._build_config(
        args,
        source,
        protocol,
        manifest,
        content,
        pretrained,
        baseline,
        dependency,
        canonical_sha256(manifest["identity"]),
        "prospective_resource_evidence",
    )
    return _normalized_config_contract(config)


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
    if len(epochs) < 5:
        raise ValueError("convergence decision requires at least five epochs")
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
    path = _safe_path(path, "health approval")
    approval_sha256 = sha256_file(path)
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
        "source_delta_path",
        "source_delta_sha256",
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
        or payload["health_attempt_id"] == expected_attempt_id
        or payload["external_final_accessed"] is not False
    ):
        raise ValueError("health approval is not bound to this attempt")
    for prefix in ("health_run_record", "health_artifact_index"):
        evidence = _safe_path(payload[f"{prefix}_path"], prefix)
        if (
            evidence.is_symlink()
            or sha256_file(evidence) != payload[f"{prefix}_sha256"]
        ):
            raise ValueError(f"{prefix} bytes do not match approval")
    run_record = _read_json_object(
        _safe_path(payload["health_run_record_path"], "health run record"),
        "health run record",
    )
    index_path = _safe_path(
        payload["health_artifact_index_path"], "health artifact index"
    )
    index = _read_json_object(index_path, "health artifact index")
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
    artifact_paths = _indexed_health_artifacts(index_path, index)
    artifacts = index["artifacts"]
    if artifacts["run_record.json"] != payload["health_run_record_sha256"]:
        raise ValueError("health run record is not bound by artifact index")
    config = _read_json_object(artifact_paths["config.json"], "health config")
    source = _read_json_object(artifact_paths["source.json"], "health source")
    if config.get("source") != source:
        raise ValueError("health source artifact differs from health config")
    source_delta = _validate_source_delta(
        Path(payload["source_delta_path"]),
        str(payload["source_delta_sha256"]),
        source,
    )
    payload["health_checkpoint_path"] = str(
        artifact_paths["best_checkpoint.pth"].resolve()
    )
    payload["health_checkpoint_sha256"] = artifacts["best_checkpoint.pth"]
    payload["health_contract"] = _normalized_config_contract(config)
    payload["health_contract"]["source"] = source_delta["convergence_source"]
    payload["approval_sha256"] = approval_sha256
    payload["source_delta"] = source_delta
    return payload


def validate_convergence_contract(
    args: argparse.Namespace, approval: Mapping[str, Any]
) -> dict[str, Any]:
    current = _prospective_contract(args)
    if current != approval.get("health_contract"):
        raise ValueError(
            "prospective convergence contract differs from approved health"
        )
    health_checkpoint = Path(str(approval["health_checkpoint_path"])).resolve()
    if args.pretrained.resolve() == health_checkpoint:
        raise ValueError("convergence must not initialize from the health checkpoint")
    if sha256_file(args.pretrained) == approval["health_checkpoint_sha256"]:
        raise ValueError("convergence pretrained bytes equal the health checkpoint")
    return current


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
        "forbidden_initialization": {
            "path": approval["health_checkpoint_path"],
            "sha256": approval["health_checkpoint_sha256"],
        },
        "gates": [
            {
                "type": "decision_receipt",
                "path": str(args.health_approval.resolve()),
                "sha256": sha256_file(args.health_approval),
                "expected_attempt_id": approval["health_attempt_id"],
                "allowed_to": [args.attempt_id],
            }
        ],
        "source_delta_sha256": approval["source_delta_sha256"],
        "wandb": {
            "entity": engine.WANDB_ENTITY,
            "project": engine.WANDB_PROJECT,
            "id": args.attempt_id,
            "resume": "never",
            "mode": "online",
            "name": args.wandb_run_name,
            "group": "task1-teammate-l05-veto-mps-convergence",
            "job_type": "reviewed-mps-convergence-50e",
        },
        "external_final_accessed": False,
    }


def _queue_receipt_path(args: argparse.Namespace) -> Path:
    return _safe_path(
        args.artifact_root / "queue_specs" / f"{args.attempt_id}.json",
        "queue spec receipt",
        require_absolute=False,
    )


def _bind_queue_receipt(
    args: argparse.Namespace, spec: Mapping[str, Any], *, execute: bool
) -> tuple[Path, str]:
    path = _queue_receipt_path(args)
    if execute:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                "execute requires the immutable reviewed queue spec"
            )
        persisted = _read_json_object(path, "queue spec receipt")
        if persisted != dict(spec):
            raise ValueError("persisted queue spec differs from prospective execution")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_once(path, dict(spec))
    return path, sha256_file(path)


def _validate_queue_approval(
    path: Path, *, attempt_id: str, queue_spec_sha256: str
) -> str:
    resolved = _safe_path(path, "queue approval")
    digest = sha256_file(resolved)
    payload = _read_json_object(resolved, "queue approval")
    if (
        set(payload)
        != {
            "schema_version",
            "status",
            "reviewer",
            "attempt_id",
            "queue_spec_sha256",
            "external_final_accessed",
        }
        or payload.get("status") != "independently_reviewed_approved"
        or not str(payload.get("reviewer", "")).strip()
        or payload.get("attempt_id") != attempt_id
        or payload.get("queue_spec_sha256") != queue_spec_sha256
        or payload.get("external_final_accessed") is not False
    ):
        raise ValueError("queue approval does not authorize this exact spec")
    return digest


def _seal_completion(
    args: argparse.Namespace,
    approval: Mapping[str, Any],
    base_index: Mapping[str, Any],
    queue_receipt_sha256: str,
) -> dict[str, Any]:
    artifact_dir = _safe_path(
        args.artifact_root / args.attempt_id, "convergence artifact root"
    )
    if (
        set(base_index)
        != {
            "schema_version",
            "attempt_id",
            "phase",
            "execution_family",
            "status",
            "artifacts",
        }
        or base_index.get("schema_version") != 1
        or base_index.get("status") != "completed"
        or base_index.get("attempt_id") != args.attempt_id
        or base_index.get("phase") != "convergence_50e"
        or base_index.get("execution_family") != "apple_mps_resource_adjusted"
    ):
        raise ValueError("base artifact index is not the completed convergence attempt")
    index_path = artifact_dir / "artifact_index.json"
    persisted_index = _read_json_object(index_path, "base artifact index")
    if persisted_index != dict(base_index):
        raise ValueError("base artifact index bytes differ from returned index")
    artifacts = persisted_index.get("artifacts")
    expected_artifacts = set(CONVERGENCE_ARTIFACT_NAMES)
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("base artifact index does not seal the exact artifact set")
    artifact_paths: dict[str, Path] = {}
    for name in CONVERGENCE_ARTIFACT_NAMES:
        path = _safe_path(artifact_dir / name, f"base artifact {name}")
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != artifacts[name]
        ):
            raise ValueError(f"base indexed artifact bytes differ: {name}")
        artifact_paths[name] = path
    if sha256_file(args.health_approval) != approval["approval_sha256"]:
        raise ValueError("health approval changed during convergence execution")
    if sha256_file(_queue_receipt_path(args)) != queue_receipt_sha256:
        raise ValueError("queue spec changed during convergence execution")
    config = _read_json_object(artifact_paths["config.json"], "convergence config")
    run_record = _read_json_object(
        artifact_paths["run_record.json"], "convergence run record"
    )
    if (
        config.get("execution_receipt_sha256") != queue_receipt_sha256
        or run_record.get("execution_receipt_sha256") != queue_receipt_sha256
        or config.get("execution_approval_sha256") != args.execution_approval_sha256
        or run_record.get("execution_approval_sha256") != args.execution_approval_sha256
    ):
        raise ValueError("base config/run record do not bind the queue receipt")
    wandb = run_record.get("wandb_completion")
    if (
        not isinstance(wandb, dict)
        or wandb.get("state") != "finished"
        or wandb.get("identity_verified") is not True
        or wandb.get("verification") != "wandb_api_post_finish"
        or tuple(
            wandb.get(key)
            for key in ("entity", "project", "id", "name", "group", "job_type")
        )
        != (
            engine.WANDB_ENTITY,
            engine.WANDB_PROJECT,
            args.attempt_id,
            args.wandb_run_name,
            engine.WANDB_GROUP,
            engine.WANDB_JOB_TYPE,
        )
    ):
        raise ValueError("durable W&B completion identity is invalid")
    continuation = engine.torch.load(
        artifact_paths[CONTINUATION_CHECKPOINT], map_location="cpu", weights_only=False
    )
    required_continuation = {
        "model",
        "optimizer",
        "scheduler",
        "completed_epochs",
        "global_optimizer_updates",
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "mps_rng_state",
    }
    scheduler = (
        continuation.get("scheduler") if isinstance(continuation, dict) else None
    )
    if (
        not isinstance(continuation, dict)
        or set(continuation) != required_continuation
        or not isinstance(continuation.get("model"), dict)
        or not isinstance(continuation.get("optimizer"), dict)
        or not isinstance(scheduler, dict)
        or scheduler.get("kind") != "cosine"
        or scheduler.get("warmup_epochs") != 5
        or scheduler.get("initial_lr") != 5e-5
        or scheduler.get("next_epoch") != EPOCHS
        or not isinstance(scheduler.get("last_learning_rates"), list)
        or not scheduler["last_learning_rates"]
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in scheduler["last_learning_rates"]
        )
        or not isinstance(continuation.get("python_rng_state"), tuple)
        or not isinstance(continuation.get("numpy_rng_state"), tuple)
        or not isinstance(continuation.get("torch_rng_state"), engine.torch.Tensor)
        or not isinstance(continuation.get("mps_rng_state"), engine.torch.Tensor)
        or continuation.get("completed_epochs") != EPOCHS
        or continuation.get("global_optimizer_updates")
        != EPOCHS * engine.TRAIN_OPTIMIZER_STEPS
    ):
        raise ValueError("epoch-50 continuation state is incomplete")
    epochs_payload = _read_json_object(artifact_paths["epochs.json"], "epochs")
    decision = convergence_decision(epochs_payload["epochs"], health_gate_passed=True)
    if decision["completed_epochs"] != EPOCHS or decision["final_epoch"] != EPOCHS:
        raise ValueError("completion sealing requires exact contiguous 50 epochs")
    write_json_once(artifact_dir / "convergence_decision.json", decision)
    final = {
        "schema_version": 2,
        "attempt_id": args.attempt_id,
        "phase": "convergence_50e",
        "status": "completed_pending_independent_review",
        "base_artifact_index_sha256": sha256_file(index_path),
        "health_approval_sha256": approval["approval_sha256"],
        "queue_spec_sha256": queue_receipt_sha256,
        "queue_approval_sha256": args.execution_approval_sha256,
        "config_sha256": artifacts["config.json"],
        "run_record_sha256": artifacts["run_record.json"],
        "best_checkpoint_sha256": artifacts["best_checkpoint.pth"],
        "continuation_checkpoint_sha256": artifacts[CONTINUATION_CHECKPOINT],
        "convergence_decision_sha256": sha256_file(
            artifact_dir / "convergence_decision.json"
        ),
        "auto_launched_150": False,
        "external_final_accessed": False,
    }
    write_json_once(artifact_dir / "convergence_index.json", final)
    handoff = {
        "schema_version": 3,
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
        "base_artifact_index_sha256": final["base_artifact_index_sha256"],
        "convergence_index_sha256": sha256_file(
            artifact_dir / "convergence_index.json"
        ),
        "config_sha256": final["config_sha256"],
        "best_checkpoint_sha256": final["best_checkpoint_sha256"],
        "continuation_checkpoint_sha256": final["continuation_checkpoint_sha256"],
        "initialization_policy": "fresh_from_reviewed_pretrained",
        "external_final_accessed": False,
        "launch_eligible": False,
    }
    write_json_once(artifact_dir / "issue93_champion_handoff.pending.json", handoff)
    return final


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--health-approval", type=Path, required=True)
    parser.add_argument("--queue-approval", type=Path)
    extra, _ = parser.parse_known_args(raw)
    engine_argv = list(raw)
    for option in ("--health-approval", "--queue-approval"):
        if option in engine_argv:
            index = engine_argv.index(option)
            del engine_argv[index : index + 2]
    args = engine.parse_args(engine_argv)
    args.health_approval = _safe_path(extra.health_approval, "health approval")
    args.queue_approval = (
        _safe_path(extra.queue_approval, "queue approval")
        if extra.queue_approval is not None
        else None
    )
    raw_spec = list(raw)
    if "--queue-approval" in raw_spec:
        index = raw_spec.index("--queue-approval")
        del raw_spec[index : index + 2]
    args.raw_argv = [item for item in raw_spec if item != "--execute"]
    health_default = f"task1-teammate-l05-veto-mps-health-{args.attempt_id}"
    if args.wandb_run_name == health_default:
        args.wandb_run_name = (
            f"task1-teammate-l05-veto-mps-convergence-50e-{args.attempt_id}"
        )
    return args


def _write_convergence_failure(
    args: argparse.Namespace, error: BaseException, stage: str
) -> None:
    artifact_dir = args.artifact_root / args.attempt_id
    if not artifact_dir.is_dir():
        return
    receipt = {
        "schema_version": 1,
        "attempt_id": args.attempt_id,
        "phase": "convergence_50e_sealing",
        "status": "failed",
        "stage": stage,
        "error_type": type(error).__name__,
        "signal": error.signal_name
        if isinstance(error, engine.RunInterrupted)
        else None,
        "external_final_accessed": False,
    }
    path = artifact_dir / "convergence_failure.json"
    if not path.exists():
        write_json_once(path, receipt)


def run(args: argparse.Namespace) -> dict[str, Any]:
    approval = validate_health_approval(args.health_approval, args.attempt_id)
    validate_convergence_contract(args, approval)
    spec = queue_spec(args, approval)
    queue_path, queue_sha256 = _bind_queue_receipt(
        args, spec, execute=bool(args.execute)
    )
    if not args.execute:
        return {
            "status": "dry_run",
            "review_required": True,
            "protocol": "fresh_50e",
            "queue_spec": spec,
            "queue_spec_path": str(queue_path),
            "queue_spec_sha256": queue_sha256,
            "launch_approved": False,
        }
    if args.queue_approval is None:
        raise ValueError("execute requires an independent exact queue approval")
    args.execution_approval_sha256 = _validate_queue_approval(
        args.queue_approval,
        attempt_id=args.attempt_id,
        queue_spec_sha256=queue_sha256,
    )
    engine.EPOCHS = EPOCHS
    engine.PHASE = "convergence_50e"
    engine.ISSUE_URL = ISSUE_URL
    engine.WANDB_GROUP = "task1-teammate-l05-veto-mps-convergence"
    engine.WANDB_JOB_TYPE = "reviewed-mps-convergence-50e"
    engine.MPS_ARTIFACT_NAMES = CONVERGENCE_ARTIFACT_NAMES
    args.execution_receipt_sha256 = queue_sha256
    args.require_wandb_completion_verification = True
    previous_handlers = engine._install_interrupt_handlers()
    try:
        base = engine.run(args)
        return _seal_completion(args, approval, base, queue_sha256)
    except BaseException as error:
        _write_convergence_failure(args, error, "engine_or_convergence_finalization")
        raise
    finally:
        engine._restore_interrupt_handlers(previous_handlers)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
