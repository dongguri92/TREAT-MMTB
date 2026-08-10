"""Review-gated teammate EVA-X lambda=0.5 + combo-veto reproduction.

Dry-run is the default. Scored execution requires an explicit reviewer and
immutable canonical dataset, baseline, dependency, source, and W&B identities.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

import datasets
from models import modeltype
from reproduction import (
    CLS_THRESHOLD,
    DATASET_SCOPE,
    EXPECTED_CUDA_VERSION,
    EXPECTED_TRAIN_CASES,
    EXPECTED_VALIDATION_CASES,
    MIN_PIXELS,
    VETO_THRESHOLD,
    WANDB_ENTITY,
    WANDB_PROJECT,
    aggregate_native_cases,
    build_paired_regression,
    canonical_sha256,
    load_canonical_manifest,
    load_pinned_baseline,
    read_json,
    reject_external_final_path,
    resolve_non_external_path,
    sha256_file,
    source_identity,
    validate_canonical_content,
    validate_coverage,
    validate_dataset_identity,
    validate_epoch_evidence,
    validate_pretrained,
    validate_runtime_dependencies,
    write_json_once,
)
from training import fit

REPO_ROOT = Path(__file__).resolve().parent
LOCK_PATH = REPO_ROOT / "requirements-reproduction.lock"
ISSUE_URL = "https://github.com/choco9966/TREAT-MMTB-2026/issues/95"
PHASE_EPOCHS = {"health": 5, "convergence": 50}
ARTIFACT_NAMES = (
    "config.json",
    "source.json",
    "case_identity.json",
    "dependency.lock",
    "epochs.json",
    "best_epoch_cases.json",
    "regression.json",
    "score.json",
    "best_checkpoint.pth",
    "run_record.json",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def protocol_contract(phase: str) -> dict[str, Any]:
    return {
        "phase": phase,
        "epochs": PHASE_EPOCHS[phase],
        "model": "evax_seg",
        "variant": "small",
        "channels": 1,
        "target_size": 1024,
        "physical_batch_size": 8,
        "effective_batch_size": 8,
        "lambda_cls": 0.5,
        "optimizer": "adamw",
        "initial_lr": 5e-5,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "loss": "dicece",
        "crop_frac": 0.15,
        "clahe_clip": 2.0,
        "seed": 42,
        "checkpoint_selection": "max_0.7_accuracy_plus_0.3_dice",
        "validation_space": "native",
        "inference": {
            "detection": "combo",
            "cls_threshold": CLS_THRESHOLD,
            "t_veto": VETO_THRESHOLD,
            "min_pixels": MIN_PIXELS,
        },
    }


def build_config(
    args: argparse.Namespace,
    source: dict[str, str],
    pretrained: dict[str, Any],
    device: torch.device | str,
    manifest: dict[str, Any] | None = None,
    content: dict[str, str] | None = None,
    baseline: dict[str, Any] | None = None,
    dependency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = protocol_contract(args.phase)
    return {
        "schema_version": 2,
        "issue_url": ISSUE_URL,
        "attempt_id": args.attempt_id,
        "reviewed_by": args.reviewed_by,
        "fresh_attempt": True,
        "resume_policy": "never",
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "num_workers": args.num_workers,
        "device": str(device),
        "source": source,
        "pretrained": pretrained,
        "dataset_scope": DATASET_SCOPE,
        "manifest": manifest,
        "content": content,
        "baseline": None
        if baseline is None
        else {key: value for key, value in baseline.items() if key != "cases"},
        "dependency": dependency,
        "wandb": {
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "mode": "online",
            "id": args.attempt_id,
            "name": args.wandb_run_name,
            "resume": "never",
        },
        "external_final_isolation": {
            "dataset_scope": DATASET_SCOPE,
            "canonical_bytes_verified": content is not None,
            "external_final_test_untouched": True,
        },
    }


def _start_wandb(config: dict[str, Any]) -> Any:
    import wandb

    wandb_config = config["wandb"]
    public_config = _wandb_public_config(config)
    if canonical_sha256(public_config) != wandb_config["config_sha256"]:
        raise RuntimeError("W&B public config hash differs from the sealed config")
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id=wandb_config["id"],
        name=wandb_config["name"],
        mode="online",
        resume="never",
        group="task1-teammate-l05-veto-reproduction",
        job_type=config["protocol"]["phase"],
        config=public_config,
    )
    run.define_metric("train/global_step")
    run.define_metric("train/*", step_metric="train/global_step")
    run.define_metric("validation/global_step")
    run.define_metric("validation/*", step_metric="validation/global_step")
    run.define_metric("epoch")
    run.define_metric("epoch/*", step_metric="epoch")
    return run


def _wandb_public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return aggregate/hash-only W&B config without case identities or paths."""
    manifest = config.get("manifest")
    baseline = config.get("baseline")
    dependency = config.get("dependency")
    content = config.get("content")
    return {
        "schema_version": config.get("schema_version"),
        "issue_url": config.get("issue_url"),
        "attempt_id": config.get("attempt_id"),
        "protocol": config.get("protocol"),
        "protocol_sha256": config.get("protocol_sha256"),
        "source": config.get("source"),
        "pretrained": config.get("pretrained"),
        "dataset": {
            "scope": config.get("dataset_scope"),
            "train_case_count": EXPECTED_TRAIN_CASES,
            "validation_case_count": EXPECTED_VALIDATION_CASES,
            "manifest_sha256": manifest.get("manifest_sha256")
            if isinstance(manifest, dict)
            else None,
            "case_identity_sha256": config.get("case_identity_sha256"),
            "combined_content_sha256": content.get("combined_content_sha256")
            if isinstance(content, dict)
            else None,
        },
        "baseline": {
            "baseline_id": baseline.get("baseline_id"),
            "score_sha256": baseline.get("score_sha256"),
            "run_record_sha256": baseline.get("run_record_sha256"),
        }
        if isinstance(baseline, dict)
        else None,
        "dependency": {
            "lock_sha256": dependency.get("lock_sha256"),
            "python": dependency.get("python"),
            "torch": dependency.get("torch"),
            "torchvision": dependency.get("distributions", {}).get("torchvision"),
            "cuda": dependency.get("cuda"),
        }
        if isinstance(dependency, dict)
        else None,
        "external_final_isolation": config.get("external_final_isolation"),
    }


def _relative_artifact_hashes(artifact_dir: Path) -> dict[str, str]:
    return {name: sha256_file(artifact_dir / name) for name in ARTIFACT_NAMES}


def _validate_health_index(
    index_path: Path,
    current_config: dict[str, Any],
    expected_source: dict[str, str],
) -> dict[str, Any]:
    index = read_json(index_path)
    if index.get("status") != "completed" or index.get("phase") != "health":
        raise ValueError("health artifact index is not a completed health attempt")
    hashes = index.get("artifacts")
    if not isinstance(hashes, dict) or set(hashes) != set(ARTIFACT_NAMES):
        raise ValueError("health artifact index coverage is incomplete")
    root = index_path.parent.resolve()
    for name, expected_hash in hashes.items():
        if Path(name).name != name or not isinstance(expected_hash, str):
            raise ValueError("health artifact names/hashes must be sealed and relative")
        if sha256_file(root / name) != expected_hash:
            raise ValueError(f"health artifact hash mismatch: {name}")
    config = read_json(root / "config.json")
    source = read_json(root / "source.json")
    identity = read_json(root / "case_identity.json")
    epochs = read_json(root / "epochs.json")
    best_cases = read_json(root / "best_epoch_cases.json")
    regression = read_json(root / "regression.json")
    score = read_json(root / "score.json")
    record = read_json(root / "run_record.json")
    if source != expected_source:
        raise ValueError("source changed between health and convergence")
    stable_fields = (
        "pretrained",
        "dataset_scope",
        "manifest",
        "content",
        "baseline",
        "dependency",
    )
    if any(config.get(field) != current_config.get(field) for field in stable_fields):
        raise ValueError("health immutable input identity differs from convergence")
    expected_health = protocol_contract("health")
    expected_convergence = protocol_contract("convergence")
    health_semantics = {
        key: value
        for key, value in expected_health.items()
        if key not in {"phase", "epochs"}
    }
    convergence_semantics = {
        key: value
        for key, value in expected_convergence.items()
        if key not in {"phase", "epochs"}
    }
    if (
        current_config.get("protocol") != expected_convergence
        or current_config.get("protocol_sha256")
        != canonical_sha256(expected_convergence)
        or config.get("protocol") != expected_health
        or config.get("protocol_sha256") != canonical_sha256(expected_health)
        or health_semantics != convergence_semantics
        or canonical_sha256(identity) != current_config.get("case_identity_sha256")
        or identity.get("dataset_scope") != DATASET_SCOPE
        or len(identity.get("train", [])) != EXPECTED_TRAIN_CASES
        or len(identity.get("validation", [])) != EXPECTED_VALIDATION_CASES
        or not isinstance(epochs.get("epochs"), list)
    ):
        raise ValueError("health protocol or case identity is invalid")
    validate_epoch_evidence(epochs["epochs"], 5, 55)
    cases = best_cases.get("cases")
    if not isinstance(cases, list):
        raise TypeError("health best-epoch case evidence is missing")
    if not all(isinstance(case, dict) and "case_id" in case for case in cases):
        raise TypeError("health best-epoch cases must contain case identities")
    validate_coverage(identity["validation"], [str(case["case_id"]) for case in cases])
    case_metrics = aggregate_native_cases(cases, identity["validation"])
    selected_epoch = score.get("selected_epoch")
    if not isinstance(selected_epoch, int) or not 1 <= selected_epoch <= 5:
        raise ValueError("health selected epoch is outside the completed phase")
    selected_evidence = epochs["epochs"][selected_epoch - 1]
    selected_metrics = {
        key: selected_evidence[key]
        for key in (
            "classification_accuracy",
            "dice",
            "weighted_composite",
            "coverage",
        )
    }
    if (
        score.get("attempt_id") != config.get("attempt_id")
        or score.get("phase") != "health"
        or score.get("dataset_scope") != DATASET_SCOPE
        or score.get("external_final_test_untouched") is not True
        or score.get("metrics") != selected_metrics
        or score.get("metrics")
        != {
            key: case_metrics[key]
            for key in (
                "classification_accuracy",
                "dice",
                "weighted_composite",
                "coverage",
            )
        }
        or score.get("decision") != expected_health["inference"]
        or score.get("regression_sha256") != sha256_file(root / "regression.json")
        or regression.get("case_count") != EXPECTED_VALIDATION_CASES
    ):
        raise ValueError("health score or regression semantics are invalid")
    regression_cases = regression.get("cases")
    if not isinstance(regression_cases, list) or not all(
        isinstance(case, dict) and "case_id" in case for case in regression_cases
    ):
        raise TypeError("health regression per-case evidence is missing")
    validate_coverage(
        identity["validation"],
        [str(case["case_id"]) for case in regression_cases],
    )
    wandb_evidence = record.get("wandb")
    if (
        record.get("status") != "completed"
        or record.get("attempt_id") != config.get("attempt_id")
        or record.get("phase") != "health"
        or record.get("reviewed_by") != config.get("reviewed_by")
        or not str(record.get("reviewed_by", "")).strip()
        or record.get("completed_epochs") != 5
        or record.get("completed_train_steps") != 275
        or record.get("train_steps_per_epoch") != 55
        or record.get("validation_steps_per_epoch") != 111
        or record.get("requested_epochs") != 5
        or record.get("selected_epoch") != selected_epoch
        or record.get("metrics") != selected_metrics
        or record.get("wandb_finished") is not True
        or not isinstance(wandb_evidence, dict)
        or wandb_evidence.get("id") != config["attempt_id"]
        or wandb_evidence.get("entity") != WANDB_ENTITY
        or wandb_evidence.get("project") != WANDB_PROJECT
        or wandb_evidence.get("mode") != "online"
        or wandb_evidence.get("resume") != "never"
        or wandb_evidence.get("config_sha256")
        != config.get("wandb", {}).get("config_sha256")
        or not str(wandb_evidence.get("url", "")).startswith("https://")
    ):
        raise ValueError(
            "health completion, reviewer, step, or W&B evidence is invalid"
        )
    return {
        "attempt_id": config["attempt_id"],
        "artifact_index_sha256": sha256_file(index_path),
        "phase_transition_sha256": canonical_sha256(
            {"health": expected_health, "convergence": expected_convergence}
        ),
        "reviewed_by": record["reviewed_by"],
    }


def _safe_failure(
    artifact_dir: Path, args: argparse.Namespace, stage: str, error: BaseException
) -> None:
    path = artifact_dir / "failure.json"
    if artifact_dir.exists() and not path.exists():
        write_json_once(
            path,
            {
                "schema_version": 1,
                "attempt_id": args.attempt_id,
                "phase": args.phase,
                "stage": stage,
                "error_type": type(error).__name__,
                "external_final_test_untouched": True,
            },
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    bound_paths = (
        "manifest",
        "baseline_score",
        "baseline_run_record",
        "pretrained",
        "train_dcm_dir",
        "train_mask_dir",
        "val_dcm_dir",
        "val_mask_dir",
        "artifact_root",
        "health_artifact_index",
    )
    for label in bound_paths:
        value = getattr(args, label, None)
        if value is not None:
            reject_external_final_path(value, label)
    for label in bound_paths:
        value = getattr(args, label, None)
        if value is not None:
            setattr(args, label, resolve_non_external_path(value, label))
    artifact_dir = args.artifact_root / args.attempt_id
    if artifact_dir.exists():
        raise FileExistsError(
            "attempt directory already exists; every retry must be fresh"
        )
    if not args.execute:
        return {
            "status": "dry_run",
            "phase": args.phase,
            "epochs": PHASE_EPOCHS[args.phase],
            "attempt_id": args.attempt_id,
            "review_required": True,
        }
    if not str(args.reviewed_by or "").strip():
        raise ValueError("--execute requires a non-empty --reviewed-by attestation")

    artifact_dir.mkdir(parents=True, exist_ok=False)
    wandb_run: Any = None
    stage = "source_identity"
    try:
        source = source_identity(REPO_ROOT)
        stage = "canonical_inputs"
        manifest = load_canonical_manifest(args.manifest)
        content = validate_canonical_content(
            manifest["identity"],
            args.train_dcm_dir,
            args.train_mask_dir,
            args.val_dcm_dir,
            args.val_mask_dir,
        )
        baseline = load_pinned_baseline(
            args.baseline_score,
            args.baseline_run_record,
            manifest["identity"]["validation"],
        )
        pretrained = validate_pretrained(args.pretrained)
        stage = "dependency_identity"
        dependency = validate_runtime_dependencies(LOCK_PATH)
        dependency["lock_sha256"] = sha256_file(LOCK_PATH)
        device = _device(args.device)
        if device.type != "cuda":
            raise RuntimeError("scored reproduction requires a CUDA device")
        if torch.version.cuda != EXPECTED_CUDA_VERSION:
            raise RuntimeError("runtime CUDA differs from the verified CUDA 11.8 build")
        if not hasattr(torch.serialization, "safe_globals"):
            raise RuntimeError("runtime torch lacks the required safe_globals API")
        dependency.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else str(device)
                ),
            }
        )
        config = build_config(
            args,
            source,
            pretrained,
            device,
            manifest,
            content,
            baseline,
            dependency,
        )
        stage = "dataset_setup"
        datasets.TRAIN_DCM_DIR = str(args.train_dcm_dir)
        datasets.TRAIN_MASK_DIR = str(args.train_mask_dir)
        datasets.VAL_DCM_DIR = str(args.val_dcm_dir)
        datasets.VAL_MASK_DIR = str(args.val_mask_dir)
        _set_seed(config["protocol"]["seed"])
        train_loader, val_loader = datasets.dataloader(
            batch_size=8,
            target_size=1024,
            clahe_clip=2.0,
            num_workers=args.num_workers,
            seed=42,
            crop_frac=0.15,
        )
        identity, identity_sha256 = validate_dataset_identity(
            train_loader, val_loader, manifest["identity"]
        )
        if len(train_loader) != 55 or len(val_loader) != 111:
            raise ValueError(
                "loader step counts differ from the sealed 55/111 contract"
            )
        config.update(
            {
                "case_identity_sha256": identity_sha256,
                "train_steps_per_epoch": 55,
                "validation_steps_per_epoch": 111,
            }
        )
        config["wandb"]["config_sha256"] = canonical_sha256(
            _wandb_public_config(config)
        )
        stage = "health_gate"
        health_gate = None
        if args.phase == "convergence":
            if args.health_artifact_index is None:
                raise ValueError("convergence requires --health-artifact-index")
            health_gate = _validate_health_index(
                args.health_artifact_index, config, source
            )
        stage = "model_setup"
        model = modeltype(
            "evax_seg",
            in_channels=1,
            img_size=1024,
            pretrained_path=str(args.pretrained),
            variant="small",
        ).to(device)
        config["parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        write_json_once(artifact_dir / "config.json", config)
        write_json_once(artifact_dir / "source.json", source)
        write_json_once(artifact_dir / "case_identity.json", identity)
        shutil.copyfile(LOCK_PATH, artifact_dir / "dependency.lock")
        stage = "wandb_start"
        wandb_run = _start_wandb(config)
        if str(wandb_run.id) != args.attempt_id:
            raise TypeError("W&B did not preserve the sealed attempt ID")
        started = time.perf_counter()
        checkpoint_path = artifact_dir / "best_checkpoint.pth"
        stage = "training"
        details = cast(
            dict[str, Any],
            fit(
                model,
                train_loader,
                val_loader,
                device,
                max_epochs=PHASE_EPOCHS[args.phase],
                lambda_cls=0.5,
                initial_lr=5e-5,
                ckpt_path=str(checkpoint_path),
                patience=None,
                batch_dice=True,
                optimizer_name="adamw",
                scheduler="cosine",
                warmup_epochs=5,
                loss_name="dicece",
                wandb_run=wandb_run,
                reproduction_expected_ids=identity["validation"],
                return_details=True,
            ),
        )
        stage = "artifact_finalization"
        epochs = details["epochs"]
        validate_epoch_evidence(epochs, PHASE_EPOCHS[args.phase], 55)
        if details["completed_train_steps"] != PHASE_EPOCHS[args.phase] * 55:
            raise RuntimeError("completed optimizer-step evidence is incomplete")
        best = details["best_native_metrics"]
        if not isinstance(best, dict):
            raise TypeError("native validation did not produce best-epoch metrics")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["reproduction"] = {
            "attempt_id": args.attempt_id,
            "selected_epoch": details["best_epoch"],
            "source": source,
            "config_sha256": sha256_file(artifact_dir / "config.json"),
            "pretrained_sha256": pretrained["sha256"],
            "case_identity_sha256": identity_sha256,
        }
        torch.save(checkpoint, checkpoint_path)
        write_json_once(artifact_dir / "epochs.json", {"epochs": epochs})
        write_json_once(
            artifact_dir / "best_epoch_cases.json", {"cases": best["cases"]}
        )
        regression = build_paired_regression(best["cases"], baseline["cases"])
        write_json_once(artifact_dir / "regression.json", regression)
        score = {
            "schema_version": 2,
            "attempt_id": args.attempt_id,
            "phase": args.phase,
            "dataset_scope": DATASET_SCOPE,
            "external_final_test_untouched": True,
            "selected_epoch": details["best_epoch"],
            "metrics": {
                key: best[key]
                for key in (
                    "classification_accuracy",
                    "dice",
                    "weighted_composite",
                    "coverage",
                )
            },
            "decision": config["protocol"]["inference"],
            "baseline": config["baseline"],
            "regression_sha256": sha256_file(artifact_dir / "regression.json"),
        }
        write_json_once(artifact_dir / "score.json", score)
        run_url = wandb_run.get_url()
        if (
            not str(run_url).startswith("https://")
            or str(wandb_run.id) != args.attempt_id
            or str(wandb_run.entity) != WANDB_ENTITY
            or str(wandb_run.project) != WANDB_PROJECT
        ):
            raise RuntimeError(
                "W&B run identity or URL differs from the sealed contract"
            )
        for name, value in score["metrics"].items():
            if name != "coverage":
                wandb_run.summary[f"best/{name}"] = value
        wandb_run.summary["best/epoch"] = details["best_epoch"]
        wandb_run.summary["regression/fixed"] = regression["taxonomy_counts"]["fixed"]
        wandb_run.summary["regression/regressed"] = regression["taxonomy_counts"][
            "regressed"
        ]
        wandb_run.finish()
        wandb_run = None
        run_record = {
            "schema_version": 2,
            "attempt_id": args.attempt_id,
            "phase": args.phase,
            "status": "completed",
            "reviewed_by": args.reviewed_by,
            "external_final_test_untouched": True,
            "requested_epochs": PHASE_EPOCHS[args.phase],
            "completed_epochs": details["completed_epochs"],
            "completed_train_steps": details["completed_train_steps"],
            "train_steps_per_epoch": 55,
            "validation_steps_per_epoch": 111,
            "selected_epoch": details["best_epoch"],
            "runtime_seconds": time.perf_counter() - started,
            "metrics": score["metrics"],
            "wandb_finished": True,
            "wandb": {**config["wandb"], "url": run_url},
            "health_gate": health_gate,
            "artifacts": list(ARTIFACT_NAMES),
        }
        write_json_once(artifact_dir / "run_record.json", run_record)
        artifact_index = {
            "schema_version": 2,
            "attempt_id": args.attempt_id,
            "phase": args.phase,
            "status": "completed",
            "artifacts": _relative_artifact_hashes(artifact_dir),
        }
        write_json_once(artifact_dir / "artifact_index.json", artifact_index)
        return artifact_index
    except BaseException as error:
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=1)
            except BaseException as finish_error:  # noqa: BLE001
                _safe_failure(artifact_dir, args, "wandb_failure_finish", finish_error)
        _safe_failure(artifact_dir, args, stage, error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASE_EPOCHS), required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/reproduction")
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--baseline-run-record", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--train-dcm-dir", type=Path, required=True)
    parser.add_argument("--train-mask-dir", type=Path, required=True)
    parser.add_argument("--val-dcm-dir", type=Path, required=True)
    parser.add_argument("--val-mask-dir", type=Path, required=True)
    parser.add_argument("--health-artifact-index", type=Path)
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    parser.add_argument("--reviewed-by")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.attempt_id) is None:
        parser.error("--attempt-id must be a single safe path component")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    path_names = (
        "artifact_root",
        "manifest",
        "baseline_score",
        "baseline_run_record",
        "pretrained",
        "train_dcm_dir",
        "train_mask_dir",
        "val_dcm_dir",
        "val_mask_dir",
        "health_artifact_index",
    )
    for name in path_names:
        value = getattr(args, name)
        if value is not None:
            reject_external_final_path(value, name)
    for name in path_names:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve_non_external_path(value, name))
    args.wandb_run_name = args.wandb_run_name or (
        f"task1-teammate-l05-veto-{args.phase}-{args.attempt_id}"
    )
    return args


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
