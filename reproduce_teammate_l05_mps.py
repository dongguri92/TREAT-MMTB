"""Review-gated Apple-MPS health amendment for Issue #106.

This runner is separate from the sealed Linux/CUDA reproduction. Dry-run is
the default. Reviewed execution is bootstrap-only and requires a disposable
1024x1024 train/validation/next-epoch resource gate before W&B can start.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

import datasets
from models import modeltype
from reproduction import (
    CLS_THRESHOLD,
    DATASET_SCOPE,
    EXPECTED_TRAIN_CASES,
    EXPECTED_VALIDATION_CASES,
    MIN_PIXELS,
    VETO_THRESHOLD,
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_paired_regression,
    canonical_sha256,
    load_canonical_manifest,
    load_pinned_baseline,
    reject_external_final_path,
    resolve_non_external_path,
    sha256_file,
    source_identity,
    validate_canonical_content,
    validate_dataset_identity,
    validate_epoch_evidence,
    validate_mps_allocator_environment,
    validate_mps_runtime_dependencies,
    validate_pretrained,
    write_json_once,
)
from training import compute_lr, fit, make_optimizer
from utils import DiceCELoss

REPO_ROOT = Path(__file__).resolve().parent
MPS_LOCK_PATH = REPO_ROOT / "requirements-reproduction-mps.lock"
MPS_RESOURCE_CONTRACT_PATH = REPO_ROOT / "mps_resource_contract.json"
MPS_BOOTSTRAP_PATH = REPO_ROOT / "reproduce_teammate_l05_mps_bootstrap.py"
ISSUE_URL = "https://github.com/choco9966/TREAT-MMTB-2026/issues/106"
PARENT_ISSUE_URL = "https://github.com/choco9966/TREAT-MMTB-2026/issues/95"
EPOCHS = 5
TARGET_SIZE = 1024
PHYSICAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EFFECTIVE_BATCH_SIZE = 8
TRAIN_MICRO_STEPS = 440
TRAIN_OPTIMIZER_STEPS = 55
FEASIBILITY_OPTIMIZER_STEPS = 1
ACCEPTANCE_FIRST_EPOCH_OPTIMIZER_STEPS = 55
ACCEPTANCE_TRANSITION_OPTIMIZER_STEPS = 21
ACCEPTANCE_SOAK_OPTIMIZER_STEPS = (
    ACCEPTANCE_FIRST_EPOCH_OPTIMIZER_STEPS
    + ACCEPTANCE_TRANSITION_OPTIMIZER_STEPS
)
ACCEPTANCE_SOAK_MICRO_STEPS = (
    ACCEPTANCE_SOAK_OPTIMIZER_STEPS * GRADIENT_ACCUMULATION_STEPS
)
ACCEPTANCE_SOAK_VALIDATION_STEPS = EXPECTED_VALIDATION_CASES
MPS_ARTIFACT_NAMES = (
    "config.json",
    "source.json",
    "case_identity.json",
    "dependency.lock",
    "resource_evidence.json",
    "acceptance_soak_heartbeat.jsonl",
    "epochs.json",
    "best_epoch_cases.json",
    "regression.json",
    "score.json",
    "best_checkpoint.pth",
    "run_record.json",
)


class MPSFeasibilityFailure(RuntimeError):
    def __init__(self, evidence: dict[str, Any]):
        super().__init__("sealed 1024 MPS feasibility probe failed")
        self.evidence = evidence


class RunInterrupted(KeyboardInterrupt):
    def __init__(self, signum: int):
        self.signum = signum
        self.signal_name = signal.Signals(signum).name
        super().__init__(f"run interrupted by {self.signal_name}")


def _install_interrupt_handlers() -> dict[signal.Signals, Any]:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("MPS reproduction must run in the main thread")
    previous: dict[signal.Signals, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise RunInterrupted(signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupt)
    except BaseException:
        _restore_interrupt_handlers(previous)
        raise
    return previous


def _restore_interrupt_handlers(previous: dict[signal.Signals, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mps_protocol_contract() -> dict[str, Any]:
    return {
        "phase": "health",
        "epochs": EPOCHS,
        "execution_family": "apple_mps_resource_adjusted",
        "historical_cuda_equivalence_claimed": False,
        "model": "evax_seg",
        "variant": "small",
        "channels": 1,
        "target_size": TARGET_SIZE,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "batch_dice": True,
        "loss_accumulation_semantics": "mean_of_8_microbatch_multitask_losses",
        "train_micro_steps_per_epoch": TRAIN_MICRO_STEPS,
        "train_optimizer_steps_per_epoch": TRAIN_OPTIMIZER_STEPS,
        "dropped_train_micro_steps_per_epoch": 4,
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
        "feasibility": {
            "input_size": [TARGET_SIZE, TARGET_SIZE],
            "probe": "exact_8_microbatch_adamw_optimizer_update",
            "optimizer_updates": FEASIBILITY_OPTIMIZER_STEPS,
            "must_pass_before_wandb": True,
            "failure_policy": "sealed_resource_evidence_only",
            "automatic_512_fallback": False,
        },
        "acceptance_soak": {
            "probe": "train_validation_next_epoch_resource_lifecycle",
            "optimizer_updates": ACCEPTANCE_SOAK_OPTIMIZER_STEPS,
            "micro_steps": ACCEPTANCE_SOAK_MICRO_STEPS,
            "first_epoch_optimizer_updates": (
                ACCEPTANCE_FIRST_EPOCH_OPTIMIZER_STEPS
            ),
            "validation_resource_steps": ACCEPTANCE_SOAK_VALIDATION_STEPS,
            "next_epoch_optimizer_updates": (
                ACCEPTANCE_TRANSITION_OPTIMIZER_STEPS
            ),
            "wandb_forbidden": True,
            "must_pass_before_scientific_run": True,
            "disposable_model_discarded_before_training": True,
        },
        "inference": {
            "detection": "combo",
            "cls_threshold": CLS_THRESHOLD,
            "t_veto": VETO_THRESHOLD,
            "min_pixels": MIN_PIXELS,
        },
    }


def _memory_snapshot() -> dict[str, int | None]:
    def value(name: str) -> int | None:
        function = getattr(torch.mps, name, None)
        if not callable(function):
            return None
        try:
            return int(cast(Any, function()))
        except RuntimeError:
            return None

    return {
        "current_allocated_bytes": value("current_allocated_memory"),
        "driver_allocated_bytes": value("driver_allocated_memory"),
        "recommended_max_bytes": value("recommended_max_memory"),
    }


def _load_mps_resource_contract() -> dict[str, Any]:
    with MPS_RESOURCE_CONTRACT_PATH.open(encoding="utf-8") as stream:
        contract = json.load(stream)
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise RuntimeError("invalid MPS resource contract")
    return contract


def _validate_bootstrap_proof(expected_role: str) -> dict[str, Any]:
    encoded = os.environ.get("TREAT_MMTB_MPS_BOOTSTRAP_PROOF")
    digest = os.environ.get("TREAT_MMTB_MPS_BOOTSTRAP_PROOF_SHA256")
    if not encoded or not digest:
        raise RuntimeError("execute requires the stdlib-only MPS bootstrap")
    if canonical_sha256(json.loads(encoded)) != digest:
        raise RuntimeError("MPS bootstrap proof digest is invalid")
    proof = json.loads(encoded)
    contract = _load_mps_resource_contract()
    expected_child_argv = [
        os.environ.get("TREAT_MMTB_MPS_WORKER_PYTHON", ""),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    if (
        proof.get("schema_version") != 1
        or proof.get("role") != expected_role
        or proof.get("allocator") != contract["allocator"]
        or proof.get("host") != contract["host"]
        or proof.get("contract_sha256") != sha256_file(MPS_RESOURCE_CONTRACT_PATH)
        or proof.get("launcher_sha256") != sha256_file(MPS_BOOTSTRAP_PATH)
        or proof.get("worker_sha256") != sha256_file(Path(__file__).resolve())
        or proof.get("child_argv_sha256")
        != canonical_sha256(expected_child_argv)
        or proof.get("parent_pid") != os.getppid()
        or not re.fullmatch(r"[0-9a-f]{64}", str(proof.get("nonce", "")))
        or proof.get("torch_imported_in_bootstrap") is not False
    ):
        raise RuntimeError("MPS bootstrap proof differs from reviewed contract")
    allocator = validate_mps_allocator_environment()
    if allocator["values"] != proof["allocator"]:
        raise RuntimeError("allocator environment differs from bootstrap proof")
    return {**proof, "proof_sha256": digest}


def _supervisor_progress(event: dict[str, Any]) -> None:
    descriptor = os.environ.get("TREAT_MMTB_MPS_PROGRESS_FD")
    if descriptor is None:
        raise RuntimeError("supervised execute requires progress descriptor")
    payload = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
    os.write(int(descriptor), payload.encode("utf-8"))


def _validate_memory_headroom(snapshot: dict[str, int | None]) -> float:
    driver = snapshot.get("driver_allocated_bytes")
    recommended = snapshot.get("recommended_max_bytes")
    if (
        not isinstance(driver, int)
        or not isinstance(recommended, int)
        or recommended <= 0
    ):
        raise RuntimeError("MPS memory headroom APIs are required")
    headroom = 1.0 - (driver / recommended)
    required = float(_load_mps_resource_contract()["memory"]["minimum_headroom_ratio"])
    if headroom < required:
        raise RuntimeError("MPS memory headroom fell below reviewed minimum")
    return headroom


def _run_mps_optimizer_probe(
    loader: Any,
    device: torch.device,
    pretrained_path: Path,
    *,
    optimizer_updates: int,
    probe_name: str,
    heartbeat_path: Path | None = None,
    validation_loader: Any = None,
    validation_after_updates: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Exercise train/validation/transition MPS resources without W&B or scores."""
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("optimizer probe requires the verified MPS device")
    if optimizer_updates < 1:
        raise ValueError("optimizer probe requires at least one update")
    empty_cache = getattr(torch.mps, "empty_cache", None)
    synchronize = getattr(torch.mps, "synchronize", None)
    stage = "cache_prepare"
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    heartbeat_stream: Any = None
    batch = image = mask = cls = None
    segmentation = classification = None
    segmentation_loss = classification_loss = loss = gradient_norm = None
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "probe": probe_name,
        "target_size": TARGET_SIZE,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "requested_optimizer_updates": optimizer_updates,
        "requested_micro_steps": optimizer_updates
        * GRADIENT_ACCUMULATION_STEPS,
        "wandb_started": False,
        "updates": [],
        "validation": [],
    }

    def emit(record: dict[str, Any]) -> None:
        if heartbeat_stream is not None:
            heartbeat_stream.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
            heartbeat_stream.flush()
            os.fsync(heartbeat_stream.fileno())
        if progress_callback is not None:
            progress_callback(record)

    try:
        if heartbeat_path is not None:
            heartbeat_stream = heartbeat_path.open("x", encoding="utf-8")
        if callable(empty_cache):
            empty_cache()
        evidence["memory_before"] = _memory_snapshot()
        evidence["memory_before_headroom_ratio"] = _validate_memory_headroom(
            evidence["memory_before"]
        )
        stage = "model_construction"
        model = cast(
            torch.nn.Module,
            modeltype(
                "evax_seg",
                in_channels=1,
                img_size=TARGET_SIZE,
                pretrained_path=str(pretrained_path),
                variant="small",
            ),
        )
        stage = "device_transfer"
        model = cast(torch.nn.Module, model.to(device))
        model.train()
        cast(Any, model).return_cls = True
        stage = "optimizer_construction"
        optimizer = make_optimizer(model, initial_lr=5e-5, optimizer_name="adamw")
        probe_lr = compute_lr(0, EPOCHS, 5e-5, "cosine", 5)
        for group in optimizer.param_groups:
            group["lr"] = probe_lr
        evidence["optimizer"] = {
            "name": "adamw",
            "learning_rate": probe_lr,
            "gradient_clip_norm": 12,
        }
        segmentation_loss_fn = DiceCELoss(batch_dice=True)
        classification_loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer.zero_grad(set_to_none=True)
        iterator = iter(loader)
        started = time.perf_counter()
        completed_micro_steps = 0
        for update_index in range(optimizer_updates):
            if (
                validation_loader is not None
                and validation_after_updates == update_index
            ):
                stage = "validation_resource_traversal"
                model.eval()
                validation_ids: list[str] = []
                with torch.no_grad():
                    for validation_index, validation_batch in enumerate(
                        validation_loader, start=1
                    ):
                        batch = validation_batch
                        validation_batch_ids = [
                            str(value) for value in batch.get("id", [])
                        ]
                        if len(validation_batch_ids) != 1:
                            raise ValueError("validation resource identity is missing")
                        validation_ids.extend(validation_batch_ids)
                        image = batch["image"].to(device)
                        mask = batch["mask"].to(device)
                        cls = batch["cls"].to(device)
                        segmentation, classification = model(image)
                        segmentation_main = (
                            segmentation[0]
                            if isinstance(segmentation, (list, tuple))
                            else segmentation
                        )
                        segmentation_loss = segmentation_loss_fn(
                            segmentation_main, mask
                        )
                        classification_loss = classification_loss_fn(
                            classification, cls
                        )
                        loss = segmentation_loss + 0.5 * classification_loss
                        if not all(
                            bool(torch.isfinite(value).item())
                            for value in (
                                segmentation_loss,
                                classification_loss,
                                loss,
                            )
                        ):
                            raise FloatingPointError(
                                "validation resource loss is non-finite"
                            )
                        _ = segmentation_main.argmax(1).cpu()
                        _ = torch.softmax(segmentation_main, dim=1)[:, 1].cpu()
                        _ = torch.sigmoid(classification).cpu()
                        if callable(synchronize):
                            synchronize()
                        memory = _memory_snapshot()
                        headroom = _validate_memory_headroom(memory)
                        validation_record = {
                            "phase": "validation_resource",
                            "validation_step": validation_index,
                            "finite_loss": True,
                            "elapsed_seconds": time.perf_counter() - started,
                            "memory": memory,
                            "headroom_ratio": headroom,
                        }
                        cast(
                            list[dict[str, Any]], evidence["validation"]
                        ).append(validation_record)
                        emit(validation_record)
                        batch = image = mask = cls = None
                        segmentation = classification = None
                        segmentation_loss = classification_loss = loss = None
                if (
                    len(validation_ids) != ACCEPTANCE_SOAK_VALIDATION_STEPS
                    or len(set(validation_ids)) != ACCEPTANCE_SOAK_VALIDATION_STEPS
                ):
                    raise ValueError(
                        "validation resource traversal is not exact 111 unique cases"
                    )
                evidence["validation_identity_sha256"] = canonical_sha256(
                    sorted(validation_ids)
                )
                evidence["completed_validation_steps"] = len(validation_ids)
                model.train()
                iterator = iter(loader)
                probe_lr = compute_lr(1, EPOCHS, 5e-5, "cosine", 5)
                for group in optimizer.param_groups:
                    group["lr"] = probe_lr
                emit(
                    {
                        "phase": "next_epoch_transition",
                        "completed_optimizer_updates": update_index,
                        "learning_rate": probe_lr,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
            update_case_ids: list[str] = []
            last_losses: dict[str, float] = {}
            for _ in range(GRADIENT_ACCUMULATION_STEPS):
                stage = "microbatch_load"
                try:
                    batch = next(iterator)
                except StopIteration as error:
                    raise ValueError(
                        "probe loader exhausted before required optimizer path"
                    ) from error
                stage = "microbatch_identity"
                batch_ids = [str(value) for value in batch.get("id", [])]
                if len(batch_ids) != PHYSICAL_BATCH_SIZE:
                    raise ValueError("probe microbatch identity is missing")
                update_case_ids.extend(batch_ids)
                stage = "microbatch_device_transfer"
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                cls = batch["cls"].to(device)
                if tuple(image.shape) != (1, 1, TARGET_SIZE, TARGET_SIZE):
                    raise ValueError("probe batch is not exact 1x1x1024x1024")
                stage = "forward"
                segmentation, classification = model(image)
                stage = "loss"
                segmentation_loss = segmentation_loss_fn(segmentation, mask)
                classification_loss = classification_loss_fn(classification, cls)
                loss = segmentation_loss + 0.5 * classification_loss
                if not all(
                    bool(torch.isfinite(value).item())
                    for value in (segmentation_loss, classification_loss, loss)
                ):
                    raise FloatingPointError("probe loss is non-finite")
                last_losses = {
                    "segmentation_loss": float(segmentation_loss.item()),
                    "classification_loss": float(classification_loss.item()),
                    "total_loss": float(loss.item()),
                }
                stage = "backward"
                (loss / GRADIENT_ACCUMULATION_STEPS).backward()
                completed_micro_steps += 1
                batch = image = mask = cls = None
                segmentation = classification = None
                segmentation_loss = classification_loss = loss = None
            if len(set(update_case_ids)) != GRADIENT_ACCUMULATION_STEPS:
                raise ValueError("optimizer update did not use 8 distinct microbatches")
            stage = "gradient_clip"
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise FloatingPointError("probe gradient norm is non-finite")
            stage = "optimizer_step"
            optimizer.step()
            stage = "optimizer_zero_grad"
            optimizer.zero_grad(set_to_none=True)
            stage = "synchronize"
            if callable(synchronize):
                synchronize()
            memory = _memory_snapshot()
            headroom = _validate_memory_headroom(memory)
            update_evidence = {
                "phase": (
                    "next_epoch_train"
                    if validation_after_updates is not None
                    and update_index >= validation_after_updates
                    else "first_epoch_train"
                ),
                "optimizer_update": update_index + 1,
                "completed_micro_steps": completed_micro_steps,
                "distinct_microbatches": len(set(update_case_ids)),
                "microbatch_identity_sha256": canonical_sha256(
                    sorted(update_case_ids)
                ),
                "finite_losses": True,
                "finite_gradient_norm": True,
                "gradient_norm": float(gradient_norm.item()),
                "last_microbatch_losses": last_losses,
                "elapsed_seconds": time.perf_counter() - started,
                "memory": memory,
                "headroom_ratio": headroom,
            }
            cast(list[dict[str, Any]], evidence["updates"]).append(update_evidence)
            emit(update_evidence)
            evidence["heartbeat_records_written"] = len(evidence["updates"]) + len(
                evidence["validation"]
            )
        memory_after = _memory_snapshot()
        evidence.update(
            {
                "status": "passed",
                "finite_loss": True,
                "completed_optimizer_updates": optimizer_updates,
                "completed_micro_steps": completed_micro_steps,
                "memory_after_optimizer_path": memory_after,
                "memory_after_headroom_ratio": _validate_memory_headroom(
                    memory_after
                ),
            }
        )
        if heartbeat_stream is not None:
            heartbeat_stream.close()
            heartbeat_stream = None
            evidence["heartbeat_sha256"] = sha256_file(cast(Path, heartbeat_path))
        return evidence
    except BaseException as error:
        if heartbeat_stream is not None:
            heartbeat_stream.flush()
            os.fsync(heartbeat_stream.fileno())
            evidence["heartbeat_sha256"] = sha256_file(cast(Path, heartbeat_path))
        evidence.update(
            {
                "status": "failed",
                "failure_stage": stage,
                "error_type": type(error).__name__,
                "out_of_memory": "out of memory" in str(error).lower(),
                "memory_at_failure": _memory_snapshot(),
                "heartbeat_records_written": len(evidence["updates"])
                + len(evidence["validation"]),
            }
        )
        if isinstance(error, RunInterrupted):
            evidence.update(
                {
                    "reason": "signal_interruption",
                    "signal": error.signal_name,
                    "signal_number": error.signum,
                }
            )
        raise MPSFeasibilityFailure(evidence) from error
    finally:
        if heartbeat_stream is not None:
            heartbeat_stream.close()
        batch = image = mask = cls = None
        segmentation = classification = None
        segmentation_loss = classification_loss = loss = gradient_norm = None
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if model is not None:
            zero_grad = getattr(model, "zero_grad", None)
            if callable(zero_grad):
                try:
                    zero_grad(set_to_none=True)
                except RuntimeError:
                    pass
            del model
        if optimizer is not None:
            del optimizer
        gc.collect()
        if callable(empty_cache):
            try:
                empty_cache()
            except RuntimeError:
                pass
        if callable(synchronize):
            try:
                synchronize()
            except RuntimeError:
                pass


def _resource_evidence(
    args: argparse.Namespace,
    source: dict[str, str],
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    content: dict[str, str],
    pretrained: dict[str, Any],
    baseline: dict[str, Any],
    dependency: dict[str, Any],
    identity_sha256: str,
    probe: dict[str, Any],
    acceptance_soak: dict[str, Any],
    allocator: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "issue_url": ISSUE_URL,
        "parent_issue_url": PARENT_ISSUE_URL,
        "attempt_id": args.attempt_id,
        "reviewed_by": args.reviewed_by,
        "protocol_sha256": canonical_sha256(protocol),
        "source": source,
        "manifest_sha256": manifest["manifest_sha256"],
        "combined_content_sha256": content["combined_content_sha256"],
        "case_identity_sha256": identity_sha256,
        "pretrained_sha256": pretrained["sha256"],
        "baseline_score_sha256": baseline["score_sha256"],
        "baseline_run_record_sha256": baseline["run_record_sha256"],
        "dependency_lock_sha256": dependency["lock_sha256"],
        "runtime": {
            "python": dependency["python"],
            "platform_system": dependency["platform_system"],
            "platform_machine": dependency["platform_machine"],
            "torch": dependency["distributions"]["torch"],
            "torchvision": dependency["distributions"]["torchvision"],
            "mps_built": dependency["mps_built"],
            "mps_available": dependency["mps_available"],
            "mps_cpu_fallback": dependency["mps_cpu_fallback"],
        },
        "probe": probe,
        "acceptance_soak": acceptance_soak,
        "mps_allocator": allocator,
        "bootstrap": bootstrap,
        "automatic_512_fallback_started": False,
        "wandb_started_before_probe_completion": False,
        "external_final_test_untouched": True,
    }


def _build_config(
    args: argparse.Namespace,
    source: dict[str, str],
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    content: dict[str, str],
    pretrained: dict[str, Any],
    baseline: dict[str, Any],
    dependency: dict[str, Any],
    identity_sha256: str,
    resource_sha256: str,
    allocator: dict[str, Any],
    resource_gate_receipt_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "issue_url": ISSUE_URL,
        "parent_issue_url": PARENT_ISSUE_URL,
        "attempt_id": args.attempt_id,
        "reviewed_by": args.reviewed_by,
        "fresh_attempt": True,
        "resume_policy": "never",
        "protocol": protocol,
        "protocol_sha256": canonical_sha256(protocol),
        "num_workers": args.num_workers,
        "device": "mps",
        "source": source,
        "pretrained": pretrained,
        "dataset_scope": DATASET_SCOPE,
        "manifest": {
            "manifest_sha256": manifest["manifest_sha256"],
            "train_case_count": EXPECTED_TRAIN_CASES,
            "validation_case_count": EXPECTED_VALIDATION_CASES,
        },
        "content": content,
        "baseline": {key: value for key, value in baseline.items() if key != "cases"},
        "dependency": dependency,
        "case_identity_sha256": identity_sha256,
        "resource_evidence_sha256": resource_sha256,
        "mps_allocator": allocator,
        "resource_gate_receipt_sha256": resource_gate_receipt_sha256,
        "train_micro_steps_per_epoch": TRAIN_MICRO_STEPS,
        "train_optimizer_steps_per_epoch": TRAIN_OPTIMIZER_STEPS,
        "validation_steps_per_epoch": EXPECTED_VALIDATION_CASES,
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
            "canonical_bytes_verified": True,
            "external_final_test_untouched": True,
        },
    }


def _wandb_public_config(config: dict[str, Any]) -> dict[str, Any]:
    dependency = config["dependency"]
    return {
        "schema_version": config["schema_version"],
        "issue_url": config["issue_url"],
        "parent_issue_url": config["parent_issue_url"],
        "attempt_id": config["attempt_id"],
        "protocol": config["protocol"],
        "protocol_sha256": config["protocol_sha256"],
        "source": config["source"],
        "pretrained": config["pretrained"],
        "dataset": {
            "scope": config["dataset_scope"],
            "train_case_count": EXPECTED_TRAIN_CASES,
            "validation_case_count": EXPECTED_VALIDATION_CASES,
            "manifest_sha256": config["manifest"]["manifest_sha256"],
            "case_identity_sha256": config["case_identity_sha256"],
            "combined_content_sha256": config["content"]["combined_content_sha256"],
        },
        "baseline": config["baseline"],
        "dependency": {
            "lock_sha256": dependency["lock_sha256"],
            "python": dependency["python"],
            "platform_system": dependency["platform_system"],
            "platform_machine": dependency["platform_machine"],
            "torch": dependency["distributions"]["torch"],
            "torchvision": dependency["distributions"]["torchvision"],
            "mps_built": dependency["mps_built"],
            "mps_available": dependency["mps_available"],
            "mps_cpu_fallback": dependency["mps_cpu_fallback"],
        },
        "resource_evidence_sha256": config["resource_evidence_sha256"],
        "mps_allocator": config["mps_allocator"],
        "resource_gate_receipt_sha256": config["resource_gate_receipt_sha256"],
        "external_final_isolation": config["external_final_isolation"],
    }


def _start_wandb(config: dict[str, Any]) -> Any:
    import wandb

    public_config = _wandb_public_config(config)
    if canonical_sha256(public_config) != config["wandb"]["config_sha256"]:
        raise RuntimeError("W&B public config hash differs from sealed MPS config")
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        id=config["wandb"]["id"],
        name=config["wandb"]["name"],
        mode="online",
        resume="never",
        group="task1-teammate-l05-veto-mps-health",
        job_type="reviewed-mps-health",
        config=public_config,
    )
    run.define_metric("train/micro_step")
    run.define_metric("train/*", step_metric="train/micro_step")
    run.define_metric("validation/global_step")
    run.define_metric("validation/*", step_metric="validation/global_step")
    run.define_metric("epoch")
    run.define_metric("epoch/*", step_metric="epoch")
    return run


def _validate_mps_epochs(epochs: list[dict[str, Any]]) -> None:
    validate_epoch_evidence(epochs, EPOCHS, TRAIN_OPTIMIZER_STEPS)
    for epoch, evidence in enumerate(epochs, start=1):
        if evidence.get("micro_steps") != TRAIN_MICRO_STEPS:
            raise ValueError("MPS epoch evidence has wrong microstep count")
        if evidence.get("completed_train_steps") != epoch * TRAIN_OPTIMIZER_STEPS:
            raise ValueError("MPS cumulative optimizer-step evidence is incomplete")


def _load_resource_gate(
    args: argparse.Namespace,
    source: dict[str, str],
    identity_sha256: str,
    pretrained_sha256: str,
) -> tuple[dict[str, Any], str]:
    receipt_path = args.resource_gate_receipt
    if receipt_path is None:
        raise RuntimeError("scientific run requires a supervisor resource-gate receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    index_path = receipt_path.with_name("supervisor_index.json")
    supervisor_index = json.loads(index_path.read_text(encoding="utf-8"))
    receipt_sha256 = sha256_file(receipt_path)
    expected_gate_id = f"{args.attempt_id}-resource-gate"
    if (
        receipt.get("status") != "completed"
        or receipt.get("role") != "resource_gate"
        or receipt.get("attempt_id") != expected_gate_id
        or supervisor_index.get("receipt_sha256") != receipt_sha256
        or supervisor_index.get("status") != "completed"
    ):
        raise RuntimeError("resource-gate supervisor receipt is incomplete")
    gate_dir = args.artifact_root / expected_gate_id
    gate_index_path = gate_dir / "acceptance_soak_index.json"
    gate_resource_path = gate_dir / "resource_evidence.json"
    gate_index = json.loads(gate_index_path.read_text(encoding="utf-8"))
    resource = json.loads(gate_resource_path.read_text(encoding="utf-8"))
    if (
        gate_index.get("status") != "passed"
        or gate_index.get("wandb_started") is not False
        or gate_index.get("scientific_training_started") is not False
        or gate_index.get("resource_evidence_sha256")
        != sha256_file(gate_resource_path)
        or resource.get("source") != source
        or resource.get("case_identity_sha256") != identity_sha256
        or resource.get("pretrained_sha256") != pretrained_sha256
        or resource.get("acceptance_soak", {}).get("status") != "passed"
    ):
        raise RuntimeError("resource-gate evidence differs from scientific inputs")
    return resource, receipt_sha256


def _seal_resource_failure(
    artifact_dir: Path, args: argparse.Namespace, resource: dict[str, Any]
) -> dict[str, Any]:
    write_json_once(artifact_dir / "resource_evidence.json", resource)
    artifacts = {
        "resource_evidence.json": sha256_file(
            artifact_dir / "resource_evidence.json"
        )
    }
    heartbeat_path = artifact_dir / "acceptance_soak_heartbeat.jsonl"
    if heartbeat_path.exists():
        artifacts[heartbeat_path.name] = sha256_file(heartbeat_path)
    index = {
        "schema_version": 1,
        "attempt_id": args.attempt_id,
        "phase": "health",
        "status": "resource_infeasible",
        "resource_evidence_sha256": sha256_file(
            artifact_dir / "resource_evidence.json"
        ),
        "artifacts": artifacts,
        "automatic_512_fallback_started": False,
        "wandb_started": False,
        "external_final_test_untouched": True,
    }
    write_json_once(artifact_dir / "resource_index.json", index)
    return index


def _safe_failure(
    artifact_dir: Path,
    args: argparse.Namespace,
    stage: str,
    error: BaseException,
    *,
    wandb_finish_succeeded: bool | None = None,
    wandb_finish_error: BaseException | None = None,
) -> None:
    path = artifact_dir / "failure.json"
    if artifact_dir.exists() and not path.exists():
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "attempt_id": args.attempt_id,
            "phase": "health",
            "status": "failed",
            "stage": stage,
            "error_type": type(error).__name__,
            "external_final_test_untouched": True,
        }
        if isinstance(error, RunInterrupted):
            receipt.update(
                {
                    "reason": "signal_interruption",
                    "signal": error.signal_name,
                    "signal_number": error.signum,
                    "wandb_exit_code_1_requested": wandb_finish_succeeded is not None,
                    "wandb_finish_succeeded": wandb_finish_succeeded,
                }
            )
        if wandb_finish_error is not None:
            receipt["wandb_finish_error_type"] = type(wandb_finish_error).__name__
        heartbeat = artifact_dir / "acceptance_soak_heartbeat.jsonl"
        if heartbeat.exists():
            receipt["partial_heartbeat_sha256"] = sha256_file(heartbeat)
            receipt["partial_heartbeat_records"] = len(
                heartbeat.read_text(encoding="utf-8").splitlines()
            )
        write_json_once(path, receipt)


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = args.artifact_root / args.attempt_id
    if artifact_dir.exists():
        raise FileExistsError("attempt directory already exists; retries must be fresh")
    if not args.execute:
        return {
            "status": "dry_run",
            "phase": "health",
            "epochs": EPOCHS,
            "attempt_id": args.attempt_id,
            "protocol": mps_protocol_contract(),
            "review_required": True,
        }
    if not str(args.reviewed_by or "").strip():
        raise ValueError("--execute requires a non-empty --reviewed-by attestation")
    if not args.internal_worker:
        raise RuntimeError("execute must be launched by the stdlib-only bootstrap")

    role = "resource_gate" if args.acceptance_soak_only else "scientific_run"
    bootstrap = _validate_bootstrap_proof(role)
    allocator = validate_mps_allocator_environment()

    artifact_dir.mkdir(parents=True, exist_ok=False)
    wandb_run: Any = None
    stage = "source_identity"
    previous_handlers = _install_interrupt_handlers()
    try:
        stage = "source_identity"
        source = source_identity(REPO_ROOT)
        protocol = mps_protocol_contract()
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
        stage = "mps_dependency_identity"
        dependency = validate_mps_runtime_dependencies(MPS_LOCK_PATH)
        dependency["lock_sha256"] = sha256_file(MPS_LOCK_PATH)
        if torch.device("mps").type != "mps":
            raise RuntimeError("MPS amendment cannot use another device")

        stage = "dataset_setup"
        datasets.TRAIN_DCM_DIR = str(args.train_dcm_dir)
        datasets.TRAIN_MASK_DIR = str(args.train_mask_dir)
        datasets.VAL_DCM_DIR = str(args.val_dcm_dir)
        datasets.VAL_MASK_DIR = str(args.val_mask_dir)
        _set_seed(protocol["seed"])
        probe_loader, probe_val_loader = datasets.dataloader(
            batch_size=PHYSICAL_BATCH_SIZE,
            target_size=TARGET_SIZE,
            clahe_clip=protocol["clahe_clip"],
            num_workers=args.num_workers,
            seed=protocol["seed"],
            crop_frac=protocol["crop_frac"],
        )
        identity, identity_sha256 = validate_dataset_identity(
            probe_loader, probe_val_loader, manifest["identity"]
        )
        if len(probe_loader) != EXPECTED_TRAIN_CASES or len(probe_val_loader) != 111:
            raise ValueError("MPS loaders differ from exact 444/111 microstep contract")

        if args.acceptance_soak_only:
            stage = "mps_1024_feasibility"
            try:
                probe = _run_mps_optimizer_probe(
                    probe_loader,
                    torch.device("mps"),
                    args.pretrained,
                    optimizer_updates=FEASIBILITY_OPTIMIZER_STEPS,
                    probe_name="exact_8_microbatch_adamw_optimizer_update",
                    progress_callback=_supervisor_progress,
                )
            except MPSFeasibilityFailure as error:
                resource = _resource_evidence(
                    args,
                    source,
                    protocol,
                    manifest,
                    content,
                    pretrained,
                    baseline,
                    dependency,
                    identity_sha256,
                    error.evidence,
                    {"status": "not_started"},
                    allocator,
                    bootstrap,
                )
                return _seal_resource_failure(artifact_dir, args, resource)

            stage = "mps_1024_acceptance_soak"
            _set_seed(protocol["seed"])
            try:
                acceptance_soak = _run_mps_optimizer_probe(
                    probe_loader,
                    torch.device("mps"),
                    args.pretrained,
                    optimizer_updates=ACCEPTANCE_SOAK_OPTIMIZER_STEPS,
                    probe_name="no_wandb_train_validation_transition_soak",
                    heartbeat_path=(
                        artifact_dir / "acceptance_soak_heartbeat.jsonl"
                    ),
                    validation_loader=probe_val_loader,
                    validation_after_updates=(
                        ACCEPTANCE_FIRST_EPOCH_OPTIMIZER_STEPS
                    ),
                    progress_callback=_supervisor_progress,
                )
            except MPSFeasibilityFailure as error:
                resource = _resource_evidence(
                    args,
                    source,
                    protocol,
                    manifest,
                    content,
                    pretrained,
                    baseline,
                    dependency,
                    identity_sha256,
                    probe,
                    error.evidence,
                    allocator,
                    bootstrap,
                )
                return _seal_resource_failure(artifact_dir, args, resource)

            resource = _resource_evidence(
                args,
                source,
                protocol,
                manifest,
                content,
                pretrained,
                baseline,
                dependency,
                identity_sha256,
                probe,
                acceptance_soak,
                allocator,
                bootstrap,
            )
            write_json_once(artifact_dir / "resource_evidence.json", resource)
            resource_sha256 = sha256_file(artifact_dir / "resource_evidence.json")
            del probe_loader, probe_val_loader
            gc.collect()
            torch.mps.empty_cache()
            torch.mps.synchronize()
            _supervisor_progress(
                {
                    "phase": "resource_process_cleanup",
                    "status": "passed",
                    "memory": _memory_snapshot(),
                }
            )
            index = {
                "schema_version": 1,
                "attempt_id": args.attempt_id,
                "phase": "acceptance_soak",
                "status": "passed",
                "resource_evidence_sha256": resource_sha256,
                "heartbeat_sha256": sha256_file(
                    artifact_dir / "acceptance_soak_heartbeat.jsonl"
                ),
                "wandb_started": False,
                "scientific_training_started": False,
                "external_final_test_untouched": True,
            }
            write_json_once(artifact_dir / "acceptance_soak_index.json", index)
            return index

        stage = "resource_gate_validation"
        resource, resource_gate_receipt_sha256 = _load_resource_gate(
            args, source, identity_sha256, pretrained["sha256"]
        )
        shutil.copyfile(
            args.artifact_root
            / f"{args.attempt_id}-resource-gate"
            / "resource_evidence.json",
            artifact_dir / "resource_evidence.json",
        )
        shutil.copyfile(
            args.artifact_root
            / f"{args.attempt_id}-resource-gate"
            / "acceptance_soak_heartbeat.jsonl",
            artifact_dir / "acceptance_soak_heartbeat.jsonl",
        )
        resource_sha256 = sha256_file(artifact_dir / "resource_evidence.json")

        stage = "fresh_training_setup"
        del probe_loader, probe_val_loader
        gc.collect()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        _set_seed(protocol["seed"])
        train_loader, val_loader = datasets.dataloader(
            batch_size=PHYSICAL_BATCH_SIZE,
            target_size=TARGET_SIZE,
            clahe_clip=protocol["clahe_clip"],
            num_workers=args.num_workers,
            seed=protocol["seed"],
            crop_frac=protocol["crop_frac"],
        )
        training_identity, training_identity_sha256 = validate_dataset_identity(
            train_loader, val_loader, manifest["identity"]
        )
        if training_identity != identity or training_identity_sha256 != identity_sha256:
            raise RuntimeError("dataset identity changed after feasibility probe")
        model = modeltype(
            "evax_seg",
            in_channels=1,
            img_size=TARGET_SIZE,
            pretrained_path=str(args.pretrained),
            variant="small",
        ).to(torch.device("mps"))
        config = _build_config(
            args,
            source,
            protocol,
            manifest,
            content,
            pretrained,
            baseline,
            dependency,
            identity_sha256,
            resource_sha256,
            allocator,
            resource_gate_receipt_sha256,
        )
        config["parameter_count"] = sum(p.numel() for p in model.parameters())
        config["wandb"]["config_sha256"] = canonical_sha256(
            _wandb_public_config(config)
        )
        write_json_once(artifact_dir / "config.json", config)
        write_json_once(artifact_dir / "source.json", source)
        write_json_once(artifact_dir / "case_identity.json", identity)
        shutil.copyfile(MPS_LOCK_PATH, artifact_dir / "dependency.lock")

        stage = "wandb_start"
        wandb_run = _start_wandb(config)
        if str(wandb_run.id) != args.attempt_id:
            raise TypeError("W&B did not preserve the sealed MPS attempt ID")
        started = time.perf_counter()
        checkpoint_path = artifact_dir / "best_checkpoint.pth"
        stage = "training"
        details = cast(
            dict[str, Any],
            fit(
                model,
                train_loader,
                val_loader,
                torch.device("mps"),
                max_epochs=EPOCHS,
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
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                progress_callback=_supervisor_progress,
            ),
        )

        stage = "artifact_finalization"
        epochs = details["epochs"]
        _validate_mps_epochs(epochs)
        if details["completed_train_steps"] != EPOCHS * TRAIN_OPTIMIZER_STEPS:
            raise RuntimeError("MPS optimizer-step evidence is incomplete")
        if details["completed_micro_steps"] != EPOCHS * TRAIN_MICRO_STEPS:
            raise RuntimeError("MPS microstep evidence is incomplete")
        best = details["best_native_metrics"]
        if not isinstance(best, dict):
            raise TypeError("native validation did not produce MPS health metrics")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["reproduction"] = {
            "attempt_id": args.attempt_id,
            "selected_epoch": details["best_epoch"],
            "source": source,
            "config_sha256": sha256_file(artifact_dir / "config.json"),
            "resource_evidence_sha256": resource_sha256,
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
            "schema_version": 1,
            "issue_url": ISSUE_URL,
            "parent_issue_url": PARENT_ISSUE_URL,
            "attempt_id": args.attempt_id,
            "phase": "health",
            "execution_family": "apple_mps_resource_adjusted",
            "historical_cuda_equivalence_claimed": False,
            "dataset_scope": DATASET_SCOPE,
            "external_final_test_untouched": True,
            "selected_epoch": details["best_epoch"],
            "metrics": {
                name: best[name]
                for name in (
                    "classification_accuracy",
                    "dice",
                    "weighted_composite",
                    "coverage",
                )
            },
            "decision": protocol["inference"],
            "baseline": config["baseline"],
            "resource_evidence_sha256": resource_sha256,
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
            raise RuntimeError("W&B identity differs from sealed MPS contract")
        for name, value in score["metrics"].items():
            if name != "coverage":
                wandb_run.summary[f"best/{name}"] = value
        wandb_run.summary["best/epoch"] = details["best_epoch"]
        wandb_run.summary["resource/evidence_sha256"] = resource_sha256
        wandb_run.summary["regression/fixed"] = regression["taxonomy_counts"]["fixed"]
        wandb_run.summary["regression/regressed"] = regression["taxonomy_counts"][
            "regressed"
        ]
        wandb_run.finish()
        wandb_run = None
        run_record = {
            "schema_version": 1,
            "issue_url": ISSUE_URL,
            "parent_issue_url": PARENT_ISSUE_URL,
            "attempt_id": args.attempt_id,
            "phase": "health",
            "execution_family": "apple_mps_resource_adjusted",
            "historical_cuda_equivalence_claimed": False,
            "status": "completed",
            "reviewed_by": args.reviewed_by,
            "external_final_test_untouched": True,
            "requested_epochs": EPOCHS,
            "completed_epochs": details["completed_epochs"],
            "completed_train_micro_steps": details["completed_micro_steps"],
            "completed_optimizer_steps": details["completed_train_steps"],
            "train_micro_steps_per_epoch": TRAIN_MICRO_STEPS,
            "train_optimizer_steps_per_epoch": TRAIN_OPTIMIZER_STEPS,
            "validation_steps_per_epoch": EXPECTED_VALIDATION_CASES,
            "selected_epoch": details["best_epoch"],
            "runtime_seconds": time.perf_counter() - started,
            "metrics": score["metrics"],
            "resource_evidence_sha256": resource_sha256,
            "wandb_finished": True,
            "wandb": {**config["wandb"], "url": run_url},
            "artifacts": list(MPS_ARTIFACT_NAMES),
        }
        write_json_once(artifact_dir / "run_record.json", run_record)
        artifact_index = {
            "schema_version": 1,
            "attempt_id": args.attempt_id,
            "phase": "health",
            "execution_family": "apple_mps_resource_adjusted",
            "status": "completed",
            "artifacts": {
                name: sha256_file(artifact_dir / name)
                for name in MPS_ARTIFACT_NAMES
            },
        }
        write_json_once(artifact_dir / "artifact_index.json", artifact_index)
        return artifact_index
    except BaseException as error:
        finish_succeeded: bool | None = None
        finish_error: BaseException | None = None
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=1)
                finish_succeeded = True
            except BaseException as caught_finish_error:
                finish_succeeded = False
                finish_error = caught_finish_error
        _safe_failure(
            artifact_dir,
            args,
            stage,
            error,
            wandb_finish_succeeded=finish_succeeded,
            wandb_finish_error=finish_error,
        )
        raise
    finally:
        _restore_interrupt_handlers(previous_handlers)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/reproduction-mps")
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--baseline-run-record", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--train-dcm-dir", type=Path, required=True)
    parser.add_argument("--train-mask-dir", type=Path, required=True)
    parser.add_argument("--val-dcm-dir", type=Path, required=True)
    parser.add_argument("--val-mask-dir", type=Path, required=True)
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--reviewed-by")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--acceptance-soak-only",
        action="store_true",
        help="run the mandatory no-W&B epoch/validation/transition gate and exit",
    )
    parser.add_argument("--resource-gate-receipt", type=Path)
    parser.add_argument(
        "--internal-worker", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.attempt_id) is None:
        parser.error("--attempt-id must be a single safe path component")
    if args.num_workers != 0:
        parser.error("reviewed Apple-MPS contract requires --num-workers 0")
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
    )
    for name in path_names:
        value = getattr(args, name)
        reject_external_final_path(value, name)
    for name in path_names:
        value = getattr(args, name)
        setattr(args, name, resolve_non_external_path(value, name))
    if args.resource_gate_receipt is not None:
        reject_external_final_path(args.resource_gate_receipt, "resource_gate_receipt")
        args.resource_gate_receipt = resolve_non_external_path(
            args.resource_gate_receipt, "resource_gate_receipt"
        )
    args.wandb_run_name = args.wandb_run_name or (
        f"task1-teammate-l05-veto-mps-health-{args.attempt_id}"
    )
    return args


def main(argv: list[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") in {"failed", "resource_infeasible"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
