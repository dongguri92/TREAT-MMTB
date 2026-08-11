"""Review-gated Apple-MPS health amendment for Issue #106.

This runner is separate from the sealed Linux/CUDA reproduction. Dry-run is
the default. Reviewed execution is bootstrap-only and requires a disposable
1024x1024 train/validation/next-epoch resource gate before W&B can start.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
from github_approval import verify_scientific_approval
from models import modeltype
from mps_evidence import validate_gate_artifacts
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
from training import (
    accumulated_train_step,
    compute_lr,
    fit,
    make_optimizer,
    validation_step,
)
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
    "bootstrap_proof.json",
    "dependency.lock",
    "resource_evidence.json",
    "acceptance_soak_heartbeat.jsonl",
    "epochs.json",
    "best_epoch_cases.json",
    "regression.json",
    "score.json",
    "best_checkpoint.pth",
    "checkpoint_receipt.json",
    "wandb_terminal.json",
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
    loader_start = proof.get("loader_start")
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
        or not isinstance(loader_start, dict)
        or not isinstance(loader_start.get("components"), list)
        or len(loader_start["components"]) != GRADIENT_ACCUMULATION_STEPS
        or loader_start.get("fingerprint")
        != canonical_sha256(loader_start["components"])
    ):
        raise RuntimeError("MPS bootstrap proof differs from reviewed contract")
    allocator = validate_mps_allocator_environment()
    if allocator["values"] != proof["allocator"]:
        raise RuntimeError("allocator environment differs from bootstrap proof")
    return {**proof, "proof_sha256": digest}


def _revalidate_scientific_approval(
    bootstrap: dict[str, Any],
    args: argparse.Namespace,
    source_git_commit: str,
) -> dict[str, Any]:
    sealed = bootstrap.get("soak_approval")
    if not isinstance(sealed, dict) or args.resource_gate_receipt is None:
        raise RuntimeError("scientific worker requires sealed GitHub approval")
    authoritative = verify_scientific_approval(
        str(sealed.get("review_url", "")),
        args.attempt_id,
        args.resource_gate_receipt,
        source_git_commit,
        _load_mps_resource_contract()["approval"],
    )
    if authoritative != sealed:
        raise RuntimeError("GitHub approval changed after supervisor sealing")
    return authoritative


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


def _case_digest(case_id: str) -> str:
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def _batch_fingerprint_component(batch: dict[str, Any]) -> dict[str, Any]:
    ids = [str(value) for value in batch.get("id", [])]
    tensors: dict[str, Any] = {}
    for name in ("image", "mask", "cls"):
        tensor = batch[name].detach().cpu().contiguous()
        tensors[name] = {
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    return {
        "case_sha256_ordered": [_case_digest(value) for value in ids],
        "tensors": tensors,
    }


def _batches_fingerprint(batches: list[dict[str, Any]]) -> str:
    return canonical_sha256([_batch_fingerprint_component(batch) for batch in batches])


def _fingerprint_callbacks(expected: str) -> tuple[list[dict[str, Any]], Any, Any]:
    components: list[dict[str, Any]] = []

    def observe(batch: dict[str, Any], _index: int) -> None:
        components.append(_batch_fingerprint_component(batch))

    def validate() -> None:
        if len(components) != GRADIENT_ACCUMULATION_STEPS:
            raise ValueError("first optimizer group fingerprint is incomplete")
        if canonical_sha256(components) != expected:
            raise ValueError("first optimizer group differs from sealed preview")

    return components, observe, validate


def _loader_start_fingerprint(loader: Any) -> str:
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(GRADIENT_ACCUMULATION_STEPS)]
    return _batches_fingerprint(batches)


def _cleanup_mps_boundary() -> dict[str, Any]:
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()
    memory = _memory_snapshot()
    return {
        "status": "passed",
        "gc_collected": True,
        "empty_cache_completed": True,
        "synchronize_completed": True,
        "memory": memory,
        "headroom_ratio": _validate_memory_headroom(memory),
    }


def _critical_memory_evidence(
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    memory = _memory_snapshot()
    headroom = _validate_memory_headroom(memory)
    evidence = {
        "phase": "backward_complete_pre_adamw_memory",
        "memory": memory,
        "headroom_ratio": headroom,
        "tensor_scalar_materialized": False,
        "explicit_mps_synchronize_called": False,
        **(context or {}),
    }
    if progress_callback is not None:
        progress_callback(evidence)
    return evidence


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
    expected_start_fingerprint: str | None = None,
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
    batch = None
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
    emitted_records = 0
    minimum_headroom = 1.0

    def emit(record: dict[str, Any]) -> None:
        nonlocal emitted_records
        if heartbeat_stream is not None:
            heartbeat_stream.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
            heartbeat_stream.flush()
            os.fsync(heartbeat_stream.fileno())
        emitted_records += 1
        evidence["heartbeat_records_written"] = emitted_records
        if progress_callback is not None:
            progress_callback(record)

    def observe_critical_memory(context: dict[str, Any] | None) -> dict[str, Any]:
        nonlocal minimum_headroom
        critical = _critical_memory_evidence(progress_callback, context)
        minimum_headroom = min(
            minimum_headroom, critical["headroom_ratio"]
        )
        return critical

    def set_stage(value: str) -> None:
        nonlocal stage
        stage = value

    try:
        if heartbeat_path is not None:
            heartbeat_stream = heartbeat_path.open("x", encoding="utf-8")
        if callable(empty_cache):
            empty_cache()
        evidence["memory_before"] = _memory_snapshot()
        evidence["memory_before_headroom_ratio"] = _validate_memory_headroom(
            evidence["memory_before"]
        )
        minimum_headroom = min(
            minimum_headroom, evidence["memory_before_headroom_ratio"]
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
        iterator = iter(loader)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        completed_micro_steps = 0
        first_epoch_case_ids: list[str] = []
        next_epoch_case_ids: list[str] = []
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
                        validation_events: list[str] = []
                        validation_result = validation_step(
                            model,
                            batch,
                            segmentation_loss_fn,
                            0.5,
                            device,
                            use_amp=False,
                            materialize_log_loss=True,
                            native_case_materialization=True,
                            stage_callback=validation_events.append,
                        )
                        if not math.isfinite(float(validation_result["loss"])):
                            raise FloatingPointError(
                                "validation resource loss is non-finite"
                            )
                        memory = _memory_snapshot()
                        headroom = _validate_memory_headroom(memory)
                        minimum_headroom = min(minimum_headroom, headroom)
                        validation_record = {
                            "phase": "validation_resource",
                            "validation_step": validation_index,
                            "finite_loss": True,
                            "case_sha256": _case_digest(validation_batch_ids[0]),
                            "operation_events": validation_events,
                            "elapsed_seconds": time.perf_counter() - started,
                            "memory": memory,
                            "headroom_ratio": headroom,
                        }
                        cast(
                            list[dict[str, Any]], evidence["validation"]
                        ).append(validation_record)
                        emit(validation_record)
                        batch = None
                if (
                    len(validation_ids) != ACCEPTANCE_SOAK_VALIDATION_STEPS
                    or len(set(validation_ids)) != ACCEPTANCE_SOAK_VALIDATION_STEPS
                ):
                    raise ValueError(
                        "validation resource traversal is not exact 111 unique cases"
                    )
                evidence["validation_identity_sha256"] = canonical_sha256(
                    sorted(_case_digest(value) for value in validation_ids)
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
            update_components: list[dict[str, Any]] = []
            fingerprint_observer = fingerprint_validator = None
            if update_index == 0 and expected_start_fingerprint is not None:
                (
                    update_components,
                    fingerprint_observer,
                    fingerprint_validator,
                ) = _fingerprint_callbacks(expected_start_fingerprint)

            bound_iterator = iterator

            def update_batches(
                iterator_: Any = bound_iterator,
                case_ids: list[str] = update_case_ids,
                current_update_index: int = update_index,
            ) -> Any:
                nonlocal completed_micro_steps, stage
                for _ in range(GRADIENT_ACCUMULATION_STEPS):
                    stage = "microbatch_load"
                    try:
                        current = next(iterator_)
                    except StopIteration as error:
                        raise ValueError(
                            "probe loader exhausted before required optimizer path"
                        ) from error
                    stage = "microbatch_identity"
                    batch_ids = [str(value) for value in current.get("id", [])]
                    if len(batch_ids) != PHYSICAL_BATCH_SIZE:
                        raise ValueError("probe microbatch identity is missing")
                    case_ids.extend(batch_ids)
                    if (
                        validation_after_updates is not None
                        and current_update_index >= validation_after_updates
                    ):
                        next_epoch_case_ids.extend(batch_ids)
                    else:
                        first_epoch_case_ids.extend(batch_ids)
                    if tuple(current["image"].shape) != (
                        1, 1, TARGET_SIZE, TARGET_SIZE
                    ):
                        raise ValueError("probe batch is not exact 1x1x1024x1024")
                    completed_micro_steps += 1
                    yield current

            def validate_group(
                case_ids: list[str] = update_case_ids,
                validator: Any = fingerprint_validator,
                components: list[dict[str, Any]] = update_components,
            ) -> None:
                if len(set(case_ids)) != GRADIENT_ACCUMULATION_STEPS:
                    raise ValueError(
                        "optimizer update did not use 8 distinct microbatches"
                    )
                if validator is not None:
                    validator()
                    evidence["loader_start_fingerprint"] = (
                        expected_start_fingerprint
                    )
                    evidence["loader_start_components"] = components

            step = accumulated_train_step(
                model,
                update_batches(),
                optimizer,
                segmentation_loss_fn,
                0.5,
                device,
                scaler=None,
                use_amp=False,
                critical_memory_observer=observe_critical_memory,
                critical_memory_context={
                    "probe": probe_name,
                    "optimizer_update": update_index + 1,
                },
                stage_callback=set_stage,
                accumulation=GRADIENT_ACCUMULATION_STEPS,
                batch_observer=fingerprint_observer,
                group_validator=validate_group,
            )
            memory = _memory_snapshot()
            headroom = _validate_memory_headroom(memory)
            minimum_headroom = min(minimum_headroom, headroom)
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
                    sorted(_case_digest(value) for value in update_case_ids)
                ),
                "microbatch_case_sha256_ordered": [
                    _case_digest(value) for value in update_case_ids
                ],
                "finite_losses": True,
                "gradient_clip_completed": True,
                "last_microbatch_losses": step["microbatches"][-1],
                "critical_memory": step["critical_memory"],
                "elapsed_seconds": time.perf_counter() - started,
                "memory": memory,
                "headroom_ratio": headroom,
            }
            cast(list[dict[str, Any]], evidence["updates"]).append(update_evidence)
            emit(update_evidence)
        memory_after = _memory_snapshot()
        expected_first = (
            validation_after_updates * GRADIENT_ACCUMULATION_STEPS
            if validation_after_updates is not None
            else optimizer_updates * GRADIENT_ACCUMULATION_STEPS
        )
        expected_next = (
            (optimizer_updates - validation_after_updates)
            * GRADIENT_ACCUMULATION_STEPS
            if validation_after_updates is not None
            else 0
        )
        if (
            len(first_epoch_case_ids) != expected_first
            or len(set(first_epoch_case_ids)) != expected_first
            or len(next_epoch_case_ids) != expected_next
            or len(set(next_epoch_case_ids)) != expected_next
        ):
            raise ValueError("soak epoch identities are not exact unique cohorts")
        evidence.update(
            {
                "status": "passed",
                "finite_loss": True,
                "completed_optimizer_updates": optimizer_updates,
                "completed_micro_steps": completed_micro_steps,
                "first_epoch_unique_case_count": len(set(first_epoch_case_ids)),
                "next_epoch_unique_case_count": len(set(next_epoch_case_ids)),
                "first_epoch_identity_sha256": canonical_sha256(
                    sorted(_case_digest(value) for value in first_epoch_case_ids)
                ),
                "next_epoch_identity_sha256": canonical_sha256(
                    sorted(_case_digest(value) for value in next_epoch_case_ids)
                ),
                "memory_after_optimizer_path": memory_after,
                "memory_after_headroom_ratio": _validate_memory_headroom(
                    memory_after
                ),
                "minimum_headroom_ratio": min(
                    minimum_headroom,
                    _validate_memory_headroom(memory_after),
                ),
            }
        )
        if heartbeat_stream is not None:
            heartbeat_stream.close()
            heartbeat_stream = None
            evidence["heartbeat_sha256"] = sha256_file(cast(Path, heartbeat_path))
        return evidence
    except BaseException as error:
        evidence_errors: list[dict[str, str]] = []

        def record_evidence_error(
            evidence_stage: str, evidence_error: BaseException
        ) -> None:
            evidence_errors.append(
                {
                    "stage": evidence_stage,
                    "error_type": type(evidence_error).__name__,
                }
            )

        if heartbeat_stream is not None:
            try:
                heartbeat_stream.flush()
            except BaseException as evidence_error:
                record_evidence_error("failure_heartbeat_flush", evidence_error)
            try:
                os.fsync(heartbeat_stream.fileno())
            except BaseException as evidence_error:
                record_evidence_error("failure_heartbeat_fsync", evidence_error)
            try:
                evidence["heartbeat_sha256"] = sha256_file(
                    cast(Path, heartbeat_path)
                )
            except BaseException as evidence_error:
                record_evidence_error("failure_heartbeat_hash", evidence_error)
        failure_memory: dict[str, Any] = {}
        try:
            failure_memory = _memory_snapshot()
        except BaseException as evidence_error:
            record_evidence_error("failure_memory_snapshot", evidence_error)
        evidence.update(
            {
                "status": "failed",
                "failure_stage": stage,
                "error_type": type(error).__name__,
                "out_of_memory": "out of memory" in str(error).lower(),
                "memory_at_failure": failure_memory,
                "heartbeat_records_written": emitted_records,
            }
        )
        if evidence_errors:
            evidence["failure_evidence_errors"] = evidence_errors
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
        active_error = sys.exc_info()[0] is not None
        cleanup_errors: list[dict[str, str]] = []
        if heartbeat_stream is not None:
            try:
                heartbeat_stream.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(
                    {
                        "stage": "heartbeat_close",
                        "error_type": type(cleanup_error).__name__,
                    }
                )
        batch = None
        if optimizer is not None:
            try:
                optimizer.zero_grad(set_to_none=True)
            except BaseException as cleanup_error:
                cleanup_errors.append(
                    {
                        "stage": "optimizer_zero_grad",
                        "error_type": type(cleanup_error).__name__,
                    }
                )
        if model is not None:
            zero_grad = getattr(model, "zero_grad", None)
            if callable(zero_grad):
                try:
                    zero_grad(set_to_none=True)
                except BaseException as cleanup_error:
                    cleanup_errors.append(
                        {
                            "stage": "model_zero_grad",
                            "error_type": type(cleanup_error).__name__,
                        }
                    )
            del model
        if optimizer is not None:
            del optimizer
        try:
            gc.collect()
        except BaseException as cleanup_error:
            cleanup_errors.append(
                {"stage": "gc_collect", "error_type": type(cleanup_error).__name__}
            )
        if callable(empty_cache):
            try:
                empty_cache()
            except BaseException as cleanup_error:
                cleanup_errors.append(
                    {
                        "stage": "empty_cache",
                        "error_type": type(cleanup_error).__name__,
                    }
                )
        if callable(synchronize):
            try:
                synchronize()
            except BaseException as cleanup_error:
                cleanup_errors.append(
                    {
                        "stage": "synchronize_cleanup",
                        "error_type": type(cleanup_error).__name__,
                    }
                )
        cleanup_memory: dict[str, Any] = {}
        cleanup_headroom: float | None = None
        try:
            cleanup_memory = _memory_snapshot()
            cleanup_headroom = _validate_memory_headroom(cleanup_memory)
        except BaseException as cleanup_error:
            cleanup_errors.append(
                {
                    "stage": "cleanup_memory_headroom",
                    "error_type": type(cleanup_error).__name__,
                }
            )
        evidence["cleanup"] = {
            "status": "failed" if cleanup_errors else "passed",
            "errors": cleanup_errors,
            "memory": cleanup_memory,
            "headroom_ratio": cleanup_headroom,
        }
        if cleanup_errors and not active_error:
            evidence.update(
                {
                    "status": "failed",
                    "failure_stage": "cleanup",
                    "error_type": cleanup_errors[0]["error_type"],
                }
            )
            raise MPSFeasibilityFailure(evidence)


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
    identity: dict[str, Any],
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
        "canonical_case_sha256": {
            "train": [_case_digest(value) for value in identity["train"]],
            "validation": [
                _case_digest(value) for value in identity["validation"]
            ],
        },
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


def _verify_wandb_terminal(
    attempt_id: str,
    config: dict[str, Any],
    score: dict[str, Any],
    regression: dict[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Query the authoritative W&B API after finish before sealing completion."""
    import wandb

    api = wandb.Api(timeout=30)
    remote = None
    state = ""
    for _ in range(12):
        remote = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{attempt_id}")
        state = str(remote.state).lower()
        if str(remote.id) == attempt_id and state == "finished":
            break
        time.sleep(5)
    if remote is None or str(remote.id) != attempt_id or state != "finished":
        raise RuntimeError("W&B run is not authoritatively finished")
    remote_config = dict(remote.config)
    expected_config = _wandb_public_config(config)
    if remote_config != expected_config:
        raise RuntimeError("authoritative W&B config differs from sealed config")
    remote_summary = dict(remote.summary)
    expected_summary = {
        **{
            f"best/{name}": value
            for name, value in score["metrics"].items()
            if name != "coverage"
        },
        "best/epoch": score["selected_epoch"],
        "resource/evidence_sha256": score["resource_evidence_sha256"],
        "regression/fixed": regression["taxonomy_counts"]["fixed"],
        "regression/regressed": regression["taxonomy_counts"]["regressed"],
        "regression/sha256": score["regression_sha256"],
        "checkpoint/sha256": checkpoint_sha256,
    }
    if any(remote_summary.get(key) != value for key, value in expected_summary.items()):
        raise RuntimeError("authoritative W&B summary differs from sealed evidence")
    return {
        "schema_version": 1,
        "id": str(remote.id),
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "state": state,
        "url": str(remote.url),
        "config_sha256": canonical_sha256(remote_config),
        "summary_sha256": canonical_sha256(expected_summary),
        "verified_summary": expected_summary,
    }


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
    expected_receipt_path = (
        args.artifact_root
        / ".supervisor"
        / expected_gate_id
        / "supervisor_receipt.json"
    ).resolve()
    if receipt_path.resolve() != expected_receipt_path:
        raise RuntimeError("resource-gate receipt is outside canonical supervisor path")
    gate_dir = args.artifact_root / expected_gate_id
    minimum_headroom = float(
        _load_mps_resource_contract()["memory"]["minimum_headroom_ratio"]
    )
    chain = validate_gate_artifacts(
        gate_dir,
        expected_gate_id,
        minimum_headroom,
        receipt_path.with_name("progress.jsonl"),
        receipt.get("bootstrap_proof_canonical_sha256"),
    )
    approval = _validate_bootstrap_proof("scientific_run").get("soak_approval")
    if (
        receipt.get("status") != "completed"
        or receipt.get("role") != "resource_gate"
        or receipt.get("attempt_id") != expected_gate_id
        or supervisor_index.get("receipt_sha256") != receipt_sha256
        or supervisor_index.get("status") != "completed"
        or receipt.get("child_completion_sha256")
        != chain["gate_index_sha256"]
        or receipt.get("evidence_chain") != chain
        or receipt.get("bootstrap_proof_canonical_sha256")
        != chain["bootstrap_proof_sha256"]
        or not isinstance(approval, dict)
        or approval.get("status") != "approved"
        or approval.get("attempt_id") != args.attempt_id
        or approval.get("gate_attempt_id") != expected_gate_id
        or approval.get("source_git_commit") != source["git_commit"]
        or approval.get("resource_gate_receipt_sha256") != receipt_sha256
        or not str(approval.get("approval_sha256", "")).strip()
    ):
        raise RuntimeError("resource-gate supervisor receipt is incomplete")
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
        or resource.get("acceptance_soak", {}).get(
            "loader_start_fingerprint"
        ) is None
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
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.attempt_id) is None:
        raise ValueError("attempt id must be a single safe path component")
    for name in (
        "artifact_root",
        "manifest",
        "baseline_score",
        "baseline_run_record",
        "pretrained",
        "train_dcm_dir",
        "train_mask_dir",
        "val_dcm_dir",
        "val_mask_dir",
    ):
        value = Path(getattr(args, name))
        reject_external_final_path(value, name)
        reject_external_final_path(value.resolve(), name)
    if args.resource_gate_receipt is not None:
        reject_external_final_path(
            Path(args.resource_gate_receipt), "resource_gate_receipt"
        )
        reject_external_final_path(
            Path(args.resource_gate_receipt).resolve(), "resource_gate_receipt"
        )
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
        if not datasets._HAS_ALBU or datasets.A is None:
            raise RuntimeError(
                "reviewed MPS execution requires Albumentations 2 import"
            )
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
                    expected_start_fingerprint=bootstrap["loader_start"][
                        "fingerprint"
                    ],
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
                    identity,
                    error.evidence,
                    {"status": "not_started"},
                    allocator,
                    bootstrap,
                )
                return _seal_resource_failure(artifact_dir, args, resource)

            stage = "feasibility_cleanup"
            del probe_loader, probe_val_loader
            _cleanup_mps_boundary()
            _set_seed(protocol["seed"])
            preview_loader, preview_val_loader = datasets.dataloader(
                batch_size=PHYSICAL_BATCH_SIZE,
                target_size=TARGET_SIZE,
                clahe_clip=protocol["clahe_clip"],
                num_workers=args.num_workers,
                seed=protocol["seed"],
                crop_frac=protocol["crop_frac"],
            )
            preview_identity, preview_identity_sha256 = validate_dataset_identity(
                preview_loader, preview_val_loader, manifest["identity"]
            )
            if preview_identity != identity or preview_identity_sha256 != identity_sha256:
                raise RuntimeError("preview loader identity differs after feasibility")
            loader_start_fingerprint = _loader_start_fingerprint(preview_loader)
            if (
                loader_start_fingerprint
                != bootstrap["loader_start"]["fingerprint"]
            ):
                raise RuntimeError(
                    "loader start differs from independent canonical-input verifier"
                )
            del preview_loader, preview_val_loader
            _cleanup_mps_boundary()
            _set_seed(protocol["seed"])
            probe_loader, probe_val_loader = datasets.dataloader(
                batch_size=PHYSICAL_BATCH_SIZE,
                target_size=TARGET_SIZE,
                clahe_clip=protocol["clahe_clip"],
                num_workers=args.num_workers,
                seed=protocol["seed"],
                crop_frac=protocol["crop_frac"],
            )
            soak_identity, soak_identity_sha256 = validate_dataset_identity(
                probe_loader, probe_val_loader, manifest["identity"]
            )
            if soak_identity != identity or soak_identity_sha256 != identity_sha256:
                raise RuntimeError("soak loader identity differs from sealed preview")

            stage = "mps_1024_acceptance_soak"
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
                    expected_start_fingerprint=loader_start_fingerprint,
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
                    identity,
                    probe,
                    error.evidence,
                    allocator,
                    bootstrap,
                )
                return _seal_resource_failure(artifact_dir, args, resource)

            del probe_loader, probe_val_loader
            cleanup = _cleanup_mps_boundary()
            _supervisor_progress(
                {
                    "phase": "resource_process_cleanup",
                    **cleanup,
                }
            )
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
                identity,
                probe,
                acceptance_soak,
                allocator,
                bootstrap,
            )
            write_json_once(artifact_dir / "resource_evidence.json", resource)
            resource_sha256 = sha256_file(artifact_dir / "resource_evidence.json")
            index = {
                "schema_version": 2,
                "attempt_id": args.attempt_id,
                "phase": "acceptance_soak",
                "status": "passed",
                "resource_evidence_sha256": resource_sha256,
                "heartbeat_sha256": sha256_file(
                    artifact_dir / "acceptance_soak_heartbeat.jsonl"
                ),
                "optimizer_updates": ACCEPTANCE_SOAK_OPTIMIZER_STEPS,
                "micro_steps": ACCEPTANCE_SOAK_MICRO_STEPS,
                "validation_steps": ACCEPTANCE_SOAK_VALIDATION_STEPS,
                "heartbeat_records": acceptance_soak[
                    "heartbeat_records_written"
                ],
                "cleanup": cleanup,
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
        loader_start_fingerprint = _loader_start_fingerprint(probe_loader)
        if loader_start_fingerprint != resource["acceptance_soak"].get(
            "loader_start_fingerprint"
        ):
            raise RuntimeError("scientific loader start differs from reviewed soak")
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
        _cleanup_mps_boundary()
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
        (
            _first_group_components,
            first_group_observer,
            first_group_validator,
        ) = _fingerprint_callbacks(loader_start_fingerprint)
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
        write_json_once(artifact_dir / "bootstrap_proof.json", bootstrap)
        shutil.copyfile(MPS_LOCK_PATH, artifact_dir / "dependency.lock")

        stage = "authoritative_github_approval_revalidation"
        _revalidate_scientific_approval(bootstrap, args, source["git_commit"])

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
                critical_memory_observer=lambda context: _critical_memory_evidence(
                    _supervisor_progress, context
                ),
                first_group_batch_observer=first_group_observer,
                first_group_validator=first_group_validator,
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
        checkpoint_receipt = {
            "schema_version": 1,
            "attempt_id": args.attempt_id,
            "selected_epoch": details["best_epoch"],
            "source_git_commit": source["git_commit"],
            "config_sha256": sha256_file(artifact_dir / "config.json"),
            "resource_evidence_sha256": resource_sha256,
            "pretrained_sha256": pretrained["sha256"],
            "case_identity_sha256": identity_sha256,
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
        write_json_once(
            artifact_dir / "checkpoint_receipt.json", checkpoint_receipt
        )
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
        run_url = wandb_run.url
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
        wandb_run.summary["regression/sha256"] = score["regression_sha256"]
        wandb_run.summary["checkpoint/sha256"] = checkpoint_receipt[
            "checkpoint_sha256"
        ]
        wandb_run.finish()
        wandb_run = None
        stage = "wandb_terminal_verification"
        wandb_terminal = _verify_wandb_terminal(
            args.attempt_id,
            config,
            score,
            regression,
            checkpoint_receipt["checkpoint_sha256"],
        )
        if wandb_terminal["url"] != str(run_url):
            raise RuntimeError("W&B terminal URL differs from sealed run")
        write_json_once(artifact_dir / "wandb_terminal.json", wandb_terminal)
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
            "checkpoint_receipt_sha256": sha256_file(
                artifact_dir / "checkpoint_receipt.json"
            ),
            "wandb_terminal_sha256": sha256_file(
                artifact_dir / "wandb_terminal.json"
            ),
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
        _supervisor_progress(
            {
                "phase": "scientific_completion",
                "attempt_id": args.attempt_id,
                "artifact_index_sha256": sha256_file(
                    artifact_dir / "artifact_index.json"
                ),
                "checkpoint_sha256": checkpoint_receipt["checkpoint_sha256"],
                "selected_epoch": details["best_epoch"],
                "wandb": wandb_terminal,
            }
        )
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
