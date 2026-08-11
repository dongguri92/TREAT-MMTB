"""Stdlib-only verification for the Issue #106 MPS evidence chain."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

FIRST_EPOCH_UPDATES = 55
NEXT_EPOCH_UPDATES = 21
TOTAL_UPDATES = FIRST_EPOCH_UPDATES + NEXT_EPOCH_UPDATES
GRADIENT_ACCUMULATION_STEPS = 8
TOTAL_MICROSTEPS = TOTAL_UPDATES * GRADIENT_ACCUMULATION_STEPS
VALIDATION_STEPS = 111
TARGET_SIZE = 1024
PHYSICAL_BATCH_SIZE = 1
EFFECTIVE_BATCH_SIZE = 8
FEASIBILITY_PROBE = "exact_8_microbatch_adamw_optimizer_update"
FEASIBILITY_LEARNING_RATE = 1e-5
CLS_THRESHOLD = 0.5
VETO_THRESHOLD = 0.005
MIN_PIXELS = 0
HEARTBEAT_ROWS = TOTAL_UPDATES + VALIDATION_STEPS + 1
VALIDATION_OPERATION_EVENTS = [
    "device_transfers",
    "forward",
    "loss_item",
    "log_loss_item",
    "prepared_masks_cpu",
    "classification_sum_item",
    "probabilities_cpu",
    "native_metadata_cpu",
]
SCIENCE_ARTIFACTS = {
    "config.json", "source.json", "case_identity.json", "bootstrap_proof.json",
    "dependency.lock",
    "resource_evidence.json", "acceptance_soak_heartbeat.jsonl", "epochs.json",
    "best_epoch_cases.json", "regression.json", "score.json",
    "best_checkpoint.pth", "checkpoint_receipt.json", "wandb_terminal.json",
    "run_record.json",
}
BF16_PRECISION_CONTRACT = {
    "mode": "mixed_precision",
    "autocast_device_type": "mps",
    "autocast_dtype": "bfloat16",
    "parameter_dtype": "float32",
    "loss_compute_dtype": "float32",
    "optimizer_master_state_dtype": "float32",
    "gradient_scaler": False,
    "fallback_policy": "fail_closed",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is not a canonical SHA-256 digest")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _read_heartbeat(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError("heartbeat rows must contain objects")
        rows.append(row)
    return rows


def _validate_phase_order(rows: list[dict[str, Any]]) -> None:
    if len(rows) != HEARTBEAT_ROWS:
        raise ValueError("gate heartbeat row count differs from exact contract")
    first = rows[:FIRST_EPOCH_UPDATES]
    validation = rows[
        FIRST_EPOCH_UPDATES : FIRST_EPOCH_UPDATES + VALIDATION_STEPS
    ]
    transition = rows[FIRST_EPOCH_UPDATES + VALIDATION_STEPS]
    second = rows[FIRST_EPOCH_UPDATES + VALIDATION_STEPS + 1 :]
    if [row.get("phase") for row in first] != [
        "first_epoch_train"
    ] * FIRST_EPOCH_UPDATES:
        raise ValueError("first-epoch heartbeat phase order is invalid")
    if [row.get("optimizer_update") for row in first] != list(
        range(1, FIRST_EPOCH_UPDATES + 1)
    ):
        raise ValueError("first-epoch optimizer update sequence is invalid")
    if [row.get("completed_micro_steps") for row in first] != list(
        range(
            GRADIENT_ACCUMULATION_STEPS,
            FIRST_EPOCH_UPDATES * GRADIENT_ACCUMULATION_STEPS + 1,
            GRADIENT_ACCUMULATION_STEPS,
        )
    ):
        raise ValueError("first-epoch microstep sequence is invalid")
    if [row.get("phase") for row in validation] != [
        "validation_resource"
    ] * VALIDATION_STEPS:
        raise ValueError("validation heartbeat phase order is invalid")
    if [row.get("validation_step") for row in validation] != list(
        range(1, VALIDATION_STEPS + 1)
    ):
        raise ValueError("validation heartbeat sequence is invalid")
    if (
        transition.get("phase") != "next_epoch_transition"
        or transition.get("completed_optimizer_updates") != FIRST_EPOCH_UPDATES
    ):
        raise ValueError("next-epoch transition heartbeat is invalid")
    if [row.get("phase") for row in second] != [
        "next_epoch_train"
    ] * NEXT_EPOCH_UPDATES:
        raise ValueError("next-epoch heartbeat phase order is invalid")
    if [row.get("optimizer_update") for row in second] != list(
        range(FIRST_EPOCH_UPDATES + 1, TOTAL_UPDATES + 1)
    ):
        raise ValueError("next-epoch optimizer update sequence is invalid")
    if [row.get("completed_micro_steps") for row in second] != list(
        range(
            (FIRST_EPOCH_UPDATES + 1) * GRADIENT_ACCUMULATION_STEPS,
            TOTAL_MICROSTEPS + 1,
            GRADIENT_ACCUMULATION_STEPS,
        )
    ):
        raise ValueError("next-epoch microstep sequence is invalid")


def _headroom(record: dict[str, Any]) -> float:
    memory = record.get("memory", record)
    if not isinstance(memory, dict):
        raise TypeError("memory evidence is incomplete")
    driver = memory.get("driver_allocated_bytes")
    recommended = memory.get("recommended_max_bytes")
    if not isinstance(driver, int) or not isinstance(recommended, int) or recommended <= 0:
        raise ValueError("memory evidence is incomplete")
    computed = 1.0 - driver / recommended
    observed = record.get("headroom_ratio")
    if not isinstance(observed, (int, float)) or abs(observed - computed) > 1e-12:
        raise ValueError("memory headroom was not recomputed from raw bytes")
    return float(computed)


def _validate_raw_gate_rows(
    rows: list[dict[str, Any]], resource: dict[str, Any], minimum: float
) -> float:
    soak = resource.get("acceptance_soak", {})
    updates = soak.get("updates")
    validation = soak.get("validation")
    if not isinstance(updates, list) or not isinstance(validation, list):
        raise TypeError("raw gate update/validation evidence is missing")
    if len(updates) != TOTAL_UPDATES or len(validation) != VALIDATION_STEPS:
        raise ValueError("raw gate evidence count differs from contract")
    canonical = resource.get("canonical_case_sha256", {})
    bootstrap = resource.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise TypeError("bootstrap evidence is missing")
    proof_sha256 = bootstrap.get("proof_sha256")
    proof_body = {key: value for key, value in bootstrap.items() if key != "proof_sha256"}
    if (
        _require_sha256(proof_sha256, "bootstrap proof")
        != canonical_sha256(proof_body)
        or bootstrap.get("dataset_identity", {}).get("canonical_case_sha256")
        != canonical
    ):
        raise ValueError("canonical case digests differ from sealed bootstrap source")
    train_ids = canonical.get("train")
    validation_ids = canonical.get("validation")
    if (
        not isinstance(train_ids, list)
        or len(train_ids) != 444
        or len(set(train_ids)) != 444
        or not isinstance(validation_ids, list)
        or len(validation_ids) != VALIDATION_STEPS
        or len(set(validation_ids)) != VALIDATION_STEPS
    ):
        raise ValueError("canonical case digest source is incomplete")
    for digest in [*train_ids, *validation_ids]:
        _require_sha256(digest, "case identity")
    if rows[:FIRST_EPOCH_UPDATES] != updates[:FIRST_EPOCH_UPDATES]:
        raise ValueError("first-epoch heartbeat does not bind raw updates")
    if rows[FIRST_EPOCH_UPDATES:FIRST_EPOCH_UPDATES + VALIDATION_STEPS] != validation:
        raise ValueError("heartbeat does not bind raw validation")
    if rows[-NEXT_EPOCH_UPDATES:] != updates[-NEXT_EPOCH_UPDATES:]:
        raise ValueError("next-epoch heartbeat does not bind raw updates")
    headrooms: list[float] = []

    def validate_update(update: Any) -> list[str]:
        if not isinstance(update, dict):
            raise TypeError("optimizer update proof is incomplete")
        critical = update.get("critical_memory")
        identities = update.get("microbatch_case_sha256_ordered")
        if (
            update.get("distinct_microbatches") != GRADIENT_ACCUMULATION_STEPS
            or update.get("finite_losses") is not True
            or update.get("gradient_clip_completed") is not True
            or update.get("precision") != BF16_PRECISION_CONTRACT
            or update.get("master_state")
            != {
                "parameter_dtypes": ["torch.float32"],
                "optimizer_state_dtypes": ["torch.float32"],
            }
            or not isinstance(identities, list)
            or len(identities) != GRADIENT_ACCUMULATION_STEPS
            or len(set(identities)) != GRADIENT_ACCUMULATION_STEPS
            or any(value not in train_ids for value in identities)
            or update.get("microbatch_identity_sha256")
            != canonical_sha256(sorted(identities))
            or not isinstance(critical, dict)
            or critical.get("phase") != "backward_complete_pre_adamw_memory"
            or critical.get("tensor_scalar_materialized") is not False
            or critical.get("explicit_mps_synchronize_called") is not False
        ):
            raise ValueError("optimizer proof is incomplete")
        headrooms.extend((_headroom(critical), _headroom(update)))
        return identities

    probe = resource.get("probe")
    if not isinstance(probe, dict):
        raise TypeError("feasibility probe evidence is missing")
    probe_updates = probe.get("updates")
    probe_optimizer = probe.get("optimizer")
    sealed_loader_start = bootstrap.get("loader_start")
    probe_loader_components = probe.get("loader_start_components")
    if (
        resource.get("precision") != BF16_PRECISION_CONTRACT
        or probe.get("precision") != BF16_PRECISION_CONTRACT
        or probe.get("precision_runtime")
        != {**BF16_PRECISION_CONTRACT, "runtime_probe": "passed"}
        or probe.get("status") != "passed"
        or probe.get("probe") != FEASIBILITY_PROBE
        or probe.get("target_size") != TARGET_SIZE
        or probe.get("physical_batch_size") != PHYSICAL_BATCH_SIZE
        or probe.get("gradient_accumulation_steps")
        != GRADIENT_ACCUMULATION_STEPS
        or probe.get("effective_batch_size") != EFFECTIVE_BATCH_SIZE
        or probe.get("requested_optimizer_updates") != 1
        or probe.get("requested_micro_steps") != GRADIENT_ACCUMULATION_STEPS
        or probe.get("wandb_started") is not False
        or probe.get("completed_optimizer_updates") != 1
        or probe.get("completed_micro_steps") != GRADIENT_ACCUMULATION_STEPS
        or not isinstance(probe_updates, list)
        or len(probe_updates) != 1
        or not isinstance(probe_optimizer, dict)
        or probe_optimizer.get("name") != "adamw"
        or not _same_number(
            probe_optimizer.get("learning_rate"), FEASIBILITY_LEARNING_RATE
        )
        or probe_optimizer.get("gradient_clip_norm") != 12
        or not isinstance(probe_loader_components, list)
        or len(probe_loader_components) != GRADIENT_ACCUMULATION_STEPS
        or probe.get("loader_start_fingerprint")
        != canonical_sha256(probe_loader_components)
        or not isinstance(sealed_loader_start, dict)
        or sealed_loader_start.get("components") != probe_loader_components
        or sealed_loader_start.get("fingerprint")
        != probe.get("loader_start_fingerprint")
        or probe.get("cleanup", {}).get("status") != "passed"
    ):
        raise ValueError("feasibility probe differs from exact contract")
    probe_identities = validate_update(probe_updates[0])
    if [
        case_id
        for component in probe_loader_components
        for case_id in component.get("case_sha256_ordered", [])
        if isinstance(component, dict)
    ] != probe_identities:
        raise ValueError("feasibility update differs from sealed loader start")
    probe_cleanup = probe.get("cleanup")
    if not isinstance(probe_cleanup, dict):
        raise TypeError("feasibility cleanup evidence is incomplete")
    headrooms.append(_headroom(probe_cleanup))
    for memory_key, ratio_key in (
        ("memory_before", "memory_before_headroom_ratio"),
        ("memory_after_optimizer_path", "memory_after_headroom_ratio"),
    ):
        headrooms.append(_headroom({
            "memory": soak.get(memory_key),
            "headroom_ratio": soak.get(ratio_key),
        }))
    first_epoch_ids: list[str] = []
    next_epoch_ids: list[str] = []
    for update_index, update in enumerate(updates):
        identities = validate_update(update)
        target = (
            first_epoch_ids
            if update_index < FIRST_EPOCH_UPDATES
            else next_epoch_ids
        )
        target.extend(identities)
    if (
        len(first_epoch_ids) != 440
        or len(set(first_epoch_ids)) != 440
        or len(next_epoch_ids) != 168
        or len(set(next_epoch_ids)) != 168
        or soak.get("first_epoch_unique_case_count") != 440
        or soak.get("next_epoch_unique_case_count") != 168
        or soak.get("first_epoch_identity_sha256")
        != canonical_sha256(sorted(first_epoch_ids))
        or soak.get("next_epoch_identity_sha256")
        != canonical_sha256(sorted(next_epoch_ids))
    ):
        raise ValueError("soak epoch identity coverage is invalid")
    for row in validation:
        _require_sha256(row.get("case_sha256"), "validation case")
        if (
            row.get("finite_loss") is not True
            or row.get("precision") != BF16_PRECISION_CONTRACT
            or row.get("operation_events") != VALIDATION_OPERATION_EVENTS
        ):
            raise ValueError("validation resource proof is incomplete")
        headrooms.append(_headroom(row))
    observed_validation = [row["case_sha256"] for row in validation]
    if observed_validation != validation_ids:
        raise ValueError("validation identity order differs from canonical source")
    if soak.get("validation_identity_sha256") != canonical_sha256(
        sorted(observed_validation)
    ):
        raise ValueError("validation identity digest does not recompute")
    loader_components = soak.get("loader_start_components")
    if (
        not isinstance(loader_components, list)
        or len(loader_components) != GRADIENT_ACCUMULATION_STEPS
        or soak.get("loader_start_fingerprint")
        != canonical_sha256(loader_components)
        or not isinstance(sealed_loader_start, dict)
        or sealed_loader_start.get("components") != loader_components
        or sealed_loader_start.get("fingerprint")
        != soak.get("loader_start_fingerprint")
    ):
        raise ValueError("loader fingerprint does not recompute")
    for component in loader_components:
        if not isinstance(component, dict):
            raise TypeError("loader component is invalid")
        component_ids = component.get("case_sha256_ordered")
        tensors = component.get("tensors")
        if (
            not isinstance(component_ids, list)
            or len(component_ids) != 1
            or component_ids[0] not in train_ids
            or not isinstance(tensors, dict)
            or set(tensors) != {"image", "mask", "cls"}
        ):
            raise ValueError("loader component source is incomplete")
        for tensor in tensors.values():
            if not isinstance(tensor, dict) or not isinstance(tensor.get("shape"), list):
                raise TypeError("loader tensor component is incomplete")
            _require_sha256(tensor.get("sha256"), "loader tensor")
    if [
        case_id
        for component in loader_components
        for case_id in component["case_sha256_ordered"]
    ] != updates[0].get("microbatch_case_sha256_ordered"):
        raise ValueError("soak first update differs from sealed loader start")
    recomputed = min(headrooms)
    if recomputed < minimum or abs(
        float(soak.get("minimum_headroom_ratio", -1)) - recomputed
    ) > 1e-12:
        raise ValueError("minimum headroom summary differs from raw evidence")
    return recomputed


def _validate_supervisor_progress(
    progress_path: Path,
    heartbeat_rows: list[dict[str, Any]],
    resource: dict[str, Any],
    cleanup: dict[str, Any],
) -> str:
    progress = _read_heartbeat(progress_path)
    probe_update = resource["probe"]["updates"][0]
    expected = [probe_update["critical_memory"], probe_update]
    for row in heartbeat_rows:
        if row.get("phase") in {"first_epoch_train", "next_epoch_train"}:
            expected.append(row["critical_memory"])
        expected.append(row)
    expected.append({"phase": "resource_process_cleanup", **cleanup})
    if progress != expected:
        raise ValueError("supervisor progress differs from exact gate sequence")
    for row in progress:
        if row.get("phase") in {
            "backward_complete_pre_adamw_memory", "resource_process_cleanup"
        }:
            _headroom(row)
    return sha256_file(progress_path)


def validate_gate_artifacts(
    gate_dir: Path,
    attempt_id: str,
    minimum_headroom_ratio: float,
    supervisor_progress_path: Path | None = None,
    expected_bootstrap_proof_sha256: str | None = None,
) -> dict[str, Any]:
    index_path = gate_dir / "acceptance_soak_index.json"
    resource_path = gate_dir / "resource_evidence.json"
    heartbeat_path = gate_dir / "acceptance_soak_heartbeat.jsonl"
    index = _read_object(index_path)
    resource = _read_object(resource_path)
    rows = _read_heartbeat(heartbeat_path)
    _validate_phase_order(rows)
    recomputed_minimum = _validate_raw_gate_rows(
        rows, resource, minimum_headroom_ratio
    )
    resource_sha256 = sha256_file(resource_path)
    heartbeat_sha256 = sha256_file(heartbeat_path)
    soak = resource.get("acceptance_soak", {})
    cleanup = index.get("cleanup", {})
    if (
        index.get("schema_version") != 2
        or index.get("attempt_id") != attempt_id
        or index.get("status") != "passed"
        or index.get("precision") != BF16_PRECISION_CONTRACT
        or index.get("resource_evidence_sha256") != resource_sha256
        or index.get("heartbeat_sha256") != heartbeat_sha256
        or index.get("optimizer_updates") != TOTAL_UPDATES
        or index.get("micro_steps") != TOTAL_MICROSTEPS
        or index.get("validation_steps") != VALIDATION_STEPS
        or index.get("heartbeat_records") != HEARTBEAT_ROWS
        or index.get("wandb_started") is not False
        or index.get("scientific_training_started") is not False
        or index.get("external_final_test_untouched") is not True
        or cleanup.get("status") != "passed"
        or cleanup.get("gc_collected") is not True
        or cleanup.get("empty_cache_completed") is not True
        or cleanup.get("synchronize_completed") is not True
        or soak.get("status") != "passed"
        or soak.get("completed_optimizer_updates") != TOTAL_UPDATES
        or soak.get("completed_micro_steps") != TOTAL_MICROSTEPS
        or soak.get("completed_validation_steps") != VALIDATION_STEPS
        or soak.get("heartbeat_records_written") != HEARTBEAT_ROWS
        or cleanup.get("headroom_ratio", -1) < minimum_headroom_ratio
    ):
        raise ValueError("gate artifact chain differs from exact contract")
    progress_sha256 = None
    if supervisor_progress_path is not None:
        progress_sha256 = _validate_supervisor_progress(
            supervisor_progress_path, rows, resource, cleanup
        )
    bootstrap_sha256 = resource.get("bootstrap", {}).get("proof_sha256")
    if (
        expected_bootstrap_proof_sha256 is not None
        and bootstrap_sha256 != expected_bootstrap_proof_sha256
    ):
        raise ValueError("child bootstrap proof differs from supervisor proof")
    return {
        "gate_index_sha256": sha256_file(index_path),
        "resource_evidence_sha256": resource_sha256,
        "heartbeat_sha256": heartbeat_sha256,
        "heartbeat_records": len(rows),
        "optimizer_updates": TOTAL_UPDATES,
        "micro_steps": TOTAL_MICROSTEPS,
        "validation_steps": VALIDATION_STEPS,
        "minimum_headroom_ratio": recomputed_minimum,
        "bootstrap_proof_sha256": bootstrap_sha256,
        "supervisor_progress_sha256": progress_sha256,
    }


def _same_number(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and abs(float(left) - float(right)) <= 1e-12
    )


def _validate_scientific_progress(
    progress_path: Path,
    attempt_id: str,
    artifact_index_sha256: str,
    checkpoint_sha256: str,
    selected_epoch: int,
    wandb_terminal: dict[str, Any],
) -> str:
    rows = _read_heartbeat(progress_path)
    cursor = 0
    global_step = 0
    for epoch in range(1, 6):
        for optimizer_step in range(1, 56):
            global_step += 1
            critical = rows[cursor]
            cursor += 1
            if (
                critical.get("phase") != "backward_complete_pre_adamw_memory"
                or critical.get("epoch") != epoch
                or critical.get("optimizer_step_in_epoch") != optimizer_step
                or critical.get("global_optimizer_step") != global_step
            ):
                raise ValueError("scientific critical-memory sequence is invalid")
            _headroom(critical)
            for offset in range(1, 9):
                train = rows[cursor]
                cursor += 1
                expected_micro = (optimizer_step - 1) * 8 + offset
                if (
                    train.get("phase") != "scientific_train"
                    or train.get("epoch") != epoch
                    or train.get("micro_step") != expected_micro
                    or train.get("optimizer_step") != global_step
                    or train.get("optimizer_step_completed") is not (offset == 8)
                ):
                    raise ValueError("scientific train progress sequence is invalid")
        for validation_step in range(1, 112):
            validation = rows[cursor]
            cursor += 1
            if (
                validation.get("phase") != "scientific_validation"
                or validation.get("epoch") != epoch
                or validation.get("validation_step") != validation_step
            ):
                raise ValueError("scientific validation progress sequence is invalid")
    if cursor >= len(rows):
        raise ValueError("scientific completion progress is missing")
    completion = rows[cursor]
    cursor += 1
    if (
        cursor != len(rows)
        or completion.get("phase") != "scientific_completion"
        or completion.get("attempt_id") != attempt_id
        or completion.get("artifact_index_sha256") != artifact_index_sha256
        or completion.get("checkpoint_sha256") != checkpoint_sha256
        or completion.get("selected_epoch") != selected_epoch
        or completion.get("wandb") != wandb_terminal
    ):
        raise ValueError("scientific completion progress is invalid")
    return sha256_file(progress_path)


def validate_scientific_artifacts(
    science_dir: Path,
    attempt_id: str,
    supervisor_progress_path: Path | None = None,
    expected_bootstrap_proof_sha256: str | None = None,
    independent_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    index_path = science_dir / "artifact_index.json"
    index = _read_object(index_path)
    artifacts = index.get("artifacts")
    if (
        index.get("schema_version") != 1
        or index.get("attempt_id") != attempt_id
        or index.get("status") != "completed"
        or not isinstance(artifacts, dict)
        or set(artifacts) != SCIENCE_ARTIFACTS
    ):
        raise ValueError("scientific artifact index is incomplete")
    for name, digest in artifacts.items():
        path = science_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError("scientific artifact hash differs")
    run_record = _read_object(science_dir / "run_record.json")
    epochs = _read_object(science_dir / "epochs.json").get("epochs")
    score = _read_object(science_dir / "score.json")
    cases = _read_object(science_dir / "best_epoch_cases.json").get("cases")
    identity = _read_object(science_dir / "case_identity.json")
    config = _read_object(science_dir / "config.json")
    source = _read_object(science_dir / "source.json")
    bootstrap = _read_object(science_dir / "bootstrap_proof.json")
    resource = _read_object(science_dir / "resource_evidence.json")
    regression = _read_object(science_dir / "regression.json")
    checkpoint_receipt = _read_object(science_dir / "checkpoint_receipt.json")
    wandb_terminal = _read_object(science_dir / "wandb_terminal.json")
    bootstrap_sha = bootstrap.get("proof_sha256")
    bootstrap_body = {
        key: value for key, value in bootstrap.items() if key != "proof_sha256"
    }
    if (
        _require_sha256(bootstrap_sha, "scientific bootstrap proof")
        != canonical_sha256(bootstrap_body)
        or (
            expected_bootstrap_proof_sha256 is not None
            and bootstrap_sha != expected_bootstrap_proof_sha256
        )
        or bootstrap.get("dataset_identity", {}).get("canonical_case_sha256")
        != resource.get("canonical_case_sha256")
        or config.get("resource_gate_receipt_sha256")
        != bootstrap.get("soak_approval", {}).get("resource_gate_receipt_sha256")
        or source.get("git_commit")
        != bootstrap.get("soak_approval", {}).get("source_git_commit")
    ):
        raise ValueError("scientific artifacts differ from sealed bootstrap source")
    if not isinstance(epochs, list) or len(epochs) != 5:
        raise ValueError("scientific epoch evidence is incomplete")
    if not isinstance(cases, list) or len(cases) != 111:
        raise ValueError("scientific case evidence is incomplete")
    case_ids = [row.get("case_id") for row in cases if isinstance(row, dict)]
    expected_ids = identity.get("validation")
    expected_digests = [
        hashlib.sha256(str(case_id).encode("utf-8")).hexdigest()
        for case_id in expected_ids
    ] if isinstance(expected_ids, list) else []
    if (
        not isinstance(expected_ids, list)
        or len(expected_ids) != 111
        or len(set(expected_ids)) != 111
        or case_ids != expected_ids
        or expected_digests != resource.get("canonical_case_sha256", {}).get(
            "validation"
        )
    ):
        raise ValueError("scientific case identity coverage is incomplete")
    correct = [row.get("correct") for row in cases]
    dice_values = [row.get("dice") for row in cases if row.get("dice") is not None]
    if (
        any(value not in (0, 1) for value in correct)
        or not dice_values
        or any(not isinstance(value, (int, float)) for value in dice_values)
    ):
        raise ValueError("scientific raw case metrics are incomplete")
    accuracy = sum(correct) / len(correct)
    dice = sum(float(value) for value in dice_values) / len(dice_values)
    weighted = 0.7 * accuracy + 0.3 * dice
    metrics = score.get("metrics", {})
    exact_coverage = {
        "expected": 111,
        "observed": 111,
        "unique": 111,
        "missing": [],
        "unexpected": [],
        "duplicates": 0,
    }
    if not all(
        _same_number(metrics.get(name), value)
        for name, value in (
            ("classification_accuracy", accuracy),
            ("dice", dice),
            ("weighted_composite", weighted),
        )
    ) or metrics.get("coverage") != exact_coverage:
        raise ValueError("scientific aggregate metrics do not recompute")
    error_types = {
        (1, 1): "true_positive",
        (0, 1): "false_positive",
        (1, 0): "false_negative",
        (0, 0): "true_negative",
    }
    for row in cases:
        if not isinstance(row, dict):
            raise TypeError("scientific native case row is incomplete")
        truth = row.get("truth")
        prediction = row.get("prediction")
        correct_value = row.get("correct")
        case_dice = row.get("dice")
        cls_probability = row.get("cls_probability")
        segmentation_probability = row.get("segmentation_max_probability")
        pixels = row.get("predicted_pixels_native")
        truth_pixels = row.get("truth_pixels_native")
        intersection_pixels = row.get("intersection_pixels_native")
        branch = row.get("decision_branch")
        threshold = row.get("applied_segmentation_threshold")
        expected_prediction = (
            int(pixels > MIN_PIXELS) if isinstance(pixels, int) else None
        )
        expected_truth = (
            int(truth_pixels > 0) if isinstance(truth_pixels, int) else None
        )
        denominator = (
            pixels + truth_pixels
            if isinstance(pixels, int) and isinstance(truth_pixels, int)
            else -1
        )
        expected_dice = (
            None
            if denominator == 0
            else 2 * intersection_pixels / denominator
            if denominator > 0 and isinstance(intersection_pixels, int)
            else math.nan
        )
        if isinstance(cls_probability, (int, float)) and isinstance(
            segmentation_probability, (int, float)
        ):
            cls_positive = float(cls_probability) >= CLS_THRESHOLD
            seg_positive = float(segmentation_probability) >= 0.5
            if cls_positive == seg_positive:
                expected_branch, expected_threshold = "agreement", 0.5
            elif cls_positive and float(segmentation_probability) >= VETO_THRESHOLD:
                expected_branch, expected_threshold = (
                    "cls_positive_veto_recovery", VETO_THRESHOLD
                )
            elif cls_positive:
                expected_branch, expected_threshold = (
                    "cls_positive_below_veto_empty", None
                )
            else:
                expected_branch, expected_threshold = (
                    "cls_negative_seg_positive", 0.5
                )
        else:
            expected_branch, expected_threshold = None, None
        if (
            truth not in (0, 1)
            or prediction not in (0, 1)
            or prediction != expected_prediction
            or truth != expected_truth
            or correct_value != int(truth == prediction)
            or row.get("error_type") != error_types[(truth, prediction)]
            or (
                case_dice is not None
                and (
                    not isinstance(case_dice, (int, float))
                    or not math.isfinite(float(case_dice))
                    or not 0 <= float(case_dice) <= 1
                )
            )
            or not isinstance(cls_probability, (int, float))
            or not 0 <= float(cls_probability) <= 1
            or not isinstance(segmentation_probability, (int, float))
            or not 0 <= float(segmentation_probability) <= 1
            or not isinstance(pixels, int)
            or pixels < 0
            or not isinstance(truth_pixels, int)
            or truth_pixels < 0
            or not isinstance(intersection_pixels, int)
            or intersection_pixels < 0
            or intersection_pixels > min(pixels, truth_pixels)
            or branch != expected_branch
            or threshold != expected_threshold
            or (
                branch == "cls_positive_below_veto_empty"
                and pixels != 0
            )
            or (
                expected_dice is None
                and case_dice is not None
            )
            or (
                expected_dice is not None
                and not _same_number(case_dice, expected_dice)
            )
        ):
            raise ValueError("scientific native case semantics are invalid")
    for number, epoch in enumerate(epochs, start=1):
        if not isinstance(epoch, dict):
            raise TypeError("scientific epoch evidence is incomplete")
        coverage = epoch.get("coverage", {})
        if (
            epoch.get("epoch") != number
            or epoch.get("optimizer_steps") != 55
            or epoch.get("micro_steps") != 440
            or epoch.get("completed_train_steps") != number * 55
            or coverage.get("expected") != 111
            or coverage.get("observed") != 111
            or coverage.get("unique") != 111
            or coverage.get("duplicates") != 0
            or coverage.get("missing") != []
            or coverage.get("unexpected") != []
            or not _same_number(
                epoch.get("weighted_composite"),
                0.7 * float(epoch.get("classification_accuracy", math.nan))
                + 0.3 * float(epoch.get("dice", math.nan)),
            )
        ):
            raise ValueError("scientific epoch coverage is incomplete")
    selected_epoch = score.get("selected_epoch")
    best_epoch = max(epochs, key=lambda row: float(row["weighted_composite"]))
    if (
        not isinstance(selected_epoch, int)
        or selected_epoch != best_epoch.get("epoch")
        or not all(
            _same_number(metrics.get(name), best_epoch.get(name))
            for name in ("classification_accuracy", "dice", "weighted_composite")
        )
    ):
        raise ValueError("scientific best epoch selection does not recompute")
    regression_rows = regression.get("cases")
    if not isinstance(regression_rows, list) or len(regression_rows) != 111:
        raise ValueError("scientific regression rows are incomplete")
    regression_ids = [
        row.get("case_id") for row in regression_rows if isinstance(row, dict)
    ]
    if regression_ids != sorted(expected_ids) or len(set(regression_ids)) != 111:
        raise ValueError("scientific regression identity coverage is invalid")
    candidate = {row["case_id"]: row for row in cases}
    taxonomy_counts = {
        "fixed": 0, "regressed": 0, "unchanged_correct": 0, "unchanged_error": 0
    }
    for row in regression_rows:
        if not isinstance(row, dict) or row.get("case_id") not in candidate:
            raise ValueError("scientific regression identity is invalid")
        child = candidate[row["case_id"]]
        if row.get("candidate_correct") is not bool(child["correct"]):
            raise ValueError("scientific regression correctness differs from raw case")
        parent_correct = row.get("baseline_correct")
        child_correct = bool(child["correct"])
        expected_taxonomy = (
            "fixed" if child_correct and parent_correct is False
            else "regressed" if not child_correct and parent_correct is True
            else "unchanged_correct" if child_correct
            else "unchanged_error"
        )
        if row.get("taxonomy") != expected_taxonomy:
            raise ValueError("scientific regression taxonomy does not recompute")
        taxonomy_counts[expected_taxonomy] += 1
    if regression.get("taxonomy_counts") != taxonomy_counts:
        raise ValueError("scientific regression counts do not recompute")
    if (
        regression.get("case_count") != 111
        or regression.get("baseline_id") != "P2-B3-resolution-degradation-1e"
        or config.get("baseline", {}).get("score_sha256")
        != resource.get("baseline_score_sha256")
        or config.get("baseline", {}).get("run_record_sha256")
        != resource.get("baseline_run_record_sha256")
    ):
        raise ValueError("scientific regression baseline binding is invalid")
    wandb = run_record.get("wandb", {})
    if (
        run_record.get("status") != "completed"
        or run_record.get("wandb_finished") is not True
        or run_record.get("external_final_test_untouched") is not True
        or run_record.get("completed_epochs") != 5
        or run_record.get("completed_train_micro_steps") != 2200
        or run_record.get("completed_optimizer_steps") != 275
        or run_record.get("metrics") != score.get("metrics")
        or run_record.get("selected_epoch") != score.get("selected_epoch")
        or set(run_record.get("artifacts", [])) != SCIENCE_ARTIFACTS
        or wandb.get("id") != attempt_id
        or wandb.get("entity") != "kimhyeonwoo2431-individual"
        or wandb.get("project") != "treat-mmtb-task1"
        or not str(wandb.get("url", "")).startswith("https://wandb.ai/")
        or (science_dir / "best_checkpoint.pth").stat().st_size <= 0
    ):
        raise ValueError("scientific run record is incomplete")
    checkpoint_sha256 = sha256_file(science_dir / "best_checkpoint.pth")
    if (
        checkpoint_receipt.get("attempt_id") != attempt_id
        or checkpoint_receipt.get("selected_epoch") != selected_epoch
        or checkpoint_receipt.get("source_git_commit") != source.get("git_commit")
        or checkpoint_receipt.get("config_sha256")
        != sha256_file(science_dir / "config.json")
        or checkpoint_receipt.get("resource_evidence_sha256")
        != sha256_file(science_dir / "resource_evidence.json")
        or checkpoint_receipt.get("pretrained_sha256")
        != resource.get("pretrained_sha256")
        or checkpoint_receipt.get("case_identity_sha256")
        != canonical_sha256(identity)
        or checkpoint_receipt.get("checkpoint_sha256") != checkpoint_sha256
        or run_record.get("checkpoint_receipt_sha256")
        != sha256_file(science_dir / "checkpoint_receipt.json")
    ):
        raise ValueError("scientific checkpoint semantic binding is invalid")
    if (
        wandb_terminal.get("id") != attempt_id
        or wandb_terminal.get("entity") != "kimhyeonwoo2431-individual"
        or wandb_terminal.get("project") != "treat-mmtb-task1"
        or wandb_terminal.get("state") != "finished"
        or wandb_terminal.get("url") != wandb.get("url")
        or wandb_terminal.get("config_sha256")
        != config.get("wandb", {}).get("config_sha256")
        or wandb_terminal.get("verified_summary", {}).get("best/epoch")
        != selected_epoch
        or wandb_terminal.get("verified_summary", {}).get(
            "resource/evidence_sha256"
        )
        != sha256_file(science_dir / "resource_evidence.json")
        or wandb_terminal.get("verified_summary", {}).get("regression/sha256")
        != sha256_file(science_dir / "regression.json")
        or wandb_terminal.get("verified_summary", {}).get("checkpoint/sha256")
        != checkpoint_sha256
        or run_record.get("wandb_terminal_sha256")
        != sha256_file(science_dir / "wandb_terminal.json")
    ):
        raise ValueError("scientific W&B terminal evidence is invalid")
    if (
        not isinstance(independent_verification, dict)
        or independent_verification.get("attempt_id") != attempt_id
        or independent_verification.get("checkpoint_sha256") != checkpoint_sha256
        or independent_verification.get("regression_sha256")
        != sha256_file(science_dir / "regression.json")
        or independent_verification.get("wandb_id") != attempt_id
        or independent_verification.get("wandb_state") != "finished"
        or independent_verification.get("wandb_url") != wandb.get("url")
        or independent_verification.get("wandb_config_sha256")
        != wandb_terminal.get("config_sha256")
        or independent_verification.get("wandb_summary_sha256")
        != wandb_terminal.get("summary_sha256")
    ):
        raise ValueError("independent checkpoint/W&B verification is missing")
    checkpoint_metadata = independent_verification.get("checkpoint_metadata")
    if (
        not isinstance(checkpoint_metadata, dict)
        or checkpoint_metadata.get("native_epoch") != selected_epoch - 1
        or checkpoint_metadata.get("selected_epoch") != selected_epoch
        or not _same_number(
            checkpoint_metadata.get("native_best_metric"),
            metrics.get("weighted_composite"),
        )
        or not _same_number(
            checkpoint_metadata.get("selected_weighted_composite"),
            metrics.get("weighted_composite"),
        )
        or not isinstance(checkpoint_metadata.get("model_keys"), list)
        or not checkpoint_metadata["model_keys"]
        or not isinstance(checkpoint_metadata.get("optimizer_keys"), list)
        or not checkpoint_metadata["optimizer_keys"]
        or independent_verification.get("checkpoint_metadata_sha256")
        != canonical_sha256(checkpoint_metadata)
    ):
        raise ValueError("independent checkpoint metadata verification is invalid")
    progress_sha256 = None
    if supervisor_progress_path is not None:
        progress_sha256 = _validate_scientific_progress(
            supervisor_progress_path,
            attempt_id,
            sha256_file(index_path),
            checkpoint_sha256,
            selected_epoch,
            wandb_terminal,
        )
    return {
        "artifact_index_sha256": sha256_file(index_path),
        "run_record_sha256": sha256_file(science_dir / "run_record.json"),
        "checkpoint_sha256": checkpoint_sha256,
        "score_sha256": sha256_file(science_dir / "score.json"),
        "case_evidence_sha256": sha256_file(science_dir / "best_epoch_cases.json"),
        "supervisor_progress_sha256": progress_sha256,
    }
