"""Stdlib-only verification for the Issue #106 MPS evidence chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FIRST_EPOCH_UPDATES = 55
NEXT_EPOCH_UPDATES = 21
TOTAL_UPDATES = FIRST_EPOCH_UPDATES + NEXT_EPOCH_UPDATES
GRADIENT_ACCUMULATION_STEPS = 8
TOTAL_MICROSTEPS = TOTAL_UPDATES * GRADIENT_ACCUMULATION_STEPS
VALIDATION_STEPS = 111
HEARTBEAT_ROWS = TOTAL_UPDATES + VALIDATION_STEPS + 1
SCIENCE_ARTIFACTS = {
    "config.json", "source.json", "case_identity.json", "dependency.lock",
    "resource_evidence.json", "acceptance_soak_heartbeat.jsonl", "epochs.json",
    "best_epoch_cases.json", "regression.json", "score.json",
    "best_checkpoint.pth", "run_record.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _read_heartbeat(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("heartbeat rows must contain objects")
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
        raise ValueError("memory evidence is incomplete")
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
        raise ValueError("raw gate update/validation evidence is missing")
    if len(updates) != TOTAL_UPDATES or len(validation) != VALIDATION_STEPS:
        raise ValueError("raw gate evidence count differs from contract")
    if rows[:FIRST_EPOCH_UPDATES] != updates[:FIRST_EPOCH_UPDATES]:
        raise ValueError("first-epoch heartbeat does not bind raw updates")
    if rows[FIRST_EPOCH_UPDATES:FIRST_EPOCH_UPDATES + VALIDATION_STEPS] != validation:
        raise ValueError("heartbeat does not bind raw validation")
    if rows[-NEXT_EPOCH_UPDATES:] != updates[-NEXT_EPOCH_UPDATES:]:
        raise ValueError("next-epoch heartbeat does not bind raw updates")
    headrooms: list[float] = []
    for memory_key, ratio_key in (
        ("memory_before", "memory_before_headroom_ratio"),
        ("memory_after_optimizer_path", "memory_after_headroom_ratio"),
    ):
        headrooms.append(_headroom({
            "memory": soak.get(memory_key),
            "headroom_ratio": soak.get(ratio_key),
        }))
    for update in updates:
        critical = update.get("critical_memory")
        if (
            update.get("distinct_microbatches") != GRADIENT_ACCUMULATION_STEPS
            or update.get("finite_losses") is not True
            or update.get("finite_gradient_norm") is not True
            or len(str(update.get("microbatch_identity_sha256", ""))) != 64
            or not isinstance(critical, dict)
            or critical.get("phase") != "backward_complete_pre_adamw_memory"
            or critical.get("tensor_scalar_materialized") is not False
            or critical.get("explicit_mps_synchronize_called") is not False
        ):
            raise ValueError("optimizer proof is incomplete")
        headrooms.extend((_headroom(critical), _headroom(update)))
    for row in validation:
        if row.get("finite_loss") is not True:
            raise ValueError("validation resource proof is incomplete")
        headrooms.append(_headroom(row))
    if len(str(soak.get("validation_identity_sha256", ""))) != 64:
        raise ValueError("validation identity digest is missing")
    if len(str(soak.get("loader_start_fingerprint", ""))) != 64:
        raise ValueError("loader fingerprint is missing")
    recomputed = min(headrooms)
    if recomputed < minimum or abs(
        float(soak.get("minimum_headroom_ratio", -1)) - recomputed
    ) > 1e-12:
        raise ValueError("minimum headroom summary differs from raw evidence")
    return recomputed


def _validate_supervisor_progress(
    progress_path: Path, heartbeat_rows: list[dict[str, Any]]
) -> str:
    progress = _read_heartbeat(progress_path)
    cursor = 0
    cleanup_index = None
    critical_count = 0
    for index, row in enumerate(progress):
        if row.get("phase") == "backward_complete_pre_adamw_memory":
            _headroom(row)
            critical_count += 1
        if row.get("phase") == "resource_process_cleanup":
            _headroom(row)
            cleanup_index = index
        if cursor < len(heartbeat_rows) and row == heartbeat_rows[cursor]:
            cursor += 1
    if cursor != len(heartbeat_rows):
        raise ValueError("supervisor progress does not contain the gate heartbeat")
    if cleanup_index is None or cleanup_index <= 0 or critical_count < TOTAL_UPDATES + 1:
        raise ValueError("supervisor progress lifecycle evidence is incomplete")
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
            supervisor_progress_path, rows
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


def validate_scientific_artifacts(science_dir: Path, attempt_id: str) -> dict[str, Any]:
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
    if not isinstance(epochs, list) or len(epochs) != 5:
        raise ValueError("scientific epoch evidence is incomplete")
    if not isinstance(cases, list) or len(cases) != 111:
        raise ValueError("scientific case evidence is incomplete")
    case_ids = [row.get("case_id") for row in cases if isinstance(row, dict)]
    if len(case_ids) != 111 or len(set(case_ids)) != 111:
        raise ValueError("scientific case identity coverage is incomplete")
    for number, epoch in enumerate(epochs, start=1):
        if not isinstance(epoch, dict):
            raise ValueError("scientific epoch evidence is incomplete")
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
        ):
            raise ValueError("scientific epoch coverage is incomplete")
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
    return {
        "artifact_index_sha256": sha256_file(index_path),
        "run_record_sha256": sha256_file(science_dir / "run_record.json"),
        "checkpoint_sha256": sha256_file(science_dir / "best_checkpoint.pth"),
        "score_sha256": sha256_file(science_dir / "score.json"),
        "case_evidence_sha256": sha256_file(science_dir / "best_epoch_cases.json"),
    }
