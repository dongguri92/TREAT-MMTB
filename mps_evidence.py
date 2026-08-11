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


def validate_gate_artifacts(
    gate_dir: Path, attempt_id: str, minimum_headroom_ratio: float
) -> dict[str, Any]:
    index_path = gate_dir / "acceptance_soak_index.json"
    resource_path = gate_dir / "resource_evidence.json"
    heartbeat_path = gate_dir / "acceptance_soak_heartbeat.jsonl"
    index = _read_object(index_path)
    resource = _read_object(resource_path)
    rows = _read_heartbeat(heartbeat_path)
    _validate_phase_order(rows)
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
        or soak.get("minimum_headroom_ratio", -1) < minimum_headroom_ratio
    ):
        raise ValueError("gate artifact chain differs from exact contract")
    return {
        "gate_index_sha256": sha256_file(index_path),
        "resource_evidence_sha256": resource_sha256,
        "heartbeat_sha256": heartbeat_sha256,
        "heartbeat_records": len(rows),
        "optimizer_updates": TOTAL_UPDATES,
        "micro_steps": TOTAL_MICROSTEPS,
        "validation_steps": VALIDATION_STEPS,
        "minimum_headroom_ratio": soak["minimum_headroom_ratio"],
        "bootstrap_proof_sha256": resource.get("bootstrap", {}).get(
            "proof_sha256"
        ),
    }
