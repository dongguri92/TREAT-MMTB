"""Sealing and scoring helpers for the teammate lambda=0.5 reproduction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils import dice_metric

EXPECTED_TRAIN_CASES = 444
EXPECTED_VALIDATION_CASES = 111
DATASET_SCOPE = "task1_train_all_internal_labeled_validation"
CANONICAL_MANIFEST_SHA256 = (
    "98484d6d96b9f6898393331d0493fa4d22ed6af0057b29210e14d32be7d5aef8"
)
CANONICAL_TRAIN_CONTENT_SHA256 = (
    "11f2a977436d6a5df33075218f4464cecbef0ebd9deeaa034ca3fefd0fbfdfd3"
)
CANONICAL_VALIDATION_CONTENT_SHA256 = (
    "c73896c79068368eb7869cd8d5efdeeb8c68fde08e9923080cff5f224d6cc7db"
)
CANONICAL_COMBINED_CONTENT_SHA256 = (
    "e2ce106f14837603fc021496ab10bd7d9d9976d190efb4a151788f00d24709ca"
)
CURRENT_BASELINE_ID = "P2-B3-resolution-degradation-1e"
CURRENT_BASELINE_ATTEMPT_ID = "p2-resolution-seed2026-attempt-1"
CURRENT_BASELINE_DATASET_SCOPE = "internal_labeled_validation"
CURRENT_BASELINE_SCORE_SHA256 = (
    "9e50f50687887c834cc5ef6723295b100ca86bfc2b215b0d929169401d2af278"
)
CURRENT_BASELINE_RUN_RECORD_SHA256 = (
    "a76593dd2e2d0a9f332db7ea20872e2eec2bc6e5eee019311b48a546997f74ce"
)
WANDB_ENTITY = "kimhyeonwoo2431-individual"
WANDB_PROJECT = "treat-mmtb-task1"
CLS_THRESHOLD = 0.5
VETO_THRESHOLD = 0.005
MIN_PIXELS = 0
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_SAMPLES = 10_000
EXPECTED_PYTHON = (3, 11)
EXPECTED_PLATFORM_SYSTEM = "Linux"
EXPECTED_PLATFORM_MACHINE = "x86_64"
EXPECTED_TORCH_VERSION = "2.5.1+cu118"
EXPECTED_TORCHVISION_VERSION = "0.20.1+cu118"
EXPECTED_CUDA_VERSION = "11.8"
EXPECTED_PRETRAINED_NAME = "eva_x_small_patch16_merged520k_mim.pt"
EXPECTED_PRETRAINED_SHA256 = (
    "135d70a6988b5aacfe4848e1c2a0d524b2c076536fcaccdce88b636d302316c2"
)
EXPECTED_PRETRAINED_BYTES = 307_569_543
EXPECTED_PRETRAINED_SOURCE = (
    "https://huggingface.co/MapleF/eva_x/resolve/"
    "35ddcd6dab6ca99bbdb6cb45c8d1b093aefbd0ee/"
    "eva_x_small_patch16_merged520k_mim.pt"
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{Path(path).name} must contain a JSON object")
    return payload


def source_identity(repo_root: Path | str) -> dict[str, str]:
    root = Path(repo_root).resolve()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        text=True,
    )
    if status.strip():
        raise RuntimeError("scored attempts require a clean source worktree")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
    ).strip()
    return {"git_commit": commit, "git_tree_sha1": tree}


def reject_external_final_path(path: Path | str, label: str) -> None:
    """Reject a lexically forbidden path without touching the filesystem."""
    normalized = [
        "".join(ch if ch.isalnum() else "_" for ch in part.lower())
        for part in Path(path).parts
    ]
    forbidden = re.compile(r"external_final|external_test|final_test|test_final")
    if any(forbidden.search(part) for part in normalized):
        raise ValueError(f"{label} must not reference the external final test")


def resolve_non_external_path(path: Path | str, label: str) -> Path:
    """Reject named external-final paths before resolution, then reject targets."""
    reject_external_final_path(path, label)
    resolved = Path(path).resolve()
    reject_external_final_path(resolved, label)
    return resolved


def validate_pretrained(path: Path | str) -> dict[str, Any]:
    pretrained_path = Path(path)
    reject_external_final_path(pretrained_path, "pretrained")
    byte_size = pretrained_path.stat().st_size
    digest = sha256_file(pretrained_path)
    if pretrained_path.name != EXPECTED_PRETRAINED_NAME:
        raise ValueError("pretrained filename is not the intended EVA-X small weight")
    if byte_size != EXPECTED_PRETRAINED_BYTES or digest != EXPECTED_PRETRAINED_SHA256:
        raise ValueError("pretrained bytes are not the pinned EVA-X small weight")
    return {
        "name": EXPECTED_PRETRAINED_NAME,
        "byte_size": byte_size,
        "sha256": digest,
        "source": EXPECTED_PRETRAINED_SOURCE,
    }


def load_canonical_manifest(path: Path | str) -> dict[str, Any]:
    manifest_path = Path(path)
    digest = sha256_file(manifest_path)
    if digest != CANONICAL_MANIFEST_SHA256:
        raise ValueError("manifest hash is not the pinned canonical 444/111 manifest")
    payload = read_json(manifest_path)
    if (
        payload.get("dataset_scope") != DATASET_SCOPE
        or payload.get("train_case_count") != EXPECTED_TRAIN_CASES
        or payload.get("validation_case_count") != EXPECTED_VALIDATION_CASES
    ):
        raise ValueError("manifest dataset scope/count contract is invalid")
    split = payload.get("split")
    validation_cases = payload.get("validation_cases")
    if not isinstance(split, dict) or not isinstance(validation_cases, dict):
        raise TypeError("canonical manifest split/validation_cases are required")
    train_ids = split.get("train")
    prepared_validation_ids = split.get("val")
    if not isinstance(train_ids, list) or not isinstance(prepared_validation_ids, list):
        raise TypeError("canonical manifest train/val IDs must be arrays")
    if set(validation_cases) != set(prepared_validation_ids):
        raise ValueError("canonical validation metadata coverage is inconsistent")
    validation_ids: list[str] = []
    for prepared_id in prepared_validation_ids:
        metadata = validation_cases.get(prepared_id)
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("source_case_id"), str
        ):
            raise TypeError("canonical validation source identities are incomplete")
        validation_ids.append(metadata["source_case_id"])
    normalized_train = [str(value) for value in train_ids]
    if (
        len(normalized_train) != EXPECTED_TRAIN_CASES
        or len(set(normalized_train)) != EXPECTED_TRAIN_CASES
        or len(validation_ids) != EXPECTED_VALIDATION_CASES
        or len(set(validation_ids)) != EXPECTED_VALIDATION_CASES
        or set(normalized_train) & set(validation_ids)
    ):
        raise ValueError(
            "canonical manifest identities are not exact 444/111 disjoint sets"
        )
    identity = {
        "dataset_scope": DATASET_SCOPE,
        "train": sorted(normalized_train),
        "validation": sorted(validation_ids),
    }
    return {
        "manifest_sha256": digest,
        "identity": identity,
        "case_identity_sha256": canonical_sha256(identity),
    }


def _directory_ids(directory: Path, suffix: str) -> list[str]:
    return sorted(
        path.name[: -len(suffix)]
        for path in directory.glob(f"*{suffix}")
        if path.is_file()
    )


def _content_rows(
    case_ids: Sequence[str], dcm_dir: Path, mask_dir: Path
) -> list[dict[str, str]]:
    if _directory_ids(dcm_dir, ".dcm") != sorted(case_ids):
        raise ValueError(
            "DICOM directory identities differ from the canonical manifest"
        )
    if _directory_ids(mask_dir, ".nii.gz") != sorted(case_ids):
        raise ValueError("mask directory identities differ from the canonical manifest")
    return [
        {
            "case_id": case_id,
            "dicom_sha256": sha256_file(dcm_dir / f"{case_id}.dcm"),
            "mask_sha256": sha256_file(mask_dir / f"{case_id}.nii.gz"),
        }
        for case_id in sorted(case_ids)
    ]


def validate_canonical_content(
    identity: Mapping[str, Any],
    train_dcm_dir: Path,
    train_mask_dir: Path,
    validation_dcm_dir: Path,
    validation_mask_dir: Path,
) -> dict[str, str]:
    train_ids = identity.get("train")
    validation_ids = identity.get("validation")
    if not isinstance(train_ids, list) or not isinstance(validation_ids, list):
        raise TypeError("canonical case identity is incomplete")
    train_rows = _content_rows(train_ids, train_dcm_dir, train_mask_dir)
    validation_rows = _content_rows(
        validation_ids, validation_dcm_dir, validation_mask_dir
    )
    evidence = {
        "train_content_sha256": canonical_sha256(train_rows),
        "validation_content_sha256": canonical_sha256(validation_rows),
        "combined_content_sha256": canonical_sha256(
            {"train": train_rows, "validation": validation_rows}
        ),
    }
    expected = {
        "train_content_sha256": CANONICAL_TRAIN_CONTENT_SHA256,
        "validation_content_sha256": CANONICAL_VALIDATION_CONTENT_SHA256,
        "combined_content_sha256": CANONICAL_COMBINED_CONTENT_SHA256,
    }
    if evidence != expected:
        raise ValueError(
            "dataset bytes differ from the pinned canonical internal cohort"
        )
    return evidence


def validate_dataset_identity(
    train_loader: Any,
    val_loader: Any,
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    train_ids = sorted(str(value) for value in train_loader.dataset.ids)
    validation_ids = sorted(str(value) for value in val_loader.dataset.ids)
    identity: dict[str, Any] = {
        "dataset_scope": DATASET_SCOPE,
        "train": train_ids,
        "validation": validation_ids,
    }
    if expected_identity is None:
        if (
            len(train_ids) != EXPECTED_TRAIN_CASES
            or len(validation_ids) != EXPECTED_VALIDATION_CASES
        ):
            raise ValueError("dataset must contain exact 444/111 cohorts")
    elif identity != dict(expected_identity):
        raise ValueError("loader identities differ from the pinned canonical manifest")
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(
        validation_ids
    ):
        raise ValueError("dataset identities contain duplicates")
    if set(train_ids) & set(validation_ids):
        raise ValueError("training and validation identities overlap")
    return identity, canonical_sha256(identity)


def combo_veto_mask(
    foreground_probability: np.ndarray,
    cls_probability: float,
    cls_threshold: float = CLS_THRESHOLD,
    veto_threshold: float = VETO_THRESHOLD,
) -> np.ndarray:
    """Apply the historical 0.7356 combo rule in prepared space."""
    fg = np.asarray(foreground_probability)
    seg_probability = float(fg.max())
    cls_positive = float(cls_probability) >= cls_threshold
    seg_positive = seg_probability >= 0.5
    if cls_positive == seg_positive:
        return (fg >= 0.5).astype(np.uint8)
    if cls_positive and seg_probability >= veto_threshold:
        return (fg >= veto_threshold).astype(np.uint8)
    if cls_positive:
        return np.zeros(fg.shape, dtype=np.uint8)
    return (fg >= 0.5).astype(np.uint8)


def restore_native_mask(
    prepared_mask: np.ndarray,
    pad_info: Sequence[int],
    crop_shape: Sequence[int],
    native_shape: Sequence[int],
) -> np.ndarray:
    top, left, height, width = (int(value) for value in pad_info)
    crop_height, crop_width = (int(value) for value in crop_shape)
    native_height, native_width = (int(value) for value in native_shape)
    valid = np.asarray(prepared_mask)[top : top + height, left : left + width]
    crop = cv2.resize(
        valid.astype(np.uint8),
        (crop_width, crop_height),
        interpolation=cv2.INTER_NEAREST,
    )
    native = np.zeros((native_height, native_width), dtype=np.uint8)
    copy_height = min(crop_height, native_height)
    copy_width = min(crop_width, native_width)
    native[:copy_height, :copy_width] = crop[:copy_height, :copy_width]
    return native


def validate_coverage(
    expected_ids: Sequence[str], observed_ids: Sequence[str]
) -> dict[str, Any]:
    expected = [str(value) for value in expected_ids]
    observed = [str(value) for value in observed_ids]
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    duplicates = len(observed) - len(set(observed))
    coverage = {
        "expected": len(expected),
        "observed": len(observed),
        "unique": len(set(observed)),
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
    }
    if (
        len(expected) != EXPECTED_VALIDATION_CASES
        or len(observed) != len(expected)
        or missing
        or unexpected
        or duplicates
    ):
        raise ValueError(f"invalid validation coverage: {coverage}")
    return coverage


def aggregate_native_cases(
    cases: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]
) -> dict[str, Any]:
    coverage = validate_coverage(expected_ids, [str(case["case_id"]) for case in cases])
    accuracy = float(np.mean([int(case["correct"]) for case in cases]))
    dice_values = [
        np.nan if case.get("dice") is None else float(case["dice"]) for case in cases
    ]
    dice = float(np.nanmean(dice_values))
    composite = 0.7 * accuracy + 0.3 * dice
    if not all(math.isfinite(value) for value in (accuracy, dice, composite)):
        raise ValueError("native validation metrics must be finite")
    return {
        "classification_accuracy": accuracy,
        "dice": dice,
        "weighted_composite": composite,
        "coverage": coverage,
        "cases": list(cases),
    }


def native_case_record(
    case_id: str,
    foreground_probability: np.ndarray,
    cls_probability: float,
    native_mask: np.ndarray,
    pad_info: Sequence[int],
    crop_shape: Sequence[int],
    native_shape: Sequence[int],
) -> dict[str, Any]:
    prepared = combo_veto_mask(foreground_probability, cls_probability)
    predicted = restore_native_mask(prepared, pad_info, crop_shape, native_shape)
    truth = (np.asarray(native_mask).squeeze() > 0).astype(np.uint8)
    predicted_present = int(predicted.sum() > MIN_PIXELS)
    truth_present = int(truth.sum() > 0)
    dice = dice_metric(predicted, truth)
    error_type = (
        "true_positive"
        if predicted_present and truth_present
        else "false_positive"
        if predicted_present
        else "false_negative"
        if truth_present
        else "true_negative"
    )
    return {
        "case_id": str(case_id),
        "truth": truth_present,
        "prediction": predicted_present,
        "correct": int(predicted_present == truth_present),
        "dice": None if np.isnan(dice) else float(dice),
        "error_type": error_type,
        "cls_probability": float(cls_probability),
        "segmentation_max_probability": float(np.asarray(foreground_probability).max()),
        "predicted_pixels_native": int(predicted.sum()),
    }


def load_pinned_baseline(
    score_path: Path,
    run_record_path: Path,
    validation_ids: Sequence[str],
) -> dict[str, Any]:
    score_hash = sha256_file(score_path)
    record_hash = sha256_file(run_record_path)
    if score_hash != CURRENT_BASELINE_SCORE_SHA256:
        raise ValueError("baseline score is not the pinned current internal baseline")
    if record_hash != CURRENT_BASELINE_RUN_RECORD_SHA256:
        raise ValueError(
            "baseline run record is not the pinned current internal baseline"
        )
    score = read_json(score_path)
    record = read_json(run_record_path)
    if (
        score.get("experiment_id") != CURRENT_BASELINE_ID
        or score.get("attempt_id") != CURRENT_BASELINE_ATTEMPT_ID
        or score.get("dataset_scope") != CURRENT_BASELINE_DATASET_SCOPE
        or score.get("external_final_test_untouched") is not True
        or record.get("experiment_id") != CURRENT_BASELINE_ID
        or record.get("attempt_id") != CURRENT_BASELINE_ATTEMPT_ID
        or record.get("status") != "completed"
        or record.get("decision") != "promote"
        or record.get("train_case_count") != EXPECTED_TRAIN_CASES
        or record.get("validation_case_count") != EXPECTED_VALIDATION_CASES
    ):
        raise ValueError("pinned baseline status/identity contract is invalid")
    aggregate = score.get("aggregate")
    if not isinstance(aggregate, dict) or not isinstance(aggregate.get("cases"), list):
        raise TypeError("pinned baseline per-case evidence is missing")
    cases = aggregate["cases"]
    case_ids = [str(case.get("case_id")) for case in cases if isinstance(case, dict)]
    validate_coverage(validation_ids, case_ids)
    return {
        "baseline_id": CURRENT_BASELINE_ID,
        "attempt_id": CURRENT_BASELINE_ATTEMPT_ID,
        "score_sha256": score_hash,
        "run_record_sha256": record_hash,
        "cases": cases,
    }


def _mcnemar_exact(fixed: int, regressed: int) -> dict[str, Any]:
    discordant = fixed + regressed
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index) for index in range(min(fixed, regressed) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "fixed": fixed,
        "regressed": regressed,
        "discordant": discordant,
        "two_sided_exact_p_value": p_value,
    }


def build_paired_regression(
    candidate_cases: Sequence[Mapping[str, Any]],
    baseline_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate = {str(case["case_id"]): case for case in candidate_cases}
    baseline = {str(case["case_id"]): case for case in baseline_cases}
    if set(candidate) != set(baseline) or len(candidate) != EXPECTED_VALIDATION_CASES:
        raise ValueError("candidate and pinned baseline case coverage differ")
    rows: list[dict[str, Any]] = []
    fixed = regressed = unchanged_correct = unchanged_error = 0
    for case_id in sorted(candidate):
        child = candidate[case_id]
        parent = baseline[case_id]
        child_truth = bool(child["truth"])
        parent_truth = bool(parent["gt_has_cavity"])
        if child_truth != parent_truth:
            raise ValueError("candidate and baseline ground truth identities differ")
        child_correct = bool(child["correct"])
        parent_correct = bool(parent["cavity_correct"])
        if child_correct and not parent_correct:
            taxonomy = "fixed"
            fixed += 1
        elif parent_correct and not child_correct:
            taxonomy = "regressed"
            regressed += 1
        elif child_correct:
            taxonomy = "unchanged_correct"
            unchanged_correct += 1
        else:
            taxonomy = "unchanged_error"
            unchanged_error += 1
        child_dice = child.get("dice")
        parent_dice = parent.get("dice")
        rows.append(
            {
                "case_id": case_id,
                "taxonomy": taxonomy,
                "candidate_correct": child_correct,
                "baseline_correct": parent_correct,
                "candidate_dice": child_dice,
                "baseline_dice": parent_dice,
                "paired_dice_delta": (
                    float(child_dice) - float(parent_dice)
                    if child_dice is not None and parent_dice is not None
                    else None
                ),
            }
        )
    child_correct_array = np.asarray(
        [float(row["candidate_correct"]) for row in rows], dtype=np.float64
    )
    parent_correct_array = np.asarray(
        [float(row["baseline_correct"]) for row in rows], dtype=np.float64
    )
    child_dice_array = np.asarray(
        [
            np.nan if row["candidate_dice"] is None else row["candidate_dice"]
            for row in rows
        ],
        dtype=np.float64,
    )
    parent_dice_array = np.asarray(
        [
            np.nan if row["baseline_dice"] is None else row["baseline_dice"]
            for row in rows
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(rows), size=(BOOTSTRAP_SAMPLES, len(rows)), endpoint=False
    )
    child_accuracy = child_correct_array[indices].mean(axis=1)
    parent_accuracy = parent_correct_array[indices].mean(axis=1)
    child_dice = np.nanmean(child_dice_array[indices], axis=1)
    parent_dice = np.nanmean(parent_dice_array[indices], axis=1)
    deltas = {
        "classification_accuracy": child_accuracy - parent_accuracy,
        "dice": child_dice - parent_dice,
        "weighted_composite": (
            0.7 * child_accuracy
            + 0.3 * child_dice
            - (0.7 * parent_accuracy + 0.3 * parent_dice)
        ),
    }
    point_deltas = {
        "classification_accuracy": float(
            child_correct_array.mean() - parent_correct_array.mean()
        ),
        "dice": float(np.nanmean(child_dice_array) - np.nanmean(parent_dice_array)),
    }
    point_deltas["weighted_composite"] = (
        0.7 * point_deltas["classification_accuracy"] + 0.3 * point_deltas["dice"]
    )
    bootstrap = {
        name: {
            "point_delta": point_deltas[name],
            "percentile_95_ci": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ],
        }
        for name, values in deltas.items()
    }
    return {
        "schema_version": 1,
        "baseline_id": CURRENT_BASELINE_ID,
        "case_count": len(rows),
        "taxonomy_counts": {
            "fixed": fixed,
            "regressed": regressed,
            "unchanged_correct": unchanged_correct,
            "unchanged_error": unchanged_error,
        },
        "mcnemar": _mcnemar_exact(fixed, regressed),
        "paired_bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
            "resampling_unit": "case_id",
            "metrics": bootstrap,
        },
        "cases": rows,
    }


def locked_versions(lock_path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if (
            not line
            or raw_line[:1].isspace()
            or line.startswith(("#", "--"))
        ):
            continue
        requirement = line.removesuffix("\\").strip()
        requirement = requirement.split(" ;", 1)[0].strip()
        if "==" not in requirement:
            raise ValueError("dependency lock entries must use exact == pins")
        name, version = requirement.split("==", 1)
        versions[name.strip()] = version.strip()
    if not versions:
        raise ValueError("dependency lock must contain pinned distributions")
    return versions


def validate_runtime_dependencies(lock_path: Path) -> dict[str, Any]:
    expected = locked_versions(lock_path)
    observed: dict[str, str] = {}
    mismatches: dict[str, dict[str, str]] = {}
    for distribution, wanted in expected.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        observed[distribution] = actual
        if actual != wanted:
            mismatches[distribution] = {"expected": wanted, "observed": actual}
    if mismatches:
        raise RuntimeError(
            f"runtime dependency versions differ from lock: {mismatches}"
        )
    runtime = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "distributions": observed,
    }
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError("scored reproduction requires CPython 3.11")
    if platform.python_implementation() != "CPython":
        raise RuntimeError("scored reproduction requires CPython")
    if (
        runtime["platform_system"] != EXPECTED_PLATFORM_SYSTEM
        or runtime["platform_machine"] != EXPECTED_PLATFORM_MACHINE
    ):
        raise RuntimeError("scored reproduction requires Linux x86_64")
    if observed.get("torch") != EXPECTED_TORCH_VERSION:
        raise RuntimeError("runtime torch build differs from the verified CUDA build")
    if observed.get("torchvision") != EXPECTED_TORCHVISION_VERSION:
        raise RuntimeError("runtime torchvision build differs from the verified CUDA build")
    return runtime


def validate_epoch_evidence(
    epochs: Sequence[Mapping[str, Any]],
    requested_epochs: int,
    train_steps_per_epoch: int,
) -> None:
    if len(epochs) != requested_epochs:
        raise ValueError("epoch evidence count differs from the requested phase")
    finite_fields = (
        "train_loss",
        "validation_loss",
        "classification_accuracy",
        "dice",
        "weighted_composite",
        "runtime_seconds",
        "validation_seconds",
    )
    for expected_epoch, evidence in enumerate(epochs, start=1):
        if evidence.get("epoch") != expected_epoch:
            raise ValueError("epoch evidence ordering is incomplete")
        if not all(
            math.isfinite(float(evidence.get(name, math.nan))) for name in finite_fields
        ):
            raise ValueError("epoch evidence contains non-finite metrics")
        coverage = evidence.get("coverage")
        if not isinstance(coverage, dict) or coverage != {
            "expected": EXPECTED_VALIDATION_CASES,
            "observed": EXPECTED_VALIDATION_CASES,
            "unique": EXPECTED_VALIDATION_CASES,
            "missing": [],
            "unexpected": [],
            "duplicates": 0,
        }:
            raise ValueError("epoch evidence does not prove exact validation coverage")
        if evidence.get("optimizer_steps") != train_steps_per_epoch:
            raise ValueError("epoch evidence does not prove exact optimizer steps")
        if evidence.get("completed_train_steps") != expected_epoch * train_steps_per_epoch:
            raise ValueError("epoch evidence cumulative optimizer steps are incomplete")
    if requested_epochs * train_steps_per_epoch <= 0:
        raise ValueError("epoch evidence requires positive optimizer-step coverage")


def write_json_once(path: Path | str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(data)
