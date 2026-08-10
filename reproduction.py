"""Sealed helpers for the Task 1 teammate lambda=0.5 reproduction."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from utils import dice_metric


EXPECTED_TRAIN_CASES = 444
EXPECTED_VALIDATION_CASES = 111
WANDB_ENTITY = "kimhyeonwoo2431-individual"
WANDB_PROJECT = "treat-mmtb-task1"
CLS_THRESHOLD = 0.5
VETO_THRESHOLD = 0.01
MIN_PIXELS = 0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_identity(repo_root):
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


def reject_external_final_path(path, label):
    normalized = [
        "".join(ch if ch.isalnum() else "_" for ch in part.lower())
        for part in Path(path).resolve().parts
    ]
    forbidden = ("external_final", "external_test", "final_test", "test_final")
    if any(any(token in part for token in forbidden) for part in normalized):
        raise ValueError(f"{label} must not reference the external final test")


def validate_dataset_identity(train_loader, val_loader):
    train_ids = [str(value) for value in train_loader.dataset.ids]
    validation_ids = [str(value) for value in val_loader.dataset.ids]
    if len(train_ids) != EXPECTED_TRAIN_CASES or len(set(train_ids)) != len(train_ids):
        raise ValueError("training cohort must contain exactly 444 unique cases")
    if (
        len(validation_ids) != EXPECTED_VALIDATION_CASES
        or len(set(validation_ids)) != len(validation_ids)
    ):
        raise ValueError("validation cohort must contain exactly 111 unique cases")
    if set(train_ids) & set(validation_ids):
        raise ValueError("training and validation identities overlap")
    identity = {"train": sorted(train_ids), "validation": sorted(validation_ids)}
    return identity, canonical_sha256(identity)


def combo_veto_mask(
    foreground_probability,
    cls_probability,
    cls_threshold=CLS_THRESHOLD,
    veto_threshold=VETO_THRESHOLD,
):
    """Apply the repository-native combo/veto rule in prepared space."""
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


def restore_native_mask(prepared_mask, pad_info, crop_shape, native_shape):
    top, left, height, width = (int(value) for value in pad_info)
    crop_height, crop_width = (int(value) for value in crop_shape)
    native_height, native_width = (int(value) for value in native_shape)
    valid = np.asarray(prepared_mask)[top:top + height, left:left + width]
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


def validate_coverage(expected_ids, observed_ids):
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


def aggregate_native_cases(cases, expected_ids):
    coverage = validate_coverage(expected_ids, [case["case_id"] for case in cases])
    accuracy = float(np.mean([case["correct"] for case in cases]))
    dice_values = [
        np.nan if case["dice"] is None else case["dice"] for case in cases
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
        "cases": cases,
    }


def native_case_record(case_id, foreground_probability, cls_probability,
                       native_mask, pad_info, crop_shape, native_shape):
    prepared = combo_veto_mask(foreground_probability, cls_probability)
    predicted = restore_native_mask(prepared, pad_info, crop_shape, native_shape)
    truth = (np.asarray(native_mask).squeeze() > 0).astype(np.uint8)
    predicted_present = int(predicted.sum() > MIN_PIXELS)
    truth_present = int(truth.sum() > 0)
    dice = dice_metric(predicted, truth)
    if predicted_present and truth_present:
        error_type = "true_positive"
    elif predicted_present:
        error_type = "false_positive"
    elif truth_present:
        error_type = "false_negative"
    else:
        error_type = "true_negative"
    return {
        "case_id": str(case_id),
        "truth": truth_present,
        "prediction": predicted_present,
        "correct": int(predicted_present == truth_present),
        "dice": None if np.isnan(dice) else float(dice),
        "error_type": error_type,
        "cls_probability": float(cls_probability),
        "segmentation_max_probability": float(
            np.asarray(foreground_probability).max()
        ),
        "predicted_pixels_native": int(predicted.sum()),
    }


def write_json_once(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(data)
