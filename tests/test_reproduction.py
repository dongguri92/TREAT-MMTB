import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reproduce_teammate_l05 import _phase_record, build_config, parse_args, run
from reproduction import (
    EXPECTED_TRAIN_CASES,
    EXPECTED_VALIDATION_CASES,
    aggregate_native_cases,
    combo_veto_mask,
    restore_native_mask,
    validate_dataset_identity,
    write_json_once,
)


def test_combo_veto_matches_native_repository_rule():
    strong_segmentation = np.array([[0.0, 0.8]], dtype=np.float32)
    weak_but_accepted = np.array([[0.0, 0.02]], dtype=np.float32)
    vetoed = np.array([[0.0, 0.009]], dtype=np.float32)

    assert combo_veto_mask(strong_segmentation, 0.1).tolist() == [[0, 1]]
    assert combo_veto_mask(weak_but_accepted, 0.9).tolist() == [[0, 1]]
    assert combo_veto_mask(vetoed, 0.9).tolist() == [[0, 0]]


def test_restore_native_mask_removes_padding_and_restores_crop():
    prepared = np.zeros((6, 6), dtype=np.uint8)
    prepared[1:5, 2:4] = 1
    restored = restore_native_mask(
        prepared,
        pad_info=(1, 2, 4, 2),
        crop_shape=(4, 4),
        native_shape=(5, 4),
    )
    assert restored.shape == (5, 4)
    assert restored[:4].sum() == 16
    assert restored[4].sum() == 0


def test_aggregate_requires_exact_111_unique_case_coverage():
    ids = [str(index) for index in range(EXPECTED_VALIDATION_CASES)]
    cases = [
        {
            "case_id": case_id,
            "correct": 1,
            "dice": None,
        }
        for case_id in ids
    ]
    cases[0]["dice"] = 0.5
    aggregate = aggregate_native_cases(cases, ids)
    assert aggregate["coverage"] == {
        "expected": 111,
        "observed": 111,
        "unique": 111,
        "missing": [],
        "unexpected": [],
        "duplicates": 0,
    }
    assert aggregate["classification_accuracy"] == 1.0
    assert aggregate["dice"] == 0.5

    with pytest.raises(ValueError, match="invalid validation coverage"):
        aggregate_native_cases(cases[:-1], ids)


def test_dataset_identity_requires_exact_non_overlapping_444_111():
    train = SimpleNamespace(
        dataset=SimpleNamespace(ids=[f"train-{i}" for i in range(EXPECTED_TRAIN_CASES)])
    )
    validation = SimpleNamespace(
        dataset=SimpleNamespace(
            ids=[f"validation-{i}" for i in range(EXPECTED_VALIDATION_CASES)]
        )
    )
    identity, digest = validate_dataset_identity(train, validation)
    assert len(identity["train"]) == 444
    assert len(identity["validation"]) == 111
    assert len(digest) == 64


def _healthy_record():
    return {
        "attempt_id": "health-attempt",
        "phase": "health",
        "status": "completed",
        "completed_epochs": 5,
        "external_final_test_untouched": True,
        "metrics": {
            "classification_accuracy": 0.9,
            "dice": 0.3,
            "weighted_composite": 0.72,
            "coverage": {
                "expected": 111,
                "observed": 111,
                "unique": 111,
                "missing": [],
                "unexpected": [],
                "duplicates": 0,
            },
        },
    }


def test_convergence_health_gate_accepts_only_completed_exact_health_record(tmp_path):
    path = tmp_path / "health.json"
    path.write_text(json.dumps(_healthy_record()), encoding="utf-8")
    assert _phase_record(path)["attempt_id"] == "health-attempt"

    broken = _healthy_record()
    broken["metrics"]["coverage"]["unique"] = 110
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="completed healthy"):
        _phase_record(path)


def test_launcher_is_review_gated_dry_run_with_fixed_protocol(tmp_path):
    common = [
        "--phase", "health",
        "--attempt-id", "health-001",
        "--artifact-root", str(tmp_path / "artifacts"),
        "--pretrained", str(tmp_path / "pretrained.pt"),
        "--train-dcm-dir", str(tmp_path / "train_dcm"),
        "--train-mask-dir", str(tmp_path / "train_mask"),
        "--val-dcm-dir", str(tmp_path / "val_dcm"),
        "--val-mask-dir", str(tmp_path / "val_mask"),
    ]
    args = parse_args(common)
    config = build_config(
        args, {"git_commit": "a", "git_tree_sha1": "b"}, "c", "cpu"
    )
    assert config["lambda_cls"] == 0.5
    assert config["target_size"] == 1024
    assert config["batch_size"] == config["effective_batch_size"] == 8
    assert config["inference"] == {
        "detection": "combo",
        "cls_threshold": 0.5,
        "t_veto": 0.01,
        "min_pixels": 0,
    }
    assert config["wandb"] == {
        "entity": "kimhyeonwoo2431-individual",
        "project": "treat-mmtb-task1",
        "mode": "online",
        "run_name": "task1-teammate-l05-veto-health-health-001",
    }
    result = run(args)
    assert result["status"] == "dry_run"
    assert result["epochs"] == 5
    assert result["review_required"] is True
    assert not (tmp_path / "artifacts").exists()

    args = parse_args(common + ["--execute"])
    with pytest.raises(ValueError, match="reviewed-by"):
        run(args)

    with pytest.raises(SystemExit):
        parse_args([*common[:3], "../escaped", *common[4:]])


def test_write_json_once_refuses_attempt_artifact_overwrite(tmp_path):
    path = tmp_path / "record.json"
    write_json_once(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_json_once(path, {"value": 2})
