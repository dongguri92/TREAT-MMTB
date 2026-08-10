import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import reproduction
import reproduce_teammate_l05 as launcher
from reproduce_teammate_l05 import (
    ARTIFACT_NAMES,
    _validate_health_index,
    build_config,
    parse_args,
    protocol_contract,
    run,
)
from reproduction import (
    EXPECTED_TRAIN_CASES,
    EXPECTED_VALIDATION_CASES,
    aggregate_native_cases,
    build_paired_regression,
    combo_veto_mask,
    restore_native_mask,
    validate_canonical_content,
    validate_dataset_identity,
    write_json_once,
)


@pytest.mark.parametrize(
    ("segmentation", "classification", "expected"),
    [
        (0.004999, 0.499999, 0),
        (0.005, 0.5, 1),
        (0.004999, 0.5, 0),
        (0.5, 0.499999, 1),
        (0.5, 0.5, 1),
    ],
)
def test_combo_veto_golden_parity_boundaries(
    segmentation: float, classification: float, expected: int
) -> None:
    foreground = np.array([[0.0, segmentation]], dtype=np.float64)
    assert int(combo_veto_mask(foreground, classification)[0, 1]) == expected


def test_restore_native_mask_removes_padding_and_restores_crop() -> None:
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


def test_aggregate_requires_exact_111_unique_case_coverage() -> None:
    ids = [str(index) for index in range(EXPECTED_VALIDATION_CASES)]
    cases = [{"case_id": case_id, "correct": 1, "dice": 0.5} for case_id in ids]
    aggregate = aggregate_native_cases(cases, ids)
    assert aggregate["coverage"]["unique"] == 111
    assert aggregate["classification_accuracy"] == 1.0
    assert aggregate["dice"] == 0.5
    with pytest.raises(ValueError, match="invalid validation coverage"):
        aggregate_native_cases(cases[:-1], ids)


def test_dataset_identity_requires_exact_non_overlapping_444_111() -> None:
    train = SimpleNamespace(
        dataset=SimpleNamespace(ids=[f"train-{i}" for i in range(EXPECTED_TRAIN_CASES)])
    )
    validation = SimpleNamespace(
        dataset=SimpleNamespace(
            ids=[f"validation-{i}" for i in range(EXPECTED_VALIDATION_CASES)]
        )
    )
    identity, digest = validate_dataset_identity(train, validation)
    assert identity["dataset_scope"] == reproduction.DATASET_SCOPE
    assert len(digest) == 64


def test_canonical_content_binds_bytes_and_rejects_extra_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directories = [tmp_path / name for name in ("td", "tm", "vd", "vm")]
    for directory in directories:
        directory.mkdir()
    identity = {"train": ["train"], "validation": ["validation"]}
    (directories[0] / "train.dcm").write_bytes(b"train-dcm")
    (directories[1] / "train.nii.gz").write_bytes(b"train-mask")
    (directories[2] / "validation.dcm").write_bytes(b"val-dcm")
    (directories[3] / "validation.nii.gz").write_bytes(b"val-mask")
    rows_train = [
        {
            "case_id": "train",
            "dicom_sha256": reproduction.sha256_file(directories[0] / "train.dcm"),
            "mask_sha256": reproduction.sha256_file(directories[1] / "train.nii.gz"),
        }
    ]
    rows_validation = [
        {
            "case_id": "validation",
            "dicom_sha256": reproduction.sha256_file(directories[2] / "validation.dcm"),
            "mask_sha256": reproduction.sha256_file(
                directories[3] / "validation.nii.gz"
            ),
        }
    ]
    monkeypatch.setattr(
        reproduction,
        "CANONICAL_TRAIN_CONTENT_SHA256",
        reproduction.canonical_sha256(rows_train),
    )
    monkeypatch.setattr(
        reproduction,
        "CANONICAL_VALIDATION_CONTENT_SHA256",
        reproduction.canonical_sha256(rows_validation),
    )
    monkeypatch.setattr(
        reproduction,
        "CANONICAL_COMBINED_CONTENT_SHA256",
        reproduction.canonical_sha256(
            {"train": rows_train, "validation": rows_validation}
        ),
    )
    assert (
        validate_canonical_content(identity, *directories)["combined_content_sha256"]
        == reproduction.CANONICAL_COMBINED_CONTENT_SHA256
    )
    (directories[0] / "extra.dcm").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="identities differ"):
        validate_canonical_content(identity, *directories)


def test_paired_regression_has_exact_taxonomy_mcnemar_and_bootstrap() -> None:
    candidate = []
    baseline = []
    for index in range(EXPECTED_VALIDATION_CASES):
        candidate_correct = index != 1
        baseline_correct = index != 0
        candidate.append(
            {
                "case_id": str(index),
                "truth": 1,
                "correct": candidate_correct,
                "dice": 0.6,
            }
        )
        baseline.append(
            {
                "case_id": str(index),
                "gt_has_cavity": True,
                "cavity_correct": baseline_correct,
                "dice": 0.5,
            }
        )
    evidence = build_paired_regression(candidate, baseline)
    assert evidence["taxonomy_counts"] == {
        "fixed": 1,
        "regressed": 1,
        "unchanged_correct": 109,
        "unchanged_error": 0,
    }
    assert evidence["mcnemar"]["two_sided_exact_p_value"] == 1.0
    assert evidence["paired_bootstrap"]["samples"] == 10_000
    assert evidence["paired_bootstrap"]["metrics"]["dice"][
        "point_delta"
    ] == pytest.approx(0.1)


def _cli(tmp_path: Path) -> list[str]:
    values = {
        "manifest": "manifest.json",
        "baseline-score": "baseline-score.json",
        "baseline-run-record": "baseline-record.json",
        "pretrained": "pretrained.pt",
        "train-dcm-dir": "train_dcm",
        "train-mask-dir": "train_mask",
        "val-dcm-dir": "val_dcm",
        "val-mask-dir": "val_mask",
    }
    argv = [
        "--phase",
        "health",
        "--attempt-id",
        "health-001",
        "--artifact-root",
        str(tmp_path / "artifacts"),
    ]
    for flag, value in values.items():
        argv.extend((f"--{flag}", str(tmp_path / value)))
    return argv


def test_launcher_dry_run_is_fixed_and_does_not_leak_paths(tmp_path: Path) -> None:
    args = parse_args(_cli(tmp_path))
    config = build_config(args, {"git_commit": "a", "git_tree_sha1": "b"}, "c", "cpu")
    protocol = config["protocol"]
    assert protocol["physical_batch_size"] == protocol["effective_batch_size"] == 8
    assert protocol["inference"]["t_veto"] == 0.005
    assert config["wandb"]["id"] == "health-001"
    assert config["wandb"]["resume"] == "never"
    serialized = str(config)
    assert str(tmp_path) not in serialized
    result = run(args)
    assert result == {
        "status": "dry_run",
        "phase": "health",
        "epochs": 5,
        "attempt_id": "health-001",
        "review_required": True,
    }
    assert not (tmp_path / "artifacts").exists()


def test_health_index_recomputes_every_artifact_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    source = {"git_commit": "a", "git_tree_sha1": "b"}
    identity = {
        "dataset_scope": reproduction.DATASET_SCOPE,
        "train": [f"train-{index}" for index in range(EXPECTED_TRAIN_CASES)],
        "validation": [
            f"validation-{index}" for index in range(EXPECTED_VALIDATION_CASES)
        ],
    }
    protocol = protocol_contract("health")
    config = {
        "attempt_id": "health-001",
        "reviewed_by": "reviewer",
        "protocol": protocol,
        "protocol_sha256": reproduction.canonical_sha256(protocol),
        "pretrained_sha256": "pretrained",
        "dataset_scope": reproduction.DATASET_SCOPE,
        "manifest": {"manifest_sha256": "manifest"},
        "content": {"combined_content_sha256": "content"},
        "baseline": {"score_sha256": "baseline"},
        "dependency": {"lock_sha256": "lock"},
        "case_identity_sha256": reproduction.canonical_sha256(identity),
    }
    coverage = {
        "expected": 111,
        "observed": 111,
        "unique": 111,
        "missing": [],
        "unexpected": [],
        "duplicates": 0,
    }
    epochs = [
        {
            "epoch": epoch,
            "train_loss": 1.0,
            "validation_loss": 1.0,
            "classification_accuracy": 0.5,
            "dice": 0.5,
            "weighted_composite": 0.5,
            "runtime_seconds": 1.0,
            "validation_seconds": 1.0,
            "coverage": coverage,
        }
        for epoch in range(1, 6)
    ]
    record = {
        "status": "completed",
        "reviewed_by": "reviewer",
        "completed_epochs": 5,
        "completed_train_steps": 275,
        "train_steps_per_epoch": 55,
        "validation_steps_per_epoch": 111,
        "wandb_finished": True,
        "wandb": {
            "id": "health-001",
            "entity": reproduction.WANDB_ENTITY,
            "project": reproduction.WANDB_PROJECT,
            "mode": "online",
            "resume": "never",
            "url": "https://wandb.example/health-001",
        },
    }
    payloads = {
        "config.json": config,
        "source.json": source,
        "case_identity.json": identity,
        "epochs.json": {"epochs": epochs},
        "best_epoch_cases.json": {"cases": []},
        "regression.json": {"case_count": 111},
        "score.json": {"status": "scored"},
        "run_record.json": record,
    }
    for name, payload in payloads.items():
        write_json_once(tmp_path / name, payload)
    (tmp_path / "dependency.lock").write_text("package==1\n", encoding="utf-8")
    (tmp_path / "best_checkpoint.pth").write_bytes(b"checkpoint")
    index = {
        "status": "completed",
        "phase": "health",
        "artifacts": {
            name: reproduction.sha256_file(tmp_path / name) for name in ARTIFACT_NAMES
        },
    }
    write_json_once(tmp_path / "artifact_index.json", index)
    assert (
        _validate_health_index(tmp_path / "artifact_index.json", config, source)[
            "attempt_id"
        ]
        == "health-001"
    )
    (tmp_path / "score.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_health_index(tmp_path / "artifact_index.json", config, source)


def test_setup_failure_is_sanitized_and_cannot_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = parse_args(_cli(tmp_path) + ["--execute", "--reviewed-by", "reviewer"])

    def fail_source_identity(_root: Path) -> dict[str, str]:
        raise RuntimeError("sensitive /local/input/path")

    monkeypatch.setattr(launcher, "source_identity", fail_source_identity)
    with pytest.raises(RuntimeError, match="sensitive"):
        run(args)
    attempt = tmp_path / "artifacts" / "health-001"
    failure = reproduction.read_json(attempt / "failure.json")
    assert failure["stage"] == "source_identity"
    assert "error" not in failure
    assert not (attempt / "run_record.json").exists()
    assert not (attempt / "artifact_index.json").exists()


def test_write_json_once_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    write_json_once(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_json_once(path, {"value": 2})
