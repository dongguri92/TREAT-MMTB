import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import reproduce_teammate_l05 as launcher
import reproduction
from models_evax import _load_weights_only_checkpoint
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


def test_canonical_content_rejects_nested_symlink_before_any_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directories = [tmp_path / name for name in ("td", "tm", "vd", "vm")]
    for directory in directories:
        directory.mkdir()
    identity = {"train": ["train"], "validation": ["validation"]}
    (directories[0] / "train.dcm").write_bytes(b"train-dcm")
    (directories[1] / "train.nii.gz").write_bytes(b"train-mask")
    (directories[2] / "validation.dcm").write_bytes(b"val-dcm")
    forbidden = tmp_path / "external-final" / "validation.nii.gz"
    forbidden.parent.mkdir()
    forbidden.write_bytes(b"must-not-be-read")
    (directories[3] / "validation.nii.gz").symlink_to(forbidden)

    def unexpected_hash(_path: Path | str) -> str:
        raise AssertionError("dataset bytes must not be read before symlink rejection")

    monkeypatch.setattr(reproduction, "sha256_file", unexpected_hash)
    with pytest.raises(ValueError, match="symlinked dataset entries"):
        validate_canonical_content(identity, *directories)


def test_torch_251_safe_globals_allowlist_loads_numpy_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "numpy-checkpoint.pt"
    import torch

    torch.save({"numpy_scalar": np.float64(1.25)}, checkpoint)
    loaded = _load_weights_only_checkpoint(checkpoint)
    assert float(loaded["numpy_scalar"]) == 1.25


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
    config = build_config(
        args,
        {"git_commit": "a", "git_tree_sha1": "b"},
        {"name": "weight.pt", "byte_size": 1, "sha256": "c", "source": "pinned"},
        "cpu",
    )
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


def test_external_final_path_is_rejected_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def unexpected_resolve(_self: Path) -> Path:
        nonlocal called
        called = True
        raise AssertionError("filesystem resolution must not occur")

    monkeypatch.setattr(Path, "resolve", unexpected_resolve)
    argv = _cli(tmp_path)
    argv[argv.index("--manifest") + 1] = str(tmp_path / "external-final.json")
    with pytest.raises(ValueError, match="external final"):
        parse_args(argv)
    assert called is False


def test_pretrained_validation_pins_name_size_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weight = tmp_path / reproduction.EXPECTED_PRETRAINED_NAME
    weight.write_bytes(b"verified-weight")
    monkeypatch.setattr(reproduction, "EXPECTED_PRETRAINED_BYTES", weight.stat().st_size)
    monkeypatch.setattr(
        reproduction, "EXPECTED_PRETRAINED_SHA256", reproduction.sha256_file(weight)
    )
    assert reproduction.validate_pretrained(weight)["sha256"] == reproduction.sha256_file(
        weight
    )
    weight.write_bytes(b"different-weight")
    with pytest.raises(ValueError, match="pinned EVA-X"):
        reproduction.validate_pretrained(weight)


def test_dependency_lock_pins_verified_torch_api_environment() -> None:
    lock = Path(__file__).resolve().parents[1] / "requirements-reproduction.lock"
    versions = reproduction.locked_versions(lock)
    assert versions["torch"] == reproduction.EXPECTED_TORCH_VERSION
    assert versions["torchvision"] == reproduction.EXPECTED_TORCHVISION_VERSION
    assert versions["timm"] == "1.0.22"
    assert "--hash=sha256:" in lock.read_text(encoding="utf-8")


def test_wandb_config_contains_only_aggregate_and_hash_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = parse_args(_cli(tmp_path))
    config = build_config(
        args,
        {"git_commit": "a", "git_tree_sha1": "b"},
        {"name": "weight.pt", "byte_size": 1, "sha256": "c", "source": "pinned"},
        "cuda",
        manifest={
            "manifest_sha256": "manifest",
            "identity": {"train": ["secret-train"], "validation": ["secret-val"]},
        },
        content={"combined_content_sha256": "content"},
        baseline={
            "baseline_id": "baseline",
            "score_sha256": "score",
            "run_record_sha256": "record",
            "cases": [{"case_id": "secret-val"}],
        },
        dependency={
            "lock_sha256": "lock",
            "python": "3.11.0",
            "torch": "2.5.1+cu118",
            "cuda": "11.8",
            "distributions": {"torchvision": "0.20.1+cu118"},
        },
    )
    config["case_identity_sha256"] = "identity"
    public = launcher._wandb_public_config(config)
    serialized = str(public)
    assert "secret-train" not in serialized
    assert "secret-val" not in serialized
    assert str(tmp_path) not in serialized
    assert public["dataset"] == {
        "scope": reproduction.DATASET_SCOPE,
        "train_case_count": 444,
        "validation_case_count": 111,
        "manifest_sha256": "manifest",
        "case_identity_sha256": "identity",
        "combined_content_sha256": "content",
    }
    captured: dict[str, object] = {}

    class FakeWandbRun:
        def define_metric(self, *_args: object, **_kwargs: object) -> None:
            pass

    def fake_init(**kwargs: object) -> FakeWandbRun:
        captured.update(kwargs)
        return FakeWandbRun()

    config["wandb"]["config_sha256"] = reproduction.canonical_sha256(public)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    launcher._start_wandb(config)
    assert captured["config"] == public


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
    health_protocol = protocol_contract("health")
    convergence_protocol = protocol_contract("convergence")
    health_config = {
        "attempt_id": "health-001",
        "reviewed_by": "reviewer",
        "num_workers": 4,
        "protocol": health_protocol,
        "protocol_sha256": reproduction.canonical_sha256(health_protocol),
        "pretrained": {"sha256": "pretrained"},
        "dataset_scope": reproduction.DATASET_SCOPE,
        "manifest": {"manifest_sha256": "manifest"},
        "content": {"combined_content_sha256": "content"},
        "baseline": {"score_sha256": "baseline"},
        "dependency": {"lock_sha256": "lock"},
        "case_identity_sha256": reproduction.canonical_sha256(identity),
        "wandb": {"config_sha256": "wandb-public"},
    }
    current_config = dict(health_config)
    current_config["protocol"] = convergence_protocol
    current_config["protocol_sha256"] = reproduction.canonical_sha256(
        convergence_protocol
    )
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
            "classification_accuracy": 1.0,
            "dice": 0.5,
            "weighted_composite": 0.85,
            "runtime_seconds": 1.0,
            "validation_seconds": 1.0,
            "coverage": coverage,
            "optimizer_steps": 55,
            "completed_train_steps": epoch * 55,
        }
        for epoch in range(1, 6)
    ]
    metrics = {
        "classification_accuracy": 1.0,
        "dice": 0.5,
        "weighted_composite": 0.85,
        "coverage": coverage,
    }
    record = {
        "attempt_id": "health-001",
        "phase": "health",
        "status": "completed",
        "reviewed_by": "reviewer",
        "requested_epochs": 5,
        "completed_epochs": 5,
        "completed_train_steps": 275,
        "train_steps_per_epoch": 55,
        "validation_steps_per_epoch": 111,
        "selected_epoch": 5,
        "metrics": metrics,
        "wandb_finished": True,
        "wandb": {
            "id": "health-001",
            "entity": reproduction.WANDB_ENTITY,
            "project": reproduction.WANDB_PROJECT,
            "mode": "online",
            "resume": "never",
            "config_sha256": "wandb-public",
            "url": "https://wandb.example/health-001",
        },
    }
    cases = [
        {"case_id": case_id, "correct": 1, "dice": 0.5}
        for case_id in identity["validation"]
    ]
    regression = {"case_count": 111, "cases": cases}
    payloads = {
        "config.json": health_config,
        "source.json": source,
        "case_identity.json": identity,
        "epochs.json": {"epochs": epochs},
        "best_epoch_cases.json": {"cases": cases},
        "regression.json": regression,
        "run_record.json": record,
    }
    for name, payload in payloads.items():
        write_json_once(tmp_path / name, payload)
    write_json_once(
        tmp_path / "score.json",
        {
            "attempt_id": "health-001",
            "phase": "health",
            "dataset_scope": reproduction.DATASET_SCOPE,
            "external_final_test_untouched": True,
            "selected_epoch": 5,
            "metrics": metrics,
            "decision": health_protocol["inference"],
            "regression_sha256": reproduction.sha256_file(
                tmp_path / "regression.json"
            ),
        },
    )
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
        _validate_health_index(
            tmp_path / "artifact_index.json", current_config, source
        )[
            "attempt_id"
        ]
        == "health-001"
    )
    drifted_config = dict(current_config)
    drifted_config["protocol"] = dict(convergence_protocol)
    drifted_config["protocol"]["lambda_cls"] = 0.25
    drifted_config["protocol_sha256"] = reproduction.canonical_sha256(
        drifted_config["protocol"]
    )
    with pytest.raises(ValueError, match="health config differs"):
        _validate_health_index(
            tmp_path / "artifact_index.json", drifted_config, source
        )
    worker_drift = dict(current_config)
    worker_drift["num_workers"] = 8
    with pytest.raises(ValueError, match="health config differs"):
        _validate_health_index(
            tmp_path / "artifact_index.json", worker_drift, source
        )
    (tmp_path / "score.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_health_index(
            tmp_path / "artifact_index.json", current_config, source
        )


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
