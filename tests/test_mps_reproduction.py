import argparse
import importlib.metadata
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import reproduce_teammate_l05_mps as mps
import reproduction


def _cli(tmp_path: Path) -> list[str]:
    return [
        "--attempt-id",
        "mps-health-001",
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--baseline-score",
        str(tmp_path / "score.json"),
        "--baseline-run-record",
        str(tmp_path / "run_record.json"),
        "--pretrained",
        str(tmp_path / reproduction.EXPECTED_PRETRAINED_NAME),
        "--train-dcm-dir",
        str(tmp_path / "train-dcm"),
        "--train-mask-dir",
        str(tmp_path / "train-mask"),
        "--val-dcm-dir",
        str(tmp_path / "validation-dcm"),
        "--val-mask-dir",
        str(tmp_path / "validation-mask"),
    ]


def test_mps_protocol_is_distinct_and_preregistered() -> None:
    protocol = mps.mps_protocol_contract()
    assert protocol["execution_family"] == "apple_mps_resource_adjusted"
    assert protocol["historical_cuda_equivalence_claimed"] is False
    assert protocol["target_size"] == 1024
    assert protocol["physical_batch_size"] == 1
    assert protocol["gradient_accumulation_steps"] == 8
    assert protocol["effective_batch_size"] == 8
    assert protocol["batch_dice"] is True
    assert (
        protocol["loss_accumulation_semantics"]
        == "mean_of_8_microbatch_multitask_losses"
    )
    assert protocol["train_micro_steps_per_epoch"] == 440
    assert protocol["train_optimizer_steps_per_epoch"] == 55
    assert protocol["lambda_cls"] == 0.5
    assert protocol["inference"]["t_veto"] == 0.005
    assert protocol["feasibility"]["automatic_512_fallback"] is False


def test_mps_dry_run_requires_review_before_execution(tmp_path: Path) -> None:
    args = mps.parse_args(_cli(tmp_path))
    result = mps.run(args)
    assert result["status"] == "dry_run"
    assert result["review_required"] is True
    args.execute = True
    with pytest.raises(ValueError, match="reviewed-by"):
        mps.run(args)
    assert not (tmp_path / "artifacts" / args.attempt_id).exists()


def test_resource_failure_emits_only_sealed_resource_evidence(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "attempt"
    artifact_dir.mkdir()
    args = argparse.Namespace(attempt_id="mps-health-001")
    resource = {
        "attempt_id": args.attempt_id,
        "probe": {"status": "failed", "out_of_memory": True},
        "automatic_512_fallback_started": False,
        "wandb_started": False,
        "external_final_test_untouched": True,
    }
    index = mps._seal_resource_failure(artifact_dir, args, resource)
    assert index["status"] == "resource_infeasible"
    assert index["automatic_512_fallback_started"] is False
    assert index["wandb_started"] is False
    assert {path.name for path in artifact_dir.iterdir()} == {
        "resource_evidence.json",
        "resource_index.json",
    }
    assert index["resource_evidence_sha256"] == reproduction.sha256_file(
        artifact_dir / "resource_evidence.json"
    )
    with pytest.raises(FileExistsError):
        mps._seal_resource_failure(artifact_dir, args, resource)


def test_execute_routes_failed_probe_to_resource_only_without_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = mps.parse_args(
        _cli(tmp_path) + ["--execute", "--reviewed-by", "review-url"]
    )
    identity = {
        "dataset_scope": reproduction.DATASET_SCOPE,
        "train": [f"train-{index}" for index in range(444)],
        "validation": [f"validation-{index}" for index in range(111)],
    }

    class SizedLoader:
        def __init__(self, length: int):
            self.length = length

        def __len__(self) -> int:
            return self.length

    class ProbeModel:
        def to(self, _device: object) -> "ProbeModel":
            return self

    monkeypatch.setattr(
        mps, "source_identity", lambda _root: {"git_commit": "c", "git_tree_sha1": "t"}
    )
    monkeypatch.setattr(
        mps,
        "load_canonical_manifest",
        lambda _path: {"manifest_sha256": "manifest", "identity": identity},
    )
    monkeypatch.setattr(
        mps,
        "validate_canonical_content",
        lambda *_args: {"combined_content_sha256": "content"},
    )
    monkeypatch.setattr(
        mps,
        "load_pinned_baseline",
        lambda *_args: {
            "baseline_id": "baseline",
            "score_sha256": "score",
            "run_record_sha256": "record",
            "cases": [],
        },
    )
    monkeypatch.setattr(
        mps,
        "validate_pretrained",
        lambda _path: {"sha256": "pretrained", "byte_size": 1},
    )
    monkeypatch.setattr(
        mps,
        "validate_mps_runtime_dependencies",
        lambda _path: {
            "python": "3.14.7",
            "platform_system": "Darwin",
            "platform_machine": "arm64",
            "distributions": {"torch": "2.13.0", "torchvision": "0.28.0"},
            "mps_built": True,
            "mps_available": True,
            "mps_cpu_fallback": False,
        },
    )
    monkeypatch.setattr(mps, "sha256_file", lambda _path: "sealed-hash")
    monkeypatch.setattr(
        mps.datasets,
        "dataloader",
        lambda **_kwargs: (SizedLoader(444), SizedLoader(111)),
    )
    monkeypatch.setattr(
        mps,
        "validate_dataset_identity",
        lambda *_args: (identity, "identity-hash"),
    )
    monkeypatch.setattr(mps, "modeltype", lambda *_args, **_kwargs: ProbeModel())
    monkeypatch.setattr(
        mps,
        "_mps_feasibility_probe",
        lambda *_args: (_ for _ in ()).throw(
            mps.MPSFeasibilityFailure(
                {"status": "failed", "out_of_memory": True}
            )
        ),
    )
    monkeypatch.setattr(
        mps,
        "_start_wandb",
        lambda _config: pytest.fail("W&B must not start after failed probe"),
    )
    monkeypatch.setattr(
        mps.torch.backends.mps, "empty_cache", lambda: None, raising=False
    )

    result = mps.run(args)
    attempt = args.artifact_root / args.attempt_id
    assert result["status"] == "resource_infeasible"
    assert {path.name for path in attempt.iterdir()} == {
        "resource_evidence.json",
        "resource_index.json",
    }


def test_mps_runtime_gate_is_exact_and_disables_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "mps.lock"
    lock.write_text("torch==2.13.0\ntorchvision==0.28.0\n", encoding="utf-8")
    versions = {"torch": "2.13.0", "torchvision": "0.28.0"}
    monkeypatch.setattr(
        importlib.metadata, "version", lambda distribution: versions[distribution]
    )
    monkeypatch.setattr(reproduction.sys, "version_info", (3, 14, 7))
    monkeypatch.setattr(reproduction.platform, "python_version", lambda: "3.14.7")
    monkeypatch.setattr(
        reproduction.platform, "python_implementation", lambda: "CPython"
    )
    monkeypatch.setattr(reproduction.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(reproduction.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(reproduction.torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(reproduction.torch.backends.mps, "is_available", lambda: True)
    runtime = reproduction.validate_mps_runtime_dependencies(lock)
    assert runtime["mps_available"] is True
    assert runtime["mps_cpu_fallback"] is False
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="fallback"):
        reproduction.validate_mps_runtime_dependencies(lock)


def test_mps_lock_pins_verified_direct_environment() -> None:
    lock = Path(__file__).resolve().parents[1] / "requirements-reproduction-mps.lock"
    versions = reproduction.locked_versions(lock)
    assert versions["torch"] == reproduction.EXPECTED_MPS_TORCH_VERSION
    assert versions["torchvision"] == reproduction.EXPECTED_MPS_TORCHVISION_VERSION
    assert versions["timm"] == "1.0.22"
    assert "--hash=sha256:" in lock.read_text(encoding="utf-8")


def test_mps_public_wandb_config_excludes_case_identity_values() -> None:
    config = {
        "schema_version": 3,
        "issue_url": mps.ISSUE_URL,
        "parent_issue_url": mps.PARENT_ISSUE_URL,
        "attempt_id": "mps-health-001",
        "protocol": mps.mps_protocol_contract(),
        "protocol_sha256": "protocol",
        "source": {"git_commit": "commit", "git_tree_sha1": "tree"},
        "pretrained": {"sha256": "pretrained"},
        "dataset_scope": reproduction.DATASET_SCOPE,
        "manifest": {"manifest_sha256": "manifest"},
        "case_identity_sha256": "identity-hash",
        "content": {"combined_content_sha256": "content"},
        "baseline": {"score_sha256": "score", "run_record_sha256": "record"},
        "dependency": {
            "lock_sha256": "lock",
            "python": "3.14.7",
            "platform_system": "Darwin",
            "platform_machine": "arm64",
            "distributions": {"torch": "2.13.0", "torchvision": "0.28.0"},
            "mps_built": True,
            "mps_available": True,
            "mps_cpu_fallback": False,
        },
        "resource_evidence_sha256": "resource",
        "external_final_isolation": {"external_final_test_untouched": True},
    }
    public = mps._wandb_public_config(config)
    serialized = str(public)
    assert "secret-case-id" not in serialized
    assert public["dataset"]["train_case_count"] == 444
    assert public["dataset"]["validation_case_count"] == 111
    assert public["dependency"]["platform_machine"] == "arm64"


def test_mps_epoch_evidence_requires_440_microsteps() -> None:
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
            "dice": 0.25,
            "weighted_composite": 0.425,
            "runtime_seconds": 1.0,
            "validation_seconds": 1.0,
            "coverage": coverage,
            "optimizer_steps": 55,
            "micro_steps": 440,
            "completed_train_steps": epoch * 55,
        }
        for epoch in range(1, 6)
    ]
    mps._validate_mps_epochs(epochs)
    epochs[0]["micro_steps"] = 444
    with pytest.raises(ValueError, match="microstep"):
        mps._validate_mps_epochs(epochs)
