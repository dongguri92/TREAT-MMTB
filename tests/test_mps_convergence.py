import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import reproduce_teammate_l05_mps_convergence as convergence
from reproduction import sha256_file


def _epoch(
    epoch: int, composite: float, validation_loss: float
) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "weighted_composite": composite,
        "validation_loss": validation_loss,
    }


def _health_approval(tmp_path: Path, *, allowed_to: str = "conv-001") -> Path:
    health = tmp_path / "health"
    health.mkdir()
    protocol = convergence.engine.mps_protocol_contract()
    source_payload = {"git_commit": "commit", "git_tree_sha1": "tree"}
    config_payload = {
        "protocol": protocol,
        "num_workers": 0,
        "device": "mps",
        "source": source_payload,
        "pretrained": {"sha256": "p" * 64, "byte_size": 1},
        "dataset_scope": "train_plus_internal_validation_only",
        "manifest": {
            "manifest_sha256": "m" * 64,
            "train_case_count": 444,
            "validation_case_count": 111,
        },
        "content": {"combined_content_sha256": "c" * 64},
        "baseline": {"score_sha256": "s" * 64, "run_record_sha256": "r" * 64},
        "dependency": {"lock_sha256": "d" * 64},
        "case_identity_sha256": "i" * 64,
        "train_micro_steps_per_epoch": 440,
        "train_optimizer_steps_per_epoch": 55,
        "validation_steps_per_epoch": 111,
        "external_final_isolation": {
            "dataset_scope": "train_plus_internal_validation_only",
            "canonical_bytes_verified": True,
            "external_final_test_untouched": True,
        },
        "wandb": {
            "entity": convergence.engine.WANDB_ENTITY,
            "project": convergence.engine.WANDB_PROJECT,
            "mode": "online",
            "resume": "never",
        },
    }
    (health / "config.json").write_text(json.dumps(config_payload))
    (health / "source.json").write_text(json.dumps(source_payload))
    for name in convergence.engine.MPS_ARTIFACT_NAMES:
        path = health / name
        if not path.exists() and name != "run_record.json":
            path.write_bytes(b"checkpoint" if name == "best_checkpoint.pth" else b"{}")
    run_record = health / "run_record.json"
    artifact_index = health / "artifact_index.json"
    run_record.write_text(
        json.dumps(
            {
                "attempt_id": "health-003",
                "phase": "health",
                "status": "completed",
                "completed_epochs": 5,
                "wandb_finished": True,
            }
        )
    )
    artifact_index.write_text(
        json.dumps(
            {
                "attempt_id": "health-003",
                "status": "completed",
                "artifacts": {
                    name: sha256_file(health / name)
                    for name in convergence.engine.MPS_ARTIFACT_NAMES
                },
            }
        )
    )
    source_delta = tmp_path / "source-delta.json"
    source_delta.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "independently_reviewed",
                "reviewer": "reviewer",
                "health_source": source_payload,
                "convergence_source": convergence._head_source(),
                "execution_surface": convergence._execution_surface(),
                "external_final_accessed": False,
            }
        )
    )
    approval = tmp_path / "health-approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "issue": 95,
                "status": "approved",
                "review_state": "independently_reviewed",
                "reviewer": "reviewer",
                "health_attempt_id": "health-003",
                "health_run_record_path": str(run_record.resolve()),
                "health_run_record_sha256": sha256_file(run_record),
                "health_artifact_index_path": str(artifact_index.resolve()),
                "health_artifact_index_sha256": sha256_file(artifact_index),
                "source_delta_path": str(source_delta.resolve()),
                "source_delta_sha256": sha256_file(source_delta),
                "allowed_to": allowed_to,
                "external_final_accessed": False,
            }
        )
    )
    return approval


def test_convergence_extension_requires_all_gates_and_never_auto_launches() -> None:
    rows = [
        _epoch(epoch, 0.50 + epoch * 0.001, 1.0 - epoch * 0.002)
        for epoch in range(1, 51)
    ]
    result = convergence.convergence_decision(rows, health_gate_passed=True)
    assert result["extension_to_150_eligible"] is True
    assert result["auto_launch"] is False
    assert result["next_action"] == "request_fresh_reviewed_continuation_to_epoch_150"
    assert "optimizer_scheduler_and_rng_state" in result["extension_semantics"]


def test_convergence_rejects_plateau_nonfinite_and_incomplete_trajectory() -> None:
    plateau = [_epoch(epoch, 0.5, 1.0) for epoch in range(1, 51)]
    assert (
        convergence.convergence_decision(plateau, health_gate_passed=True)[
            "extension_to_150_eligible"
        ]
        is False
    )
    with pytest.raises(ValueError, match="at least five"):
        convergence.convergence_decision(
            [_epoch(1, 0.5, 1.0), _epoch(3, 0.6, 0.9)], health_gate_passed=True
        )
    bad = [_epoch(epoch, 0.5, 1.0) for epoch in range(1, 51)]
    bad[-1]["validation_loss"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        convergence.convergence_decision(bad, health_gate_passed=True)


def test_health_approval_is_attempt_and_byte_bound(tmp_path: Path) -> None:
    approval = _health_approval(tmp_path)
    assert (
        convergence.validate_health_approval(approval, "conv-001")["health_attempt_id"]
        == "health-003"
    )
    with pytest.raises(ValueError, match="not bound"):
        convergence.validate_health_approval(approval, "conv-002")
    record = tmp_path / "health" / "run_record.json"
    record.write_text("{}")
    with pytest.raises(ValueError, match="bytes"):
        convergence.validate_health_approval(approval, "conv-001")


def test_queue_spec_is_fresh_single_attempt_and_no_resume(tmp_path: Path) -> None:
    approval_path = _health_approval(tmp_path)
    approval = convergence.validate_health_approval(approval_path, "conv-001")
    args = argparse.Namespace(
        attempt_id="conv-001",
        raw_argv=["--attempt-id", "conv-001", "--health-approval", str(approval_path)],
        health_approval=approval_path,
        artifact_root=tmp_path / "artifacts",
        wandb_run_name="convergence-conv-001",
    )
    spec = convergence.queue_spec(args, approval)
    assert spec["attempt"] == spec["max_attempts"] == 1
    assert spec["fresh_initialization"] is True
    assert spec["wandb"]["resume"] == "never"
    assert spec["gates"][0]["allowed_to"] == ["conv-001"]
    assert spec["external_final_accessed"] is False
    assert spec["forbidden_initialization"]["sha256"] == sha256_file(
        tmp_path / "health" / "best_checkpoint.pth"
    )
    queue_path, queue_hash = convergence._bind_queue_receipt(args, spec, execute=False)
    assert queue_hash == sha256_file(queue_path)
    assert convergence._bind_queue_receipt(args, spec, execute=True)[1] == queue_hash
    queue_approval = tmp_path / "queue-approval.json"
    queue_approval.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "independently_reviewed_approved",
                "reviewer": "reviewer",
                "attempt_id": "conv-001",
                "queue_spec_sha256": queue_hash,
                "external_final_accessed": False,
            }
        )
    )
    assert convergence._validate_queue_approval(
        queue_approval, attempt_id="conv-001", queue_spec_sha256=queue_hash
    ) == sha256_file(queue_approval)


def test_health_approval_rejects_same_attempt_and_parent_symlink(
    tmp_path: Path,
) -> None:
    same = _health_approval(tmp_path, allowed_to="health-003")
    with pytest.raises(ValueError, match="not bound"):
        convergence.validate_health_approval(same, "health-003")

    external = tmp_path / "external_final"
    external.mkdir()
    real_approval = _health_approval(external)
    alias = tmp_path / "benign"
    alias.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="external final|symlink"):
        convergence.validate_health_approval(alias / real_approval.name, "conv-001")


@pytest.mark.parametrize("name", list(convergence.engine.MPS_ARTIFACT_NAMES))
def test_health_approval_rejects_every_indexed_artifact_substitution(
    tmp_path: Path, name: str
) -> None:
    approval = _health_approval(tmp_path)
    (tmp_path / "health" / name).write_bytes(b"substituted")
    with pytest.raises(ValueError, match="bytes (do not match|differ)"):
        convergence.validate_health_approval(approval, "conv-001")


def test_contract_rejects_config_drift_and_health_checkpoint_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approval_path = _health_approval(tmp_path)
    approval = convergence.validate_health_approval(approval_path, "conv-001")
    expected = approval["health_contract"]
    pretrained = tmp_path / "pretrained.pt"
    pretrained.write_bytes(b"pretrained")
    args = argparse.Namespace(pretrained=pretrained)
    monkeypatch.setattr(convergence, "_prospective_contract", lambda _args: expected)
    assert convergence.validate_convergence_contract(args, approval) == expected
    drifted = {**expected, "num_workers": 2}
    monkeypatch.setattr(convergence, "_prospective_contract", lambda _args: drifted)
    with pytest.raises(ValueError, match="differs from approved health"):
        convergence.validate_convergence_contract(args, approval)
    args.pretrained = tmp_path / "health" / "best_checkpoint.pth"
    monkeypatch.setattr(convergence, "_prospective_contract", lambda _args: expected)
    with pytest.raises(ValueError, match="must not initialize"):
        convergence.validate_convergence_contract(args, approval)


def _completed_base(
    artifact_dir: Path, attempt_id: str, rows: list[dict[str, float | int]]
) -> tuple[dict[str, object], str]:
    queue_path = artifact_dir.parent / "queue_specs" / f"{attempt_id}.json"
    queue_path.parent.mkdir(parents=True)
    queue_path.write_text("{}")
    queue_sha = sha256_file(queue_path)
    approval_sha = "a" * 64
    artifact_dir.mkdir(parents=True)
    for name in convergence.CONVERGENCE_ARTIFACT_NAMES:
        (artifact_dir / name).write_bytes(b"{}")
    (artifact_dir / "epochs.json").write_text(json.dumps({"epochs": rows}))
    (artifact_dir / "config.json").write_text(
        json.dumps(
            {
                "execution_receipt_sha256": queue_sha,
                "execution_approval_sha256": approval_sha,
            }
        )
    )
    (artifact_dir / "run_record.json").write_text(
        json.dumps(
            {
                "execution_receipt_sha256": queue_sha,
                "execution_approval_sha256": approval_sha,
                "wandb_completion": {
                    "state": "finished",
                    "identity_verified": True,
                    "verification": "wandb_api_post_finish",
                    "entity": convergence.engine.WANDB_ENTITY,
                    "project": convergence.engine.WANDB_PROJECT,
                    "id": attempt_id,
                    "name": f"convergence-{attempt_id}",
                    "group": convergence.engine.WANDB_GROUP,
                    "job_type": convergence.engine.WANDB_JOB_TYPE,
                },
            }
        )
    )
    convergence.engine.torch.save(
        {
            "model": {},
            "optimizer": {},
            "scheduler": {
                "kind": "cosine",
                "warmup_epochs": 5,
                "initial_lr": 5e-5,
                "next_epoch": 50,
                "last_learning_rates": [1e-7],
            },
            "completed_epochs": 50,
            "global_optimizer_updates": 2750,
            "python_rng_state": (),
            "numpy_rng_state": (),
            "torch_rng_state": convergence.engine.torch.tensor(
                [], dtype=convergence.engine.torch.uint8
            ),
            "mps_rng_state": convergence.engine.torch.tensor(
                [], dtype=convergence.engine.torch.uint8
            ),
        },
        artifact_dir / convergence.CONTINUATION_CHECKPOINT,
    )
    base: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "attempt_id": attempt_id,
        "phase": "convergence_50e",
        "execution_family": "apple_mps_resource_adjusted",
        "artifacts": {
            name: sha256_file(artifact_dir / name)
            for name in convergence.CONVERGENCE_ARTIFACT_NAMES
        },
    }
    (artifact_dir / "artifact_index.json").write_text(json.dumps(base))
    return base, queue_sha


def test_completion_seals_decision_pending_handoff_and_final_index(
    tmp_path: Path,
) -> None:
    approval_path = _health_approval(tmp_path)
    approval = convergence.validate_health_approval(approval_path, "conv-001")
    artifact_dir = tmp_path / "artifacts" / "conv-001"
    rows = [
        _epoch(epoch, 0.50 + epoch * 0.001, 1.0 - epoch * 0.002)
        for epoch in range(1, 51)
    ]
    base, queue_sha = _completed_base(artifact_dir, "conv-001", rows)
    args = argparse.Namespace(
        attempt_id="conv-001",
        artifact_root=tmp_path / "artifacts",
        health_approval=approval_path,
        wandb_run_name="convergence-conv-001",
        execution_approval_sha256="a" * 64,
    )
    final = convergence._seal_completion(args, approval, base, queue_sha)
    assert final["auto_launched_150"] is False
    handoff = json.loads(
        (artifact_dir / "issue93_champion_handoff.pending.json").read_text()
    )
    assert handoff["launch_eligible"] is False
    assert handoff["review_state"] == "pending_independent_review"
    assert handoff["convergence_index_sha256"] == sha256_file(
        artifact_dir / "convergence_index.json"
    )
    with pytest.raises(FileExistsError):
        convergence._seal_completion(args, approval, base, queue_sha)


def test_completion_rejects_self_consistent_indexed_epoch_substitution(
    tmp_path: Path,
) -> None:
    approval_path = _health_approval(tmp_path)
    approval = convergence.validate_health_approval(approval_path, "conv-001")
    artifact_dir = tmp_path / "artifacts" / "conv-001"
    rows = [_epoch(epoch, 0.5, 1.0) for epoch in range(1, 51)]
    base, queue_sha = _completed_base(artifact_dir, "conv-001", rows)
    (artifact_dir / "epochs.json").write_text(
        json.dumps({"epochs": [_epoch(epoch, 0.9, 0.1) for epoch in range(1, 51)]})
    )
    args = argparse.Namespace(
        attempt_id="conv-001",
        artifact_root=tmp_path / "artifacts",
        health_approval=approval_path,
        wandb_run_name="convergence-conv-001",
        execution_approval_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="indexed artifact bytes differ"):
        convergence._seal_completion(args, approval, base, queue_sha)


def test_completion_rejects_incomplete_continuation_and_approval_toctou(
    tmp_path: Path,
) -> None:
    approval_path = _health_approval(tmp_path)
    approval = convergence.validate_health_approval(approval_path, "conv-001")
    artifact_dir = tmp_path / "artifacts" / "conv-001"
    rows = [_epoch(epoch, 0.5, 1.0) for epoch in range(1, 51)]
    base, queue_sha = _completed_base(artifact_dir, "conv-001", rows)
    continuation_path = artifact_dir / convergence.CONTINUATION_CHECKPOINT
    continuation = convergence.engine.torch.load(
        continuation_path, map_location="cpu", weights_only=False
    )
    del continuation["mps_rng_state"]
    convergence.engine.torch.save(continuation, continuation_path)
    base["artifacts"][convergence.CONTINUATION_CHECKPOINT] = sha256_file(  # type: ignore[index]
        continuation_path
    )
    (artifact_dir / "artifact_index.json").write_text(json.dumps(base))
    args = argparse.Namespace(
        attempt_id="conv-001",
        artifact_root=tmp_path / "artifacts",
        health_approval=approval_path,
        wandb_run_name="convergence-conv-001",
        execution_approval_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="continuation state is incomplete"):
        convergence._seal_completion(args, approval, base, queue_sha)

    approval_path.write_text("{}")
    with pytest.raises(ValueError, match="health approval changed"):
        convergence._seal_completion(args, approval, base, queue_sha)


def test_convergence_finalization_failure_receipt_is_durable(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "conv-001"
    artifact_dir.mkdir(parents=True)
    args = argparse.Namespace(
        attempt_id="conv-001", artifact_root=tmp_path / "artifacts"
    )
    convergence._write_convergence_failure(
        args, convergence.engine.RunInterrupted(15), "convergence_finalization"
    )
    receipt = json.loads((artifact_dir / "convergence_failure.json").read_text())
    assert receipt["signal"] == "SIGTERM"
    assert receipt["status"] == "failed"
    convergence._write_convergence_failure(args, RuntimeError("later"), "later")
    assert (
        json.loads((artifact_dir / "convergence_failure.json").read_text()) == receipt
    )


@pytest.mark.parametrize("count", [1, 4, 49])
def test_partial_epochs_never_seal_completion(tmp_path: Path, count: int) -> None:
    approval_path = _health_approval(tmp_path)
    approval = convergence.validate_health_approval(approval_path, "conv-001")
    artifact_dir = tmp_path / "artifacts" / "conv-001"
    base, queue_sha = _completed_base(
        artifact_dir,
        "conv-001",
        [_epoch(i, 0.5, 1.0) for i in range(1, count + 1)],
    )
    args = argparse.Namespace(
        attempt_id="conv-001",
        artifact_root=tmp_path / "artifacts",
        health_approval=approval_path,
        wandb_run_name="convergence-conv-001",
        execution_approval_sha256="a" * 64,
    )
    with pytest.raises(ValueError):
        convergence._seal_completion(args, approval, base, queue_sha)
