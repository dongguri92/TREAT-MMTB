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
                    "run_record.json": sha256_file(run_record),
                    "best_checkpoint.pth": "a" * 64,
                },
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
    with pytest.raises(ValueError, match="contiguous"):
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
    )
    spec = convergence.queue_spec(args, approval)
    assert spec["attempt"] == spec["max_attempts"] == 1
    assert spec["fresh_initialization"] is True
    assert spec["wandb"]["resume"] == "never"
    assert spec["gates"][0]["allowed_to"] == ["conv-001"]
    assert spec["external_final_accessed"] is False


def test_completion_seals_decision_pending_handoff_and_final_index(
    tmp_path: Path,
) -> None:
    approval_path = _health_approval(tmp_path)
    approval = convergence.validate_health_approval(approval_path, "conv-001")
    artifact_dir = tmp_path / "artifacts" / "conv-001"
    artifact_dir.mkdir(parents=True)
    rows = [
        _epoch(epoch, 0.50 + epoch * 0.001, 1.0 - epoch * 0.002)
        for epoch in range(1, 51)
    ]
    (artifact_dir / "epochs.json").write_text(json.dumps({"epochs": rows}))
    base = {"status": "completed", "attempt_id": "conv-001", "phase": "convergence_50e"}
    (artifact_dir / "artifact_index.json").write_text(json.dumps(base))
    args = argparse.Namespace(
        attempt_id="conv-001",
        artifact_root=tmp_path / "artifacts",
        health_approval=approval_path,
    )
    final = convergence._seal_completion(args, approval, base)
    assert final["auto_launched_150"] is False
    handoff = json.loads(
        (artifact_dir / "issue93_champion_handoff.pending.json").read_text()
    )
    assert handoff["launch_eligible"] is False
    assert handoff["review_state"] == "pending_independent_review"
    with pytest.raises(FileExistsError):
        convergence._seal_completion(args, approval, base)


@pytest.mark.parametrize("count", [1, 4, 49])
def test_partial_epochs_never_seal_completion(tmp_path: Path, count: int) -> None:
    approval_path = _health_approval(tmp_path)
    approval = convergence.validate_health_approval(approval_path, "conv-001")
    artifact_dir = tmp_path / "artifacts" / "conv-001"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "epochs.json").write_text(
        json.dumps({"epochs": [_epoch(i, 0.5, 1.0) for i in range(1, count + 1)]})
    )
    base = {"status": "completed", "attempt_id": "conv-001", "phase": "convergence_50e"}
    (artifact_dir / "artifact_index.json").write_text(json.dumps(base))
    args = argparse.Namespace(
        attempt_id="conv-001",
        artifact_root=tmp_path / "artifacts",
        health_approval=approval_path,
    )
    with pytest.raises(ValueError):
        convergence._seal_completion(args, approval, base)
