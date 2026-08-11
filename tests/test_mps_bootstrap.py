import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import mps_evidence

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = ROOT / "reproduce_teammate_l05_mps_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("mps_bootstrap", BOOTSTRAP_PATH)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def test_bootstrap_imports_before_torch() -> None:
    script = (
        "import runpy,sys;"
        f"sys.path.insert(0,{str(ROOT)!r});"
        f"runpy.run_path({str(BOOTSTRAP_PATH)!r},run_name='bootstrap_import_test');"
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


def test_sealed_environment_sets_exact_allocator_and_binds_worker_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = bootstrap._load_contract()
    monkeypatch.setattr(bootstrap, "_host_receipt", lambda: contract["host"])
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    for name in contract["allocator"]:
        monkeypatch.delenv(name, raising=False)
    child_argv = [sys.executable, str(bootstrap.WORKER_PATH), "--internal-worker"]
    environment, proof = bootstrap._sealed_environment(
        contract, "resource_gate", child_argv
    )
    assert {
        name: environment[name] for name in contract["allocator"]
    } == contract["allocator"]
    assert environment[bootstrap.WORKER_PYTHON_ENV] == sys.executable
    assert proof["torch_imported_in_bootstrap"] is False
    assert proof["child_argv_sha256"] == bootstrap._sha256_bytes(
        bootstrap._canonical_bytes(child_argv)
    )


def test_sealed_environment_rejects_allocator_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = bootstrap._load_contract()
    monkeypatch.setattr(bootstrap, "_host_receipt", lambda: contract["host"])
    monkeypatch.setenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")
    with pytest.raises(RuntimeError, match="altered allocator"):
        bootstrap._sealed_environment(contract, "resource_gate", ["worker"])


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_valid_gate(tmp_path: Path, attempt_id: str) -> Path:
    rows = [
        {"phase": "first_epoch_train", "optimizer_update": index,
         "completed_micro_steps": index * 8}
        for index in range(1, 56)
    ]
    rows.extend(
        {"phase": "validation_resource", "validation_step": index}
        for index in range(1, 112)
    )
    rows.append({"phase": "next_epoch_transition", "completed_optimizer_updates": 55})
    rows.extend(
        {"phase": "next_epoch_train", "optimizer_update": index,
         "completed_micro_steps": index * 8}
        for index in range(56, 77)
    )
    heartbeat = tmp_path / "acceptance_soak_heartbeat.jsonl"
    heartbeat.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    resource = tmp_path / "resource_evidence.json"
    _write_json(
        resource,
        {
            "bootstrap": {"proof_sha256": "b" * 64},
            "acceptance_soak": {
                "status": "passed",
                "completed_optimizer_updates": 76,
                "completed_micro_steps": 608,
                "completed_validation_steps": 111,
                "heartbeat_records_written": 188,
                "minimum_headroom_ratio": 0.25,
            },
        },
    )
    index = tmp_path / "acceptance_soak_index.json"
    _write_json(
        index,
        {
            "schema_version": 2,
            "attempt_id": attempt_id,
            "status": "passed",
            "resource_evidence_sha256": mps_evidence.sha256_file(resource),
            "heartbeat_sha256": mps_evidence.sha256_file(heartbeat),
            "optimizer_updates": 76,
            "micro_steps": 608,
            "validation_steps": 111,
            "heartbeat_records": 188,
            "wandb_started": False,
            "scientific_training_started": False,
            "external_final_test_untouched": True,
            "cleanup": {
                "status": "passed",
                "gc_collected": True,
                "empty_cache_completed": True,
                "synchronize_completed": True,
            },
        },
    )
    return index


def _write_valid_science(tmp_path: Path, attempt_id: str) -> Path:
    run_record = tmp_path / "run_record.json"
    _write_json(
        run_record,
        {
            "status": "completed",
            "wandb_finished": True,
            "external_final_test_untouched": True,
        },
    )
    index = tmp_path / "artifact_index.json"
    _write_json(
        index,
        {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "status": "completed",
            "artifacts": {"run_record.json": bootstrap._sha256_file(run_record)},
        },
    )
    return index


@pytest.mark.parametrize("role", ["resource_gate", "scientific_run"])
def test_supervisor_accepts_only_exact_child_completion(
    tmp_path: Path, role: str
) -> None:
    path = (
        _write_valid_gate(tmp_path, "sealed-attempt")
        if role == "resource_gate"
        else _write_valid_science(tmp_path, "sealed-attempt")
    )
    valid, digest, chain = bootstrap._validate_child_completion(
        path, role, "sealed-attempt", 0, 0.1
    )
    assert valid is True
    assert digest == bootstrap._sha256_file(path)
    assert chain is not None
    assert bootstrap._validate_child_completion(
        path, role, "other-attempt", 0, 0.1
    )[0] is False
    assert bootstrap._validate_child_completion(
        path, role, "sealed-attempt", 1, 0.1
    ) == (False, None, None)


def test_supervisor_rejects_zero_exit_without_completion(tmp_path: Path) -> None:
    assert bootstrap._validate_child_completion(
        tmp_path / "missing.json", "resource_gate", "sealed-attempt", 0, 0.1
    ) == (False, None, None)


def test_supervisor_rejects_tampered_gate_chain(tmp_path: Path) -> None:
    path = _write_valid_gate(tmp_path, "sealed-attempt")
    heartbeat = tmp_path / "acceptance_soak_heartbeat.jsonl"
    heartbeat.write_text(heartbeat.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    assert bootstrap._validate_child_completion(
        path, "resource_gate", "sealed-attempt", 0, 0.1
    ) == (False, None, None)


def test_bootstrap_rejects_internal_worker_bypass() -> None:
    with pytest.raises(SystemExit, match="rejects internal worker"):
        bootstrap.main(["--internal-worker"])


def test_bootstrap_rejects_supervisor_receipt_and_duplicate_attempt_bypass() -> None:
    with pytest.raises(SystemExit, match="owns the resource gate receipt"):
        bootstrap.main(["--resource-gate-receipt", "forged.json"])
    with pytest.raises(SystemExit, match="duplicate --attempt-id"):
        bootstrap.main(
            ["--attempt-id", "first", "--attempt-id", "second", "--execute"]
        )


def test_dry_run_delegates_without_supervisor_or_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> object:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    assert bootstrap.main(["--attempt-id", "dry-run"]) == 0
    assert observed["command"] == [
        sys.executable,
        str(bootstrap.WORKER_PATH),
        "--attempt-id",
        "dry-run",
    ]


def test_supervisor_receipts_are_write_once(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    bootstrap._write_json_once(receipt, {"status": "failed"})
    with pytest.raises(FileExistsError):
        bootstrap._write_json_once(receipt, {"status": "completed"})


def test_execute_never_auto_chains_gate_and_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, list[str]]] = []

    def supervise(
        argv: list[str], role: str, artifact_root: Path, attempt_id: str,
        approval: dict[str, object] | None = None,
    ) -> Path:
        calls.append((role, argv))
        return artifact_root / f"{attempt_id}-{role}.json"

    monkeypatch.setattr(bootstrap, "_supervise", supervise)
    common = [
        "--attempt-id", "sealed", "--artifact-root", str(tmp_path),
        "--execute", "--reviewed-by", "review",
    ]
    assert bootstrap.main(common + ["--acceptance-soak-only"]) == 0
    assert [role for role, _ in calls] == ["resource_gate"]

    gate_receipt = (
        tmp_path / ".supervisor" / "sealed-resource-gate" / "supervisor_receipt.json"
    )
    gate_receipt.parent.mkdir(parents=True)
    _write_json(gate_receipt, {"status": "completed"})
    approval = tmp_path / "approval.json"
    monkeypatch.setattr(bootstrap, "_git_head", lambda: "reviewed-head")
    _write_json(
        approval,
        {
            "schema_version": 1,
            "status": "approved",
            "attempt_id": "sealed",
            "gate_attempt_id": "sealed-resource-gate",
            "source_git_commit": "reviewed-head",
            "resource_gate_receipt_sha256": bootstrap._sha256_file(gate_receipt),
            "reviewed_by": "independent-reviewer",
            "review_url": "https://example.invalid/review",
            "external_final_test_untouched": True,
        },
    )
    assert bootstrap.main(
        common + ["--scientific-run", "--soak-approval", str(approval)]
    ) == 0
    assert [role for role, _ in calls] == ["resource_gate", "scientific_run"]
