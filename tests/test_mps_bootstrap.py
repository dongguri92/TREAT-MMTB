import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = ROOT / "reproduce_teammate_l05_mps_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("mps_bootstrap", BOOTSTRAP_PATH)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def test_bootstrap_imports_before_torch() -> None:
    script = (
        "import runpy,sys;"
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


@pytest.mark.parametrize(
    ("role", "status"),
    [("resource_gate", "passed"), ("scientific_run", "completed")],
)
def test_supervisor_accepts_only_exact_child_completion(
    tmp_path: Path, role: str, status: str
) -> None:
    path = tmp_path / "completion.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempt_id": "sealed-attempt",
                "status": status,
            }
        ),
        encoding="utf-8",
    )
    valid, digest = bootstrap._validate_child_completion(
        path, role, "sealed-attempt", 0
    )
    assert valid is True
    assert digest == bootstrap._sha256_file(path)
    assert (
        bootstrap._validate_child_completion(path, role, "other-attempt", 0)[0]
        is False
    )
    assert bootstrap._validate_child_completion(path, role, "sealed-attempt", 1) == (
        False,
        None,
    )


def test_supervisor_rejects_zero_exit_without_completion(tmp_path: Path) -> None:
    assert bootstrap._validate_child_completion(
        tmp_path / "missing.json", "resource_gate", "sealed-attempt", 0
    ) == (False, None)


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
