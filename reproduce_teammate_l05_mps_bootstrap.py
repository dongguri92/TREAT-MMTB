"""Stdlib-only allocator bootstrap and watchdog for Issue #106 MPS execution."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "mps_resource_contract.json"
WORKER_PATH = ROOT / "reproduce_teammate_l05_mps.py"
PROOF_ENV = "TREAT_MMTB_MPS_BOOTSTRAP_PROOF"
PROOF_SHA_ENV = "TREAT_MMTB_MPS_BOOTSTRAP_PROOF_SHA256"
PROGRESS_FD_ENV = "TREAT_MMTB_MPS_PROGRESS_FD"
WORKER_PYTHON_ENV = "TREAT_MMTB_MPS_WORKER_PYTHON"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _write_json_once(path: Path, value: Any) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _flag_value(argv: list[str], name: str, default: str | None = None) -> str:
    try:
        index = argv.index(name)
        return argv[index + 1]
    except (ValueError, IndexError) as error:
        if default is not None:
            return default
        raise SystemExit(f"bootstrap requires {name}") from error


def _reject_duplicate_flag(argv: list[str], name: str) -> None:
    if argv.count(name) > 1:
        raise SystemExit(f"bootstrap rejects duplicate {name}")


def _sysctl(name: str) -> str:
    return subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", name], text=True
    ).strip()


def _host_receipt() -> dict[str, Any]:
    hostname_hash = _sha256_bytes(platform.node().encode("utf-8"))
    return {
        "hostname_sha256": hostname_hash,
        "system": platform.system(),
        "machine": platform.machine(),
        "os_product_version": subprocess.check_output(
            ["/usr/bin/sw_vers", "-productVersion"], text=True
        ).strip(),
        "physical_memory_bytes": int(_sysctl("hw.memsize")),
        "model": _sysctl("hw.model"),
    }


def _load_contract() -> dict[str, Any]:
    with CONTRACT_PATH.open(encoding="utf-8") as stream:
        contract = json.load(stream)
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise RuntimeError("invalid MPS resource contract")
    return contract


def _sealed_environment(
    contract: dict[str, Any], role: str, child_argv: list[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    environment = dict(os.environ)
    allocator = contract["allocator"]
    for name, expected in allocator.items():
        observed = environment.get(name)
        if observed not in (None, expected):
            raise RuntimeError(f"altered allocator setting: {name}")
        environment[name] = expected
    host = _host_receipt()
    if host != contract["host"]:
        raise RuntimeError("host resource identity differs from reviewed contract")
    proof = {
        "schema_version": 1,
        "role": role,
        "allocator": allocator,
        "host": host,
        "contract_sha256": _sha256_file(CONTRACT_PATH),
        "launcher_sha256": _sha256_file(Path(__file__).resolve()),
        "worker_sha256": _sha256_file(WORKER_PATH),
        "child_argv_sha256": _sha256_bytes(_canonical_bytes(child_argv)),
        "parent_pid": os.getpid(),
        "created_time_ns": time.time_ns(),
        "nonce": secrets.token_hex(32),
        "torch_imported_in_bootstrap": "torch" in sys.modules,
    }
    if proof["torch_imported_in_bootstrap"]:
        raise RuntimeError("stdlib bootstrap imported torch unexpectedly")
    encoded = _canonical_bytes(proof)
    environment[PROOF_ENV] = encoded.decode("ascii")
    environment[PROOF_SHA_ENV] = _sha256_bytes(encoded)
    environment[WORKER_PYTHON_ENV] = sys.executable
    return environment, proof


def _sample_process(pid: int, output: Path, seconds: int) -> dict[str, Any]:
    command = [
        "/usr/bin/sample",
        str(pid),
        str(seconds),
        "1",
        "-file",
        str(output),
    ]
    try:
        result = subprocess.run(command, check=False, timeout=seconds + 30)
        return {"requested": True, "returncode": result.returncode}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"requested": True, "error_type": type(error).__name__}


def _terminate_child(
    child: subprocess.Popen[str], supervisor_dir: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    watchdog = contract["watchdog"]
    sample_path = supervisor_dir / "timeout_sample.txt"
    sample = _sample_process(child.pid, sample_path, watchdog["sample_seconds"])
    child.send_signal(signal.SIGTERM)
    escalated = False
    try:
        child.wait(timeout=watchdog["sigterm_grace_seconds"])
    except subprocess.TimeoutExpired:
        escalated = True
        child.kill()
        child.wait(timeout=30)
    return {
        "sample": sample,
        "sample_sha256": _sha256_file(sample_path) if sample_path.exists() else None,
        "sigterm_sent": True,
        "sigterm_grace_seconds": watchdog["sigterm_grace_seconds"],
        "safe_escalation_used": escalated,
        "returncode": child.returncode,
    }


def _validate_child_completion(
    completion_path: Path, role: str, attempt_id: str, returncode: int | None
) -> tuple[bool, str | None]:
    if returncode != 0 or not completion_path.exists():
        return False, None
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None
    expected_status = "passed" if role == "resource_gate" else "completed"
    valid = (
        completion.get("schema_version") == 1
        and completion.get("attempt_id") == attempt_id
        and completion.get("status") == expected_status
    )
    return valid, _sha256_file(completion_path)


def _supervise(
    child_argv: list[str], role: str, artifact_root: Path, attempt_id: str
) -> Path:
    contract = _load_contract()
    supervisor_dir = artifact_root / ".supervisor" / attempt_id
    supervisor_dir.mkdir(parents=True, exist_ok=False)
    heartbeat_path = supervisor_dir / "progress.jsonl"
    read_fd, write_fd = os.pipe()
    child_command = [sys.executable, str(WORKER_PATH), *child_argv, "--internal-worker"]
    environment, proof = _sealed_environment(contract, role, child_command)
    environment[PROGRESS_FD_ENV] = str(write_fd)
    _write_json_once(supervisor_dir / "bootstrap_proof.json", proof)
    child = subprocess.Popen(
        child_command,
        cwd=ROOT,
        env=environment,
        pass_fds=(write_fd,),
        text=True,
    )
    os.close(write_fd)
    selector = selectors.DefaultSelector()
    selector.register(read_fd, selectors.EVENT_READ)
    last_progress = time.monotonic()
    timed_out = False
    with heartbeat_path.open("x", encoding="utf-8") as heartbeat:
        buffer = b""
        while child.poll() is None:
            events = selector.select(timeout=1.0)
            for key, _ in events:
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    json.loads(line)
                    heartbeat.write(line.decode("utf-8") + "\n")
                    heartbeat.flush()
                    os.fsync(heartbeat.fileno())
                    last_progress = time.monotonic()
            if time.monotonic() - last_progress > contract["watchdog"][
                "progress_timeout_seconds"
            ]:
                timed_out = True
                break
        termination = (
            _terminate_child(child, supervisor_dir, contract) if timed_out else None
        )
        if child.poll() is None:
            child.wait()
    os.close(read_fd)
    child_completion_path = (
        artifact_root / attempt_id / "acceptance_soak_index.json"
        if role == "resource_gate"
        else artifact_root / attempt_id / "artifact_index.json"
    )
    child_completion_valid, child_completion_sha256 = _validate_child_completion(
        child_completion_path, role, attempt_id, child.returncode
    )
    receipt = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "role": role,
        "status": (
            "completed"
            if child.returncode == 0 and not timed_out and child_completion_valid
            else "failed"
        ),
        "returncode": child.returncode,
        "timed_out": timed_out,
        "termination": termination,
        "bootstrap_proof_sha256": _sha256_file(
            supervisor_dir / "bootstrap_proof.json"
        ),
        "progress_sha256": _sha256_file(heartbeat_path),
        "progress_records": len(
            heartbeat_path.read_text(encoding="utf-8").splitlines()
        ),
        "child_completion_sha256": child_completion_sha256,
        "child_completion_verified_by_supervisor": child_completion_valid,
    }
    receipt_path = supervisor_dir / "supervisor_receipt.json"
    _write_json_once(receipt_path, receipt)
    _write_json_once(
        supervisor_dir / "supervisor_index.json",
        {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "receipt_sha256": _sha256_file(receipt_path),
            "status": receipt["status"],
        },
    )
    if receipt["status"] != "completed":
        raise RuntimeError(f"supervised {role} child failed")
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--internal-worker" in arguments:
        raise SystemExit("bootstrap rejects internal worker flag")
    if "--resource-gate-receipt" in arguments:
        raise SystemExit("bootstrap owns the resource gate receipt")
    for singleton in (
        "--attempt-id",
        "--artifact-root",
        "--execute",
        "--acceptance-soak-only",
        "--reviewed-by",
    ):
        _reject_duplicate_flag(arguments, singleton)
    if "--execute" not in arguments:
        return subprocess.run(
            [sys.executable, str(WORKER_PATH), *arguments],
            cwd=ROOT,
            check=False,
        ).returncode
    attempt_id = _flag_value(arguments, "--attempt-id")
    artifact_root = Path(
        _flag_value(
            arguments,
            "--artifact-root",
            str(ROOT / "artifacts" / "reproduction-mps"),
        )
    ).resolve()
    gate_id = f"{attempt_id}-resource-gate"
    gate_arguments = list(arguments)
    gate_arguments[gate_arguments.index("--attempt-id") + 1] = gate_id
    if "--acceptance-soak-only" not in gate_arguments:
        gate_arguments.append("--acceptance-soak-only")
    gate_receipt = _supervise(gate_arguments, "resource_gate", artifact_root, gate_id)
    if "--acceptance-soak-only" in arguments:
        print(gate_receipt)
        return 0
    scientific_arguments = [
        value for value in arguments if value != "--acceptance-soak-only"
    ]
    scientific_arguments.extend(["--resource-gate-receipt", str(gate_receipt)])
    scientific_receipt = _supervise(
        scientific_arguments, "scientific_run", artifact_root, attempt_id
    )
    print(scientific_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
