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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mps_evidence import validate_gate_artifacts, validate_scientific_artifacts

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
    contract: dict[str, Any],
    role: str,
    child_argv: list[str],
    soak_approval: dict[str, Any] | None = None,
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
        "soak_approval": soak_approval,
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
    try:
        child.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=30)
        return {
            "sample": sample,
            "sample_sha256": (
                _sha256_file(sample_path) if sample_path.exists() else None
            ),
            "sigterm_sent": False,
            "sigterm_grace_seconds": watchdog["sigterm_grace_seconds"],
            "safe_escalation_used": False,
            "returncode": child.returncode,
        }
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
    completion_path: Path,
    role: str,
    attempt_id: str,
    returncode: int | None,
    minimum_headroom_ratio: float,
    supervisor_progress_path: Path | None = None,
    expected_bootstrap_proof_sha256: str | None = None,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    if returncode != 0 or not completion_path.exists():
        return False, None, None
    try:
        if role == "resource_gate":
            chain = validate_gate_artifacts(
                completion_path.parent,
                attempt_id,
                minimum_headroom_ratio,
                supervisor_progress_path,
                expected_bootstrap_proof_sha256,
            )
        else:
            chain = validate_scientific_artifacts(
                completion_path.parent, attempt_id
            )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False, None, None
    return True, _sha256_file(completion_path), chain


def _supervise(
    child_argv: list[str],
    role: str,
    artifact_root: Path,
    attempt_id: str,
    soak_approval: dict[str, Any] | None = None,
) -> Path:
    contract = _load_contract()
    supervisor_dir = artifact_root / ".supervisor" / attempt_id
    supervisor_dir.mkdir(parents=True, exist_ok=False)
    heartbeat_path = supervisor_dir / "progress.jsonl"
    read_fd, write_fd = os.pipe()
    child_command = [sys.executable, str(WORKER_PATH), *child_argv, "--internal-worker"]
    environment, proof = _sealed_environment(
        contract, role, child_command, soak_approval
    )
    environment[PROGRESS_FD_ENV] = str(write_fd)
    _write_json_once(supervisor_dir / "bootstrap_proof.json", proof)
    child: subprocess.Popen[str] | None = None
    selector: selectors.BaseSelector | None = None
    previous_handlers: dict[int, Any] = {}
    timed_out = False
    termination: dict[str, Any] | None = None
    supervisor_error: BaseException | None = None

    signal_events: list[int] = []

    def interrupt(signum: int, _frame: Any) -> None:
        signal_events.append(signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupt)
        child = subprocess.Popen(
            child_command,
            cwd=ROOT,
            env=environment,
            pass_fds=(write_fd,),
            text=True,
        )
        os.close(write_fd)
        write_fd = -1
        selector = selectors.DefaultSelector()
        selector.register(read_fd, selectors.EVENT_READ)
        last_progress = time.monotonic()
        with heartbeat_path.open("x", encoding="utf-8") as heartbeat:
            buffer = b""
            while child.poll() is None:
                if signal_events:
                    supervisor_error = InterruptedError(
                        "supervisor interrupted by "
                        f"{signal.Signals(signal_events[0]).name}"
                    )
                    break
                events = selector.select(timeout=1.0)
                for key, _ in events:
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("supervisor progress row must be an object")
                        heartbeat.write(line.decode("utf-8") + "\n")
                        heartbeat.flush()
                        os.fsync(heartbeat.fileno())
                        last_progress = time.monotonic()
                if time.monotonic() - last_progress > contract["watchdog"][
                    "progress_timeout_seconds"
                ]:
                    timed_out = True
                    break
            while selector.get_map():
                events = selector.select(timeout=0)
                if not events:
                    break
                for key, _ in events:
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError(
                                "supervisor progress row must be an object"
                            )
                        heartbeat.write(line.decode("utf-8") + "\n")
                        heartbeat.flush()
                        os.fsync(heartbeat.fileno())
        if timed_out or supervisor_error is not None:
            try:
                termination = _terminate_child(child, supervisor_dir, contract)
            except BaseException as termination_error:
                termination = {
                    "error_type": type(termination_error).__name__,
                    "returncode": child.poll(),
                }
        elif child.poll() is None:
            child.wait()
    except BaseException as error:
        supervisor_error = error
        if child is not None and child.poll() is None:
            try:
                termination = _terminate_child(child, supervisor_dir, contract)
            except BaseException as termination_error:
                termination = {
                    "error_type": type(termination_error).__name__,
                    "returncode": child.poll(),
                }
    finally:
        if selector is not None:
            selector.close()
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
        if child is not None and child.poll() is None:
            try:
                child.kill()
                child.wait(timeout=30)
            except BaseException as reap_error:
                supervisor_error = supervisor_error or reap_error
    if not heartbeat_path.exists():
        heartbeat_path.touch(exist_ok=False)
    if signal_events and supervisor_error is None:
        supervisor_error = InterruptedError(
            "supervisor interrupted by "
            f"{signal.Signals(signal_events[0]).name}"
        )
    child_completion_path = (
        artifact_root / attempt_id / "acceptance_soak_index.json"
        if role == "resource_gate"
        else artifact_root / attempt_id / "artifact_index.json"
    )
    (
        child_completion_valid,
        child_completion_sha256,
        evidence_chain,
    ) = _validate_child_completion(
        child_completion_path,
        role,
        attempt_id,
        child.returncode if child is not None else None,
        float(contract["memory"]["minimum_headroom_ratio"]),
        heartbeat_path,
        _sha256_bytes(_canonical_bytes(proof)),
    )
    receipt = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "role": role,
        "status": (
            "completed"
            if child is not None
            and child.returncode == 0
            and not timed_out
            and supervisor_error is None
            and child_completion_valid
            else "failed"
        ),
        "returncode": child.returncode if child is not None else None,
        "timed_out": timed_out,
        "termination": termination,
        "supervisor_error_type": (
            type(supervisor_error).__name__ if supervisor_error is not None else None
        ),
        "signals_received": [signal.Signals(value).name for value in signal_events],
        "bootstrap_proof_sha256": _sha256_file(
            supervisor_dir / "bootstrap_proof.json"
        ),
        "bootstrap_proof_canonical_sha256": _sha256_bytes(
            _canonical_bytes(proof)
        ),
        "progress_sha256": _sha256_file(heartbeat_path),
        "progress_records": len(
            heartbeat_path.read_text(encoding="utf-8").splitlines()
        ),
        "child_completion_sha256": child_completion_sha256,
        "child_completion_verified_by_supervisor": child_completion_valid,
        "evidence_chain": evidence_chain,
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
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)
    if receipt["status"] != "completed":
        raise RuntimeError(f"supervised {role} child failed")
    return receipt_path


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _load_scientific_approval(
    comment_url: str, attempt_id: str, gate_receipt: Path
) -> dict[str, Any]:
    contract = _load_contract()["approval"]
    prefixes = tuple(
        "https://github.com/"
        f"{contract['github_repository']}/{kind}/"
        for kind in ("issues", "pull")
    )
    if not comment_url.startswith(prefixes) or "#issuecomment-" not in comment_url:
        raise RuntimeError("scientific approval must be an exact GitHub issue comment")
    comment_id = comment_url.rsplit("#issuecomment-", 1)[1]
    if not comment_id.isdigit():
        raise RuntimeError("scientific approval comment ID is invalid")
    api_url = (
        "https://api.github.com/repos/"
        f"{contract['github_repository']}/issues/comments/{comment_id}"
    )
    request = urllib.request.Request(
        api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "TREAT-MMTB"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            comment = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError("GitHub approval verification is unavailable") from error
    if not isinstance(comment, dict):
        raise RuntimeError("GitHub approval response is invalid")
    marker = contract["body_marker"]
    body = str(comment.get("body", ""))
    marker_line = f"<!-- {marker} -->"
    if marker_line not in body:
        raise RuntimeError("GitHub approval marker is missing")
    try:
        approval = json.loads(body.split(marker_line, 1)[1].strip())
    except json.JSONDecodeError as error:
        raise RuntimeError("GitHub approval payload is invalid") from error
    gate_id = f"{attempt_id}-resource-gate"
    if (
        not isinstance(approval, dict)
        or approval.get("schema_version") != 1
        or approval.get("status") != "approved"
        or approval.get("attempt_id") != attempt_id
        or approval.get("gate_attempt_id") != gate_id
        or approval.get("source_git_commit") != _git_head()
        or approval.get("resource_gate_receipt_sha256")
        != _sha256_file(gate_receipt)
        or comment.get("user", {}).get("login") not in contract["allowed_reviewers"]
        or comment.get("author_association")
        not in contract["required_author_associations"]
        or approval.get("reviewed_by") != comment.get("user", {}).get("login")
        or approval.get("review_url") != comment_url
        or approval.get("external_final_test_untouched") is not True
    ):
        raise RuntimeError("scientific soak approval differs from reviewed gate")
    return {
        **approval,
        "approval_sha256": _sha256_bytes(_canonical_bytes(comment)),
        "github_api_url": api_url,
    }


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
        "--scientific-run",
        "--soak-approval",
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
    gate_only = "--acceptance-soak-only" in arguments
    scientific = "--scientific-run" in arguments
    if gate_only == scientific:
        raise SystemExit(
            "execute requires exactly one of --acceptance-soak-only or "
            "--scientific-run"
        )
    if gate_only:
        gate_arguments = list(arguments)
        gate_arguments[gate_arguments.index("--attempt-id") + 1] = gate_id
        gate_receipt = _supervise(
            gate_arguments, "resource_gate", artifact_root, gate_id
        )
        print(gate_receipt)
        return 0

    approval_url = _flag_value(arguments, "--soak-approval")
    gate_receipt = (
        artifact_root / ".supervisor" / gate_id / "supervisor_receipt.json"
    )
    approval = _load_scientific_approval(
        approval_url, attempt_id, gate_receipt
    )
    scientific_arguments = []
    skip_next = False
    for value in arguments:
        if skip_next:
            skip_next = False
            continue
        if value == "--soak-approval":
            skip_next = True
            continue
        if value == "--scientific-run":
            continue
        scientific_arguments.append(value)
    scientific_arguments.extend(["--resource-gate-receipt", str(gate_receipt)])
    scientific_receipt = _supervise(
        scientific_arguments,
        "scientific_run",
        artifact_root,
        attempt_id,
        approval,
    )
    print(scientific_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
