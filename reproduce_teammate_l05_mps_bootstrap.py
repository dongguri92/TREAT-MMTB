"""Stdlib-only allocator bootstrap and watchdog for Issue #106 MPS execution."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
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

from github_approval import verify_scientific_approval
from mps_evidence import validate_gate_artifacts, validate_scientific_artifacts

ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "mps_resource_contract.json"
WORKER_PATH = ROOT / "reproduce_teammate_l05_mps.py"
SCIENTIFIC_VERIFIER_PATH = ROOT / "verify_mps_scientific_completion.py"
LOADER_VERIFIER_PATH = ROOT / "verify_mps_loader_start.py"
PROOF_ENV = "TREAT_MMTB_MPS_BOOTSTRAP_PROOF"
PROOF_SHA_ENV = "TREAT_MMTB_MPS_BOOTSTRAP_PROOF_SHA256"
PROGRESS_FD_ENV = "TREAT_MMTB_MPS_PROGRESS_FD"
WORKER_PYTHON_ENV = "TREAT_MMTB_MPS_WORKER_PYTHON"
CANONICAL_MANIFEST_SHA256 = (
    "98484d6d96b9f6898393331d0493fa4d22ed6af0057b29210e14d32be7d5aef8"
)
SAFE_ATTEMPT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
PROTECTED_PATH_RE = re.compile(r"external(?:final|test)|(?:final|test)(?:test|final)")


def _reject_unsafe_attempt(attempt_id: str) -> None:
    if SAFE_ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise ValueError("attempt id must be a single safe path component")


def _reject_protected_path(path: Path | str, label: str) -> None:
    candidate = Path(path)
    if ".." in candidate.parts:
        raise ValueError(f"{label} must not contain path traversal")
    normalized = [
        "".join(ch for ch in part.lower() if ch.isalnum())
        for part in candidate.parts
    ]
    ancestry = normalized + [
        normalized[index] + normalized[index + 1]
        for index in range(len(normalized) - 1)
    ]
    if any(PROTECTED_PATH_RE.search(component) for component in ancestry):
        raise ValueError(f"{label} must not reference the external final test")


def _validate_bootstrap_paths(
    child_argv: list[str], artifact_root: Path, attempt_id: str
) -> None:
    _reject_unsafe_attempt(attempt_id)
    _reject_protected_path(artifact_root, "artifact_root")
    for flag in (
        "--artifact-root",
        "--manifest",
        "--baseline-score",
        "--baseline-run-record",
        "--pretrained",
        "--train-dcm-dir",
        "--train-mask-dir",
        "--val-dcm-dir",
        "--val-mask-dir",
    ):
        if flag in child_argv:
            label = flag.removeprefix("--")
            value = Path(_flag_value(child_argv, flag))
            _reject_protected_path(value, label)
            _reject_protected_path(value.resolve(), label)


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


def _stage_json(path: Path, value: Any) -> Path:
    staged = path.with_name(f".{path.name}.pending")
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with staged.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return staged


def _publish_staged(staged: Path, destination: Path) -> None:
    os.replace(staged, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


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
    loader_start: dict[str, Any] | None = None,
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
    manifest_identity = None
    if "--manifest" in child_argv:
        manifest_path = Path(_flag_value(child_argv, "--manifest")).resolve()
        manifest_sha256 = _sha256_file(manifest_path)
        if manifest_sha256 != CANONICAL_MANIFEST_SHA256:
            raise RuntimeError("manifest hash differs from pinned 444/111 contract")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split = manifest.get("split") if isinstance(manifest, dict) else None
        validation_cases = (
            manifest.get("validation_cases") if isinstance(manifest, dict) else None
        )
        if (
            not isinstance(split, dict)
            or not isinstance(validation_cases, dict)
            or not isinstance(split.get("train"), list)
            or not isinstance(split.get("val"), list)
            or len(split["train"]) != 444
            or len(split["val"]) != 111
            or set(validation_cases) != set(split["val"])
        ):
            raise RuntimeError("manifest identity differs from exact 444/111 contract")
        train_ids = [str(case_id) for case_id in split["train"]]
        validation_ids: list[str] = []
        for prepared_id in split["val"]:
            metadata = validation_cases.get(prepared_id)
            if not isinstance(metadata, dict) or not isinstance(
                metadata.get("source_case_id"), str
            ):
                raise RuntimeError("manifest validation source identity is incomplete")
            validation_ids.append(metadata["source_case_id"])
        if (
            len(set(train_ids)) != 444
            or len(set(validation_ids)) != 111
            or set(train_ids) & set(validation_ids)
        ):
            raise RuntimeError("manifest identities are not exact disjoint cohorts")
        manifest_identity = {
            "manifest_file_sha256": manifest_sha256,
            "canonical_case_sha256": {
                split: [
                    _sha256_bytes(str(case_id).encode("utf-8"))
                    for case_id in case_ids
                ]
                for split, case_ids in (
                    ("train", sorted(train_ids)),
                    ("validation", sorted(validation_ids)),
                )
            },
        }
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
        "dataset_identity": manifest_identity,
        "loader_start": loader_start,
    }
    if proof["torch_imported_in_bootstrap"]:
        raise RuntimeError("stdlib bootstrap imported torch unexpectedly")
    encoded = _canonical_bytes(proof)
    environment[PROOF_ENV] = encoded.decode("ascii")
    environment[PROOF_SHA_ENV] = _sha256_bytes(encoded)
    environment[WORKER_PYTHON_ENV] = sys.executable
    return environment, proof


def _recompute_loader_start(child_argv: list[str]) -> dict[str, Any]:
    command = [sys.executable, str(LOADER_VERIFIER_PATH)]
    for flag in (
        "--manifest",
        "--train-dcm-dir",
        "--train-mask-dir",
        "--val-dcm-dir",
        "--val-mask-dir",
    ):
        command.extend((flag, _flag_value(child_argv, flag)))
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise ValueError("independent loader-start verifier returned no evidence")
    payload = json.loads(output_lines[-1])
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("components"), list)
        or len(payload["components"]) != 8
        or payload.get("fingerprint")
        != _sha256_bytes(_canonical_bytes(payload["components"]))
    ):
        raise ValueError("independent loader-start verifier returned invalid evidence")
    return payload


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
    independent_scientific_verification: dict[str, Any] | None = None,
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
                completion_path.parent,
                attempt_id,
                supervisor_progress_path,
                expected_bootstrap_proof_sha256,
                independent_scientific_verification,
            )
    except Exception:
        return False, None, None
    return True, _sha256_file(completion_path), chain


def _verify_scientific_completion_independently(
    child_argv: list[str], artifact_root: Path, attempt_id: str
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SCIENTIFIC_VERIFIER_PATH),
        "--science-dir",
        str(artifact_root / attempt_id),
        "--attempt-id",
        attempt_id,
        "--baseline-score",
        _flag_value(child_argv, "--baseline-score"),
        "--baseline-run-record",
        _flag_value(child_argv, "--baseline-run-record"),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise ValueError("independent scientific verifier returned no evidence")
    payload = json.loads(output_lines[-1])
    if not isinstance(payload, dict):
        raise TypeError("independent scientific verifier returned invalid evidence")
    return payload


def _supervise(
    child_argv: list[str],
    role: str,
    artifact_root: Path,
    attempt_id: str,
    soak_approval: dict[str, Any] | None = None,
) -> Path:
    _validate_bootstrap_paths(child_argv, artifact_root, attempt_id)
    supervisor_dir = artifact_root / ".supervisor" / attempt_id
    heartbeat_path = supervisor_dir / "progress.jsonl"
    child_command = [sys.executable, str(WORKER_PATH), *child_argv, "--internal-worker"]
    contract: dict[str, Any] | None = None
    proof: dict[str, Any] | None = None
    read_fd = write_fd = -1
    child: subprocess.Popen[str] | None = None
    selector: selectors.BaseSelector | None = None
    previous_handlers: dict[int, Any] = {}
    timed_out = False
    termination: dict[str, Any] | None = None
    supervisor_error: BaseException | None = None

    signal_events: list[int] = []
    post_commit_signal_events: list[int] = []
    completion_commit_reached = False

    def interrupt(signum: int, _frame: Any) -> None:
        target = post_commit_signal_events if completion_commit_reached else signal_events
        target.append(signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupt)
        supervisor_dir.mkdir(parents=True, exist_ok=False)
    except BaseException:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        raise

    try:
        if signal_events:
            raise InterruptedError("supervisor interrupted before setup")
        contract = _load_contract()
        read_fd, write_fd = os.pipe()
        loader_start = _recompute_loader_start(child_argv)
        environment, proof = _sealed_environment(
            contract, role, child_command, soak_approval, loader_start
        )
        environment[PROGRESS_FD_ENV] = str(write_fd)
        _write_json_once(supervisor_dir / "bootstrap_proof.json", proof)
        if signal_events:
            raise InterruptedError("supervisor interrupted before child spawn")
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
                            raise TypeError("supervisor progress row must be an object")
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
                            raise TypeError(
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
                if contract is not None:
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
    child_completion_valid = False
    child_completion_sha256 = None
    evidence_chain = None
    independent_scientific_verification = None
    if (
        role == "scientific_run"
        and child is not None
        and child.returncode == 0
        and supervisor_error is None
        and not signal_events
    ):
        try:
            independent_scientific_verification = (
                _verify_scientific_completion_independently(
                    child_argv, artifact_root, attempt_id
                )
            )
        except BaseException as error:
            supervisor_error = error
    if contract is not None and proof is not None:
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
            independent_scientific_verification,
        )
    proof_path = supervisor_dir / "bootstrap_proof.json"
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
        "bootstrap_proof_sha256": (
            _sha256_file(proof_path) if proof_path.is_file() else None
        ),
        "bootstrap_proof_canonical_sha256": (
            _sha256_bytes(_canonical_bytes(proof)) if proof is not None else None
        ),
        "progress_sha256": _sha256_file(heartbeat_path),
        "progress_records": len(
            heartbeat_path.read_text(encoding="utf-8").splitlines()
        ),
        "child_completion_sha256": child_completion_sha256,
        "child_completion_verified_by_supervisor": child_completion_valid,
        "evidence_chain": evidence_chain,
        "independent_scientific_verification": independent_scientific_verification,
    }
    receipt_path = supervisor_dir / "supervisor_receipt.json"
    index_path = supervisor_dir / "supervisor_index.json"
    watched_signals = {signal.SIGTERM, signal.SIGINT}
    old_mask = None

    def latch_pending_signals() -> None:
        if hasattr(signal, "sigpending"):
            pending = signal.sigpending() & watched_signals
            signal_events.extend(
                signum for signum in sorted(pending) if signum not in signal_events
            )

    def discard_publication_files() -> None:
        for path in (
            receipt_path,
            index_path,
            receipt_path.with_name(f".{receipt_path.name}.pending"),
            index_path.with_name(f".{index_path.name}.pending"),
        ):
            path.unlink(missing_ok=True)

    def stage_pair() -> tuple[Path, Path]:
        staged_receipt = _stage_json(receipt_path, receipt)
        staged_index = _stage_json(
            index_path,
            {
                "schema_version": 1,
                "attempt_id": attempt_id,
                "receipt_sha256": _sha256_file(staged_receipt),
                "status": receipt["status"],
            },
        )
        return staged_receipt, staged_index

    def convert_to_failure() -> tuple[Path, Path]:
        discard_publication_files()
        receipt["status"] = "failed"
        receipt["supervisor_error_type"] = "InterruptedError"
        receipt["signals_received"] = [
            signal.Signals(value).name for value in signal_events
        ]
        return stage_pair()

    try:
        staged_receipt, staged_index = stage_pair()
        if signal_events:
            staged_receipt, staged_index = convert_to_failure()
        _publish_staged(staged_receipt, receipt_path)
        if receipt["status"] == "completed" and signal_events:
            staged_receipt, staged_index = convert_to_failure()
            _publish_staged(staged_receipt, receipt_path)
        if receipt["status"] == "completed" and hasattr(signal, "pthread_sigmask"):
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, watched_signals)
        latch_pending_signals()
        if receipt["status"] == "completed" and signal_events:
            staged_receipt, staged_index = convert_to_failure()
            _publish_staged(staged_receipt, receipt_path)
        _publish_staged(staged_index, index_path)
        latch_pending_signals()
        if receipt["status"] == "completed" and signal_events:
            staged_receipt, staged_index = convert_to_failure()
            _publish_staged(staged_receipt, receipt_path)
            _publish_staged(staged_index, index_path)
        elif receipt["status"] == "completed":
            # The completed index is durable and the blocked signal set is empty.
            # Signals observed after this assignment are post-commit notifications.
            completion_commit_reached = True
    finally:
        if old_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if receipt["status"] != "completed" or not completion_commit_reached:
        raise RuntimeError(f"supervised {role} child failed")
    return receipt_path


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _load_scientific_approval(
    comment_url: str, attempt_id: str, gate_receipt: Path
) -> dict[str, Any]:
    return verify_scientific_approval(
        comment_url,
        attempt_id,
        gate_receipt,
        _git_head(),
        _load_contract()["approval"],
        urllib.request.urlopen,
    )


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
    artifact_root_raw = Path(
        _flag_value(
            arguments,
            "--artifact-root",
            str(ROOT / "artifacts" / "reproduction-mps"),
        )
    )
    _validate_bootstrap_paths(arguments, artifact_root_raw, attempt_id)
    artifact_root = artifact_root_raw.resolve()
    _reject_protected_path(artifact_root, "artifact_root")
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
