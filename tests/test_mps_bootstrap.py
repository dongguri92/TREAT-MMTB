import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import mps_evidence
import verify_mps_scientific_completion as scientific_verifier

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


def test_bootstrap_parses_real_pinned_manifest_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = (
        ROOT.parent
        / "TREAT-MMTB-2026"
        / "artifacts/nnunet/nnUNet_raw/Dataset003_Task1InternalValidation"
        / "internal_validation_manifest.json"
    )
    if not manifest_path.is_file():
        pytest.skip("real pinned internal manifest is not mounted")
    contract = bootstrap._load_contract()
    monkeypatch.setattr(bootstrap, "_host_receipt", lambda: contract["host"])
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    _environment, proof = bootstrap._sealed_environment(
        contract,
        "resource_gate",
        ["worker", "--manifest", str(manifest_path)],
        loader_start={"components": [{}] * 8, "fingerprint": "0" * 64},
    )
    identity = proof["dataset_identity"]
    assert identity["manifest_file_sha256"] == bootstrap.CANONICAL_MANIFEST_SHA256
    assert len(identity["canonical_case_sha256"]["train"]) == 444
    assert len(identity["canonical_case_sha256"]["validation"]) == 111


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_valid_gate(tmp_path: Path, attempt_id: str) -> Path:
    def memory() -> dict[str, object]:
        return {
            "memory": {
                "driver_allocated_bytes": 75,
                "recommended_max_bytes": 100,
            },
            "headroom_ratio": 0.25,
        }

    train_ids = [f"{index + 1:064x}" for index in range(444)]
    validation_ids = [f"{index + 1000:064x}" for index in range(111)]
    updates = []
    for index in range(1, 77):
        identities = [
            train_ids[((index - 1) * 8 + offset) % 444] for offset in range(8)
        ]
        updates.append({
            "phase": "first_epoch_train" if index <= 55 else "next_epoch_train",
            "optimizer_update": index,
            "completed_micro_steps": index * 8,
            "distinct_microbatches": 8,
            "microbatch_identity_sha256": mps_evidence.canonical_sha256(
                sorted(identities)
            ),
            "microbatch_case_sha256_ordered": identities,
            "finite_losses": True,
            "gradient_clip_completed": True,
            "critical_memory": {
                "phase": "backward_complete_pre_adamw_memory",
                "tensor_scalar_materialized": False,
                "explicit_mps_synchronize_called": False,
                **memory(),
            },
            **memory(),
        })
    validation = [
        {
            "phase": "validation_resource",
            "validation_step": index,
            "finite_loss": True,
            "case_sha256": validation_ids[index - 1],
            "operation_events": mps_evidence.VALIDATION_OPERATION_EVENTS,
            **memory(),
        }
        for index in range(1, 112)
    ]
    rows = updates[:55] + validation
    rows.append({"phase": "next_epoch_transition", "completed_optimizer_updates": 55})
    rows.extend(updates[55:])
    heartbeat = tmp_path / "acceptance_soak_heartbeat.jsonl"
    heartbeat.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    resource = tmp_path / "resource_evidence.json"
    loader_components = [
        {
            "case_sha256_ordered": [train_ids[index]],
            "tensors": {
                name: {"shape": [1], "sha256": f"{100 + index * 3 + offset:064x}"}
                for offset, name in enumerate(("image", "mask", "cls"))
            },
        }
        for index in range(8)
    ]
    probe_update = {**updates[0], "phase": "first_epoch_train"}
    bootstrap_proof: dict[str, Any] = {
        "dataset_identity": {
            "canonical_case_sha256": {
                "train": train_ids, "validation": validation_ids,
            }
        },
        "loader_start": {
            "components": loader_components,
            "fingerprint": mps_evidence.canonical_sha256(loader_components),
        },
    }
    bootstrap_proof["proof_sha256"] = mps_evidence.canonical_sha256(
        bootstrap_proof
    )
    _write_json(
        resource,
        {
            "bootstrap": bootstrap_proof,
            "canonical_case_sha256": {
                "train": train_ids, "validation": validation_ids,
            },
            "probe": {
                "status": "passed",
                "probe": "exact_8_microbatch_adamw_optimizer_update",
                "target_size": 1024,
                "physical_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "effective_batch_size": 8,
                "requested_optimizer_updates": 1,
                "requested_micro_steps": 8,
                "wandb_started": False,
                "completed_optimizer_updates": 1,
                "completed_micro_steps": 8,
                "optimizer": {
                    "name": "adamw",
                    "learning_rate": 1e-5,
                    "gradient_clip_norm": 12,
                },
                "loader_start_components": loader_components,
                "loader_start_fingerprint": mps_evidence.canonical_sha256(
                    loader_components
                ),
                "updates": [probe_update],
                "cleanup": {"status": "passed", "errors": [], **memory()},
            },
            "acceptance_soak": {
                "status": "passed",
                "completed_optimizer_updates": 76,
                "completed_micro_steps": 608,
                "completed_validation_steps": 111,
                "heartbeat_records_written": 188,
                "first_epoch_unique_case_count": 440,
                "next_epoch_unique_case_count": 168,
                "first_epoch_identity_sha256": mps_evidence.canonical_sha256(
                    sorted(
                        identity
                        for update in updates[:55]
                        for identity in update["microbatch_case_sha256_ordered"]
                    )
                ),
                "next_epoch_identity_sha256": mps_evidence.canonical_sha256(
                    sorted(
                        identity
                        for update in updates[55:]
                        for identity in update["microbatch_case_sha256_ordered"]
                    )
                ),
                "minimum_headroom_ratio": 0.25,
                "memory_before": {
                    "driver_allocated_bytes": 75,
                    "recommended_max_bytes": 100,
                },
                "memory_before_headroom_ratio": 0.25,
                "memory_after_optimizer_path": {
                    "driver_allocated_bytes": 75,
                    "recommended_max_bytes": 100,
                },
                "memory_after_headroom_ratio": 0.25,
                "validation_identity_sha256": mps_evidence.canonical_sha256(
                    sorted(validation_ids)
                ),
                "loader_start_components": loader_components,
                "loader_start_fingerprint": mps_evidence.canonical_sha256(
                    loader_components
                ),
                "updates": updates,
                "validation": validation,
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
                **memory(),
            },
        },
    )
    return index


def _write_valid_science(tmp_path: Path, attempt_id: str) -> Path:
    coverage = {
        "expected": 111, "observed": 111, "unique": 111,
        "duplicates": 0, "missing": [], "unexpected": [],
    }
    expected_ids = sorted(f"case-{index}" for index in range(111))
    accuracy = 89 / 111
    dice = 89 * 0.3 / 111
    weighted = 0.7 * accuracy + 0.3 * dice
    metrics = {"classification_accuracy": accuracy, "dice": dice,
               "weighted_composite": weighted, "coverage": coverage}
    artifacts = set(mps_evidence.SCIENCE_ARTIFACTS)
    for name in artifacts - {
        "run_record.json", "epochs.json", "score.json",
        "best_epoch_cases.json", "best_checkpoint.pth", "case_identity.json",
        "source.json", "resource_evidence.json", "regression.json",
        "checkpoint_receipt.json", "wandb_terminal.json", "bootstrap_proof.json",
    }:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "best_checkpoint.pth").write_bytes(b"checkpoint")
    source = {"git_commit": "a" * 40}
    identity = {"train": [f"train-{index}" for index in range(444)],
                "validation": expected_ids}
    canonical_cases = {
        split: [bootstrap._sha256_bytes(case_id.encode()) for case_id in identity[split]]
        for split in ("train", "validation")
    }
    gate_receipt_sha = "c" * 64
    bootstrap_body = {
        "dataset_identity": {"canonical_case_sha256": canonical_cases},
        "soak_approval": {
            "resource_gate_receipt_sha256": gate_receipt_sha,
            "source_git_commit": source["git_commit"],
        },
    }
    bootstrap_proof = {
        **bootstrap_body,
        "proof_sha256": mps_evidence.canonical_sha256(bootstrap_body),
    }
    _write_json(tmp_path / "source.json", source)
    _write_json(tmp_path / "case_identity.json", identity)
    _write_json(tmp_path / "bootstrap_proof.json", bootstrap_proof)
    baseline_score_sha = "d" * 64
    baseline_record_sha = "e" * 64
    config = {
        "resource_gate_receipt_sha256": gate_receipt_sha,
        "baseline": {
            "score_sha256": baseline_score_sha,
            "run_record_sha256": baseline_record_sha,
        },
        "wandb": {"config_sha256": "1" * 64},
    }
    _write_json(tmp_path / "config.json", config)
    _write_json(tmp_path / "resource_evidence.json", {
        "pretrained_sha256": "b" * 64,
        "canonical_case_sha256": canonical_cases,
        "baseline_score_sha256": baseline_score_sha,
        "baseline_run_record_sha256": baseline_record_sha,
    })
    _write_json(tmp_path / "epochs.json", {"epochs": [
        {
            "epoch": epoch, "optimizer_steps": 55, "micro_steps": 440,
            "completed_train_steps": epoch * 55, "coverage": coverage,
            "classification_accuracy": accuracy,
            "dice": dice if epoch == 3 else dice - 0.01,
            "weighted_composite": (
                weighted if epoch == 3 else 0.7 * accuracy + 0.3 * (dice - 0.01)
            ),
        }
        for epoch in range(1, 6)
    ]})
    _write_json(tmp_path / "score.json", {"selected_epoch": 3, "metrics": metrics})
    cases = [
        {
            "case_id": case_id,
            "truth": int(index < 89),
            "prediction": 1,
            "correct": int(index < 89),
            "dice": 0.3 if index < 89 else 0.0,
            "error_type": "true_positive" if index < 89 else "false_positive",
            "cls_probability": 0.9,
            "segmentation_max_probability": 0.9,
            "predicted_pixels_native": 10,
            "truth_pixels_native": 10 if index < 89 else 0,
            "intersection_pixels_native": 3 if index < 89 else 0,
            "decision_branch": "agreement",
            "applied_segmentation_threshold": 0.5,
        }
        for index, case_id in enumerate(expected_ids)
    ]
    _write_json(tmp_path / "best_epoch_cases.json", {"cases": cases})
    regression_rows = [
        {"case_id": row["case_id"],
         "candidate_correct": bool(row["correct"]),
         "baseline_correct": bool(row["correct"]),
         "taxonomy": "unchanged_correct" if row["correct"] else "unchanged_error"}
        for row in cases
    ]
    _write_json(tmp_path / "regression.json", {
        "cases": regression_rows,
        "baseline_id": "P2-B3-resolution-degradation-1e",
        "case_count": 111,
        "taxonomy_counts": {"fixed": 0, "regressed": 0,
                            "unchanged_correct": 89, "unchanged_error": 22},
    })
    checkpoint_receipt = {
        "attempt_id": attempt_id, "selected_epoch": 3,
        "source_git_commit": source["git_commit"],
        "config_sha256": bootstrap._sha256_file(tmp_path / "config.json"),
        "resource_evidence_sha256": bootstrap._sha256_file(
            tmp_path / "resource_evidence.json"
        ),
        "pretrained_sha256": "b" * 64,
        "case_identity_sha256": mps_evidence.canonical_sha256(identity),
        "checkpoint_sha256": bootstrap._sha256_file(tmp_path / "best_checkpoint.pth"),
    }
    _write_json(tmp_path / "checkpoint_receipt.json", checkpoint_receipt)
    verified_summary = {
        "best/epoch": 3,
        "resource/evidence_sha256": bootstrap._sha256_file(
            tmp_path / "resource_evidence.json"
        ),
        "regression/sha256": bootstrap._sha256_file(tmp_path / "regression.json"),
        "checkpoint/sha256": bootstrap._sha256_file(
            tmp_path / "best_checkpoint.pth"
        ),
    }
    wandb_terminal = {
        "id": attempt_id, "entity": "kimhyeonwoo2431-individual",
        "project": "treat-mmtb-task1", "state": "finished",
        "url": f"https://wandb.ai/run/{attempt_id}",
        "config_sha256": config["wandb"]["config_sha256"],
        "summary_sha256": mps_evidence.canonical_sha256(verified_summary),
        "verified_summary": verified_summary,
    }
    _write_json(tmp_path / "wandb_terminal.json", wandb_terminal)
    run_record = tmp_path / "run_record.json"
    _write_json(
        run_record,
        {
            "status": "completed",
            "wandb_finished": True,
            "external_final_test_untouched": True,
            "completed_epochs": 5,
            "completed_train_micro_steps": 2200,
            "completed_optimizer_steps": 275,
            "selected_epoch": 3,
            "metrics": metrics,
            "checkpoint_receipt_sha256": bootstrap._sha256_file(
                tmp_path / "checkpoint_receipt.json"
            ),
            "wandb_terminal_sha256": bootstrap._sha256_file(
                tmp_path / "wandb_terminal.json"
            ),
            "artifacts": sorted(artifacts),
            "wandb": {
                "id": attempt_id,
                "entity": "kimhyeonwoo2431-individual",
                "project": "treat-mmtb-task1",
                "url": f"https://wandb.ai/run/{attempt_id}",
            },
        },
    )
    index = tmp_path / "artifact_index.json"
    _write_json(
        index,
        {
            "schema_version": 1,
            "attempt_id": attempt_id,
            "status": "completed",
            "artifacts": {
                name: bootstrap._sha256_file(tmp_path / name)
                for name in artifacts
            },
        },
    )
    return index


def _independent_science(tmp_path: Path, attempt_id: str) -> dict[str, Any]:
    terminal = json.loads((tmp_path / "wandb_terminal.json").read_text())
    score = json.loads((tmp_path / "score.json").read_text())
    metadata = {
        "native_epoch": score["selected_epoch"] - 1,
        "selected_epoch": score["selected_epoch"],
        "native_best_metric": score["metrics"]["weighted_composite"],
        "selected_weighted_composite": score["metrics"]["weighted_composite"],
        "model_keys": ["weight"],
        "optimizer_keys": ["param_groups"],
    }
    return {
        "attempt_id": attempt_id,
        "checkpoint_sha256": bootstrap._sha256_file(
            tmp_path / "best_checkpoint.pth"
        ),
        "regression_sha256": bootstrap._sha256_file(tmp_path / "regression.json"),
        "wandb_id": attempt_id,
        "wandb_state": "finished",
        "wandb_url": terminal["url"],
        "wandb_config_sha256": terminal["config_sha256"],
        "wandb_summary_sha256": terminal["summary_sha256"],
        "checkpoint_metadata": metadata,
        "checkpoint_metadata_sha256": mps_evidence.canonical_sha256(metadata),
    }


@pytest.mark.parametrize("role", ["resource_gate", "scientific_run"])
def test_supervisor_accepts_only_exact_child_completion(
    tmp_path: Path, role: str
) -> None:
    path = (
        _write_valid_gate(tmp_path, "sealed-attempt")
        if role == "resource_gate"
        else _write_valid_science(tmp_path, "sealed-attempt")
    )
    independent = (
        _independent_science(tmp_path, "sealed-attempt")
        if role == "scientific_run"
        else None
    )
    valid, digest, chain = bootstrap._validate_child_completion(
        path, role, "sealed-attempt", 0, 0.1,
        independent_scientific_verification=independent,
    )
    assert valid is True
    assert digest == bootstrap._sha256_file(path)
    assert chain is not None
    assert bootstrap._validate_child_completion(
        path, role, "other-attempt", 0, 0.1,
        independent_scientific_verification=independent,
    )[0] is False
    assert bootstrap._validate_child_completion(
        path, role, "sealed-attempt", 1, 0.1,
        independent_scientific_verification=independent,
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


def test_supervisor_recomputes_raw_progress_chain(tmp_path: Path) -> None:
    path = _write_valid_gate(tmp_path, "sealed-attempt")
    heartbeat_rows = [
        json.loads(line)
        for line in (tmp_path / "acceptance_soak_heartbeat.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    resource = json.loads((tmp_path / "resource_evidence.json").read_text())
    progress_rows = [
        resource["probe"]["updates"][0]["critical_memory"],
        resource["probe"]["updates"][0],
    ]
    for row in heartbeat_rows:
        if row.get("phase") in {"first_epoch_train", "next_epoch_train"}:
            progress_rows.append(row["critical_memory"])
        progress_rows.append(row)
    gate_index = json.loads(path.read_text())
    progress_rows.append({"phase": "resource_process_cleanup",
                          **gate_index["cleanup"]})
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in progress_rows),
        encoding="utf-8",
    )
    valid, _digest, chain = bootstrap._validate_child_completion(
        path, "resource_gate", "sealed-attempt", 0, 0.1, progress,
        resource["bootstrap"]["proof_sha256"],
    )
    assert valid is True
    assert chain is not None
    assert chain["supervisor_progress_sha256"] == bootstrap._sha256_file(progress)


def test_supervisor_rejects_summary_only_gate_and_one_file_science(
    tmp_path: Path,
) -> None:
    minimal = tmp_path / "acceptance_soak_index.json"
    _write_json(minimal, {"schema_version": 2, "attempt_id": "a", "status": "passed"})
    assert bootstrap._validate_child_completion(
        minimal, "resource_gate", "a", 0, 0.1
    ) == (False, None, None)
    run_record = tmp_path / "run_record.json"
    _write_json(run_record, {"status": "completed", "wandb_finished": True})
    science = tmp_path / "artifact_index.json"
    _write_json(science, {
        "schema_version": 1, "attempt_id": "a", "status": "completed",
        "artifacts": {"run_record.json": bootstrap._sha256_file(run_record)},
    })
    assert bootstrap._validate_child_completion(
        science, "scientific_run", "a", 0, 0.1
    ) == (False, None, None)


def test_scientific_approval_rejects_fake_domain_and_offline_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt.json"
    _write_json(receipt, {"status": "completed"})
    with pytest.raises(RuntimeError, match="exact GitHub"):
        bootstrap._load_scientific_approval(
            "https://example.invalid/review", "a", receipt
        )
    monkeypatch.setattr(
        bootstrap.urllib.request,
        "urlopen",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("offline")),
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        bootstrap._load_scientific_approval(
            "https://github.com/dongguri92/TREAT-MMTB/pull/5#issuecomment-123",
            "a",
            receipt,
        )


def test_supervisor_latches_repeated_signals_before_child_reap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Child:
        pid = 12345
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    child = Child()

    def popen(*_args: object, **_kwargs: object) -> Child:
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)
        return child

    monkeypatch.setattr(bootstrap.subprocess, "Popen", popen)
    monkeypatch.setattr(bootstrap, "_load_contract", lambda: {
        "memory": {"minimum_headroom_ratio": 0.1},
        "watchdog": {"progress_timeout_seconds": 1},
    })
    monkeypatch.setattr(
        bootstrap, "_sealed_environment",
        lambda *_args: ({}, {"schema_version": 1}),
    )
    monkeypatch.setattr(
        bootstrap,
        "_recompute_loader_start",
        lambda _argv: {"components": [{}] * 8, "fingerprint": "a" * 64},
    )

    def terminate(target: Child, *_args: object) -> dict[str, object]:
        target.returncode = -15
        return {"returncode": -15}

    monkeypatch.setattr(bootstrap, "_terminate_child", terminate)
    with pytest.raises(RuntimeError, match="supervised resource_gate child failed"):
        bootstrap._supervise([], "resource_gate", tmp_path, "signal-test")
    receipt = json.loads((
        tmp_path / ".supervisor" / "signal-test" / "supervisor_receipt.json"
    ).read_text(encoding="utf-8"))
    assert receipt["signals_received"] == ["SIGTERM", "SIGTERM"]
    assert child.returncode == -15


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
    reviewed_head = "d" * 40
    monkeypatch.setattr(bootstrap, "_git_head", lambda: reviewed_head)
    comment_url = (
        "https://github.com/dongguri92/TREAT-MMTB/pull/5#issuecomment-123"
    )
    approval = {
            "schema_version": 1,
            "status": "approved",
            "attempt_id": "sealed",
            "gate_attempt_id": "sealed-resource-gate",
            "source_git_commit": reviewed_head,
            "resource_gate_receipt_sha256": bootstrap._sha256_file(gate_receipt),
            "reviewed_by": "dongguri92",
            "review_url": comment_url,
            "external_final_test_untouched": True,
    }

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({
                "user": {"login": "dongguri92"},
                "author_association": "OWNER",
                "body": "<!-- TREAT_MMTB_SOAK_APPROVAL_V1 -->\n"
                + json.dumps(approval),
            }).encode()

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", lambda *_a, **_k: Response())
    assert bootstrap.main(
        common + ["--scientific-run", "--soak-approval", comment_url]
    ) == 0
    assert [role for role, _ in calls] == ["resource_gate", "scientific_run"]


def test_gate_rejects_rehashed_raw_identity_forgery(tmp_path: Path) -> None:
    path = _write_valid_gate(tmp_path, "sealed-attempt")
    heartbeat = tmp_path / "acceptance_soak_heartbeat.jsonl"
    rows = [json.loads(line) for line in heartbeat.read_text().splitlines()]
    rows[55]["case_sha256"] = "f" * 64
    heartbeat.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    resource_path = tmp_path / "resource_evidence.json"
    resource = json.loads(resource_path.read_text())
    sealed_proof_sha = resource["bootstrap"]["proof_sha256"]
    resource["acceptance_soak"]["validation"] = rows[55:166]
    resource["acceptance_soak"]["validation_identity_sha256"] = (
        mps_evidence.canonical_sha256(
            sorted(row["case_sha256"] for row in rows[55:166])
        )
    )
    resource["canonical_case_sha256"]["validation"][0] = "f" * 64
    resource["bootstrap"]["dataset_identity"]["canonical_case_sha256"] = (
        resource["canonical_case_sha256"]
    )
    proof_body = {
        key: value for key, value in resource["bootstrap"].items()
        if key != "proof_sha256"
    }
    resource["bootstrap"]["proof_sha256"] = mps_evidence.canonical_sha256(
        proof_body
    )
    _write_json(resource_path, resource)
    index = json.loads(path.read_text())
    index["resource_evidence_sha256"] = bootstrap._sha256_file(resource_path)
    index["heartbeat_sha256"] = bootstrap._sha256_file(heartbeat)
    _write_json(path, index)
    assert bootstrap._validate_child_completion(
        path, "resource_gate", "sealed-attempt", 0, 0.1, None, sealed_proof_sha
    ) == (False, None, None)


def test_gate_rejects_invalid_feasibility_semantics_after_rehash(
    tmp_path: Path,
) -> None:
    path = _write_valid_gate(tmp_path, "sealed-attempt")
    resource_path = tmp_path / "resource_evidence.json"
    resource = json.loads(resource_path.read_text())
    resource["probe"]["updates"][0]["finite_losses"] = False
    _write_json(resource_path, resource)
    index = json.loads(path.read_text())
    index["resource_evidence_sha256"] = bootstrap._sha256_file(resource_path)
    _write_json(path, index)
    assert bootstrap._validate_child_completion(
        path, "resource_gate", "sealed-attempt", 0, 0.1
    ) == (False, None, None)


def test_gate_rejects_minimal_feasibility_summary_after_rehash(
    tmp_path: Path,
) -> None:
    path = _write_valid_gate(tmp_path, "sealed-attempt")
    resource_path = tmp_path / "resource_evidence.json"
    resource = json.loads(resource_path.read_text())
    resource["probe"] = {
        "status": "passed",
        "completed_optimizer_updates": 1,
        "completed_micro_steps": 8,
        "updates": resource["probe"]["updates"],
        "cleanup": resource["probe"]["cleanup"],
    }
    _write_json(resource_path, resource)
    index = json.loads(path.read_text())
    index["resource_evidence_sha256"] = bootstrap._sha256_file(resource_path)
    _write_json(path, index)
    assert bootstrap._validate_child_completion(
        path, "resource_gate", "sealed-attempt", 0, 0.1
    ) == (False, None, None)


def test_gate_rejects_loader_fingerprint_redefinition_after_rehash(
    tmp_path: Path,
) -> None:
    path = _write_valid_gate(tmp_path, "sealed-attempt")
    resource_path = tmp_path / "resource_evidence.json"
    resource = json.loads(resource_path.read_text())
    sealed_proof_sha = resource["bootstrap"]["proof_sha256"]
    components = resource["acceptance_soak"]["loader_start_components"]
    components[0]["tensors"]["image"]["sha256"] = "f" * 64
    fingerprint = mps_evidence.canonical_sha256(components)
    resource["acceptance_soak"]["loader_start_fingerprint"] = fingerprint
    resource["bootstrap"]["loader_start"] = {
        "components": components,
        "fingerprint": fingerprint,
    }
    proof_body = {
        key: value
        for key, value in resource["bootstrap"].items()
        if key != "proof_sha256"
    }
    resource["bootstrap"]["proof_sha256"] = mps_evidence.canonical_sha256(
        proof_body
    )
    _write_json(resource_path, resource)
    index = json.loads(path.read_text())
    index["resource_evidence_sha256"] = bootstrap._sha256_file(resource_path)
    _write_json(path, index)
    assert bootstrap._validate_child_completion(
        path,
        "resource_gate",
        "sealed-attempt",
        0,
        0.1,
        expected_bootstrap_proof_sha256=sealed_proof_sha,
    ) == (False, None, None)


def test_science_rejects_rehashed_raw_case_forgery(tmp_path: Path) -> None:
    path = _write_valid_science(tmp_path, "sealed-attempt")
    cases_path = tmp_path / "best_epoch_cases.json"
    cases = json.loads(cases_path.read_text())
    cases["cases"][0]["correct"] = 0
    _write_json(cases_path, cases)
    index = json.loads(path.read_text())
    index["artifacts"]["best_epoch_cases.json"] = bootstrap._sha256_file(cases_path)
    _write_json(path, index)
    assert bootstrap._validate_child_completion(
        path, "scientific_run", "sealed-attempt", 0, 0.1
    ) == (False, None, None)


def test_science_rejects_impossible_prediction_pixel_semantics(
    tmp_path: Path,
) -> None:
    path = _write_valid_science(tmp_path, "sealed-attempt")
    cases_path = tmp_path / "best_epoch_cases.json"
    payload = json.loads(cases_path.read_text())
    first = payload["cases"][0]
    first["prediction"] = 0
    first["correct"] = 0
    first["error_type"] = "false_negative"
    compensating = payload["cases"][89]
    compensating["prediction"] = 0
    compensating["correct"] = 1
    compensating["error_type"] = "true_negative"
    _write_json(cases_path, payload)
    regression_path = tmp_path / "regression.json"
    regression = json.loads(regression_path.read_text())
    regression["cases"][0].update({
        "candidate_correct": False,
        "baseline_correct": True,
        "taxonomy": "regressed",
    })
    regression["cases"][89].update({
        "candidate_correct": True,
        "baseline_correct": False,
        "taxonomy": "fixed",
    })
    regression["taxonomy_counts"] = {
        "fixed": 1,
        "regressed": 1,
        "unchanged_correct": 88,
        "unchanged_error": 21,
    }
    _write_json(regression_path, regression)
    index = json.loads(path.read_text())
    index["artifacts"]["best_epoch_cases.json"] = bootstrap._sha256_file(
        cases_path
    )
    index["artifacts"]["regression.json"] = bootstrap._sha256_file(
        regression_path
    )
    _write_json(path, index)
    with pytest.raises(ValueError, match="native case semantics"):
        mps_evidence.validate_scientific_artifacts(
            tmp_path,
            "sealed-attempt",
            independent_verification=_independent_science(
                tmp_path, "sealed-attempt"
            ),
        )
    assert bootstrap._validate_child_completion(
        path,
        "scientific_run",
        "sealed-attempt",
        0,
        0.1,
        independent_scientific_verification=_independent_science(
            tmp_path, "sealed-attempt"
        ),
    ) == (False, None, None)


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"epoch": 1, "best_metric": 0.8, "model": {"w": 1},
         "optimizer": {"state": 1}},
        {"epoch": 2, "best_metric": 0.8, "model": {},
         "optimizer": {"state": 1}},
        {"epoch": 2, "best_metric": 0.8, "model": {"w": 1},
         "optimizer": {}},
    ],
)
def test_checkpoint_metadata_rejects_mismatch_and_empty_mappings(
    checkpoint: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        scientific_verifier._checkpoint_metadata(checkpoint, 3, 0.8)


def test_science_rejects_wrong_published_coverage_after_rehash(
    tmp_path: Path,
) -> None:
    path = _write_valid_science(tmp_path, "sealed-attempt")
    score_path = tmp_path / "score.json"
    score = json.loads(score_path.read_text())
    score["metrics"]["coverage"] = {
        "expected": 0,
        "observed": 0,
        "unique": 0,
        "duplicates": 0,
        "missing": [],
        "unexpected": [],
    }
    _write_json(score_path, score)
    index = json.loads(path.read_text())
    index["artifacts"]["score.json"] = bootstrap._sha256_file(score_path)
    _write_json(path, index)
    assert bootstrap._validate_child_completion(
        path,
        "scientific_run",
        "sealed-attempt",
        0,
        0.1,
        independent_scientific_verification=_independent_science(
            tmp_path, "sealed-attempt"
        ),
    ) == (False, None, None)


def test_supervisor_requires_exact_scientific_progress_sequence(tmp_path: Path) -> None:
    path = _write_valid_science(tmp_path, "sealed-attempt")
    checkpoint_sha = bootstrap._sha256_file(tmp_path / "best_checkpoint.pth")
    terminal = json.loads((tmp_path / "wandb_terminal.json").read_text())
    rows: list[dict[str, object]] = []
    global_step = 0
    for epoch in range(1, 6):
        for optimizer_step in range(1, 56):
            global_step += 1
            rows.append({
                "phase": "backward_complete_pre_adamw_memory",
                "epoch": epoch,
                "optimizer_step_in_epoch": optimizer_step,
                "global_optimizer_step": global_step,
                "memory": {"driver_allocated_bytes": 75,
                           "recommended_max_bytes": 100},
                "headroom_ratio": 0.25,
            })
            for offset in range(1, 9):
                rows.append({
                    "phase": "scientific_train", "epoch": epoch,
                    "micro_step": (optimizer_step - 1) * 8 + offset,
                    "optimizer_step": global_step,
                    "optimizer_step_completed": offset == 8,
                })
        rows.extend({
            "phase": "scientific_validation", "epoch": epoch,
            "validation_step": step,
        } for step in range(1, 112))
    rows.append({
        "phase": "scientific_completion", "attempt_id": "sealed-attempt",
        "artifact_index_sha256": bootstrap._sha256_file(path),
        "checkpoint_sha256": checkpoint_sha, "selected_epoch": 3,
        "wandb": terminal,
    })
    progress = tmp_path / "science-progress.jsonl"
    progress.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    assert bootstrap._validate_child_completion(
        path, "scientific_run", "sealed-attempt", 0, 0.1, progress,
        independent_scientific_verification=_independent_science(
            tmp_path, "sealed-attempt"
        ),
    )[0] is True
    rows[1]["micro_step"] = 2
    progress.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    assert bootstrap._validate_child_completion(
        path, "scientific_run", "sealed-attempt", 0, 0.1, progress,
        independent_scientific_verification=_independent_science(
            tmp_path, "sealed-attempt"
        ),
    ) == (False, None, None)


def test_supervisor_rejects_truncated_scientific_progress_without_escaping(
    tmp_path: Path,
) -> None:
    path = _write_valid_science(tmp_path, "sealed-attempt")
    progress = tmp_path / "truncated-progress.jsonl"
    progress.write_text("{}\n", encoding="utf-8")
    assert bootstrap._validate_child_completion(
        path,
        "scientific_run",
        "sealed-attempt",
        0,
        0.1,
        progress,
        independent_scientific_verification=_independent_science(
            tmp_path, "sealed-attempt"
        ),
    ) == (False, None, None)


def test_supervisor_seals_setup_failure_after_attempt_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "_load_contract",
        lambda: (_ for _ in ()).throw(RuntimeError("broken contract")),
    )
    with pytest.raises(RuntimeError, match="supervised resource_gate child failed"):
        bootstrap._supervise([], "resource_gate", tmp_path, "setup-failure")
    supervisor = tmp_path / ".supervisor" / "setup-failure"
    receipt = json.loads((supervisor / "supervisor_receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["supervisor_error_type"] == "RuntimeError"
    assert (supervisor / "supervisor_index.json").is_file()


def test_signal_during_receipt_staging_never_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "_load_contract",
        lambda: (_ for _ in ()).throw(RuntimeError("broken contract")),
    )
    original_stage = bootstrap._stage_json
    delivered = False

    def stage(path: Path, value: Any) -> Path:
        nonlocal delivered
        staged = original_stage(path, value)
        if path.name == "supervisor_receipt.json" and not delivered:
            delivered = True
            os.kill(os.getpid(), signal.SIGTERM)
        return staged

    monkeypatch.setattr(bootstrap, "_stage_json", stage)
    with pytest.raises(RuntimeError, match="supervised resource_gate child failed"):
        bootstrap._supervise([], "resource_gate", tmp_path, "signal-sealing")
    receipt = json.loads((
        tmp_path / ".supervisor/signal-sealing/supervisor_receipt.json"
    ).read_text())
    assert receipt["status"] == "failed"
    assert receipt["signals_received"] == ["SIGTERM"]


@pytest.mark.parametrize("signal_destination", ["supervisor_receipt.json",
                                                   "supervisor_index.json"])
def test_signal_in_success_publication_window_rolls_back_to_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_destination: str,
) -> None:
    class Child:
        pid = 12345
        returncode = 0

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(bootstrap.subprocess, "Popen", lambda *_a, **_k: Child())
    monkeypatch.setattr(bootstrap, "_load_contract", lambda: {
        "memory": {"minimum_headroom_ratio": 0.1},
        "watchdog": {"progress_timeout_seconds": 1},
    })
    monkeypatch.setattr(
        bootstrap, "_sealed_environment",
        lambda *_args: ({}, {"schema_version": 1}),
    )
    monkeypatch.setattr(
        bootstrap, "_recompute_loader_start",
        lambda _argv: {"components": [{}] * 8, "fingerprint": "a" * 64},
    )
    monkeypatch.setattr(
        bootstrap, "_validate_child_completion",
        lambda *_args, **_kwargs: (True, "b" * 64, {"verified": True}),
    )
    original_publish = bootstrap._publish_staged
    delivered = False

    def publish(staged: Path, destination: Path) -> None:
        nonlocal delivered
        original_publish(staged, destination)
        if destination.name == signal_destination and not delivered:
            delivered = True
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(bootstrap, "_publish_staged", publish)
    with pytest.raises(RuntimeError, match="supervised resource_gate child failed"):
        bootstrap._supervise([], "resource_gate", tmp_path, "publish-signal")
    supervisor = tmp_path / ".supervisor/publish-signal"
    receipt = json.loads((supervisor / "supervisor_receipt.json").read_text())
    index = json.loads((supervisor / "supervisor_index.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["signals_received"] == ["SIGTERM"]
    assert index["status"] == "failed"
    assert index["receipt_sha256"] == bootstrap._sha256_file(
        supervisor / "supervisor_receipt.json"
    )
