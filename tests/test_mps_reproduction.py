import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mps_evidence
import reproduce_teammate_l05_mps as mps
import reproduction
import training


def _healthy_memory_snapshot() -> dict[str, int]:
    return {
        "current_allocated_bytes": 1,
        "driver_allocated_bytes": 10,
        "recommended_max_bytes": 100,
    }


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


def test_worker_bootstrap_proof_binds_launcher_and_exact_child_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = mps._load_mps_resource_contract()
    worker_python = "/reviewed/python"
    child_args = ["--attempt-id", "sealed", "--internal-worker"]
    monkeypatch.setattr(mps.sys, "argv", [str(mps.MPS_BOOTSTRAP_PATH), *child_args])
    monkeypatch.setenv("TREAT_MMTB_MPS_WORKER_PYTHON", worker_python)
    proof = {
        "schema_version": 1,
        "role": "resource_gate",
        "allocator": contract["allocator"],
        "host": contract["host"],
        "contract_sha256": reproduction.sha256_file(
            mps.MPS_RESOURCE_CONTRACT_PATH
        ),
        "launcher_sha256": reproduction.sha256_file(mps.MPS_BOOTSTRAP_PATH),
        "worker_sha256": reproduction.sha256_file(Path(mps.__file__).resolve()),
        "child_argv_sha256": reproduction.canonical_sha256(
            [worker_python, str(Path(mps.__file__).resolve()), *child_args]
        ),
        "parent_pid": os.getppid(),
        "nonce": "a" * 64,
        "torch_imported_in_bootstrap": False,
        "loader_start": {
            "components": [{} for _ in range(mps.GRADIENT_ACCUMULATION_STEPS)],
            "fingerprint": reproduction.canonical_sha256(
                [{} for _ in range(mps.GRADIENT_ACCUMULATION_STEPS)]
            ),
        },
    }
    encoded = json.dumps(proof, sort_keys=True, separators=(",", ":"))
    monkeypatch.setenv("TREAT_MMTB_MPS_BOOTSTRAP_PROOF", encoded)
    monkeypatch.setenv(
        "TREAT_MMTB_MPS_BOOTSTRAP_PROOF_SHA256",
        reproduction.canonical_sha256(proof),
    )
    monkeypatch.setattr(
        mps,
        "validate_mps_allocator_environment",
        lambda: {"values": contract["allocator"]},
    )
    assert mps._validate_bootstrap_proof("resource_gate")["role"] == "resource_gate"
    proof["child_argv_sha256"] = "0" * 64
    altered = json.dumps(proof, sort_keys=True, separators=(",", ":"))
    monkeypatch.setenv("TREAT_MMTB_MPS_BOOTSTRAP_PROOF", altered)
    monkeypatch.setenv(
        "TREAT_MMTB_MPS_BOOTSTRAP_PROOF_SHA256",
        reproduction.canonical_sha256(proof),
    )
    with pytest.raises(RuntimeError, match="differs from reviewed contract"):
        mps._validate_bootstrap_proof("resource_gate")


def test_documented_repo_local_mps_venv_preserves_clean_source_identity(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text(
        (Path(__file__).resolve().parents[1] / ".gitignore").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    (repo / "tracked.txt").write_text("sealed\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "sealed",
        ],
        cwd=repo,
        check=True,
    )
    environment = repo / ".venv-reproduction-mps"
    environment.mkdir()
    (environment / "pyvenv.cfg").write_text("home = sealed\n", encoding="utf-8")

    assert reproduction.source_identity(repo)["git_commit"]


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


@pytest.mark.parametrize(
    ("failure_stage", "failure_factory"),
    [
        (
            "model_construction",
            lambda: (_ for _ in ()).throw(RuntimeError("MPS out of memory")),
        ),
        (
            "device_transfer",
            lambda: SimpleNamespace(
                to=lambda _device: (_ for _ in ()).throw(
                    RuntimeError("MPS backend out of memory")
                )
            ),
        ),
    ],
)
def test_probe_seals_model_construction_and_transfer_resource_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    failure_factory: Callable[[], object],
) -> None:
    monkeypatch.setattr(mps.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(mps, "_memory_snapshot", _healthy_memory_snapshot)
    monkeypatch.setattr(mps.torch.mps, "empty_cache", lambda: None, raising=False)
    monkeypatch.setattr(mps, "modeltype", lambda *_args, **_kwargs: failure_factory())

    with pytest.raises(mps.MPSFeasibilityFailure) as caught:
        mps._run_mps_optimizer_probe(
            object(),
            mps.torch.device("mps"),
            tmp_path / "pretrained.pt",
            optimizer_updates=1,
            probe_name="test",
        )
    assert caught.value.evidence["status"] == "failed"
    assert caught.value.evidence["failure_stage"] == failure_stage
    assert caught.value.evidence["out_of_memory"] is True


@pytest.mark.parametrize("failure_stage", ["forward", "backward"])
def test_probe_seals_forward_and_backward_resource_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    class Tensor:
        shape = (1, 1, 1024, 1024)

        def to(self, _device: object, **_kwargs: object) -> "Tensor":
            return self

    class Loss:
        def __add__(self, _other: object) -> "Loss":
            return self

        def __rmul__(self, _other: object) -> "Loss":
            return self

        def __truediv__(self, _other: object) -> "Loss":
            return self

        def detach(self) -> "Loss":
            return self

        def backward(self) -> None:
            raise RuntimeError("MPS backend out of memory")

        def item(self) -> float:
            return 1.0

    class ProbeModel:
        return_cls = False

        def to(self, _device: object) -> "ProbeModel":
            return self

        def train(self) -> None:
            pass

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True

        def __call__(self, _image: Tensor) -> tuple[object, object]:
            if failure_stage == "forward":
                raise RuntimeError("MPS backend out of memory")
            return object(), object()

    monkeypatch.setattr(mps.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(mps, "_memory_snapshot", _healthy_memory_snapshot)
    monkeypatch.setattr(mps.torch.mps, "empty_cache", lambda: None, raising=False)
    monkeypatch.setattr(mps, "modeltype", lambda *_args, **_kwargs: ProbeModel())
    monkeypatch.setattr(mps, "DiceCELoss", lambda **_kwargs: lambda *_args: Loss())
    monkeypatch.setattr(
        mps.torch.nn, "BCEWithLogitsLoss", lambda: lambda *_args: Loss()
    )
    monkeypatch.setattr(
        mps.torch, "isfinite", lambda _loss: SimpleNamespace(item=lambda: True)
    )
    monkeypatch.setattr(
        mps,
        "make_optimizer",
        lambda *_args, **_kwargs: SimpleNamespace(
            param_groups=[{"lr": 5e-5}],
            zero_grad=lambda **_kwargs: None,
            step=lambda: None,
        ),
    )
    loader = [
        {"image": Tensor(), "mask": Tensor(), "cls": Tensor(), "id": [f"case-{i}"]}
        for i in range(8)
    ]

    with pytest.raises(mps.MPSFeasibilityFailure) as caught:
        mps._run_mps_optimizer_probe(
            loader,
            mps.torch.device("mps"),
            tmp_path / "pretrained.pt",
            optimizer_updates=1,
            probe_name="test",
        )
    assert caught.value.evidence["failure_stage"] == failure_stage
    assert caught.value.evidence["out_of_memory"] is True


def test_execute_routes_failed_probe_to_resource_only_without_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = mps.parse_args(
        _cli(tmp_path)
        + ["--execute", "--reviewed-by", "review-url", "--acceptance-soak-only"]
    )
    args.internal_worker = True
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
            raise RuntimeError("MPS backend out of memory")

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
    monkeypatch.setattr(
        mps,
        "validate_mps_allocator_environment",
        lambda: {
            "values": dict(reproduction.EXPECTED_MPS_ALLOCATOR_ENVIRONMENT),
            "set_before_mps_runtime_validation": True,
            "sha256": "allocator",
        },
    )
    monkeypatch.setattr(
        mps,
        "_validate_bootstrap_proof",
        lambda _role: {
            "proof_sha256": "bootstrap",
            "loader_start": {"fingerprint": "a" * 64},
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
    monkeypatch.setattr(mps.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(mps, "_memory_snapshot", _healthy_memory_snapshot)
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


@pytest.mark.parametrize("interrupt_signal", [signal.SIGTERM, signal.SIGINT])
def test_subprocess_interrupt_closes_wandb_and_writes_safe_failure_receipt(
    tmp_path: Path, interrupt_signal: signal.Signals
) -> None:
    artifact_root = tmp_path / "artifacts"
    ready_path = tmp_path / "wandb-ready"
    finish_path = tmp_path / "wandb-finish.json"
    restored_path = tmp_path / "handlers-restored"
    harness = Path(__file__).with_name("mps_interrupt_harness.py")
    process = subprocess.Popen(
        [
            sys.executable,
            str(harness),
            str(artifact_root),
            str(ready_path),
            str(finish_path),
            str(restored_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
    )
    try:
        deadline = time.monotonic() + 20
        while not ready_path.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("interrupt harness did not reach W&B-backed training")
            time.sleep(0.02)
        assert process.poll() is None
        os.kill(process.pid, interrupt_signal)
        assert process.wait(timeout=20) == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    receipt = json.loads(
        (artifact_root / "interrupt-test" / "failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert {key: receipt[key] for key in (
        "attempt_id",
        "error_type",
        "external_final_test_untouched",
        "phase",
        "reason",
        "schema_version",
        "signal",
        "signal_number",
        "stage",
        "status",
        "wandb_exit_code_1_requested",
        "wandb_finish_succeeded",
    )} == {
        "attempt_id": "interrupt-test",
        "error_type": "RunInterrupted",
        "external_final_test_untouched": True,
        "phase": "health",
        "reason": "signal_interruption",
        "schema_version": 1,
        "signal": interrupt_signal.name,
        "signal_number": interrupt_signal.value,
        "stage": "training",
        "status": "failed",
        "wandb_exit_code_1_requested": True,
        "wandb_finish_succeeded": True,
    }
    assert receipt["partial_heartbeat_records"] == 1
    assert len(receipt["partial_heartbeat_sha256"]) == 64
    assert json.loads(finish_path.read_text(encoding="utf-8")) == {"exit_code": 1}
    assert restored_path.read_text(encoding="utf-8") == "restored\n"


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


@pytest.mark.parametrize(
    ("low", "high"),
    [(None, None), ("0.8", "1.0"), ("0.9", "1.1")],
)
def test_mps_allocator_environment_rejects_missing_or_altered_values(
    monkeypatch: pytest.MonkeyPatch, low: str | None, high: str | None
) -> None:
    for name, value in (
        ("PYTORCH_MPS_LOW_WATERMARK_RATIO", low),
        ("PYTORCH_MPS_HIGH_WATERMARK_RATIO", high),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match="allocator environment"):
        reproduction.validate_mps_allocator_environment()


def test_mps_allocator_environment_is_sealed_at_exact_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in reproduction.EXPECTED_MPS_ALLOCATOR_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    sealed = reproduction.validate_mps_allocator_environment()
    assert sealed["values"] == reproduction.EXPECTED_MPS_ALLOCATOR_ENVIRONMENT
    assert sealed["set_before_mps_runtime_validation"] is True
    assert sealed["sha256"] == reproduction.canonical_sha256(
        {
            "values": reproduction.EXPECTED_MPS_ALLOCATOR_ENVIRONMENT,
            "set_before_mps_runtime_validation": True,
        }
    )


def test_memory_headroom_fails_closed_on_missing_or_low_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mps,
        "_load_mps_resource_contract",
        lambda: {"memory": {"minimum_headroom_ratio": 0.1}},
    )
    with pytest.raises(RuntimeError, match="APIs are required"):
        mps._validate_memory_headroom({})
    with pytest.raises(RuntimeError, match="below reviewed minimum"):
        mps._validate_memory_headroom(
            {
                "driver_allocated_bytes": 95,
                "recommended_max_bytes": 100,
            }
        )


def test_mps_parser_rejects_background_workers(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        mps.parse_args(_cli(tmp_path) + ["--num-workers", "1"])


def test_mps_parser_rejects_path_traversal_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def unexpected_resolve(_self: Path) -> Path:
        nonlocal called
        called = True
        raise AssertionError("filesystem resolution must not occur")

    monkeypatch.setattr(Path, "resolve", unexpected_resolve)
    argv = _cli(tmp_path)
    argv[argv.index("--artifact-root") + 1] = "../escaped-artifacts"
    with pytest.raises(ValueError, match="path traversal"):
        mps.parse_args(argv)
    assert called is False


def test_albumentations_compose_seeds_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compose_calls: list[dict[str, object]] = []

    class FakeAlbumentations:
        def __getattr__(self, _name: str) -> Callable[..., object]:
            return lambda *_args, **_kwargs: object()

        def Compose(self, *_args: object, **kwargs: object) -> object:
            compose_calls.append(kwargs)
            return object()

    monkeypatch.setattr(mps.datasets, "_HAS_ALBU", True)
    monkeypatch.setattr(mps.datasets, "A", FakeAlbumentations())
    mps.datasets.build_geometric_aug(seed=42)
    mps.datasets.build_intensity_aug(seed=43)
    assert compose_calls[0]["seed"] == 42
    assert compose_calls[1]["seed"] == 43


def _mock_successful_optimizer_probe(
    monkeypatch: pytest.MonkeyPatch, *, fail_optimizer_step: bool = False
) -> list[dict[str, object]]:
    class Tensor:
        shape = (1, 1, 1024, 1024)

        def to(self, _device: object, **_kwargs: object) -> "Tensor":
            return self

        def cpu(self) -> "Tensor":
            return self

        def argmax(self, _dimension: int) -> "Tensor":
            return self

        def __getitem__(self, _key: object) -> "Tensor":
            return self

    class Loss:
        def __add__(self, _other: object) -> "Loss":
            return self

        def __rmul__(self, _other: object) -> "Loss":
            return self

        def __truediv__(self, _other: object) -> "Loss":
            return self

        def detach(self) -> "Loss":
            return self

        def backward(self) -> None:
            pass

        def item(self) -> float:
            return 1.0

    class ProbeModel:
        return_cls = False

        def to(self, _device: object) -> "ProbeModel":
            return self

        def train(self) -> None:
            pass

        def eval(self) -> None:
            pass

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True

        def parameters(self) -> list[object]:
            return []

        def __call__(self, _image: Tensor) -> tuple[object, object]:
            return Tensor(), Tensor()

    class Optimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 5e-5}]

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True

        def step(self) -> None:
            if fail_optimizer_step:
                raise RuntimeError("MPS Metal optimizer command buffer failed")

    monkeypatch.setattr(mps.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(mps.torch.mps, "empty_cache", lambda: None, raising=False)
    monkeypatch.setattr(mps.torch.mps, "synchronize", lambda: None, raising=False)
    monkeypatch.setattr(mps, "_memory_snapshot", _healthy_memory_snapshot)
    monkeypatch.setattr(mps, "modeltype", lambda *_args, **_kwargs: ProbeModel())
    monkeypatch.setattr(mps, "make_optimizer", lambda *_args, **_kwargs: Optimizer())
    monkeypatch.setattr(mps, "DiceCELoss", lambda **_kwargs: lambda *_args: Loss())
    monkeypatch.setattr(
        mps.torch.nn, "BCEWithLogitsLoss", lambda: lambda *_args: Loss()
    )
    monkeypatch.setattr(
        mps.torch.nn.utils, "clip_grad_norm_", lambda *_args, **_kwargs: Loss()
    )
    monkeypatch.setattr(
        mps.torch, "isfinite", lambda _value: SimpleNamespace(item=lambda: True)
    )
    monkeypatch.setattr(mps.torch, "softmax", lambda value, **_kwargs: value)
    monkeypatch.setattr(mps.torch, "sigmoid", lambda value: value)
    def fake_validation_step(
        *_args: object, stage_callback: Callable[[str], None], **_kwargs: object
    ) -> dict[str, object]:
        for event in mps_evidence.VALIDATION_OPERATION_EVENTS:
            stage_callback(event)
        return {"loss": 1.0}

    monkeypatch.setattr(mps, "validation_step", fake_validation_step)
    return [
        {
            "image": Tensor(),
            "mask": Tensor(),
            "cls": Tensor(),
            "id": [f"case-{index}"],
        }
        for index in range(mps.ACCEPTANCE_SOAK_MICRO_STEPS)
    ]


def test_no_wandb_acceptance_soak_covers_full_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _mock_successful_optimizer_probe(monkeypatch)
    validation_loader = [
        {
            "image": loader[0]["image"],
            "mask": loader[0]["mask"],
            "cls": loader[0]["cls"],
            "id": [f"validation-{index}"],
        }
        for index in range(mps.ACCEPTANCE_SOAK_VALIDATION_STEPS)
    ]
    heartbeat = tmp_path / "acceptance.jsonl"
    evidence = mps._run_mps_optimizer_probe(
        loader,
        mps.torch.device("mps"),
        tmp_path / "pretrained.pt",
        optimizer_updates=mps.ACCEPTANCE_SOAK_OPTIMIZER_STEPS,
        probe_name="acceptance",
        heartbeat_path=heartbeat,
        validation_loader=validation_loader,
        validation_after_updates=mps.ACCEPTANCE_FIRST_EPOCH_OPTIMIZER_STEPS,
    )
    assert evidence["status"] == "passed"
    assert evidence["wandb_started"] is False
    assert evidence["completed_optimizer_updates"] == 76
    assert evidence["completed_micro_steps"] == 608
    assert evidence["completed_validation_steps"] == 111
    assert len(evidence["updates"]) == 76
    assert len(evidence["validation"]) == 111
    assert all(
        row["operation_events"] == mps_evidence.VALIDATION_OPERATION_EVENTS
        for row in evidence["validation"]
    )
    assert evidence["first_epoch_unique_case_count"] == 440
    assert evidence["next_epoch_unique_case_count"] == 168
    assert all(update["distinct_microbatches"] == 8 for update in evidence["updates"])
    assert all("memory" in update for update in evidence["updates"])
    assert len(heartbeat.read_text(encoding="utf-8").splitlines()) == 188
    assert evidence["heartbeat_records_written"] == 188
    assert evidence["heartbeat_sha256"] == reproduction.sha256_file(heartbeat)


def test_first_group_fingerprint_materializes_each_batch_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _mock_successful_optimizer_probe(monkeypatch)
    calls: list[str] = []

    def component(batch: dict[str, object]) -> dict[str, object]:
        case_id = str(batch["id"][0])  # type: ignore[index]
        calls.append(case_id)
        return {"case": case_id}

    expected_components = [{"case": f"case-{index}"} for index in range(8)]
    monkeypatch.setattr(mps, "_batch_fingerprint_component", component)
    evidence = mps._run_mps_optimizer_probe(
        loader,
        mps.torch.device("mps"),
        tmp_path / "pretrained.pt",
        optimizer_updates=1,
        probe_name="single-pass-fingerprint",
        expected_start_fingerprint=reproduction.canonical_sha256(
            expected_components
        ),
    )
    assert calls == [f"case-{index}" for index in range(8)]
    assert evidence["loader_start_components"] == expected_components


def test_optimizer_path_failure_is_sealed_before_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _mock_successful_optimizer_probe(
        monkeypatch, fail_optimizer_step=True
    )
    with pytest.raises(mps.MPSFeasibilityFailure) as caught:
        mps._run_mps_optimizer_probe(
            loader,
            mps.torch.device("mps"),
            tmp_path / "pretrained.pt",
            optimizer_updates=1,
            probe_name="optimizer-failure",
        )
    assert caught.value.evidence["failure_stage"] == "optimizer_step"
    assert caught.value.evidence["wandb_started"] is False


def test_probe_cleanup_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _mock_successful_optimizer_probe(monkeypatch)
    calls = 0

    def empty_cache() -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(mps.torch.mps, "empty_cache", empty_cache, raising=False)
    with pytest.raises(mps.MPSFeasibilityFailure) as caught:
        mps._run_mps_optimizer_probe(
            loader,
            mps.torch.device("mps"),
            tmp_path / "pretrained.pt",
            optimizer_updates=1,
            probe_name="cleanup-failure",
        )
    assert caught.value.evidence["failure_stage"] == "cleanup"
    assert caught.value.evidence["cleanup"]["status"] == "failed"


def test_primary_probe_failure_survives_cleanup_headroom_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _mock_successful_optimizer_probe(
        monkeypatch, fail_optimizer_step=True
    )
    calls = 0

    def memory_snapshot() -> dict[str, int]:
        nonlocal calls
        calls += 1
        if calls >= 4:
            return {
                "current_allocated_bytes": 1,
                "driver_allocated_bytes": 95,
                "recommended_max_bytes": 100,
            }
        return _healthy_memory_snapshot()

    monkeypatch.setattr(mps, "_memory_snapshot", memory_snapshot)
    with pytest.raises(mps.MPSFeasibilityFailure) as caught:
        mps._run_mps_optimizer_probe(
            loader,
            mps.torch.device("mps"),
            tmp_path / "pretrained.pt",
            optimizer_updates=1,
            probe_name="primary-before-cleanup",
        )
    assert caught.value.evidence["failure_stage"] == "optimizer_step"
    assert caught.value.evidence["error_type"] == "RuntimeError"
    assert caught.value.evidence["cleanup"]["status"] == "failed"
    assert caught.value.evidence["cleanup"]["errors"] == [
        {"stage": "cleanup_memory_headroom", "error_type": "RuntimeError"}
    ]


def test_primary_probe_failure_survives_failure_memory_snapshot_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _mock_successful_optimizer_probe(monkeypatch, fail_optimizer_step=True)
    calls = 0

    def memory_snapshot() -> dict[str, int]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("failure snapshot unavailable")
        return _healthy_memory_snapshot()

    monkeypatch.setattr(mps, "_memory_snapshot", memory_snapshot)
    with pytest.raises(mps.MPSFeasibilityFailure) as caught:
        mps._run_mps_optimizer_probe(
            loader,
            mps.torch.device("mps"),
            tmp_path / "pretrained.pt",
            optimizer_updates=1,
            probe_name="failure-snapshot-error",
        )
    assert caught.value.evidence["failure_stage"] == "optimizer_step"
    assert caught.value.evidence["memory_at_failure"] == {}
    assert caught.value.evidence["failure_evidence_errors"] == [
        {"stage": "failure_memory_snapshot", "error_type": "RuntimeError"}
    ]


@pytest.mark.parametrize(
    ("failure_stage", "patch_target"),
    [
        ("failure_heartbeat_flush", "flush"),
        ("failure_heartbeat_fsync", "fsync"),
        ("failure_heartbeat_hash", "sha256_file"),
    ],
)
def test_primary_probe_failure_survives_heartbeat_sealing_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    patch_target: str,
) -> None:
    loader = _mock_successful_optimizer_probe(monkeypatch, fail_optimizer_step=True)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("heartbeat evidence unavailable")

    if patch_target == "flush":
        original_open = Path.open

        class FailingFlushStream:
            def __init__(self, stream: object) -> None:
                self.stream = stream

            def __getattr__(self, name: str) -> object:
                return getattr(self.stream, name)

            def flush(self) -> None:
                fail()

        def open_with_failing_flush(path: Path, *args: object, **kwargs: object) -> object:
            stream = original_open(path, *args, **kwargs)
            if path.name.endswith(".jsonl"):
                return FailingFlushStream(stream)
            return stream

        monkeypatch.setattr(Path, "open", open_with_failing_flush)
    elif patch_target == "fsync":
        monkeypatch.setattr(mps.os, "fsync", fail)
    else:
        monkeypatch.setattr(mps, "sha256_file", fail)
    with pytest.raises(mps.MPSFeasibilityFailure) as caught:
        mps._run_mps_optimizer_probe(
            loader,
            mps.torch.device("mps"),
            tmp_path / "pretrained.pt",
            optimizer_updates=1,
            probe_name=failure_stage,
            heartbeat_path=tmp_path / f"{failure_stage}.jsonl",
        )
    assert caught.value.evidence["failure_stage"] == "optimizer_step"
    assert {row["stage"] for row in caught.value.evidence["failure_evidence_errors"]} >= {
        failure_stage
    }


def test_transition_heartbeat_is_counted_when_callback_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _mock_successful_optimizer_probe(monkeypatch)
    validation_loader = [
        {
            "image": loader[0]["image"],
            "mask": loader[0]["mask"],
            "cls": loader[0]["cls"],
            "id": [f"validation-{index}"],
        }
        for index in range(mps.ACCEPTANCE_SOAK_VALIDATION_STEPS)
    ]

    def progress(record: dict[str, object]) -> None:
        if record.get("phase") == "next_epoch_transition":
            raise RuntimeError("transition callback failed")

    with pytest.raises(mps.MPSFeasibilityFailure) as caught:
        mps._run_mps_optimizer_probe(
            loader,
            mps.torch.device("mps"),
            tmp_path / "pretrained.pt",
            optimizer_updates=56,
            probe_name="transition-count",
            validation_loader=validation_loader,
            validation_after_updates=55,
            progress_callback=progress,
        )
    assert caught.value.evidence["heartbeat_records_written"] == 167


def test_accumulated_step_materializes_scalars_only_after_adamw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Tensor:
        def to(self, _device: object, **_kwargs: object) -> "Tensor":
            return self

    class Loss:
        def __add__(self, _other: object) -> "Loss":
            return self

        def __rmul__(self, _other: object) -> "Loss":
            return self

        def __truediv__(self, _other: object) -> "Loss":
            return self

        def backward(self) -> None:
            events.append("backward")

        def detach(self) -> "Loss":
            return self

        def item(self) -> float:
            events.append("item")
            return 1.0

    class Model:
        return_cls = False

        def __call__(self, _image: Tensor) -> tuple[Tensor, Tensor]:
            return Tensor(), Tensor()

        def parameters(self) -> list[object]:
            return []

    class Optimizer:
        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True

        def step(self) -> None:
            events.append("adamw")

    monkeypatch.setattr(training.torch.nn, "BCEWithLogitsLoss", lambda: lambda *_: Loss())
    monkeypatch.setattr(training.torch.nn.utils, "clip_grad_norm_", lambda *_a, **_k: Loss())
    batches = [
        {"image": Tensor(), "mask": Tensor(), "cls": Tensor()}
        for _ in range(8)
    ]
    training.accumulated_train_step(
        Model(), batches, Optimizer(), lambda *_: Loss(), 0.5,
        mps.torch.device("mps"), use_amp=False,
        critical_memory_observer=lambda: events.append("critical") or {},
    )
    assert events == (
        ["backward", "item", "item", "item", "item", "item"] * 7
        + [
            "backward", "critical", "adamw",
            "item", "item", "item", "item", "item",
        ]
    )


def test_first_scientific_group_is_observed_without_preview_advancement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    iterator_calls = 0

    class Tensor:
        def __init__(self, case_id: str):
            self.case_id = case_id

        def to(self, _device: object, **_kwargs: object) -> "Tensor":
            return self

    class Loss:
        def __add__(self, _other: object) -> "Loss":
            return self

        def __rmul__(self, _other: object) -> "Loss":
            return self

        def __truediv__(self, _other: object) -> "Loss":
            return self

        def backward(self) -> None:
            return None

        def detach(self) -> "Loss":
            return self

        def item(self) -> float:
            return 1.0

    class Loader:
        def __len__(self) -> int:
            return 8

        def __iter__(self):
            nonlocal iterator_calls
            iterator_calls += 1
            for index in range(8):
                tensor = Tensor(f"case-{index}")
                yield {"id": [tensor.case_id], "image": tensor, "mask": tensor, "cls": tensor}

    class Model:
        return_cls = False

        def train(self) -> None:
            return None

        def __call__(self, image: Tensor) -> tuple[Tensor, Tensor]:
            return image, image

        def parameters(self) -> list[object]:
            return []

    class Optimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 1.0}]

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none

        def step(self) -> None:
            return None

    monkeypatch.setattr(training.torch.nn, "BCEWithLogitsLoss", lambda: lambda *_: Loss())
    monkeypatch.setattr(training.torch.nn.utils, "clip_grad_norm_", lambda *_a, **_k: Loss())
    training.train_one_epoch(
        Model(), Loader(), Optimizer(), lambda *_: Loss(), 0.5,
        mps.torch.device("mps"), None, use_amp=False,
        gradient_accumulation_steps=8,
        first_group_batch_observer=lambda batch, _index: observed.append(batch["id"][0]),
        first_group_validator=lambda: None,
    )
    assert iterator_calls == 1
    assert observed == [f"case-{index}" for index in range(8)]


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
        "resource_gate_receipt_sha256": "resource-receipt",
        "mps_allocator": {
            "values": dict(reproduction.EXPECTED_MPS_ALLOCATOR_ENVIRONMENT),
            "set_before_mps_runtime_validation": True,
            "sha256": "allocator",
        },
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


def test_scientific_worker_requeries_github_and_rejects_forged_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "gate-receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    sealed = {"review_url": "https://github.com/dongguri92/TREAT-MMTB/pull/5#issuecomment-1"}
    args = argparse.Namespace(attempt_id="sealed", resource_gate_receipt=receipt)
    monkeypatch.setattr(
        mps,
        "verify_scientific_approval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("GitHub approval verification is unavailable")
        ),
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        mps._revalidate_scientific_approval(
            {"soak_approval": sealed}, args, "a" * 40
        )


def test_resource_probe_never_synchronizes_per_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _mock_successful_optimizer_probe(monkeypatch)
    calls = 0

    def synchronize() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(mps.torch.mps, "synchronize", synchronize, raising=False)
    mps._run_mps_optimizer_probe(
        loader,
        mps.torch.device("mps"),
        tmp_path / "pretrained.pt",
        optimizer_updates=2,
        probe_name="schedule-parity",
    )
    assert calls == 1  # final disposable-process cleanup only
