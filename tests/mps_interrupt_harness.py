"""Subprocess harness for signal-safe MPS runner regression tests."""

from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import reproduce_teammate_l05_mps as mps
import reproduction


class SizedLoader:
    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length


class Model:
    def to(self, _device: object) -> Model:
        return self

    def parameters(self) -> list[Any]:
        return []


class WandbRun:
    def __init__(self, ready_path: Path, finish_path: Path):
        self.id = "interrupt-test"
        self.entity = reproduction.WANDB_ENTITY
        self.project = reproduction.WANDB_PROJECT
        self.summary: dict[str, Any] = {}
        self._finish_path = finish_path
        ready_path.write_text("ready\n", encoding="utf-8")

    def finish(self, exit_code: int | None = None) -> None:
        self._finish_path.write_text(
            json.dumps({"exit_code": exit_code}), encoding="utf-8"
        )


def main() -> int:
    artifact_root, ready_path, finish_path, restored_path = map(
        Path, sys.argv[1:5]
    )
    identity = {
        "dataset_scope": reproduction.DATASET_SCOPE,
        "train": [f"train-{index}" for index in range(444)],
        "validation": [f"validation-{index}" for index in range(111)],
    }
    mps.source_identity = lambda _root: {  # type: ignore[assignment]
        "git_commit": "commit",
        "git_tree_sha1": "tree",
    }
    mps.load_canonical_manifest = lambda _path: {  # type: ignore[assignment]
        "manifest_sha256": "manifest",
        "identity": identity,
    }
    mps.validate_canonical_content = lambda *_args: {  # type: ignore[assignment]
        "combined_content_sha256": "content"
    }
    mps.load_pinned_baseline = lambda *_args: {  # type: ignore[assignment]
        "baseline_id": "baseline",
        "score_sha256": "score",
        "run_record_sha256": "record",
        "cases": [],
    }
    mps.validate_pretrained = lambda _path: {  # type: ignore[assignment]
        "sha256": "pretrained",
        "byte_size": 1,
    }
    mps.validate_mps_runtime_dependencies = lambda _path: {  # type: ignore[assignment]
        "python": "3.14.7",
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "distributions": {"torch": "2.13.0", "torchvision": "0.28.0"},
        "mps_built": True,
        "mps_available": True,
        "mps_cpu_fallback": False,
    }
    mps.datasets.dataloader = lambda **_kwargs: (  # type: ignore[assignment]
        SizedLoader(444),
        SizedLoader(111),
    )
    mps.validate_dataset_identity = lambda *_args: (  # type: ignore[assignment]
        identity,
        "identity-hash",
    )
    mps._mps_feasibility_probe = lambda *_args: {  # type: ignore[assignment]
        "status": "passed",
        "finite_loss": True,
    }
    mps.modeltype = lambda *_args, **_kwargs: Model()  # type: ignore[assignment]
    mps._start_wandb = lambda _config: WandbRun(  # type: ignore[assignment]
        ready_path, finish_path
    )

    def wait_for_signal(*_args: Any, **_kwargs: Any) -> None:
        while True:
            time.sleep(1)

    mps.fit = wait_for_signal  # type: ignore[assignment]
    cli = [
        "--attempt-id",
        "interrupt-test",
        "--artifact-root",
        str(artifact_root),
        "--manifest",
        "manifest.json",
        "--baseline-score",
        "score.json",
        "--baseline-run-record",
        "run_record.json",
        "--pretrained",
        reproduction.EXPECTED_PRETRAINED_NAME,
        "--train-dcm-dir",
        "train-dcm",
        "--train-mask-dir",
        "train-mask",
        "--val-dcm-dir",
        "validation-dcm",
        "--val-mask-dir",
        "validation-mask",
        "--execute",
        "--reviewed-by",
        "interrupt-regression",
    ]
    previous = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        mps.run(mps.parse_args(cli))
    except mps.RunInterrupted:
        if all(signal.getsignal(signum) == handler for signum, handler in previous.items()):
            restored_path.write_text("restored\n", encoding="utf-8")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
