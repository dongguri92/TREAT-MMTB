"""Independent post-child verifier for scientific checkpoint and W&B evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import wandb

from reproduce_teammate_l05_mps import _wandb_public_config
from reproduction import (
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_paired_regression,
    canonical_sha256,
    load_pinned_baseline,
    read_json,
    sha256_file,
)


def verify(
    science_dir: Path,
    attempt_id: str,
    baseline_score: Path,
    baseline_run_record: Path,
) -> dict[str, Any]:
    identity = read_json(science_dir / "case_identity.json")
    config = read_json(science_dir / "config.json")
    score = read_json(science_dir / "score.json")
    resource = read_json(science_dir / "resource_evidence.json")
    cases = read_json(science_dir / "best_epoch_cases.json").get("cases")
    regression = read_json(science_dir / "regression.json")
    checkpoint_receipt = read_json(science_dir / "checkpoint_receipt.json")
    if not isinstance(cases, list):
        raise TypeError("scientific case rows are missing")
    validation_ids = identity.get("validation")
    if not isinstance(validation_ids, list):
        raise TypeError("scientific validation identity is missing")
    baseline = load_pinned_baseline(
        baseline_score, baseline_run_record, validation_ids
    )
    expected_regression = build_paired_regression(cases, baseline["cases"])
    if regression != expected_regression:
        raise ValueError("paired regression differs from pinned baseline recomputation")

    checkpoint_path = science_dir / "best_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a mapping")
    reproduction = checkpoint.get("reproduction")
    expected_reproduction = {
        "attempt_id": attempt_id,
        "selected_epoch": score["selected_epoch"],
        "source": read_json(science_dir / "source.json"),
        "config_sha256": sha256_file(science_dir / "config.json"),
        "resource_evidence_sha256": sha256_file(
            science_dir / "resource_evidence.json"
        ),
        "pretrained_sha256": resource["pretrained_sha256"],
        "case_identity_sha256": canonical_sha256(identity),
    }
    if (
        reproduction != expected_reproduction
        or checkpoint_receipt.get("checkpoint_sha256") != sha256_file(checkpoint_path)
    ):
        raise ValueError("checkpoint embedded reproduction metadata is invalid")

    remote = wandb.Api(timeout=30).run(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/{attempt_id}"
    )
    if str(remote.id) != attempt_id or str(remote.state).lower() != "finished":
        raise ValueError("authoritative W&B run is not finished")
    remote_config = dict(remote.config)
    if remote_config != _wandb_public_config(config):
        raise ValueError("authoritative W&B config differs from sealed config")
    remote_summary = dict(remote.summary)
    expected_summary = {
        **{
            f"best/{name}": value
            for name, value in score["metrics"].items()
            if name != "coverage"
        },
        "best/epoch": score["selected_epoch"],
        "resource/evidence_sha256": score["resource_evidence_sha256"],
        "regression/fixed": regression["taxonomy_counts"]["fixed"],
        "regression/regressed": regression["taxonomy_counts"]["regressed"],
        "regression/sha256": score["regression_sha256"],
        "checkpoint/sha256": checkpoint_receipt["checkpoint_sha256"],
    }
    if any(remote_summary.get(key) != value for key, value in expected_summary.items()):
        raise ValueError("authoritative W&B summary differs from sealed evidence")
    return {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "checkpoint_sha256": checkpoint_receipt["checkpoint_sha256"],
        "checkpoint_metadata_sha256": canonical_sha256(reproduction),
        "regression_sha256": sha256_file(science_dir / "regression.json"),
        "wandb_id": str(remote.id),
        "wandb_state": str(remote.state).lower(),
        "wandb_url": str(remote.url),
        "wandb_config_sha256": canonical_sha256(remote_config),
        "wandb_summary_sha256": canonical_sha256(expected_summary),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--science-dir", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--baseline-run-record", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.science_dir,
                args.attempt_id,
                args.baseline_score,
                args.baseline_run_record,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
