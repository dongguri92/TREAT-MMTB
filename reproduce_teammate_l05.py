"""Review-gated teammate EVA-X lambda=0.5 + combo-veto reproduction.

The default invocation is a non-mutating dry run. A scored attempt requires
both ``--execute`` and ``--reviewed-by``. Convergence is always a fresh
50-epoch attempt and additionally requires a completed 5-epoch health record.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

import datasets
from models import modeltype
from reproduction import (
    CLS_THRESHOLD,
    EXPECTED_TRAIN_CASES,
    EXPECTED_VALIDATION_CASES,
    MIN_PIXELS,
    VETO_THRESHOLD,
    WANDB_ENTITY,
    WANDB_PROJECT,
    reject_external_final_path,
    sha256_file,
    source_identity,
    validate_dataset_identity,
    write_json_once,
)
from training import fit


REPO_ROOT = Path(__file__).resolve().parent
ISSUE_URL = "https://github.com/choco9966/TREAT-MMTB-2026/issues/95"
PHASE_EPOCHS = {"health": 5, "convergence": 50}


def _phase_record(path):
    with Path(path).open(encoding="utf-8") as stream:
        record = json.load(stream)
    coverage = record.get("metrics", {}).get("coverage", {})
    metrics = record.get("metrics", {})
    if (
        record.get("phase") != "health"
        or record.get("status") != "completed"
        or record.get("completed_epochs") != PHASE_EPOCHS["health"]
        or record.get("external_final_test_untouched") is not True
        or coverage.get("expected") != EXPECTED_VALIDATION_CASES
        or coverage.get("observed") != EXPECTED_VALIDATION_CASES
        or coverage.get("unique") != EXPECTED_VALIDATION_CASES
        or coverage.get("missing")
        or coverage.get("unexpected")
        or coverage.get("duplicates")
        or not all(
            math.isfinite(float(metrics.get(name, math.nan)))
            for name in (
                "classification_accuracy", "dice", "weighted_composite"
            )
        )
    ):
        raise ValueError("convergence requires a completed healthy 5-epoch record")
    return record


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_config(args, source, pretrained_sha256, device):
    return {
        "schema_version": 1,
        "issue_url": ISSUE_URL,
        "attempt_id": args.attempt_id,
        "phase": args.phase,
        "epochs": PHASE_EPOCHS[args.phase],
        "fresh_attempt": True,
        "resume_policy": "fresh_attempt_only",
        "reviewed_by": args.reviewed_by,
        "model": "evax_seg",
        "variant": "small",
        "channels": 1,
        "target_size": 1024,
        "batch_size": 8,
        "effective_batch_size": 8,
        "lambda_cls": 0.5,
        "optimizer": "adamw",
        "initial_lr": 5e-5,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "loss": "dicece",
        "crop_frac": 0.15,
        "clahe_clip": 2.0,
        "seed": 42,
        "num_workers": args.num_workers,
        "checkpoint_selection": "max_0.7_accuracy_plus_0.3_dice",
        "validation_space": "native",
        "validation_sampling": "deterministic_without_replacement",
        "expected_train_cases": EXPECTED_TRAIN_CASES,
        "expected_validation_cases": EXPECTED_VALIDATION_CASES,
        "inference": {
            "detection": "combo",
            "cls_threshold": CLS_THRESHOLD,
            "t_veto": VETO_THRESHOLD,
            "min_pixels": MIN_PIXELS,
        },
        "wandb": {
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "mode": "online",
            "run_name": args.wandb_run_name,
        },
        "device": str(device),
        "source": source,
        "pretrained_path": str(args.pretrained),
        "pretrained_sha256": pretrained_sha256,
        "dataset_paths": {
            "train_dcm": str(args.train_dcm_dir),
            "train_mask": str(args.train_mask_dir),
            "validation_dcm": str(args.val_dcm_dir),
            "validation_mask": str(args.val_mask_dir),
        },
        "external_final_isolation": {
            "external_final_path_available": False,
            "all_bound_paths_rejected_if_external_final": True,
            "external_final_test_untouched": True,
        },
        "health_run_record": (
            str(args.health_run_record) if args.health_run_record else None
        ),
    }


def _start_wandb(config):
    import wandb

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=config["wandb"]["run_name"],
        mode="online",
        group="task1-teammate-l05-veto-reproduction",
        job_type=config["phase"],
        config=config,
    )
    run.define_metric("train/global_step")
    run.define_metric("train/*", step_metric="train/global_step")
    run.define_metric("validation/global_step")
    run.define_metric("validation/*", step_metric="validation/global_step")
    run.define_metric("epoch")
    run.define_metric("epoch/*", step_metric="epoch")
    return run


def run(args):
    for label in (
        "train_dcm_dir", "train_mask_dir", "val_dcm_dir", "val_mask_dir",
        "pretrained", "artifact_root",
    ):
        reject_external_final_path(getattr(args, label), label)
    if args.health_run_record is not None:
        reject_external_final_path(args.health_run_record, "health_run_record")
    if args.phase == "convergence":
        if args.health_run_record is None:
            raise ValueError("convergence requires --health-run-record")
        health_record = _phase_record(args.health_run_record)
    else:
        health_record = None

    artifact_dir = args.artifact_root / args.attempt_id
    if artifact_dir.exists():
        raise FileExistsError(
            "attempt directory already exists; use a new attempt ID for every retry"
        )
    if not args.execute:
        return {
            "status": "dry_run",
            "phase": args.phase,
            "epochs": PHASE_EPOCHS[args.phase],
            "artifact_dir": str(artifact_dir),
            "review_required": True,
        }
    if not args.reviewed_by or not args.reviewed_by.strip():
        raise ValueError("--execute requires a non-empty --reviewed-by attestation")

    source = source_identity(REPO_ROOT)
    pretrained_sha256 = sha256_file(args.pretrained)
    device = _device(args.device)
    config = build_config(args, source, pretrained_sha256, device)
    artifact_dir.mkdir(parents=True, exist_ok=False)

    datasets.TRAIN_DCM_DIR = str(args.train_dcm_dir)
    datasets.TRAIN_MASK_DIR = str(args.train_mask_dir)
    datasets.VAL_DCM_DIR = str(args.val_dcm_dir)
    datasets.VAL_MASK_DIR = str(args.val_mask_dir)
    _set_seed(config["seed"])
    train_loader, val_loader = datasets.dataloader(
        batch_size=config["batch_size"],
        target_size=config["target_size"],
        clahe_clip=config["clahe_clip"],
        num_workers=config["num_workers"],
        seed=config["seed"],
        crop_frac=config["crop_frac"],
    )
    identity, identity_sha256 = validate_dataset_identity(
        train_loader, val_loader
    )
    write_json_once(artifact_dir / "case_identity.json", identity)
    config["case_identity_sha256"] = identity_sha256
    config["train_steps_per_epoch"] = len(train_loader)
    config["validation_steps_per_epoch"] = len(val_loader)

    model = modeltype(
        "evax_seg", in_channels=1, img_size=1024,
        pretrained_path=str(args.pretrained), variant="small",
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config["parameter_count"] = parameter_count
    write_json_once(artifact_dir / "config.json", config)
    write_json_once(artifact_dir / "source.json", source)
    config_sha256 = sha256_file(artifact_dir / "config.json")
    checkpoint_path = artifact_dir / "best_checkpoint.pth"
    wandb_run = _start_wandb(config)
    started = time.perf_counter()
    try:
        details = fit(
            model, train_loader, val_loader, device,
            max_epochs=config["epochs"], lambda_cls=config["lambda_cls"],
            initial_lr=config["initial_lr"], ckpt_path=str(checkpoint_path),
            patience=None, batch_dice=True, optimizer_name=config["optimizer"],
            scheduler=config["scheduler"], warmup_epochs=config["warmup_epochs"],
            loss_name=config["loss"], wandb_run=wandb_run,
            reproduction_expected_ids=identity["validation"],
            return_details=True,
        )
        best = details["best_native_metrics"]
        if best is None:
            raise RuntimeError("native validation did not produce best-epoch metrics")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint["reproduction"] = {
            "attempt_id": args.attempt_id,
            "phase": args.phase,
            "selected_epoch": details["best_epoch"],
            "source": source,
            "config_sha256": config_sha256,
            "pretrained_sha256": pretrained_sha256,
            "case_identity_sha256": identity_sha256,
        }
        torch.save(checkpoint, checkpoint_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        cases_path = artifact_dir / "best_epoch_cases.json"
        write_json_once(cases_path, {"cases": best["cases"]})
        score = {
            "schema_version": 1,
            "attempt_id": args.attempt_id,
            "phase": args.phase,
            "dataset_scope": "task1_train_444_internal_validation_111",
            "external_final_test_untouched": True,
            "selected_epoch": details["best_epoch"],
            "metrics": {
                "classification_accuracy": best["classification_accuracy"],
                "dice": best["dice"],
                "weighted_composite": best["weighted_composite"],
                "coverage": best["coverage"],
            },
            "decision": config["inference"],
            "case_identity_sha256": identity_sha256,
            "cases_sha256": sha256_file(cases_path),
        }
        score_path = artifact_dir / "score.json"
        write_json_once(score_path, score)
        runtime_seconds = time.perf_counter() - started
        run_url = wandb_run.get_url()
        run_record = {
            "schema_version": 1,
            "issue_url": ISSUE_URL,
            "attempt_id": args.attempt_id,
            "phase": args.phase,
            "status": "completed",
            "fresh_attempt": True,
            "resume_policy": "fresh_attempt_only",
            "reviewed_by": args.reviewed_by,
            "external_final_test_untouched": True,
            "requested_epochs": config["epochs"],
            "completed_epochs": details["completed_epochs"],
            "completed_train_steps": details["completed_train_steps"],
            "selected_epoch": details["best_epoch"],
            "runtime_seconds": runtime_seconds,
            "metrics": score["metrics"],
            "wandb": {
                **config["wandb"],
                "url": run_url,
            },
            "hashes": {
                "source_sha256": sha256_file(artifact_dir / "source.json"),
                "config_sha256": config_sha256,
                "source_git_commit": source["git_commit"],
                "source_git_tree_sha1": source["git_tree_sha1"],
                "pretrained_sha256": pretrained_sha256,
                "case_identity_sha256": identity_sha256,
                "score_sha256": sha256_file(score_path),
                "checkpoint_sha256": checkpoint_sha256,
                "cases_sha256": sha256_file(cases_path),
            },
            "artifacts": {
                "config": str(artifact_dir / "config.json"),
                "source": str(artifact_dir / "source.json"),
                "score": str(score_path),
                "checkpoint": str(checkpoint_path),
                "cases": str(cases_path),
            },
            "health_gate": (
                {
                    "source_attempt_id": health_record["attempt_id"],
                    "source_run_record": str(args.health_run_record),
                    "source_run_record_sha256": sha256_file(args.health_run_record),
                }
                if health_record is not None else None
            ),
        }
        run_record_path = artifact_dir / "run_record.json"
        write_json_once(run_record_path, run_record)
        artifact_index = {
            "run_record": str(run_record_path),
            "run_record_sha256": sha256_file(run_record_path),
            "score_sha256": sha256_file(score_path),
            "checkpoint_sha256": checkpoint_sha256,
            "source_sha256": sha256_file(artifact_dir / "source.json"),
            "config_sha256": config_sha256,
            "pretrained_sha256": pretrained_sha256,
        }
        write_json_once(artifact_dir / "artifact_index.json", artifact_index)
        for name, value in score["metrics"].items():
            if name != "coverage":
                wandb_run.summary[f"best/{name}"] = value
        wandb_run.summary["best/epoch"] = details["best_epoch"]
        wandb_run.summary["artifacts/checkpoint_sha256"] = checkpoint_sha256
        wandb_run.summary["artifacts/run_record_sha256"] = artifact_index[
            "run_record_sha256"
        ]
    except BaseException as error:
        write_json_once(artifact_dir / "failure.json", {
            "attempt_id": args.attempt_id,
            "phase": args.phase,
            "error_type": type(error).__name__,
            "error": str(error),
            "external_final_test_untouched": True,
        })
        wandb_run.finish(exit_code=1)
        raise
    else:
        wandb_run.finish()
    return artifact_index


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASE_EPOCHS), required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/reproduction"))
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--train-dcm-dir", type=Path, required=True)
    parser.add_argument("--train-mask-dir", type=Path, required=True)
    parser.add_argument("--val-dcm-dir", type=Path, required=True)
    parser.add_argument("--val-mask-dir", type=Path, required=True)
    parser.add_argument("--health-run-record", type=Path)
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--reviewed-by")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.attempt_id) is None:
        parser.error("--attempt-id must be a single safe path component")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    args.artifact_root = args.artifact_root.resolve()
    for name in (
        "pretrained", "train_dcm_dir", "train_mask_dir", "val_dcm_dir",
        "val_mask_dir", "health_run_record",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.wandb_run_name = args.wandb_run_name or (
        f"task1-teammate-l05-veto-{args.phase}-{args.attempt_id}"
    )
    return args


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
