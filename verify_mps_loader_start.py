"""Recompute the sealed first accumulated group directly from canonical inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets
from reproduce_teammate_l05_mps import (
    GRADIENT_ACCUMULATION_STEPS,
    PHYSICAL_BATCH_SIZE,
    TARGET_SIZE,
    _batch_fingerprint_component,
    _set_seed,
    mps_protocol_contract,
)
from reproduction import (
    canonical_sha256,
    load_canonical_manifest,
    validate_canonical_content,
    validate_dataset_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-dcm-dir", type=Path, required=True)
    parser.add_argument("--train-mask-dir", type=Path, required=True)
    parser.add_argument("--val-dcm-dir", type=Path, required=True)
    parser.add_argument("--val-mask-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_canonical_manifest(args.manifest)
    validate_canonical_content(
        manifest["identity"],
        args.train_dcm_dir,
        args.train_mask_dir,
        args.val_dcm_dir,
        args.val_mask_dir,
    )
    datasets.TRAIN_DCM_DIR = str(args.train_dcm_dir)
    datasets.TRAIN_MASK_DIR = str(args.train_mask_dir)
    datasets.VAL_DCM_DIR = str(args.val_dcm_dir)
    datasets.VAL_MASK_DIR = str(args.val_mask_dir)
    protocol = mps_protocol_contract()
    _set_seed(protocol["seed"])
    loader, val_loader = datasets.dataloader(
        batch_size=PHYSICAL_BATCH_SIZE,
        target_size=TARGET_SIZE,
        clahe_clip=protocol["clahe_clip"],
        num_workers=0,
        seed=protocol["seed"],
        crop_frac=protocol["crop_frac"],
    )
    validate_dataset_identity(loader, val_loader, manifest["identity"])
    iterator = iter(loader)
    components = [
        _batch_fingerprint_component(next(iterator))
        for _ in range(GRADIENT_ACCUMULATION_STEPS)
    ]
    print(
        json.dumps(
            {
                "components": components,
                "fingerprint": canonical_sha256(components),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
