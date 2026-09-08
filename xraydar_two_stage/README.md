# X-Raydar two-stage inference core

This directory contains the standalone inference code from our final Task 1
submission. It does not depend on `evax_baseline/` or the original experiment
workspace. This is an inference release, not an end-to-end training release:
training scripts, datasets, caches, checkpoints, and Docker archives are excluded.
The team's method and results remain documented in the repository's main README.

## Pipeline

1. Histogram-conditioned XGBoost window normalization, with DICOM polarity
   resolved once (PresentationLUTShape takes precedence over photometric type).
2. Resize the shorter side to 1024; use a top-aligned, horizontally centered
   1024-square crop. The two model input standardizations are recorded in
   `runtime_config.json`; preserve them when using the released checkpoints.
3. Average sigmoid probabilities of three X-Raydar classifiers (EMA epochs
   215, 282, 299), with threshold 0.5 and no logit bias. Negative cases skip
   segmentation and receive an empty mask.
4. For positive cases, infer U-Net and Mask2Former probabilities, restore each
   to the original image grid, and average threshold-centered logits equally
   (U-Net boundary 0.06; Mask2Former boundary 0.54). Threshold at probability 0.5.
5. If a positive case's mask is empty, rescue at half its maximum probability;
   retain a one-pixel peak fallback. Write binary uint8 NIfTI masks with original
   geometry and a matching `prediction.csv` (`our_id,cavity`).

## Required artifacts

Obtain the exact submission artifacts from the team and place them under
`weights/` (git-ignored). Renaming arbitrary pretrained weights is not sufficient.

| File | Role |
|---|---|
| `classifier_1.pt` | Cavity classifier, EMA epoch 215 |
| `classifier_2.pt` | Cavity classifier, EMA epoch 282 |
| `classifier_3.pt` | Cavity classifier, EMA epoch 299 |
| `unet.pt` | Positive-only U-Net, epoch 500 / step 2000 |
| `mask2former.pt` | Positive-only Mask2Former, epoch 350 / step 1400 |
| `window_normalizer.joblib` | Fitted window-normalization bundle |

Verify exact artifact identity with `sha256sum -c weights.sha256` from this
directory. The checksums come from the final submission's staging manifest.

Only load trusted artifacts: the checkpoint and joblib loaders deserialize
Python objects. No artifact download or training-data access is performed here.

## Run locally

Run from this directory. Python 3.12 and CUDA 11.8 PyTorch wheels are pinned by
the supplied dependency files; do not regenerate the lock to reproduce the
submission environment. CPU inference is supported but may be slow.

```bash
uv sync --frozen --no-dev --no-install-project
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python predict.py \
  --input /absolute/path/to/input --output /absolute/path/to/output \
  --classifier-weights weights/classifier_1.pt weights/classifier_2.pt weights/classifier_3.pt \
  --unet-weights weights/unet.pt --mask2former-weights weights/mask2former.pt \
  --window-model weights/window_normalizer.joblib --runtime-config runtime_config.json
uv run --no-sync python validate_outputs.py \
  --input /absolute/path/to/input --output /absolute/path/to/output
```

Input layout is `input/<case_id>/<image>.dcm`, one 2D X-ray per case. Use an empty
output directory. Default batch size is 4; reduce with `--batch-size 1` if needed.
The submission implementation prepares all input images in host memory before
batching model inference, so split very large collections into smaller runs.

## Docker

After supplying the six artifacts, build from this directory:

```bash
docker build -t treat-xraydar:final .
docker run --rm --network none --gpus all \
  -v /absolute/path/to/input:/input:ro \
  -v /absolute/path/to/output:/output \
  treat-xraydar:final
```

Building requires network access for dependencies; inference does not. The build
strictly loads all five checkpoints on CPU to detect architecture mismatches.
Omit `--gpus all` for CPU execution. GPU execution needs a compatible host NVIDIA
driver and NVIDIA Container Toolkit. The Docker context uses an allowlist to
exclude unrelated local files. Validate outputs with the local command above.

## Code and provenance

- `model.py`: standalone PyTorch/torchvision X-Raydar classifier, U-Net, and
  Mask2Former-style decoder definitions; strict checkpoint loading.
- `window_normalization.py`: fitted-window preprocessing (no fitting code).
- `predict.py`: gating, original-grid ensembling, rescue, and output writing.
- `runtime_config.json`: final submission operating points.
- `validate_outputs.py`: IDs, binary masks, geometry, and CSV/mask consistency.
- `tests/`: synthetic regression checks without private data or weights.

Core inference source is copied unchanged from our final submission; only the
Docker runtime-config source path was relocated for this public layout. The
architecture builds on X-Raydar/XNet38, torchvision Inception-v3, and the
Mask2Former design. This upload does not grant redistribution rights for external
pretrained weights or datasets; consult their original terms before sharing them.
