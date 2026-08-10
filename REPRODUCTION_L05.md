# Teammate EVA-X lambda=0.5 reproduction

This is a sealed internal-validation reproduction for
[Issue #95](https://github.com/choco9966/TREAT-MMTB-2026/issues/95). It never
uses the external-final cohort.

The fixed contract is EVA-X small, one channel, 1024 input, physical and
effective batch 8, `lambda_cls=0.5`, AdamW at `5e-5`, cosine scheduling with
five warmup epochs, and native-grid combo inference with
`cls_threshold=0.5`, `t_veto=0.005`, and `min_pixels=0`.

The scored environment is fixed to CPython 3.11 on Linux x86_64, CUDA 11.8,
PyTorch `2.5.1+cu118`, and torchvision `0.20.1+cu118`. Install the complete
hash-locked dependency graph first:

```bash
uv venv --python 3.11 .venv-reproduction
uv pip sync --python .venv-reproduction/bin/python --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  requirements-reproduction.lock
```

`requirements-reproduction.in` records the direct inputs used to compile the
lock. The launcher refuses scored execution on another Python, OS,
architecture, PyTorch, torchvision, CUDA, or dependency version, and requires
the PyTorch `safe_globals` API used by the model loader.

The only accepted pretrained file is
`eva_x_small_patch16_merged520k_mim.pt` (307,569,543 bytes, SHA-256
`135d70a6988b5aacfe4848e1c2a0d524b2c076536fcaccdce88b636d302316c2`),
downloaded from the pinned upstream EVA-X revision
`35ddcd6dab6ca99bbdb6cb45c8d1b093aefbd0ee`.

Every invocation binds the canonical 444/111 manifest and byte identities,
the promoted current baseline score/run record, pretrained weights, clean Git
source, dependency lock/runtime versions, case identities, and step counts.
Local input paths are used only for loading and are never placed in W&B config
or portable artifacts. W&B receives only aggregate counts and sealed hashes;
raw train/validation identities are kept out of the remote config.

Review a five-epoch health dry run (omit `--execute`):

```bash
python reproduce_teammate_l05.py \
  --phase health --attempt-id teammate-l05-health-001 \
  --manifest /local/internal_validation_manifest.json \
  --baseline-score /local/baseline/score.json \
  --baseline-run-record /local/baseline/run_record.json \
  --pretrained /local/eva_x_small_patch16_merged520k_mim.pt \
  --train-dcm-dir /local/train/CXR \
  --train-mask-dir /local/train/CXR_label \
  --val-dcm-dir /local/validation/CXR \
  --val-mask-dir /local/validation/CXR_label
```

A scored health run additionally requires
`--execute --reviewed-by <reviewer-or-review-url>`. W&B is fixed to entity
`kimhyeonwoo2431-individual`, project `treat-mmtb-task1`, online mode, the
attempt ID as the W&B ID, and `resume=never`.

Convergence is a fresh 50-epoch attempt and consumes the completed health
`artifact_index.json` via `--health-artifact-index`. Before model or W&B setup,
the launcher recomputes every indexed artifact hash and revalidates source,
pretrained bytes, protocol, canonical cohort/content, baseline, dependency,
case identity, reviewer, finite five-epoch metrics, 55 training steps per
epoch, exact 111-case validation coverage, and completed W&B evidence.
It also proves that health and convergence differ only by phase and requested
epoch count, and cross-checks the selected epoch, score, per-case coverage,
regression hash, and W&B public-config hash before the 50-epoch attempt starts.

Every supplied path is checked lexically for external-final names before path
resolution or any content access, then checked again after resolution to catch
symlink targets. A rejected external-final path is therefore fail-closed.

Completed attempts contain only relative artifact names. They include config,
source, case identity, dependency lock, all epoch evidence, best-epoch cases,
paired regression evidence (taxonomy, exact McNemar, deterministic 10,000-case
bootstrap), score, checkpoint, run record, and artifact index. The run record
and index are marked completed only after W&B finishes successfully. A failure
at any earlier stage writes a sanitized `failure.json` and cannot produce a
completed attempt.
