# Teammate EVA-X lambda=0.5 reproduction

This protocol reconstructs the reported internal-validation best configuration
for [main tracking Issue #95](https://github.com/choco9966/TREAT-MMTB-2026/issues/95).
It does not use or inspect the external-final test.

The fixed scored contract is EVA-X small, one channel, 1024 input, batch 8,
`lambda_cls=0.5`, AdamW at `5e-5`, cosine scheduling with five warmup epochs,
and native-grid combo-veto inference with `cls_threshold=0.5`, `t_veto=0.01`,
and `min_pixels=0`. W&B is fixed to entity
`kimhyeonwoo2431-individual`, project `treat-mmtb-task1`, online mode.

Install the repository dependencies plus `wandb`, then review the dry-run plan:

```bash
python reproduce_teammate_l05.py \
  --phase health \
  --attempt-id teammate-l05-health-001 \
  --pretrained /absolute/path/eva_x_small_patch16_merged520k_mim.pt \
  --train-dcm-dir /absolute/path/train/CXR \
  --train-mask-dir /absolute/path/train/CXR_label \
  --val-dcm-dir /absolute/path/val/CXR \
  --val-mask-dir /absolute/path/val/CXR_label
```

After independent review, run the separately sealed five-epoch health attempt by
adding `--execute --reviewed-by <reviewer-or-review-url>`. The launcher refuses
to reuse an existing attempt directory and validates exact 444/111 unique,
non-overlapping identities before W&B starts.

Only after the health `run_record.json` reports a completed finite 5-epoch run
with exact 111/111/111 coverage may a fresh 50-epoch attempt start:

```bash
python reproduce_teammate_l05.py \
  --phase convergence \
  --attempt-id teammate-l05-convergence-001 \
  --health-run-record artifacts/reproduction/teammate-l05-health-001/run_record.json \
  --pretrained /absolute/path/eva_x_small_patch16_merged520k_mim.pt \
  --train-dcm-dir /absolute/path/train/CXR \
  --train-mask-dir /absolute/path/train/CXR_label \
  --val-dcm-dir /absolute/path/val/CXR \
  --val-mask-dir /absolute/path/val/CXR_label \
  --execute --reviewed-by <reviewer-or-review-url>
```

Each completed attempt writes immutable config/source/case-identity files, the
selected checkpoint, best-epoch case records, `score.json`, `run_record.json`,
and `artifact_index.json`. The index seals the run-record, score, checkpoint,
source, config, and pretrained hashes. Retries always use a new attempt ID; this
workflow intentionally does not resume or overwrite scored attempts.
