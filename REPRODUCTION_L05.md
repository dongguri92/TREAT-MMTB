# Teammate EVA-X lambda=0.5 reproduction

This is a sealed internal-validation reproduction for
[Issue #95](https://github.com/choco9966/TREAT-MMTB-2026/issues/95). It never
uses the external-final cohort.

The fixed contract is EVA-X small, one channel, 1024 input, physical and
effective batch 8, `lambda_cls=0.5`, AdamW at `5e-5`, cosine scheduling with
five warmup epochs, and native-grid combo inference with
`cls_threshold=0.5`, `t_veto=0.005`, and `min_pixels=0`.

Install the exact environment first:

```bash
python -m pip install -r requirements-reproduction.lock
```

Every invocation binds the canonical 444/111 manifest and byte identities,
the promoted current baseline score/run record, pretrained weights, clean Git
source, dependency lock/runtime versions, case identities, and step counts.
Local input paths are used only for loading and are never placed in W&B config
or portable artifacts.

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

Completed attempts contain only relative artifact names. They include config,
source, case identity, dependency lock, all epoch evidence, best-epoch cases,
paired regression evidence (taxonomy, exact McNemar, deterministic 10,000-case
bootstrap), score, checkpoint, run record, and artifact index. The run record
and index are marked completed only after W&B finishes successfully. A failure
at any earlier stage writes a sanitized `failure.json` and cannot produce a
completed attempt.
