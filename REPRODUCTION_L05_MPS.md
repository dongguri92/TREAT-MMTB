# Apple-MPS lambda=0.5 health amendment

This is the separately reviewed Apple-MPS amendment for
[Issue #106](https://github.com/choco9966/TREAT-MMTB-2026/issues/106), whose
parent is [Issue #95](https://github.com/choco9966/TREAT-MMTB-2026/issues/95).
It does not replace the Linux x86_64/CUDA 11.8 contract in
`REPRODUCTION_L05.md` and does not claim historical CUDA equivalence.

The scientific configuration remains exact canonical 444 train / deterministic
111 internal-validation cases, EVA-X small segmentation plus classification,
`lambda_cls=0.5`, and combo veto `t_veto=0.005`. The amendment is a fresh
five-epoch health gate only. External-final data remains untouched.

## Locked MPS environment

The reviewed target is CPython 3.14.7 on Darwin arm64 with Apple MPS,
PyTorch 2.13.0, and torchvision 0.28.0. CPU fallback is forbidden. Install the
complete macOS 14+ arm64 lock without changing the CUDA lock:

```bash
MACOSX_DEPLOYMENT_TARGET=14.0 uv venv --python 3.14.7 .venv-reproduction-mps
MACOSX_DEPLOYMENT_TARGET=14.0 uv pip sync \
  --python .venv-reproduction-mps/bin/python \
  --require-hashes requirements-reproduction-mps.lock
```

The exact repo-local `.venv-reproduction-mps/` path is ignored by Git, so the
required clean-source identity gate remains valid after creating the documented
environment.

`requirements-reproduction-mps.in` records the direct inputs used to compile
the hash lock. The runner fails closed unless every locked distribution,
Python patch version, operating system, architecture, PyTorch/torchvision
version, and MPS availability match exactly. It also rejects
`PYTORCH_ENABLE_MPS_FALLBACK=1`.

Execution must start through the stdlib-only
`reproduce_teammate_l05_mps_bootstrap.py` launcher. Before importing PyTorch,
the launcher sets missing allocator values to exactly low `0.9` / high `1.0`,
rejects altered values, verifies the reviewed `Macmini9,1` host receipt, and
seals the allocator, host, launcher, worker, contract, parent PID, and exact
child command into the bootstrap proof. Direct `--execute` calls to the worker
are rejected.

## Preregistered resource contract

- Input is attempted at 1024 first and only.
- Physical batch is 1 with gradient accumulation 8, preserving effective
  batch 8.
- Batch-Dice remains enabled. Each optimizer update accumulates the mean of
  eight independently reduced microbatch multitask losses; this is the exact
  resource-adjusted MPS update rule, not a claim of numerical identity with a
  physical-batch-8 Batch-Dice reduction.
- Each epoch consumes 440 microsteps and 55 optimizer updates; the final four
  shuffled cases are dropped exactly as the physical-batch-8 CUDA loader drops
  an incomplete batch.
- Before W&B initialization, eight distinct real 1×1×1024×1024 microbatches
  must complete the exact accumulated optimizer path: finite multitask loss,
  loss/8 backward, gradient clipping at 12, unchanged AdamW step, zero-grad,
  and `torch.mps.synchronize()`.
- A mandatory no-W&B resource child completes a full first epoch of 55
  optimizer updates / 440 microsteps, traverses all 111 deterministic
  validation cases exactly once, applies the next-epoch LR transition, and
  completes another 21 optimizer updates / 168 microsteps. Every update and
  validation step records a flushed and fsynced append-only heartbeat.
- The resource contract requires `num_workers=0`, Albumentations 2 Compose
  seeds (`42` geometric, `43` intensity), no persistent workers, and at least
  10% headroom relative to the MPS recommended maximum before and throughout
  the gate. Missing memory APIs fail closed.
- The supervisor enforces a 30-minute no-progress timeout. It captures a
  process sample, sends SIGTERM, allows a 120-second cleanup grace, and only
  then escalates. Child artifacts cannot mark supervisor completion: the
  supervisor independently verifies and hashes the write-once completion
  index and emits its own immutable receipt.
- After the resource gate, all disposable tensors, model, optimizer, loaders,
  and workers are destroyed, followed by GC, MPS cache clearing, and MPS
  synchronization. Scientific training is a separate fresh child process and
  must consume the supervisor-owned gate receipt.
- Probe failure creates only immutable `resource_evidence.json` and
  `resource_index.json`. It never starts W&B or training and never changes to
  512. A 512 attempt requires a separate reviewed issue and fresh attempt.

## Review-only preflight

Dry-run is the default and does not inspect data or create artifacts:

```bash
.venv-reproduction-mps/bin/python reproduce_teammate_l05_mps_bootstrap.py \
  --attempt-id teammate-l05-mps-health-001 \
  --manifest /local/internal_validation_manifest.json \
  --baseline-score /local/baseline/score.json \
  --baseline-run-record /local/baseline/run_record.json \
  --pretrained /local/eva_x_small_patch16_merged520k_mim.pt \
  --train-dcm-dir /local/train/CXR \
  --train-mask-dir /local/train/CXR_label \
  --val-dcm-dir /local/validation/CXR \
  --val-mask-dir /local/validation/CXR_label
```

Execution additionally requires both `--execute` and
`--reviewed-by <review-url-or-reviewer>`. Do not add either until this amendment
has independent approval. W&B is fixed to entity
`kimhyeonwoo2431-individual`, project `treat-mmtb-task1`, online mode,
`resume=never`, and the fresh attempt ID.

To run only the acceptance gate and exit before W&B or scientific training,
add `--acceptance-soak-only`. The bootstrap creates a separate
`<attempt-id>-resource-gate` child attempt and supervisor receipt. Without that
flag, a passing gate is followed by a distinct scientific child. Both paths
still require `--execute`, independent review attestation, exact canonical
inputs, and a fresh immutable attempt ID.

Successful execution logs every microstep's total, segmentation, and
classification loss, accumulation boundary, optimizer-step count, and every LR
group. Every epoch logs full deterministic 111-case Accuracy, Dice, weighted
composite, coverage, runtime, 440 microsteps, and 55 optimizer updates. Local
artifacts seal source, protocol, dependency, checkpoint, canonical data,
identity, feasibility, score, paired regression, checkpoint, W&B identity, and
artifact hashes. W&B receives aggregate counts and hashes, never case IDs or
local paths.

Resource and scientific children stream privacy-safe progress out of process to
the supervisor. `SIGTERM` and `SIGINT` are controlled cancellations. An active
W&B run is finished with `exit_code=1`; a sanitized immutable `failure.json`
records the signal, finish outcome, and any partial-heartbeat hash/count; and
the prior process handlers are restored before exit.

No training or W&B run is authorized by this document alone.
