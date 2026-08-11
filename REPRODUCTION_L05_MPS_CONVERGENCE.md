# Apple-MPS 50-epoch convergence amendment

This is a separately sealed, resource-adjusted Apple-MPS convergence path for
Issue #95. It stacks on the reviewed Issue #106 five-epoch health runner but
does not resume from or load the health checkpoint. The candidate starts fresh
from the same pinned EVA-X pretrained weights and keeps the exact 444/111,
1024, physical-batch-1/accumulation-8, effective-batch-8, `lambda_cls=0.5`, and
classification-veto `0.005` contract.

`reproduce_teammate_l05_mps_convergence.py` is dry-run by default. It requires
an immutable, independently reviewed health approval receipt whose exact
health run record and artifact index hashes are bound to the new attempt ID.
Validation rehashes every file covered by that index, including the actual
`best_checkpoint.pth` bytes. Authorization files are opened once without
following a final symlink, their opened descriptor identity is checked against
the approved canonical path, and consumers parse or load the same captured
bytes rather than reopening a path. It rebuilds the prospective source,
manifest/content, pretrained, baseline, dependency, and resource contract and
requires exact equality after normalizing only attempt, phase, epoch, and W&B
run identity. The approved health head and convergence head differ, so a
separate independently reviewed source-delta receipt must bind both source
identities and the exact hashes of every execution-surface file. Both the
health checkpoint path and its bytes are forbidden as
initialization; the candidate must use the reviewed pretrained weights.
The emitted queue specification is single-attempt (`max_attempts=1`), binds the
review receipt, uses a distinct W&B identity with `resume=never`, and forbids
external-final access. Dry-run writes that specification once under
`<artifact-root>/queue_specs/`; execution requires a separate reviewer approval
bound to its exact SHA-256. A retry is a new reviewed attempt and directory.
The convergence protocol requires `num_workers=0`; this removes unobservable
persistent-worker RNG from the continuation boundary.

The successful 50-epoch path preserves the base runner's signal handling:
SIGINT/SIGTERM finish active W&B with `exit_code=1`, write an immutable
privacy-safe failure receipt, and restore prior handlers. The base artifact
index remains immutable, and convergence finalization installs its own signal
boundary and immutable failure receipt. A second `convergence_index.json` binds
the rehashed base artifacts, original health approval, queue approval/spec,
config, run record, best checkpoint, exact epoch-50 continuation checkpoint,
and convergence decision. The pending Issue #93 handoff then binds both indexes
and both checkpoints. W&B name/group/job type are convergence-specific and the
run record requires post-finish API verification of the complete identity.
The exact reviewed queue-approval bytes are copied write-once into
`queue_approval.json`, included in the base artifact index, and compared with
the revalidated original approval again during final sealing.

## Conditional epoch-150 semantics

Epoch 150 is never auto-launched. Eligibility requires a complete healthy
50-epoch trajectory, positive last-ten composite slope, non-deteriorating
validation-loss slope, more than 0.01 improvement over epoch five, and no
plateau. Eligibility produces only a request for fresh independent review.

The safest preregistered extension semantics are an exact continuation from
the distinct sealed epoch-50 checkpoint to total epoch 150. It contains model,
optimizer, scheduler/LR phase, completed epoch/update counters, and
Python/NumPy/Torch/MPS RNG state. It also contains the dedicated shuffled
DataLoader generator state, an explicit empty worker-RNG set under the
zero-worker policy, the completed 50-epoch scheduler horizon, and the exact
cosine epoch-51-through-150 rule. Machine-readable continuation fields point
only to `continuation_epoch_50.pth`; the independently selected best checkpoint
remains separate and is never a continuation source.
That continuation requires a new runner/spec/attempt/W&B ID and review receipt;
this amendment deliberately contains no executable epoch-150 path.

The Issue #93 handoff remains `pending_independent_review` and
`launch_eligible=false`. It records the composite-selected epoch budget, never
the health-gate selected epoch, and becomes launch evidence only after a
separate reviewer binds the final convergence bytes and reviewed commit.

No GPU, private data, feasibility probe, training, or W&B operation has been
performed by this remediation. `NO-LAUNCH` remains in force until the new
source-delta and queue receipts receive fresh independent approval.
