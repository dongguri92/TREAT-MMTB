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
`best_checkpoint.pth` bytes. It rebuilds the prospective source,
manifest/content, pretrained, baseline, dependency, and resource contract and
requires exact equality after normalizing only attempt, phase, epoch, and W&B
run identity. Both the health checkpoint path and its bytes are forbidden as
initialization; the candidate must use the reviewed pretrained weights.
The emitted queue specification is single-attempt (`max_attempts=1`), binds the
review receipt, uses a distinct W&B identity with `resume=never`, and forbids
external-final access. A retry is a new reviewed attempt and directory.

The successful 50-epoch path preserves the base runner's signal handling:
SIGINT/SIGTERM finish active W&B with `exit_code=1`, write an immutable
privacy-safe failure receipt, and restore prior handlers. The base artifact
index remains immutable. A second `convergence_index.json` binds that index,
the health approval, convergence decision, and pending Issue #93 handoff.

## Conditional epoch-150 semantics

Epoch 150 is never auto-launched. Eligibility requires a complete healthy
50-epoch trajectory, positive last-ten composite slope, non-deteriorating
validation-loss slope, more than 0.01 improvement over epoch five, and no
plateau. Eligibility produces only a request for fresh independent review.

The safest preregistered extension semantics are an exact continuation from
the sealed epoch-50 selected checkpoint to total epoch 150, with optimizer,
scheduler, RNG, update-count, and checkpoint hashes restored and verified.
That continuation requires a new runner/spec/attempt/W&B ID and review receipt;
this amendment deliberately contains no executable epoch-150 path.

The Issue #93 handoff remains `pending_independent_review` and
`launch_eligible=false`. It records the composite-selected epoch budget, never
the health-gate selected epoch, and becomes launch evidence only after a
separate reviewer binds the final convergence bytes and reviewed commit.

No GPU, data, feasibility probe, training, or W&B operation is performed by
the tests or dry-run documentation.
