# Metal kernel research harness

The schema-2 harness is the default for new Metal kernel searches. Schema-1
templates, runs, and evidence remain readable by `scripts/metal_autoresearch.py`;
they are not reinterpreted as schema 2.

## Acceptance policy

The primary portfolio metric is optimized-CPU PIOP wall time divided by
Metal-hybrid PIOP wall time at log 26. The minimum accepted result is 5x over
five alternating pairs. The overall median and both order-stratum medians must
clear 5x. Every completed standalone or fused kernel also has a 5x local floor;
Instruction RA retains its 7x floor.

These are acceptance floors, not search caps. A candidate may become the next
search parent while still below 5x if it improves beyond the frozen noise
threshold. The kernel is not complete until it clears its local floor. Once the
portfolio reaches 5x, the loop continues when disjoint profile attribution
predicts another 5 percent PIOP gain or a measured kernel has clear additional
headroom. A 4x analytical estimate is a reason to investigate, never an
acceptance threshold.

## Evaluation model

An evaluator owns its paired replication. The controller launches one process,
then recomputes every control/treatment ratio, the median, MAD, and both order
strata from raw arm timings. Controller-level repeats around an internally
paired evaluator are forbidden.

Each search uses ordered tiers:

1. A proxy tier ranks candidates only after its ordering is calibrated against
   the representative scale. If no calibrated proxy exists, the template marks
   it inapplicable.
2. The representative tier selects a local parent using exact results at the
   target scale.
3. The log-26 holdout runs only after the local 5x gate and decides PIOP
   acceptance. It is never used to tune the candidate.
4. The log-27 transfer tier checks that an accepted result survives the next
   trace scale before the run is marked transferred.

The current OuterRemainder template deliberately disables its proxy tier. The
next Outer phase should first test whether a cheaper scale preserves candidate
ordering; until that evidence exists, at most a small analytically selected set
should reach log 26. A failed transfer resumes from the accepted holdout; it
does not rerun representative or holdout evidence.

## Time and resource accounting

The controller records queue wait, subprocess wall time, and exclusive-machine
lease time separately. Builds, warmups, CPU controls, crashes, timeouts, and
invalid results are charged. Representative and holdout reserves cannot be
spent by proxy screening.

Each reserve protects a bounded number of future invocations from screening
work. Spending that protection does not prohibit a retry if the total budget
still has capacity. Failures and interrupted attempts consume both their
measured resources and one protected invocation.

GPU-active time is recorded only when validated device timestamps exist. When
it is unavailable, the budget uses a conservative treatment-wall or subprocess-
wall upper bound and records that it is estimated. Historical `gpu_seconds`
fields are not relabeled as device-active time.

The controller admits an evaluator only when the remaining calendar budget can
contain its full timeout. Waiting for the machine lease is bounded by the
remaining slack. Timeout and cancellation terminate the evaluator process group
before the lease is released.

## Agent and machine ownership

Up to three proposal agents may work in parallel on independent analysis or
isolated candidate artifacts. The root agent alone modifies the shared
worktree, reconciles proposals, runs builds and tests, and owns the serialized
CPU/GPU evaluator lease. Proposal agents do not run Cargo, CPU controls, or GPU
evaluators.

Candidate analysis should contain:

- the current measured cost and exact timing boundary;
- arithmetic, byte-traffic, occupancy, launch, and synchronization ceilings;
- the expected useful operations per read and resident-state assumptions;
- one falsifiable change, its predicted gain, and its likely failure mode.

This lets the root agent reject low-ceiling ideas before a target-scale build.

## Durable records

A schema-2 run contains an atomically replaced, self-digested `run.json`, append-only baseline, candidate,
tier, decision, and portfolio ledgers, per-evaluation raw output and attempt
telemetry, and snapshots of accepted editable paths. `inflight.json` makes an
interrupted evaluation recoverable. If an attempt did not seal telemetry,
recovery conservatively charges the observed in-flight interval before
restoring the accepted parent.

Production states are monotonic for one accepted parent: active, then
portfolio-accepted at log 26, then transferred at log 27. Keeping a new parent
returns the run to active. Successful stages are reused after interruption;
attempt identifiers are never reused.

## Commands

The canonical controller dispatches by contract version:

```text
python3 scripts/metal_autoresearch.py init TEMPLATE RUN_DIR
python3 scripts/metal_autoresearch.py candidate-context RUN_DIR
python3 scripts/metal_autoresearch.py trial RUN_DIR --summary TEXT --param NAME=VALUE
python3 scripts/metal_autoresearch.py status RUN_DIR
python3 scripts/metal_autoresearch.py recover RUN_DIR
python3 scripts/metal_autoresearch.py validate-production RUN_DIR
python3 scripts/metal_autoresearch.py goal-prompt GOAL_CONTRACT
```

The registry binds every template to one canonical production slot. Initialization
fails before evaluation if the template path, declared slot, registry owner, or
registry digest disagree.

## Requirement-to-code map

| Requirement | Implementation |
|---|---|
| Canonical slot identity | `kernel_registry.json`, `metal_kernel_registry.py` |
| Versioned goal/template validation | `metal_research/contracts.py` |
| Raw paired recomputation | `metal_research/paired.py` |
| Reserve-aware multidimensional budgets | `metal_research/budget.py` |
| Queue, lease, failure, and timeout telemetry | `metal_research/attempt.py` |
| Legacy-result adapters and common envelope | `metal_research/results.py` |
| Transactions, ledgers, snapshots, recovery | `metal_research/runner.py` |

## Known evidence limits

- Five same-fixture pairs measure order and runtime variation, not input-tape
  variation. Claims must say so.
- Candidate selection biases the winning local measurement. Revalidate the
  selected parent in a fresh invocation before promotion.
- A proxy may become invalid when layout, cutoff, or resident-state behavior
  changes. Sentinel misranking disables it for the phase.
- PIOP shares used in projections must be disjoint. Overlapping attribution is
  rejected rather than normalized.
