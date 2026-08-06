# Metal kernel research harness

Schema 2 is the only contract allowed to initialize a new Metal kernel search.
Schema-1 templates, runs, and evidence remain readable by
`scripts/metal_autoresearch.py` for existing-run inspection, recovery, trials, and
production validation; they are never fresh parents or reinterpreted as schema 2.

## Acceptance policy

The primary portfolio metric is optimized-CPU PIOP wall time divided by
Metal-hybrid PIOP wall time at log 26. Portfolio completion requires 5x over
five alternating pairs at log 26 and log 27. The overall median and both
order-stratum medians must clear 5x at each scale. Every completed standalone
or fused kernel also has a 5x local floor; Instruction RA retains its 7x floor.

These are acceptance floors, not search caps. A candidate may become the next
search parent while still below 5x if it improves beyond the frozen noise
threshold. A kernel may transfer while the portfolio is below 5x: its PIOP
holdout protects against integration regressions, while its local metric decides
kernel acceptance. The portfolio floor is claimed only when sealed log-26 and
log-27 PIOP evidence also clears 5x. Once the portfolio reaches 5x, the loop
continues when disjoint profile attribution predicts another 5 percent PIOP gain
or a measured kernel has clear additional headroom. A 4x analytical estimate is
a reason to investigate, never an acceptance threshold.

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
3. Fresh representative revalidation must reproduce the local floor before the
   holdout is opened.
4. The log-26 PIOP holdout runs after local acceptance and is never used to tune
   the candidate. A valid below-floor result seals the run as rejected. An
   invalid evaluator attempt may retry the same candidate, but tuning stays
   disabled.
5. The log-27 transfer tier checks that the accepted kernel survives the next
   trace scale before the run is marked kernel-transferred.

The OuterRemainder successor uses an exact log-25 proxy with one excluded warmup
and four alternating pairs. Log 25 is the smallest cheaper scale that retains
the log-26 cap of 8,192 threadgroups; log 24 changes the launch geometry. Its 1%
screen is deliberately more permissive than the representative noise gate, and
every passing candidate immediately runs the unchanged five-pair log-26 tier.
It cannot accept a candidate or satisfy the 5x floor. A scale-dependent layout,
cutoff, or occupancy change invalidates the proxy and requires a successor
contract. A failed transfer resumes from the accepted holdout; it does not rerun
representative or holdout evidence. Continuing the portfolio opens a linked
successor run with a freshly sealed holdout; a terminal run is never reopened
for tuning.

## Time and resource accounting

The controller records queue wait, subprocess wall time, and exclusive-machine
lease time separately. Builds, warmups, CPU controls, crashes, timeouts, and
invalid results are charged. Representative and holdout reserves cannot be
spent by proxy screening.

Each production reserve protects its first invocation and one retry from
screening work. Its resource pool is at least the tier cost limit multiplied by
the protected invocation count. Failures and interrupted attempts consume both
their measured resources and one protected invocation.

GPU-active time is recorded only when validated device timestamps exist. When
it is unavailable, the budget uses a conservative treatment-wall or subprocess-
wall upper bound and records that it is estimated. Historical `gpu_seconds`
fields are not relabeled as device-active time.

The controller admits an evaluator only when the remaining calendar budget can
contain its full timeout. Waiting for the machine lease is bounded by the
remaining slack. A tracked wrapper publishes a launch token, PID, and process
group before evaluator work and inherits the global lease descriptor. Timeout,
cancellation, and recovery terminate and verify that process group before the
lease is released or reacquired.

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
- a pre-registered latency target no greater than the calibrated floor divided
  by 80 percent, plus the applicable 5x or 7x speedup floor;
- one falsifiable change, its predicted gain, and its likely failure mode.

This lets the root agent reject low-ceiling ideas before a target-scale build.
Before initialization, the editable scope must also contain every production and
test file needed by the first pre-registered dataflow candidate and the next
ranked phase candidates. Discovering a missing implementation file after the
baseline supersedes the run; frozen scope is never widened in place.

## Durable records

A schema-2 run contains an atomically replaced, self-digested `run.json`,
append-only baseline, candidate, tier, decision, and kernel-validation ledgers,
per-evaluation raw output and attempt telemetry, and snapshots of accepted
editable paths. `inflight.json` makes an interrupted evaluation recoverable. If
an attempt did not seal telemetry, recovery first drains its tracked process
group, then conservatively charges the observed in-flight interval before
restoring the accepted parent.

Initialization moves from `initializing` to `active`; interrupted or failed
baseline evaluation becomes `initialization_retryable`, and `resume-init`
reuses accepted tiers while assigning a fresh ID to each missing retry. Kernel
validation then moves from `active` through `holdout_retryable` after a sealed
revalidation is recovered or when an invalid holdout may repeat,
`kernel_accepted`, and `kernel_transferred`. A valid failed holdout ends at
`holdout_rejected`. Trial admission is legal only in `active`. Successful stages
are reused after interruption; attempt identifiers are never reused.

A run-bound goal decision uses the worst overall or order-stratum speedup across
its accepted log-26 holdout and log-27 transfer, binds the accepted-parent
snapshot and both result digests, and is appended idempotently. A continuation
event names a successor run; it does not reactivate the terminal kernel run.

## Commands

The canonical controller dispatches by contract version:

```text
python3 scripts/metal_autoresearch.py init TEMPLATE RUN_DIR
python3 scripts/metal_autoresearch.py validate-template TEMPLATE
python3 scripts/metal_autoresearch.py resume-init RUN_DIR
python3 scripts/metal_autoresearch.py candidate-context RUN_DIR
python3 scripts/metal_autoresearch.py trial RUN_DIR --summary TEXT --param NAME=VALUE
python3 scripts/metal_autoresearch.py status RUN_DIR
python3 scripts/metal_autoresearch.py recover RUN_DIR
python3 scripts/metal_autoresearch.py validate-production RUN_DIR
python3 scripts/metal_autoresearch.py goal-decision GOAL_CONTRACT --run-dir RUN_DIR --shares-disjoint --candidate KERNEL:SHARE:SPEEDUP
python3 scripts/metal_autoresearch.py goal-prompt GOAL_CONTRACT
```

The registry binds every template to one canonical production slot and records its
contract schema and lifecycle. A slot may expose at most one `fresh_init` template;
schema-1 entries are `existing_runs_only`. Initialization fails before creating a
run directory or launching an evaluator if the template path, declared slot,
lifecycle, registry owner, or registry digest disagree. `validate-template` still
reports legacy templates as valid but not fresh-init eligible.

## Requirement-to-code map

| Requirement | Implementation |
|---|---|
| Canonical slot identity | `kernel_registry.json`, `metal_kernel_registry.py` |
| Versioned goal/template validation | `metal_research/contracts.py` |
| Raw paired recomputation | `metal_research/paired.py` |
| Reserve-aware multidimensional budgets | `metal_research/budget.py` |
| Queue, lease, failure, and timeout telemetry | `metal_research/attempt.py` |
| Legacy-result adapters and common envelope | `metal_research/results.py` |
| Transactions, ledgers, snapshots, recovery, goal decisions | `metal_research/runner.py` |

## Known evidence limits

- Five same-fixture pairs measure order and runtime variation, not input-tape
  variation. Claims must say so.
- Candidate selection biases the winning local measurement. Revalidate the
  selected parent in a fresh invocation before promotion.
- A holdout is permanently exposed after a valid result. Further optimization
  requires a successor run with a new sealed holdout.
- A proxy may become invalid when layout, cutoff, or resident-state behavior
  changes. Sentinel misranking disables it for the phase.
- PIOP shares used in projections must be disjoint. Overlapping attribution is
  rejected rather than normalized.
