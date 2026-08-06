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
the log-26 cap of 8,192 threadgroups; log 24 changes the launch geometry. Three
fixed sentinels calibrate its ordering against the unchanged five-pair log-26
tier. Kendall tau-b must be at least 0.8 with no material inversion. Every second
proxy rejection is audited at log 26; a material false negative disables proxy
ranking for the phase. The proxy cannot accept a candidate or satisfy the 5x
floor.

A direct representative lane is available for a high-confidence mechanism or a
disabled proxy. The caller supplies both the flag and a reason. If the one-shot
phase checkpoint is pending, the direct lane first runs log 25 only as the
checkpoint probe and ignores its ranking; after the checkpoint passes it goes
straight to log 26. A scale-dependent layout, cutoff, or occupancy change
invalidates the proxy. A failed transfer resumes from the accepted holdout; it
does not rerun representative or holdout evidence. Continuing the portfolio
opens a linked successor run with a freshly sealed holdout; a terminal run is
never reopened for tuning.

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

Up to three proposal agents may work in parallel, but their roles are distinct:
one derives the ceiling and bottleneck model, one sketches the candidate, and
one tries to falsify the model and design. They do not repeat the same broad
analysis. The root agent alone modifies the shared worktree, reconciles the
proposals, runs builds and tests, and owns the serialized CPU/GPU evaluator
lease. Proposal agents do not run Cargo, CPU controls, or GPU evaluators.

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

A phase covers one mechanism family and normally admits at most two candidates:
the primary design and one pre-registered risk-reduction variant. Its first
candidate has a one-shot, phase-local checkpoint at the cheapest exact scale.
A miss seals the negative result immediately. Expensive representative runs are
reserved for candidates that pass that checkpoint. A layout or dependency-closure
change opens a separate phase rather than widening the frozen scope.

Before freezing a fresh mechanism phase, profile one cold and one warm exact
controller cycle. Separate source assembly, compilation, fixture/device setup,
queueing, evaluation, parsing, and checkpointing; freeze a valid-candidates-per-hour
target. Compile a reduced evaluator only from its sealed transitive dependency
closure and only after its exact result contract has been validated. Version-3
preflight evidence binds the controller source closure, the exact reconstructed
shader fragments and assembled sources, and the sealed runner binary and source.
A revision label alone is not sufficient provenance.

The current OuterRemainder profile records 2.255 seconds cold and 1.939 seconds
warm for an exact log-25 controller cycle. Controller overhead is below 0.08%,
and the cold cycle implies 1,596 cycles/hour; the contract freezes a conservative
1,200-cycle/hour floor. This is controller capacity for the proxy evaluator, not
candidate-development throughput or the cadence of log-26, holdout, and transfer
runs. Each phase's wall and candidate budgets are its timebox; its contract also
names an analytical ceiling, progress checkpoint, and kill or redesign action.
Missing that checkpoint seals `phase_exhausted` rather than spending the remaining
phase budget by default.

If an inherited proxy is already known to misrank the representative evaluator,
do not spend a new phase recalibrating the unchanged proxy. Record the inherited
evidence and use the direct lane, retaining the cheap scale only for the one-shot
checkpoint. Rejected implementation code is restored out of the live shader;
the phase summary, exact hashes, and measured failure remain in the evidence
registry.

## Durable records

A schema-2 run contains an immutable, self-digested `run.json` contract and an
atomically replaced, self-digested `state.json`. Baseline, candidate, tier,
decision, proxy, and kernel-validation ledgers are append-only. Per-evaluation raw
output and attempt telemetry bind their digests, and accepted editable paths are
snapshotted. Initialization publishes the complete run directory by atomic rename;
a failed initialization removes only its private staging directory.

`inflight.json` is written before candidate admission and before every evaluator
launch. It makes admission, evaluation, and promotion recoverable. If an attempt
did not seal telemetry, recovery first drains its tracked process group, then
conservatively charges the observed in-flight interval before restoring the
accepted parent. A passed or failed phase checkpoint is stored once in
`state.json`, bound to its candidate, evaluation, and result digest, and verified
rather than rerun after interruption.

Initialization moves from `initializing` to `active`; interrupted or failed
baseline evaluation becomes `initialization_retryable`, and `resume-init` reuses
accepted tiers while assigning a fresh ID to each missing retry. Proxy state moves
from `pending_calibration` to `enabled` or `disabled`, with one terminal event in
`proxy-events.jsonl`. A missed mechanism checkpoint moves the run to
`phase_exhausted`.

Kernel validation moves from `active` through `holdout_retryable` after a sealed
revalidation is recovered or when an invalid holdout may repeat, then to
`kernel_accepted` and `kernel_transferred`. A valid failed holdout ends at
`holdout_rejected`. Trial admission is legal only in `active`. Successful stages
are reused after interruption; attempt identifiers are never reused.

A run-bound goal decision uses the worst overall or order-stratum speedup across
its accepted log-26 holdout and log-27 transfer, binds the accepted-parent
snapshot and both result digests, and is appended idempotently. A continuation
event names a successor run; it does not reactivate the terminal kernel run.

## Commands

The canonical controller dispatches by contract version:

```text
python3 scripts/metal_autoresearch.py profile-iteration TEMPLATE OUTPUT_PREFIX
python3 scripts/metal_autoresearch.py init TEMPLATE RUN_DIR
python3 scripts/metal_autoresearch.py validate-template TEMPLATE
python3 scripts/metal_autoresearch.py resume-init RUN_DIR
python3 scripts/metal_autoresearch.py calibrate-proxy RUN_DIR
python3 scripts/metal_autoresearch.py candidate-context RUN_DIR
python3 scripts/metal_autoresearch.py trial RUN_DIR --summary TEXT --param NAME=VALUE
python3 scripts/metal_autoresearch.py trial RUN_DIR --summary TEXT --direct-to-representative --direct-reason REASON
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
