# Akita Metal end-to-end prover polishing goal

## Decision and objective

Optimize the composed Akita Metal prover as one system now that commit, Jolt PIOP,
and evaluation proof all have Metal implementations. The primary acceptance bar is
at least **5x complete `jolt_prover::prove` speedup** over the optimized CPU backend
for BTreeMap, Fibonacci, and SHA-2 chain at `T = 2^28`. The timed boundary includes
all per-proof hybrid CPU work, transfers, allocation, synchronization, readback, and
proof assembly. Work may not be moved into preprocessing to improve the score.

The search objective is lexicographic:

1. maximize the worst of the three `T = 2^28` speedups until all exceed 5x;
2. preserve the 5x floor while minimizing their total Metal wall time;
3. preserve or improve the accepted Metal parent from `T = 2^25` through `2^28`;
4. reduce fixed costs and fit geometry/activity-based CPU/Metal switchovers below
   `T = 2^25`, with `T = 2^20` as the first small-scale sentinel.

Crossing 5x does not end the campaign. Continue while the analytical queue contains
a bounded change with credible material end-to-end upside. Stop when that queue is
empty, correctness or evaluator integrity is blocked, or a tranche budget is reached.

## Starting performance budget

The matched Perfetto runs in
[akita-metal-perfetto-t28-analysis.md](akita-metal-perfetto-t28-analysis.md) are the
initial analytical anchors. They are single observed runs, not final release claims.

| Workload | Optimized CPU | Metal | Speedup | 5x Metal ceiling | Remaining gap |
|---|---:|---:|---:|---:|---:|
| BTreeMap | 166.55 s | 56.34 s | 2.96x | 33.31 s | 23.03 s |
| Fibonacci | 215.18 s | 45.72 s | 4.71x | 43.04 s | 2.68 s |
| SHA-2 chain | 213.70 s | 42.45 s | 5.03x | 42.74 s | clears by 0.29 s |

These are the frozen starting controls. After fold-index fusion and hot-only
out-of-place RAM compaction, the accepted paired checkpoint measures 48.08 s for
BTreeMap, or 3.464x against the frozen CPU anchor, leaving 14.77 s to the 5x ceiling.
Fibonacci and SHA-2 retain the frozen T28 values until the next milestone sweep.
Ranges below are planning estimates to falsify, not measured promises.

| Priority | Mechanism | Current predicted opportunity |
|---|---|---:|
| 1 | Measure and retime the Stage 4/5 compatibility-scatter prefetch | up to 1.2 s on BTreeMap |
| 2 | Reduce BTreeMap commit traffic/locality cost | at least 1.8 s needed for a 5x commit phase |
| 3 | Remove commit wrapper, row-generation, and synchronization residue | 0.3--0.6 s per workload |
| 4 | Reprice the retained RAM message/cycle/reduction residue | at most 0.51 s to its terminal bar |
| 5 | Generalize the bytecode address carrier from `log_K = 13` to 14 | at most 1.34 s on SHA-2 |
| 6 | Remove remaining lazy-first-bind, product-output, and command gaps | reprice after priorities 1--5 |

The first high-activity RAM tranche is complete at Jolt `1799ff816`: stable address
bucketing, a cycle-tiled frontier, scalar cold-address owners, cooperative hot-address
compaction, and the CPU address tail preserve the existing relation and yield a
verified 4.14 s BTreeMap improvement. Peak RSS is 80.08 GiB. The implementation is
not finished: the hottest address contains 20,729,173 of 65,195,206 accesses, so one
threadgroup still serializes 31.8% of the message work. RAM kernels remain 2.97 s
GPU-active against a 0.57 s analytical floor.

Fixed hot-message chunks and 1,024-thread compaction groups are retained, reducing
RAM GPU-active time from 2.97 s to 1.82 s without material RSS growth. The latter
misses its 1.7 s terminal bar. Hot-only out-of-place count/prefix/scatter is now also
retained: exact T25 and T28 proofs pass, RAM GPU-active time is 1.2246 s, and the
complete BTreeMap prover is 48.08 s at 80.08 GiB. The intended early rounds account
for the gain; their GPU time fell by 0.6105 s while the late rounds stayed flat. The
remaining RAM residue is only about 0.51 s to its terminal bar and must be separated
by subphase before another RAM edit. Full RAM design and measurements live in
[akita-metal-high-activity-ram.md](akita-metal-high-activity-ram.md).

The shared opening boundary is now measured well enough to isolate its two index
lifecycles. Before fusion, Akita reported 50,897,879,040 bytes (47.402 GiB) of
one-shot deferred opening indices at BTreeMap T28: 30.469 GiB for fold records/counts
and 16.934 GiB for coefficient records/offsets. Their construction took about 3.00 s,
of which about 1.10 s was GPU-active, while the opening command interval was about
5.54 s against 2.62 s of GPU-active work.

The accepted fold candidate now partitions records in the decompose/fold consumer,
removes the 30.469 GiB global fold index, and leaves the retained-index route intact
for smaller geometries. It passes exact retained/fused parity, reconstructs every
digit, verifies the full proof, and measures 49.29--49.55 s at BTreeMap T28 with
81.93--82.03 GiB peak RSS. The fixed T28 harness measures 1.813 s GPU-active for the
fused packed fold, 87% of its 1.58 s modeled floor. The remaining opening index is
exactly 18,182,307,840 bytes (16.934 GiB), with about 0.99 s integrated index wall
time; the opening command interval is 4.53--4.57 s against 2.82--2.84 s GPU-active.
These counters overlap and are not additive. A follow-up candidate generated the
coefficient buckets inside the consumer and removed all 18,182,307,840 remaining
index bytes. It passed exact retained/fused parity and full proof verification;
Fibonacci T25 took 5.89 s and BTreeMap T28 took 49.30 s at 80.09 GiB RSS. That is
only 0.12 s faster than the 49.42 s accepted-parent mean and 0.01 s slower than its
best observation. The opening command interval fell only to 4.44 s while GPU-active
time increased from 2.82--2.84 s to 3.13 s. The candidate missed the predeclared
0.5 s materiality and 49.05 s terminal bars, so it was rejected and reverted. Its
30 KiB one-threadgroup-per-window schedule traded nominal allocation removal for
real occupancy and device work; window tuning is not eligible without a new model.

## Main execution plan

1. **Lock the accepted state without rerunning it.** Audit both worktrees, record the
   paired revisions and evaluator/workload digests, and preserve the local Jolt path
   overrides. The frozen CPU anchors remain valid until CPU code, protocol, workload
   generation, flags, machine, or timing boundary changes. A treatment without
   successful proof verification and complete route/fallback telemetry is invalid.
2. **Model hot-address RAM compaction before changing it.** Resolve the exact state
   record, address-segment, round-transition, message-read, and output-ownership
   boundary in Jolt. Price compulsory state traffic, fp128 arithmetic, prefix/scatter
   work, dispatches, and the auxiliary hot-state plane. Commit one predicted complete-
   prover saving and a falsifier before editing a kernel. The design is admissible
   only if worst-case T28 RSS remains below 90 GiB and its modeled ceiling is at least
   0.5 s.
3. **Parallelize only the skewed hot-address tail.** Keep cold addresses on the
   accepted in-place path. For public-geometry-selected hot segments, give multiple
   threadgroups disjoint output ownership through a count/prefix/scatter schedule and
   alternate between the existing span and one hot-only auxiliary plane. Preserve
   per-address cycle order, odd-boundary behavior, round challenges, message values,
   transcript, proof bytes, and verifier behavior. Keep message extraction ordered
   after the completed compaction, matching the current bound-then-message phase.
   Do not combine this with Stage 4/5 retiming or commit work.
4. **Gate the RAM candidate cheaply.** First run focused CPU/Metal kernel parity over
   skewed, multi-chunk, odd-length, and boundary-crossing segments. Then run one
   verified BTreeMap T25 sentinel. Admit one BTreeMap T28 treatment only if telemetry
   confirms the intended multi-group route and state-plane lifecycle. Retain it only
   if it saves at least 0.5 s complete-prover time or 5% of the affected span, keeps
   RSS below 90 GiB, and introduces no silent fallback. Rerun once only for ambiguity,
   surprise, or parent promotion.
5. **Rerank the remaining disjoint ceilings.** After the RAM result, recompute the
   complete-prover critical path and choose exactly one of Stage 4/5 compatibility-
   scatter retiming, BTreeMap commit traffic/locality, commit wrapper and row-
   generation residue, or a remaining Stage 1/6b gap. The rejected coefficient-index
   fusion is closed evidence, not a tuning branch; reopen it only if a different
   ownership model removes the observed occupancy cost on paper.
6. **Close shared and workload-specific residue.** Extend the bytecode address carrier
   to `log_K = 14` using public geometry if its SHA-2 CPU island remains material;
   remove commit wrapper, row-generation, synchronization, and lazy-bind residue in
   measured order. Keep generic fp128 kernels and residency policy in Akita and keep
   Jolt witness geometry and cross-stage orchestration here.
7. **Rerank only at milestones.** Run the three-workload T28 Metal matrix only after
   the model predicts a material change to the worst-workload score. Compare against
   the frozen CPU anchors and the accepted Metal parent, promote one paired parent,
   and derive the next queue from new stage deltas. Do not keep polishing a kernel
   after its total remaining ceiling becomes immaterial.
8. **Fit and validate the deployment envelope.** Once all three T28 workloads clear
   5x with margin, check T25--T28, fit public geometry/activity-based CPU/Metal
   crossovers, and probe T20 plus the scales bracketing each crossover. Finish with
   order-reversed CPU/Metal confirmation pairs, exact proof verification, memory,
   formatting, relevant nextest suites, both clippy modes, and removal of search
   variants and obsolete telemetry.

The plan is deliberately sequential: BTreeMap sets the first objective, while
Fibonacci and SHA-2 act as regression and shared-overhead witnesses. A new protocol
change is not a substitute for an unexplained implementation gap; only a bounded
public schedule or batching change with a written ceiling argument is eligible.

## Fixed evaluator contract

Build once outside the measurement gate:

```bash
cargo build --release -p jolt-prover --example modular_benchmark \
  --features prover-fixtures,metal
```

Run the resulting binary without `--format` for ordinary timings:

```bash
./target/release/examples/modular_benchmark \
  --name fibonacci --scale 28 --backend {optimized|metal}
./target/release/examples/modular_benchmark \
  --name sha2-chain --scale 28 --backend {optimized|metal}
./target/release/examples/modular_benchmark \
  --name btreemap --scale 28 --target-trace-size 150000000 \
  --backend {optimized|metal}
```

The BTreeMap override is part of the workload identity: its default 90%-of-domain
target overflows the `2^28` trace domain. Score the reported `jolt_prover::prove`
wall time, require `PROOF_VERIFIED ... value=true`, and retain route/fallback and peak
RSS telemetry. Shape-only preparation may remain outside the score only when it is
witness- and transcript-independent and reusable across proofs; all per-proof work
must stay inside the timed boundary. Use `--format chrome` only for a decision-blocking
trace, never as a treatment timing.

## Architecture and boundaries

Keep generic fp128 commitment and opening kernels, residency, and scheduling in the
Akita fork. Keep Jolt witness geometry, cross-stage resource scheduling, PIOP kernels,
and the adapter in this repository. Route on public geometry and measured activity,
never workload names. Maintain one accepted parent in each repository and record the
paired revisions for every retained candidate.

The CPU and Metal paths use the same verifier statement and soundness target. Hybrid
CPU work is allowed when it is an intentional timed algorithmic choice; silent
fallback is not. Avoid protocol changes by default. A minor public-shape-derived
schedule or batching change is allowed only after a written traffic/latency model
shows that the unchanged protocol has an inadequate ceiling. Isolate it from kernel
changes and update [akita-metal-protocol-changes.md](akita-metal-protocol-changes.md),
the prover, and the verifier together. Do not make invasive protocol changes in this
campaign.

For small scales, fit each major family to an affine cost model such as
`CPU(work) = c*work` and `Metal(work) = launch + transfer + g*work`. Select a route
from geometry and activity with a safety margin. Threshold tuning can prevent a
small-scale regression; improving `T = 2^20` materially requires reducing the Metal
fixed term through fusion, reuse, or fewer command boundaries.

## Lean research loop

Use sequential hill climbing from the accepted paired revisions. Keep only three
small run artifacts under an ignored `benchmark-runs/akita-metal-e2e-polish/`
directory: the current analytical model, an append-only candidate ledger, and raw
machine-readable observations. Before the first edit, audit both worktrees, preserve
all existing changes, and make coherent local checkpoints for the accepted Jolt and
Akita parents. Add only telemetry required by the initial model, then freeze the
benchmark command, timing boundary, parser, and workload generation. Record both
revisions, evaluator and workload digests, reference artifacts and their invalidation
rule, machine identity, a 12-measured-candidate tranche, and the stop conditions.

For each candidate:

1. name one mechanism, the affected boundary, predicted complete-prover saving,
   implementation cost, and falsifying observation;
2. reject it analytically if its ceiling cannot move an end-to-end decision;
3. make one scoped edit and run the smallest exact parity/correctness test;
4. run normally one warm candidate-only sentinel: BTreeMap `T = 2^25` first for a
   RAM kernel and `T = 2^28` only after affected-span telemetry is credible;
   BTreeMap `T = 2^28` for cross-stage scheduling, SHA-2 `T = 2^28` for
   `log_K = 14`, Fibonacci `T = 2^25` first for opening work, or `T = 2^20` for
   fixed-cost routing;
5. compare with the frozen reference or accepted Metal parent, update only the
   affected model terms, and keep, discard, or mark the result invalid;
6. rerun once only when the result is near the decision threshold, surprising, or
   being promoted to a new accepted parent.

A routine gate has a hard 120-second execution budget excluding incremental
compilation. Do not run repeated CPU controls, a three-workload matrix, Criterion,
or Perfetto for each candidate. Capture a new trace only when an unexpected stage
delta or missing counter prevents the next decision. A provisional improvement must
clear measured noise and complexity cost; use 0.5 s complete-prover saving or 5% of
the affected span as the default materiality bar. A smaller change may remain only
when it is a simple removal of waste and introduces no maintenance surface.

Fail closed on incorrect output, verifier failure, missing metrics, evaluator drift,
unexplained fallback, timeout, non-finite timing, or unrelated worktree changes.
Incremental compilation is allowed to take the time it needs; it is not a reason to
inflate the measurement gate. At a tranche boundary, checkpoint the accepted parents,
refresh the ranked queue from the accumulated model, and report the next tranche. Do
not substitute a broad validation sweep for that checkpoint.

## Milestones and final validation

Run a three-workload `T = 2^28` Metal milestone only when the analytical model
predicts a meaningful change to the worst-workload score. Freeze untraced optimized
CPU anchors once the evaluator and build are stable; rerun them only if CPU code,
protocol, schedule, workload generation, compiler flags, machine, or timing boundary
changes.

Final acceptance requires:

- two order-reversed CPU/Metal pairs for each `T = 2^28` workload, with the worst
  valid pair above 5x and enough margin to survive measured run noise;
- one verified Metal sweep for all three workloads at `T = 2^25, 2^26, 2^27, 2^28`,
  with no material regression from the accepted parent;
- `T = 2^20` and the two scales surrounding each fitted route crossover, showing
  that the hybrid selector chooses the faster path within its safety margin;
- exact focused CPU/Metal parity, full proof verification, formatting, relevant
  tests, and both required clippy modes;
- peak RSS at most 90 GiB at `T = 2^28`, no process swapping, and no undocumented
  fallback on a qualified route;
- removal of search variants, experimental knobs, raw logs, obsolete instrumentation,
  and code paths not selected by the final design.

After the 5x floor is accepted, continue with the same loop until no simple candidate
has at least roughly 0.5 s or 1% credible `T = 2^28` upside, or until further progress
would require an invasive protocol change. Report rejected candidates and remaining
analytical floors rather than hiding negative results.

## Copy/paste launch prompt

```text
Create and pursue the goal in specs/akita-metal-e2e-polish-goal.md. Work from the
current feat/akita-metal Jolt branch at
/Users/mgeorghiades/worktrees/jolt/bright-ridge/jolt and Akita
perf/metal-commit-eval-proof at
/Users/mgeorghiades/worktrees/akita-metal-eval-proof. The accepted runtime sources are
Jolt 9fb538461 and Akita a454c7575; later commits may contain analysis, exact negative
experiments, reverts, and result documentation. First audit both worktrees and record
their actual heads and trees. Preserve the intentionally local Jolt Cargo.lock and
.cargo/config.toml path overrides. Do not push. Rebuild before the first treatment;
an existing release binary may still be linked against the reverted Akita candidate.

Optimize the composed Akita Metal prover across those two worktrees. The hard bar is
at least 5x complete jolt_prover::prove speedup over the optimized CPU backend for
BTreeMap, Fibonacci, and SHA-2 chain at T=2^28, maximizing the worst workload first.
Once all three clear 5x with credible margin, continue removing material analytical
bottlenecks while preserving that floor, T25--T28 performance, exact proof
verification, and the 90 GiB RSS guard. Reduce fixed costs and fit public
geometry/activity-based hybrid switchovers around smaller workloads, including T20.

Freeze and use the evaluator contract in the document. BTreeMap T28 must include
--target-trace-size 150000000. Routine treatment runs must omit --format, and every
accepted result must print successful proof verification. Do not move witness- or
transcript-dependent work outside the timed proving boundary. Reuse the frozen CPU
anchors unless an explicit invalidation condition changes.

The retained RAM work, fixed hot-message chunks, 1,024-thread compaction, deferred
fold-index fusion, and hot-only out-of-place count/prefix/scatter are already
implemented; do not redo them. The latter passes exact lockstep and full proof
verification, reduces BTreeMap T28 RAM GPU-active time from 1.823 s to 1.2246 s, and
brings the complete prover from the accepted 49.42 s mean to 48.08 s at 80.08 GiB
RSS. Treat 48.08 s as the current Metal parent; it is 3.464x against the frozen
166.548 s CPU anchor and is 14.77 s above the 5x ceiling. Its 0.5146 s remaining
ceiling to the RAM terminal bar requires subphase telemetry before more RAM work.

Do not repeat the rejected deferred-coefficient experiment. It removed the remaining
18,182,307,840 index bytes, passed exact parity and proof verification, measured
Fibonacci T25 at 5.89 s and BTreeMap T28 at 49.30 s / 80.09 GiB, but saved only
0.12 s against the accepted-parent mean. Opening GPU-active time rose from roughly
2.83 s to 3.13 s, so the one-group-per-window schedule exchanged virtual/private
allocation pressure for real occupancy and device work. It missed the predeclared
0.5 s and 49.05 s bars and was reverted. Preserve this as negative evidence; do not
tune its window without a materially different analytical ownership model.

Resume by reranking the disjoint post-RAM ceilings. First localize the Stage 4/5
compatibility-scatter construction and prefetch boundary using existing traces and
source ownership; add narrow telemetry rather than taking a new Perfetto trace if the
missing construction interval cannot be recovered. Compare overlap with scheduling
after Stage 4 and retain a timing change only on the combined Stage 4 + Stage 5
critical path. Then reprice BTreeMap commit traffic/locality, commit wrapper and row-
generation residue, the retained RAM subphases, the `log_K=14` SHA-2 carrier, and
remaining Stage 1/6b gaps. Select one mechanism whose disjoint end-to-end ceiling is
at least 0.5 s, and commit its boundary, floor, prediction, and falsifier before code.

Use an analysis-led sequential loop: one mechanism, one prediction and falsifier,
one scoped edit, one focused correctness check, and normally one warm affected-
workload treatment. A routine measurement gate is at most 120 seconds excluding
compilation. Do not run repeated baselines, full workload matrices, Criterion, or
Perfetto during ordinary iterations. Repeat only ambiguous, surprising, or parent-
promoting results. Run a full T28 workload milestone only when the model predicts a
material change to the worst-workload score.

Keep generic fp128 kernels, opening residency, and scheduling in Akita; keep Jolt
witness geometry, PIOP kernels, adapter logic, and cross-stage orchestration in Jolt.
Route by public geometry/activity rather than workload name, charge all hybrid CPU
work and shifted preparation to proving, and fail closed on qualified paths. Avoid
protocol changes by default. Only a bounded, documented minor change with a written
analytical need is in scope, and it must update prover and verifier together without
reducing soundness.

Keep accepted changes bisectable. Maintain only a terse ignored analytical model,
append-only event ledger, and raw observations; preserve negative results and keep
search machinery and obsolete telemetry out of production. Do not stop merely on
the first 5x observations: complete the order-reversed confirmation pairs, T25--T28
and crossover checks, proof/parity/memory gates, formatting, relevant nextest suites,
both clippy modes, and cleanup. Continue afterward only while a bounded candidate
has at least roughly 0.5 s or 1% credible T28 upside without an invasive protocol
change. Use the engineer, gpu-kernel-analysis, autoresearch-loop, experiment-design,
result-validation, and coding-standards skills, with this document's lean gate taking
precedence over heavier historical loop defaults.
```
