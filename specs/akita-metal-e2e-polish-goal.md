# Akita Metal end-to-end prover polish

## Goal

Make the composed Akita Metal prover at least **5x faster than the optimized CPU
backend for complete `jolt_prover::prove`** on BTreeMap, Fibonacci, and SHA-2 chain
at `T = 2^28`. Optimize the worst ratio first. After all three clear 5x with
credible margin, continue while a simple, bounded candidate still has at least
roughly 0.5 seconds or 1% of credible T28 upside.

The score includes every per-proof cost: hybrid CPU work, row generation, transfers,
allocation, synchronization, readback, and proof assembly. Preprocessing may exclude
only public-shape-derived, witness-independent work that can actually be reused across
proofs. Every scored proof must verify. Peak RSS must remain at or below 90 GiB.

The secondary objective is a well-calibrated deployment envelope: preserve or improve
the accepted parent at T25--T28, then fit public geometry/activity-based CPU/Metal
switchovers and check T20 plus the scales bracketing each crossover.

## Accepted state and numerical target

The paired worktrees are:

- Jolt: `feat/akita-metal` at
  `/Users/mgeorghiades/worktrees/jolt/bright-ridge/jolt`;
- Akita: `perf/metal-commit-eval-proof` at
  `/Users/mgeorghiades/worktrees/akita-metal-eval-proof`.

The accepted runtime sources are Jolt `9fb538461` and Akita `a454c7575`. Later heads
contain documentation, rejected experiments, and exact reverts: Jolt `5f0d3e4d3` and
Akita `a027f03ca` after the first polish tranche. Audit the actual heads and trees
before resuming. Jolt's modified `Cargo.lock` and untracked `.cargo/config.toml` are
intentional local Akita path overrides. Do not commit or remove them. Do not push.

The release binary was rebuilt from those accepted runtime trees after the C1 revert.
Rebuild only after another runtime-source change.

The CPU anchors and last accepted Metal observations are:

| Workload | Optimized CPU | Accepted Metal anchor | Current ratio | 5x ceiling | Gap |
|---|---:|---:|---:|---:|---:|
| BTreeMap | 166.548 s | 48.08 s | 3.464x | 33.310 s | 14.770 s |
| Fibonacci | 215.18 s | 45.72 s | 4.71x | 43.036 s | 2.684 s |
| SHA-2 chain | 213.70 s | 42.45 s | 5.03x | 42.740 s | clears by 0.29 s |

The Fibonacci and SHA-2 values are the last valid T28 anchors, not remeasurements of
every BTreeMap-only change. Refresh the three-workload matrix only at a milestone.
The accepted BTreeMap proof peaks at 80.08 GiB.

Commit optimization alone cannot close BTreeMap's 14.77-second gap: even deleting its
entire current 12.61-second Akita root commit would leave about 35.47 seconds. The
campaign therefore has two necessary fronts: substantially reduce commit, then remove
at least one independently measured PIOP/cross-stage critical-path cost.

## What is already done

Do not repeat these campaigns:

- generic Metal PIOP, fp128 Akita commit, and eval-proof integration;
- cycle-major zero-copy packed commit input and hybrid CPU/Metal scheduling;
- high-activity BTreeMap RAM routing, fixed hot-message chunks, 1,024-thread
  compaction, and hot-only count/prefix/scatter;
- deferred fold-index fusion in eval proof;
- deferred coefficient-index fusion, which passed exact parity but saved only 0.12 s
  and increased GPU-active opening work, then was reverted;
- Stage 4/5 private grouped-output storage, which verified but regressed complete
  proving to 50.25 s, then was reverted;
- Stage 4/5 worker retiming under the existing ownership model. Moving it after Stage
  4 exposes 1.735 s to avoid at most 0.709 s of contention; moving it earlier transfers
  the same memory-heavy work into register preparation. Reopen only with a new
  work-elimination or ownership mechanism;
- two D512 root tasks per SIMDgroup. It halved modeled matrix reads at T25 from
  207.23 GB to 104.15 GB, but root GPU time regressed from 1.473 s to 1.729 s and
  complete proving from 6.45 s to 6.71 s. Register pressure or reduced issue rate
  dominates the removed traffic. Do not retry larger task-reuse factors.

The detailed evidence is in
[akita-metal-high-activity-ram.md](akita-metal-high-activity-ram.md),
[akita-metal-stage4-stage5-prefetch.md](akita-metal-stage4-stage5-prefetch.md),
[akita-metal-perfetto-t28-analysis.md](akita-metal-perfetto-t28-analysis.md), and
[akita-metal-protocol-changes.md](akita-metal-protocol-changes.md).

## Main plan

### 1. Freeze the evaluator and accepted parents

Audit both worktrees, record revision and tree IDs, and confirm that the runtime diff
from each accepted source to its current head is empty. Preserve the local Jolt path
overrides. Keep the CPU controls frozen unless CPU code, protocol, workload generation,
flags, compiler, machine, or timing boundary changes. Record any such invalidation
before another comparison.

Use one append-only ignored event ledger and one small current model under
`benchmark-runs/akita-metal-e2e-polish/`. Do not build a large search harness.

### 2. Current tranche: route pressured product-output extraction to CPU

A fresh post-A2 BTreeMap T28 trace verified in 48.98 s (diagnostic only; retain the
48.08 s untraced score). Its disjoint top-level spans are 14.219 s commit, 28.842 s
PIOP, and 5.897 s eval proof. Stage 2 is 7.656 s on Metal versus 8.006 s on optimized
CPU because several opposing deltas cancel. The clearest bounded anomaly is the final
eight-opening extraction for `ProductRemainder`:

- current Metal wall: 1.897 s;
- current Metal GPU-active: 0.081 s;
- frozen optimized-CPU wall: 0.474 s;
- BTreeMap-only exposed gap: about 1.423 s;
- Fibonacci Metal/CPU: 0.079 s / 0.553 s, so a universal CPU route would regress it;
- SHA-2 Metal/CPU: 0.893 s / 0.639 s, only about 0.25 s of upside.

Keep every Metal product uni-skip and remainder round unchanged. After all round
challenges are fixed, use the existing optimized CPU opening walk over an owning
handle to the original slice-backed witness rows only when public trace geometry and
RAM activity select it. The initial production rule is T28 with at least `2^25`
certified RAM accesses; it selects BTreeMap's 65,195,206 accesses but excludes SHA-2's
4,196,259 and Fibonacci's sparse route. Missing owning rows keep the Metal path.

The CPU walk constructs one T-element fp128 equality table, exactly 4 GiB at T28,
then extracts the same eight typed columns in one parallel pass. The frozen CPU trace
is the best measured latency control. Under the live Metal allocation pressure,
predict 0.47--0.85 s for the opening walk, 1.05--1.43 s affected-span savings, and
roughly 46.7--47.4 s complete proving before run noise. The accepted 80.08 GiB peak
projects to about 84.1 GiB, below the 90 GiB guard.

Before implementation, factor the CPU formula into one shared helper rather than
copying it. Add a red exact-parity test that forces the activity route, verifies the
same output claims as optimized CPU, and asserts that the route ran exactly once.
Then run that focused test and one verified BTreeMap T28 treatment. A T25 timing
sentinel is unnecessary because the public T28 cutoff makes its route unreachable;
the focused test must explicitly cover the off route instead. Reject if the CPU
opening walk exceeds 0.85 s, complete proving exceeds 47.58 s, RSS exceeds 90 GiB,
proof verification fails, telemetry is missing, or an ineligible geometry routes.
This changes neither protocol nor soundness.

### 3. Recompute the complete critical path

After every material result, update only the affected model terms. The remaining
BTreeMap gap determines the next tranche. Separate:

1. Akita root GPU time and overlapped CPU tail;
2. Jolt row production, commit wrapper, waits, readback, and result assembly;
3. PIOP stage wall time and GPU-active time;
4. eval-proof wall time, command interval, and GPU-active time.

Use existing counters first. Capture one new BTreeMap Perfetto trace only if the
critical-path owner cannot be identified from those counters. Old traces are
localization evidence, not an additive budget after the retained RAM/eval changes.

### 4. Attack one disjoint post-commit ceiling at a time

Rank candidates by credible complete-prover seconds divided by implementation and
correctness risk. The likely queue is:

1. the high-activity ProductRemainder output CPU hybrid above;
2. commit arithmetic reduction with one task per SIMDgroup, or wrapper,
   synchronization, and CPU-tail residue, but only if new telemetry shows it on the
   critical path;
3. the largest current BTreeMap PIOP subphase, expected among Stage 2, Stage 4, or
   Stage 6b after repricing—not chosen from the stale pre-RAM trace totals;
4. shared fixed command/materialization gaps that also create margin for Fibonacci
   and SHA-2;
5. the bounded SHA-2 `log_K = 14` bytecode carrier if its roughly 1.34-second CPU
   island remains present.

For each, commit one mechanism, boundary, lower bound, predicted end-to-end saving,
and falsifier before code. Reject candidates on paper when their entire disjoint
ceiling cannot move the objective. Do not combine unrelated kernel, scheduling, and
protocol changes in one treatment.

### 5. Milestones, small scales, and production cleanup

Run the full T28 Metal workload matrix only after the model predicts a material change
to the worst-workload score. BTreeMap remains the primary objective; Fibonacci and
SHA-2 are shared-path regression witnesses. A shared commit gain should create margin
for the latter two, but measure rather than infer the final claim.

Once all three clear 5x, run order-reversed CPU/Metal confirmation pairs, then a
verified T25--T28 Metal sweep. Fit CPU/Metal crossover rules from public geometry and
activity, and probe T20 plus the scales on either side of each threshold. Finally
remove rejected variants, experimental knobs, obsolete telemetry, and raw artifacts;
run formatting, focused exact-parity tests, relevant nextest suites, and both required
clippy modes.

Crossing 5x is a milestone, not an automatic stop. Continue with the same analytical
loop while a bounded non-invasive candidate has material upside.

## Fixed evaluator

Compilation is outside the measurement budget and may take as long as needed:

```bash
cargo build --release -p jolt-prover --example modular_benchmark \
  --features prover-fixtures,metal
```

Ordinary treatments use the untraced release binary:

```bash
./target/release/examples/modular_benchmark \
  --name fibonacci --scale 28 --backend {optimized|metal}
./target/release/examples/modular_benchmark \
  --name sha2-chain --scale 28 --backend {optimized|metal}
./target/release/examples/modular_benchmark \
  --name btreemap --scale 28 --target-trace-size 150000000 \
  --backend {optimized|metal}
```

The BTreeMap override is part of the workload identity. Score the reported
`jolt_prover::prove` wall time and require `PROOF_VERIFIED ... value=true`, route and
fallback counters, affected-span telemetry, and peak RSS. Use `--format chrome` only
when missing ownership evidence blocks the next decision.

## Lean candidate loop

Use sequential hill climbing from one accepted paired parent:

1. choose one mechanism from the current critical path;
2. state its exact boundary, compulsory traffic/work floor, predicted complete-prover
   saving, and a numerical falsifier;
3. add the smallest red exactness/parity test, then one scoped implementation;
4. run the focused correctness gate and normally one warm affected-workload sentinel;
5. promote to T28 only when affected-span telemetry supports the mechanism;
6. keep, revert, or mark invalid; rerun once only when threshold ambiguity, surprise,
   or promotion requires it;
7. update the accepted parent, negative-evidence ledger, and ranked model.

A routine execution gate is at most 120 seconds, excluding compilation. Do not run
repeated CPU controls, a three-workload matrix, Criterion, or Perfetto for ordinary
candidates. Fail closed on wrong output, verifier failure, missing metrics, evaluator
drift, unexplained fallback, timeout, non-finite timing, or unrelated worktree edits.

Keep generic fp128 kernels, residency, and scheduling in Akita. Keep Jolt witness
geometry, PIOP kernels, adapter behavior, and cross-stage orchestration in Jolt. Route
by public geometry/activity, never workload names. Hybrid CPU work is allowed and
fully timed; silent fallback is not. Avoid protocol changes. A minor public schedule
or batching change is eligible only with a written ceiling argument, isolated prover
and verifier updates, unchanged soundness, and an entry in
[akita-metal-protocol-changes.md](akita-metal-protocol-changes.md). Invasive protocol
changes are out of scope.

## Completion gate

The 5x claim requires:

- two order-reversed CPU/Metal pairs for each T28 workload, with the worse valid ratio
  above 5x and enough margin for measured noise;
- successful proof verification and no undocumented fallback on every scored route;
- peak RSS at most 90 GiB with no swapping;
- one verified Metal sweep for all three workloads at T25--T28 without a material
  regression from the accepted parent;
- T20 and the scales bracketing every fitted CPU/Metal crossover;
- focused CPU/Metal parity, formatting, relevant nextest suites, and both required
  clippy modes;
- a clean production diff containing no rejected variants, search machinery, raw
  logs, or obsolete instrumentation.

## Copy/paste goal prompt

```text
Create and pursue the goal in specs/akita-metal-e2e-polish-goal.md. Work from the
current feat/akita-metal Jolt worktree at
/Users/mgeorghiades/worktrees/jolt/bright-ridge/jolt and the
perf/metal-commit-eval-proof Akita worktree at
/Users/mgeorghiades/worktrees/akita-metal-eval-proof. Do not push. Preserve Jolt's
intentional local Cargo.lock and .cargo/config.toml path overrides. Audit both heads,
trees, and runtime diffs first. The accepted runtime sources are Jolt 9fb538461 and
Akita a454c7575; later commits contain documentation, rejected candidates, and exact
reverts. The current release binary matches those accepted runtime sources; rebuild
after any runtime-source edit.

The hard objective is at least 5x complete jolt_prover::prove speedup over optimized
CPU for BTreeMap, Fibonacci, and SHA-2 chain at T=2^28, maximizing the worst ratio
first. Preserve exact proof verification, RSS <= 90 GiB, accepted T25--T28
performance, and useful public geometry/activity-based CPU/Metal crossovers at lower
scales. Once all three clear 5x with credible margin, continue while a simple bounded
candidate has at least roughly 0.5 seconds or 1% of credible T28 upside.

Do not retry C1: two D512 tasks per SIMDgroup halved modeled matrix reads at T25 but
regressed root GPU time from 1.473s to 1.729s and was exactly reverted. The current
first tranche is the high-activity ProductRemainder output hybrid in Jolt. Preserve
all Metal product rounds; after their challenges are fixed, route only the final
eight-opening extraction through the existing optimized CPU formula when the public
geometry is at least T28, the certified RAM access count is at least 2^25, and an
owning slice-backed witness handle exists. BTreeMap has 65,195,206 accesses; SHA-2
has 4,196,259, and Fibonacci stays sparse. Missing eligibility must keep the current
Metal route.

Factor the CPU opening formula into one shared helper rather than duplicating it.
Before production code, record the exact boundary: a 4GiB T28 fp128 equality table
plus one row-extraction pass, frozen CPU output wall 0.474s versus current Metal wall
1.897s and GPU-active 0.081s, predicted CPU-hybrid wall 0.47--0.85s, projected RSS
about 84.1GiB, and a complete-prover target below 47.58s from the 48.08s parent. Add
a red test that forces the route, checks exact output-claim parity, asserts one route
hit, and covers the off condition. Run the focused test and then at most one verified
BTreeMap T28 treatment; skip a timed T25 sentinel because the T28 cutoff makes this
candidate unreachable there. Reject on output wall above 0.85s, complete proving
above 47.58s, RSS above 90GiB, verifier failure, missing telemetry, or wrong routing.
This is an implementation-only hybrid and must not change the protocol.

After that result, recompute the complete critical path. Commit work alone cannot
close BTreeMap's current 14.77-second gap to its 33.31-second 5x ceiling, so select the
next disjoint PIOP or cross-stage mechanism from current counters, taking one new
Perfetto trace only if ownership remains unresolved. Do not choose from stale stage
totals or redo the accepted RAM/eval work, rejected coefficient-index fusion,
rejected Stage 4/5 private storage, or analytically rejected worker retiming.

Use the lean loop in the document: one mechanism, one analytical prediction and
falsifier, one red correctness test, one scoped edit, one short sentinel, and T28 only
after promotion. Routine execution gates are <=120 seconds excluding compilation.
Do not run repeated CPU baselines, broad workload matrices, Criterion, or Perfetto in
ordinary iterations. Preserve negative results and one accepted paired parent.

Keep generic fp128 kernel/residency/scheduling work in Akita and Jolt geometry,
adapter, PIOP, and cross-stage orchestration in Jolt. Route only on public geometry or
activity. Charge all per-proof hybrid work to the score and fail closed on qualified
Metal paths. Avoid protocol changes; only a bounded documented minor change with a
written necessity and unchanged soundness is in scope. Use the engineer,
gpu-kernel-analysis, experiment-design, autoresearch-loop, result-validation,
kernel-parity-bench, coding-standards, and comment-style skills, with this document's
fast credibility gates taking precedence over heavier historical loop defaults.
```
