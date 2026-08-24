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

The accepted runtime sources are Jolt `9fb538461` and Akita `a454c7575`. At the
recorded C4 gate, Jolt tree `6c705b5c5` and Akita tree `58523a7b0` differed from
those sources only by documentation, rejected experiments, and exact reverts. Audit
the actual heads and trees before resuming because edits to this document advance the
Jolt head without changing runtime code. Jolt's modified `Cargo.lock` and untracked
`.cargo/config.toml` are intentional local Akita path overrides. Do not commit or
remove them. Do not push.

The current release binary was built from rejected C4 before its source revert. Rebuild
from the accepted runtime trees before another measurement; rebuild again only after a
later runtime-source change.

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
  dominates the removed traffic. Do not retry larger task-reuse factors;
- a materialized CPU ProductRemainder opening hybrid. It reduced its local extraction
  from 1.897 s to 0.762 s in two verified T28 runs, but the 4 GiB equality-table walk
  displaced later unified-memory work and regressed complete proving from 48.08 s to
  a 49.59 s mean. Do not retry it unless the table is eliminated and the cross-stage
  memory effect is priced separately;
- carry-free D512 radix-`2^16` accumulation. It preserved exact output and every
  route/traffic counter, but raising live per-lane state from 40 to 64 32-bit values
  doubled T25 root GPU time from 1.473 s to 2.958 s and raised complete proving from
  6.45 s to 8.32 s. Do not retry wide, RNS, or other larger-state accumulators;
- two-chain carry interleaving. It preserved the 40-value persistent state and exact
  traffic, but the second live carry vector raised T25 root GPU time from 1.473 s to
  1.604 s and complete proving from 6.45 s to 6.60 s. The compiler already exposes
  enough instruction-level parallelism, or transient register pressure dominates.
  Do not retry lookahead, extra chains, or other carry-scheduling variants.

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

### 2. Current tranche: specialize the uniform D512 sign quadrant

Each selected row is evaluated in two 256-coefficient bands, each held by two
128-coefficient accumulators. One band has a uniform sign. In the other, the
negacyclic boundary can intersect only one accumulator; the other accumulator is
entirely positive or entirely negative. The accepted shader nevertheless sends both
through the mixed-sign helper. It therefore computes a vector sign comparison, carry
selection, four limb selections, and wrap selection for 128 coefficients whose sign
is already known from `local_shift < 128`.

C5 adds one SIMDgroup-uniform branch in each variable-sign band. It keeps the one
boundary accumulator on the mixed helper and calls the existing positive or negative
helper for the uniform accumulator. No accumulator, carry chain, array, task, matrix
stream, barrier, scratch byte, output, protocol, transcript, or verifier state changes.
The operation count is independent of workload data for every nonzero selector.

The T28 traffic floor remains 4.388 seconds and the calibrated hot-dependent compute
term remains 7.77 seconds before treatment. C5 reduces mixed-sign preparation from
256 to 128 of the 512 coefficients added for each selected row. Counting the four
limb selections plus sign/carry/wrap selection gives a conservative 0.4--0.8-second
T28 root saving, predicting 11.7--12.1 seconds root GPU and 47.3--47.7 seconds
complete proving. The falsifiers are that Metal compiles uniform masks away already,
or that the added uniform branch costs more than the removed selects.

Add a red route-identity assertion and extend the existing negacyclic-rotation test at
shifts 1, 127, 128, 129, and 255. Reuse the exact mixed-sign, stream-boundary,
512-block, and streaming parity suite. Run one verified T25 BTreeMap sentinel and
admit T28 only if root GPU time is at most 1.43 seconds from the 1.473-second parent
with identical traffic, hybrid split, and proof. At T28 require root GPU at most
12.10 seconds plus materiality: either complete proving at most 47.58 seconds or root
GPU at most 11.884 seconds. RSS must remain at most 90 GiB. Any miss exactly reverts
without a shift-threshold or branch-layout sweep. This is an execution-only Akita
change.

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

1. the uniform D512 sign-quadrant specialization above;
2. further commit arithmetic reduction with one task per SIMDgroup, or wrapper,
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
reverts. The current release binary was built from rejected C4 before its exact source
revert. Rebuild it from the accepted runtime trees before measuring, then rebuild only
after a runtime-source edit.

The hard objective is at least 5x complete jolt_prover::prove speedup over optimized
CPU for BTreeMap, Fibonacci, and SHA-2 chain at T=2^28, maximizing the worst ratio
first. Preserve exact proof verification, RSS <= 90 GiB, accepted T25--T28
performance, and useful public geometry/activity-based CPU/Metal crossovers at lower
scales. Once all three clear 5x with credible margin, continue while a simple bounded
candidate has at least roughly 0.5 seconds or 1% of credible T28 upside.

Do not retry C1's two-task matrix reuse, C2's 4GiB CPU opening table, C3's wide
signed-digit accumulator, or C4's two-live-carry interleaving. Each preserved
exactness and was exactly reverted after its short gate; C3 doubled T25 root GPU to
2.958s and C4 regressed it from 1.473s to 1.604s.

The current tranche is D512 sign-quadrant specialization in Akita. For each selected
row, only one of the two 128-coefficient accumulators in the variable band can contain
the negacyclic sign boundary. Keep that accumulator on the mixed-sign helper and route
the other through the existing positive or negative helper using a SIMDgroup-uniform
`local_shift < 128` branch. Preserve the two accumulators, 40-value state, arithmetic
results, task grid, traffic, hybrid split, proof, transcript, and verifier. Add no
state, barriers, streams, or protocol changes. The T28 model predicts 11.7--12.1s root
GPU from 12.510s; uniform-mask compiler elimination is the main falsifier.

Use a red route identity, boundary shifts 1/127/128/129/255, and the existing exact
mixed-sign, stream-boundary, 512-block, and streaming tests. Then run one verified
BTreeMap T25 sentinel. Admit T28 only if T25 root GPU is at most 1.43s with unchanged
counters. At T28 require root GPU at most 12.10s and either complete proving at most
47.58s or root GPU at most 11.884s, with RSS at most 90GiB and exact verification.
Reject and exactly revert without a threshold or branch-layout sweep if the gate
misses.

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
