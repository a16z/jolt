# Akita Metal end-to-end prover polish

## Decision

Optimize the composed Akita/Metal prover, not isolated kernels. The hard milestone is
at least **5x complete `jolt_prover::prove` speedup** over the optimized CPU backend
for BTreeMap, Fibonacci, and SHA-2 chain at `T = 2^28`, maximizing the worst ratio
first. Once all three clear 5x with credible margin, continue while a bounded,
non-invasive candidate has at least 0.5 seconds or 1% of plausible T28 upside.

Every per-proof cost is charged: witness adaptation, hybrid CPU work, allocation,
transfers, synchronization, readback, and proof assembly. Every scored proof must
verify, silent fallback is forbidden, and peak RSS must not exceed 90 GiB. Public,
witness-independent preprocessing may be excluded only when it is genuinely reusable
across proofs.

## Accepted state and gap

Worktrees:

- Jolt: `feat/akita-metal` at
  `/Users/mgeorghiades/worktrees/jolt/bright-ridge/jolt`;
- Akita: `perf/metal-commit-eval-proof` at
  `/Users/mgeorghiades/worktrees/akita-metal-eval-proof`.

The frozen runtime parent is Jolt `6ec86d08a77d2210676c4f299d55cf7f0ab46892`
and Akita `8291c2dbcd75f413e9697b7cb7ff89942a0c9005`. At handoff, their
tree IDs were `c109b3e925f58fe0e9553eca0a17439280cd02c8` and
`58523a7b0546b540c7636248a31906074ae1e136`. Audit revisions, tree IDs,
and runtime diffs before resuming. Jolt's modified `Cargo.lock` and untracked
`.cargo/config.toml` are intentional local Akita path overrides; do not commit or
remove them. Do not push.

The current release binary contains rejected S3 code and is not a valid evaluator.
Rebuild the accepted source before the next measurement. Compilation is not part of
an experiment's time gate.

| Workload | Optimized CPU | Last credible Metal | Speedup | 5x target | Remaining gap |
|---|---:|---:|---:|---:|---:|
| BTreeMap | 166.548 s | 46.99 s | 3.544x | 33.310 s | 13.680 s |
| Fibonacci | 215.177 s | 45.719 s | 4.71x | 43.035 s | 2.684 s |
| SHA-2 chain | 213.703 s | 42.452 s | 5.03x | 42.741 s | clears by 0.289 s |

BTreeMap is the post-S1 score. Fibonacci and SHA-2 are the last credible results
from the preceding accepted parent; do not spend a matrix run merely to refresh
them. Remeasure them at the next material milestone.

The accepted post-S1 BTreeMap trace is 47.389 seconds: 14.149 seconds commit,
27.211 seconds PIOP, 6.001 seconds eval proof, and about 0.028 seconds other work.
S1 overlaps independent host and Metal Stage-6b members, cutting that stage from
7.043 to 4.994 seconds; its untraced confirmation is 46.99 seconds at 80.07 GiB.
Deleting all commit time would still leave roughly 32.84 seconds, leaving only
0.47 seconds of margin against the 5x target. The remaining plan therefore needs
both a major commit gain and non-commit critical-path savings.

## Main plan

### 1. Freeze the evaluator and refresh the disjoint budget

Keep the CPU controls frozen unless the CPU implementation, protocol, workload,
compiler, machine, flags, or timed boundary changes. Preserve one accepted paired
parent and an append-only ignored ledger under
`benchmark-runs/akita-metal-e2e-polish/`.

Rebuild the accepted source, but do not rerun a baseline unless a candidate needs a
fresh paired control. Update the BTreeMap model from the accepted S1 trace and
existing counters. Identify enough non-overlapping credible upside to cover the
13.68-second gap; use about 16.4 seconds of gross modeled upside as a planning target
for noise and interaction. A candidate whose complete disjoint ceiling is below
0.5 seconds or cannot materially move the remaining gap is rejected analytically.
Take a new Perfetto trace only when existing telemetry cannot identify the owner.

### 2. Remove cross-stage ownership and residency costs

Analyze the following as complete chains, not isolated child spans, and implement
only the largest mechanism with a credible net saving:

1. **Stage-1 source residency chain.** ProductRemainder output, the immediately
   following Instruction opening, and the Stage-3 shift prefix share a large Stage-1
   source. S3 cut Product output from 2.003 to 0.527 seconds on CPU, yet complete
   proving regressed by 0.703 seconds because Instruction output, Stage 3, and later
   Metal work inherited the residency bill. Standalone CPU Product opening routes
   are closed. Price either a late asynchronous source primer overlapped with useful
   preceding work or a compact retained Product+Instruction opening view built while
   Stage 1 is hot. Charge contention, all retained bytes, Instruction, Shift, and the
   later accelerator spans. Proceed only if the whole chain credibly saves at least
   0.5 seconds.
2. **Direct Stage-5 source consumption.** The Stage-4/5 compatibility path builds
   four dense planes totaling about 9.25 GiB. Construction costs roughly 1.55 seconds
   plus 0.18 seconds of address prefetch and overlaps Stage 4. Retiming and private
   grouped storage are closed failures. Specify a direct cycle-major or resident
   consumer that eliminates the planes, including the exact Stage-5 address-phase
   access pattern and every replacement read or computation.
3. **Integrated eval-proof fixed costs.** Isolated T28 eval proof is about 5.175
   seconds, while integrated eval is 6.001 seconds. Localize the roughly 0.83-second
   integration tax around root coefficient packing, D-role relation construction,
   and root folding. Treat 0.83 seconds as a ceiling, not an additive prediction.

The accepted two-lane scheduler leaves only about 9 ms of host work exposed in
Stage 6b. Do not optimize its total merely because it is large. The lazy Bytecode
width-8 treatment is also closed: it improved T25 but regressed the T28 Bytecode
member and Stage 6b.

### 3. Redesign the Akita commit root at the schedule or representation level

The accepted commit takes 14.149 seconds, including about 12.51 seconds in the D512
root. It uses one task per SIMDgroup, two coefficient bands, sixteen position
partials, and about forty persistent 32-bit accumulator values per lane. At T28 it
performs about 1.019 trillion fp128 coefficient additions. Modeled traffic is
1,810 GiB, with a 4.39-second bandwidth floor; the calibrated arithmetic term is
about 7.77 seconds.

The tested local instruction and accumulator variants are closed: two tasks per
group, wider carry-free digits, two live carry chains, and C5 sign-quadrant
specialization all failed their end-to-end gates. C5 reduced root GPU time to 12.088
seconds but regressed the verified proof to 49.07 seconds, so it was exactly reverted.
The next commit candidate must remove dominant work, change reuse/ownership, or use a
bounded public schedule/configuration change. Generic fp128, residency, and scheduling
work lives in Akita; workload geometry and orchestration stay in Jolt.

### 4. Re-rank the composed prover and bounded protocol knobs

After each retained treatment, recompute the complete critical path and remove
overlapping ceilings. Prefer work that helps all three workloads and creates margin
for Fibonacci and SHA-2. Route only on public geometry or activity, never workload
names. Hybrid execution is allowed when fully timed. Cross-component scheduling and
layout changes are preferred when they remove allocations, source scans, queue
serialization, or unified-memory migration paid by more than one stage.

Minor public schedule, batching, or layout changes are in scope only when a written
ceiling shows that prover-only work cannot reach the target. They must preserve the
statement and soundness, update prover and verifier together, be independently
revertible, and be recorded in
[akita-metal-protocol-changes.md](akita-metal-protocol-changes.md). Invasive protocol
changes remain out of scope.

### 5. Validate the milestone, calibrate crossovers, and clean up

Run the full T28 workload matrix only when the model predicts a material change to
the worst ratio. Once all three appear above 5x, run two order-reversed CPU/Metal
pairs per workload and score the worse valid ratio. Then run verified Metal guards at
T25--T28, fit CPU/Metal crossovers from public geometry/activity, and test T20 plus
the scales bracketing every threshold.

Before handoff, remove rejected variants, search-only switches, obsolete telemetry,
and raw artifacts. Keep the generic Metal backend and protocol-facing changes
reviewable as separate logical diffs. Run formatting, focused exact-parity tests,
relevant nextest suites, and both required clippy modes. Document remaining caveats
and anything not verified.

## Fast candidate loop

For each candidate:

1. State one mechanism, exact boundary, lower bound, predicted complete-prover
   saving, and numerical falsifier before code.
2. Add the smallest red exactness or parity test, then one scoped edit.
3. Run focused correctness and normally one warm T25 affected-workload sentinel.
4. Promote to one T28 run only when affected-span telemetry supports the mechanism.
5. Keep, exactly revert, or mark invalid. Repeat once only for threshold ambiguity,
   a surprising result, or final promotion, not to hunt a favorable sample.
6. Update the accepted parent, latency model, and negative-evidence ledger.

A routine execution gate is at most 120 seconds, excluding compilation. Do not run
repeated CPU controls, broad matrices, Criterion, or Perfetto during ordinary
iterations. Fail closed on wrong output, verifier failure, missing metrics, evaluator
drift, unexplained fallback, swapping, non-finite timing, or unrelated source edits.

Closed paths and their evidence live in
[akita-metal-high-activity-ram.md](akita-metal-high-activity-ram.md),
[akita-metal-stage4-stage5-prefetch.md](akita-metal-stage4-stage5-prefetch.md),
[akita-metal-perfetto-t28-analysis.md](akita-metal-perfetto-t28-analysis.md), and the
ignored experiment ledger. Do not retry a closed mechanism without a materially new
ownership or work-elimination argument.

## Fixed evaluator

Build:

```bash
cargo build --release -p jolt-prover --example modular_benchmark \
  --features prover-fixtures,metal
```

Score the reported `jolt_prover::prove` wall time and require
`PROOF_VERIFIED ... value=true`:

```bash
./target/release/examples/modular_benchmark \
  --name fibonacci --scale 28 --backend {optimized|metal}
./target/release/examples/modular_benchmark \
  --name sha2-chain --scale 28 --backend {optimized|metal}
./target/release/examples/modular_benchmark \
  --name btreemap --scale 28 --target-trace-size 150000000 \
  --backend {optimized|metal}
```

The BTreeMap trace-size override is part of the workload identity. Record proof
verification, route/fallback counters, affected-span telemetry, and peak RSS.

## Completion gate

Do not claim 5x or finish the goal until all of these hold:

- two order-reversed CPU/Metal pairs for each T28 workload, with the worse valid
  ratio above 5x and enough margin for observed noise;
- exact verification, no undocumented fallback, no swapping, and RSS at most 90 GiB
  for every scored run;
- verified Metal guards for all workloads at T25--T28 and tests bracketing every
  fitted lower-scale crossover, including T20 when practical;
- a production diff without rejected implementations, search machinery, raw logs, or
  obsolete instrumentation; and
- focused parity tests, formatting, relevant nextest suites, and both required clippy
  modes, with any pre-existing blocker separated from candidate diagnostics.

## Copy/paste goal prompt

```text
Create a persistent goal to execute specs/akita-metal-e2e-polish-goal.md through its
completion gate. Read the entire document before acting. Treat its evaluator,
accepted state, main plan, fast candidate loop, closed paths, and completion gate as
binding; keep working across continuations until the objective is achieved or a real
external blocker satisfies the goal-mode blocked threshold.

Work in Jolt branch feat/akita-metal at
/Users/mgeorghiades/worktrees/jolt/bright-ridge/jolt and Akita branch
perf/metal-commit-eval-proof at
/Users/mgeorghiades/worktrees/akita-metal-eval-proof. Do not push. Preserve Jolt's
intentional Cargo.lock and .cargo/config.toml path overrides and never commit them.
Audit both heads, tree IDs, and runtime diffs first. Accepted runtime sources are Jolt
6ec86d08a77d2210676c4f299d55cf7f0ab46892 and Akita
8291c2dbcd75f413e9697b7cb7ff89942a0c9005. Their recorded tree IDs are
c109b3e925f58fe0e9553eca0a17439280cd02c8 and
58523a7b0546b540c7636248a31906074ae1e136. The current release binary contains
rejected S3 code; do not score it. Rebuild the accepted source before the next
measurement. Compilation time is unrestricted and excluded from execution gates.

The hard objective is at least 5x complete jolt_prover::prove speedup over optimized
CPU for BTreeMap, Fibonacci, and SHA-2 chain at T=2^28, maximizing the worst ratio
first. Preserve exact proof verification, peak RSS <=90 GiB, accepted T25--T28
performance, and useful public geometry/activity-based lower-scale crossovers. Charge
all per-proof CPU, GPU, allocation, transfer, synchronization, and assembly work.
After all three clear 5x with credible margin, continue while a bounded non-invasive
candidate has at least 0.5 seconds or 1% of plausible T28 upside.

The current post-S1 BTreeMap score is 46.99 seconds versus 166.548 seconds CPU, or
3.544x, leaving 13.68 seconds to the 5x threshold. Its accepted trace is 47.389
seconds: 14.149 commit, 27.211 PIOP, 6.001 eval proof, and about 0.028 other. S1's
two-lane scheduler is accepted and leaves only about 9 ms of Stage-6b host work
exposed. Fibonacci and SHA-2 have not been rerun after S1; their last credible scores
are 4.71x and 5.03x. Do not refresh them until a material milestone.

Begin with an analytical refresh, not a benchmark sweep. Build a disjoint
latency/ceiling ledger with about 16.4 seconds of gross credible upside for the
13.68-second BTreeMap gap. First price the complete shared-residency chain spanning
ProductRemainder output, Instruction opening, and the Stage-3 shift prefix. S3 proved
that making Product output fast in isolation only moves the residency cost: Product
fell by 1.476 seconds while complete proving regressed by 0.703 seconds. Do not retry
standalone CPU Product opening routes. Compare a late asynchronous source primer and
a compact retained Product+Instruction opening view, charging contention, retained
memory, Instruction, Shift, and downstream Metal effects. In parallel in the model,
price direct Stage-5 consumption that actually deletes the 9.25-GiB compatibility
planes and the roughly 0.83-second integrated eval-proof tax. Implement only a
mechanism with at least 0.5 seconds of credible disjoint end-to-end saving. If none
qualifies, reject them analytically and re-rank.

For commit, do not retry two-task reuse, wide carry-free accumulation, two-chain
interleaving, sign-quadrant specialization, or other local variants of those closed
ideas. The next Akita candidate must change dominant work, reuse, ownership, or a
bounded public schedule/configuration. Keep generic fp128 kernels, residency, and
scheduling in Akita; keep Jolt geometry, adapters, PIOP kernels, and cross-stage
orchestration in Jolt. Minor documented protocol/config changes are allowed only with
an explicit necessity and ceiling argument, synchronized prover/verifier changes,
and unchanged soundness. Do not make invasive protocol changes.

Also treat S2's lazy Bytecode width-8 path as closed: it helped T25 but regressed the
T28 Bytecode member and Stage 6b. Do not retry any closed path without a materially
new work-elimination or ownership argument. Take a new Perfetto trace only when the
accepted trace and existing counters cannot identify ownership.

Use a lean sequential loop: one candidate, one prediction and falsifier, one red
correctness test, one scoped implementation, focused parity, and normally one warm
T25 sentinel. Promote to one T28 run only when affected-span telemetry supports the
mechanism. Routine execution gates are <=120 seconds excluding compilation. Do not
rerun frozen CPU controls, broad workload matrices, Criterion, or Perfetto in ordinary
iterations. Repeat a measurement only for ambiguity, surprise, or final promotion.
Keep or exactly revert each candidate and preserve an append-only evidence ledger.

After every retained result, update the accepted parent, disjoint critical path, and
ranked queue before selecting more code. Prefer cross-component scheduling or layouts
that remove repeated source scans, allocations, queue serialization, or unified-memory
migration. Route by public geometry/activity only. Keep all proof and transcript
semantics exact and fail closed on verifier errors, missing telemetry, fallback,
swapping, evaluator drift, or unrelated source changes.

Run the full three-workload T28 matrix only at material milestones. Final acceptance
requires two order-reversed CPU/Metal pairs per workload with the worse valid ratio
above 5x, exact verification, no undocumented fallback, no swapping, RSS <=90 GiB,
verified T25--T28 guards, calibrated lower-scale crossovers, production cleanup,
formatting, relevant nextest suites, and both required clippy modes. Do not stop merely
because one isolated kernel or one workload reaches 5x. After the milestone, continue
only while the document's bounded polish rule admits another candidate.
```
