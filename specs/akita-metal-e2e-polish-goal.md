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

Accepted runtime sources are Jolt `9fb53846188b6cb9481c39f74fb99007c2cfa8aa`
and Akita `a454c757597608ba20ad593be108557219103d6e`. Later commits are
documentation or exact reverts of rejected candidates. Audit revision and tree IDs
before resuming. Jolt's modified `Cargo.lock` and untracked `.cargo/config.toml` are
intentional local Akita path overrides; do not commit or remove them. Do not push.

The current release binary contains rejected C5 code and is not a valid evaluator.
Rebuild after the next runtime edit, or rebuild the accepted tree first if a baseline
measurement is needed. Compilation is not part of an experiment's time gate.

| Workload | Optimized CPU | Accepted Metal | Speedup | 5x target | Remaining gap |
|---|---:|---:|---:|---:|---:|
| BTreeMap | 166.548 s | 48.08 s | 3.464x | 33.310 s | 14.770 s |
| Fibonacci | 215.18 s | 45.72 s | 4.71x | 43.036 s | 2.684 s |
| SHA-2 chain | 213.70 s | 42.45 s | 5.03x | 42.740 s | clears by 0.29 s |

The accepted BTreeMap trace localizes 48.984 seconds as 14.219 seconds commit,
28.842 seconds PIOP, and 5.897 seconds eval proof. It is diagnostic; retain the
48.08-second untraced run as the score. Deleting all commit time would still leave
35.47 seconds, so 5x necessarily requires both commit and PIOP/cross-stage gains.

## Main plan

### 1. Freeze the evaluator and build a disjoint budget

Keep the CPU controls frozen unless the CPU implementation, protocol, workload,
compiler, machine, flags, or timed boundary changes. Preserve one accepted paired
parent and an append-only ignored ledger under
`benchmark-runs/akita-metal-e2e-polish/`.

Before another implementation, update a disjoint BTreeMap latency model from the
accepted trace and existing counters. Identify enough non-overlapping credible upside
to cover the 14.77-second gap; use roughly 20% gross headroom as a planning target for
noise and overlap, not as a promotion requirement. A candidate whose complete
disjoint ceiling cannot materially move the remaining gap is rejected analytically.
Take a new Perfetto trace only when existing telemetry cannot identify the owner.

### 2. Remove PIOP ownership and materialization costs

Analyze these two mechanisms first, then implement the one with the larger credible
end-to-end saving per correctness and engineering risk:

1. **ProductRemainder output fusion.** Stage 2 spends 1.897 seconds extracting the
   product output. A CPU treatment reduced that span to about 0.762 seconds but
   materialized and walked a 4 GiB equality table. Complete proving regressed to a
   49.59-second mean, consistent with later unified-memory displacement. Derive
   whether the required opening evaluations can be accumulated during the existing
   Metal binds or recovered from final resident state without that table. Price added
   arithmetic, traffic, command boundaries, and downstream residency before code.
2. **Direct Stage-5 source consumption.** The Stage-4/5 compatibility path builds
   four dense planes totaling about 9.25 GiB. Construction costs roughly 1.55 seconds
   plus 0.18 seconds of address prefetch and overlaps Stage 4; the trace is consistent
   with contention. Retiming and private grouped storage are closed failures. Specify
   a direct cycle-major or resident-source consumer that eliminates the planes rather
   than relabeling or moving them, including the exact Stage-5 address-phase access
   pattern and any replacement work.

After either treatment, remeasure Stage 2 and Stage 4 once. Current discrepancies are
Stage 2 RAM preparation and product output, plus Stage 4 register preparation and
rounds. Stage 6b already contains large Metal wins; do not optimize its total merely
because it is large.

### 3. Redesign the Akita commit root at the schedule or representation level

The accepted D512 root uses one task per SIMDgroup, two coefficient bands, sixteen
position partials, and about forty persistent 32-bit accumulator values per lane. At
T28 it performs about 1.019 trillion fp128 coefficient additions. Modeled traffic is
1,810 GiB, with a 4.39-second bandwidth floor; the calibrated arithmetic term is
about 7.77 seconds versus 12.51 seconds observed root GPU time.

The tested local instruction and accumulator variants are closed: two tasks per
group, wider carry-free digits, two live carry chains, and C5 sign-quadrant
specialization all failed their end-to-end gates. C5 reduced root GPU time to 12.088
seconds but regressed the verified proof to 49.07 seconds, so it was exactly reverted.
The next commit candidate must remove dominant work, change reuse/ownership, or use a
bounded public schedule/configuration change. Generic fp128, residency, and scheduling
work lives in Akita; workload geometry and orchestration stay in Jolt.

### 4. Re-rank shared fixed costs and bounded protocol knobs

After each retained treatment, recompute the complete critical path. Prefer work that
helps all three workloads and creates margin for Fibonacci and SHA-2. Route only on
public geometry or activity, never workload names. Hybrid execution is allowed when
fully timed.

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
and raw artifacts. Run formatting, focused exact-parity tests, relevant nextest
suites, and both required clippy modes. Document remaining caveats and anything not
verified.

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
Create and pursue the goal in specs/akita-metal-e2e-polish-goal.md. Read the document
fully before acting and treat its evaluator, accepted state, invariants, fast candidate
loop, closed paths, and completion gate as binding.

Work in Jolt branch feat/akita-metal at
/Users/mgeorghiades/worktrees/jolt/bright-ridge/jolt and Akita branch
perf/metal-commit-eval-proof at
/Users/mgeorghiades/worktrees/akita-metal-eval-proof. Do not push. Preserve Jolt's
intentional Cargo.lock and .cargo/config.toml path overrides and never commit them.
Audit both heads, tree IDs, and runtime diffs first. Accepted runtime sources are Jolt
9fb53846188b6cb9481c39f74fb99007c2cfa8aa and Akita
a454c757597608ba20ad593be108557219103d6e. The current release binary contains
rejected C5 code; do not score it. Rebuild after the next runtime edit, or rebuild the
accepted tree first only if a fresh baseline is genuinely needed. Compilation time is
unrestricted and excluded from experiment gates.

The hard objective is at least 5x complete jolt_prover::prove speedup over optimized
CPU for BTreeMap, Fibonacci, and SHA-2 chain at T=2^28, maximizing the worst ratio
first. Preserve exact proof verification, peak RSS <=90 GiB, accepted T25--T28
performance, and useful public geometry/activity-based lower-scale crossovers. Charge
all per-proof CPU, GPU, allocation, transfer, synchronization, and assembly work.
After all three clear 5x with credible margin, continue while a bounded non-invasive
candidate has at least 0.5 seconds or 1% of plausible T28 upside.

Begin with an analytical refresh, not a benchmark sweep. Using the accepted trace and
counters, make a disjoint latency/ceiling ledger for the 14.77-second BTreeMap gap.
Analyze ProductRemainder output fusion without a materialized equality table and
direct Stage-5 consumption that eliminates the 9.25-GiB compatibility scatter. Pick
one only after stating its exact algebra/dataflow boundary, compulsory work and
traffic, credible end-to-end saving, memory effect, and numerical falsifier. If both
ceilings are inadequate, re-rank before coding. Take a new Perfetto trace only when
existing evidence cannot identify ownership.

For commit, do not retry two-task reuse, wide carry-free accumulation, two-chain
interleaving, sign-quadrant specialization, or other local variants of those closed
ideas. The next Akita candidate must change dominant work, reuse, ownership, or a
bounded public schedule/configuration. Keep generic fp128 kernels, residency, and
scheduling in Akita; keep Jolt geometry, adapters, PIOP kernels, and cross-stage
orchestration in Jolt. Minor documented protocol/config changes are allowed only with
an explicit necessity and ceiling argument, synchronized prover/verifier changes,
and unchanged soundness. Do not make invasive protocol changes.

Use a lean sequential loop: one candidate, one prediction and falsifier, one red
correctness test, one scoped implementation, focused parity, and normally one warm
T25 sentinel. Promote to one T28 run only when affected-span telemetry supports the
mechanism. Routine execution gates are <=120 seconds excluding compilation. Do not
rerun frozen CPU controls, broad workload matrices, Criterion, or Perfetto in ordinary
iterations. Repeat a measurement only for ambiguity, surprise, or final promotion.
Keep or exactly revert each candidate and preserve an append-only evidence ledger.

Run the full three-workload T28 matrix only at material milestones. Final acceptance
requires two order-reversed CPU/Metal pairs per workload with the worse valid ratio
above 5x, exact verification, no undocumented fallback, no swapping, RSS <=90 GiB,
verified T25--T28 guards, calibrated lower-scale crossovers, production cleanup,
formatting, relevant nextest suites, and both required clippy modes. Do not stop merely
because one isolated kernel or one workload reaches 5x.
```
