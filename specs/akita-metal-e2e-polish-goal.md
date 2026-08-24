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

These are the frozen starting controls. The current provisional Jolt checkpoint has
improved BTreeMap to a two-run mean of 51.52 s (3.233x), leaving 18.21 s to the 5x
ceiling; its individual runs were 52.50 s and 50.54 s. Fibonacci and SHA-2 retain the
frozen values until the next milestone sweep. Ranges below are planning estimates to
falsify, not measured promises.

| Priority | Mechanism | Initial predicted opportunity |
|---|---|---:|
| 1 | Widen, then if needed multi-group, hot-address RAM compaction | BTreeMap 0.4--0.9 s for width alone |
| 2 | Stream/fuse the deferred opening index, decompose, and coefficient packing | Fibonacci/SHA-2 1.5--2 s; BTreeMap 2--3.5 s |
| 3 | Measure and retime the Stage 4/5 compatibility-scatter prefetch | up to 1.2 s on BTreeMap |
| 4 | Generalize the bytecode address carrier from `log_K = 13` to 14 | at most 1.34 s on SHA-2 |
| 5 | Remove commit wrapper, row-generation, and synchronization residue | 0.3--0.6 s per workload |
| 6 | Reduce BTreeMap commit traffic/locality cost | at least 1.8 s needed for 5x commit |
| 7 | Remove remaining lazy-first-bind, product-output, and command gaps | reprice after priorities 1--6 |

The first high-activity RAM tranche is complete at Jolt `1799ff816`: stable address
bucketing, a cycle-tiled frontier, scalar cold-address owners, cooperative hot-address
compaction, and the CPU address tail preserve the existing relation and yield a
verified 4.14 s BTreeMap improvement. Peak RSS is 80.08 GiB. The implementation is
not finished: the hottest address contains 20,729,173 of 65,195,206 accesses, so one
threadgroup still serializes 31.8% of the message work. RAM kernels remain 2.97 s
GPU-active against a 0.57 s analytical floor.

Fixed hot-message chunks are now retained: they reduce RAM GPU-active time from
2.97 s to 2.15 s without changing RSS, but leave one-group-per-address compaction as
the critical schedule. The immediate candidate widens that group from 256 threads to
the largest supported SIMD-aligned width capped at 1,024, without changing work or
storage. If that does not reach the predeclared 1.7 s RAM bar, separately model a
hot-only multi-group out-of-place compaction; do not combine the two mechanisms. Full
design, measurements, and the corrected chunk-boundary invariant live in
[akita-metal-high-activity-ram.md](akita-metal-high-activity-ram.md).

## Main execution plan

1. **Freeze the evaluator and parent.** Record the paired Jolt/Akita revisions,
   workload construction, compiler flags, machine state, proof-timing boundary, and
   one optimized-CPU anchor per workload. Treat a run without successful proof
   verification or complete backend-route telemetry as invalid.
2. **Finish BTreeMap's RAM parallelism first.** Retain the accepted address-segmented
   route and parallelize only the hot-address message boundary with a fixed chunk
   worklist and hierarchical reduction. First prove exact cold/hot parity, then use
   one T25 sentinel to catch dispatch or occupancy regressions and one T28 treatment
   only if the affected-span model remains credible. If the residual kernel time is
   still material, model hot-only multi-group compaction as a separate candidate.
   Reject any design that scans or sorts all `T` rows per round, exceeds 90 GiB RSS,
   or cannot plausibly save 0.5 s end to end.
3. **Remove shared opening overhead.** Stream or fuse deferred opening-index
   generation, decomposition, and coefficient packing so intermediates stay resident
   and are not materialized or transferred twice. Screen on Fibonacci first, then
   check that the same mechanism helps BTreeMap and SHA-2.
4. **Resolve the remaining geometry-specific gaps.** Reprice the Stage 4/5
   compatibility-scatter schedule, extend the bytecode address carrier to
   `log_K = 14` when public geometry qualifies, and remove commit wrapper,
   row-generation, synchronization, and traffic residue. Keep only changes with a
   measured complete-prover effect or an obvious no-cost removal of waste.
5. **Rerank at milestones.** Run all three `T = 2^28` workloads only after the model
   predicts a material change to the worst-workload score. Preserve one accepted
   paired parent and use the new stage deltas to choose the next mechanism; do not
   continue optimizing a locally hot kernel after its end-to-end ceiling becomes
   immaterial.
6. **Fit the deployment envelope.** Once all three workloads clear 5x with margin,
   check `T = 2^25` through `2^28`, fit geometry/activity-based CPU/Metal crossovers,
   and probe `T = 2^20` plus the scales bracketing each crossover. Then run final
   parity, verification, memory, formatting, test, and lint gates and remove search
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

> Create and pursue the goal in `specs/akita-metal-e2e-polish-goal.md`. Resume from
> Jolt `7acb4be74` on `feat/akita-metal` and Akita `4ccde218b` on
> `perf/metal-commit-eval-proof`; first audit both worktrees and preserve the local
> Jolt `Cargo.lock` and `.cargo/config.toml` path-override state. Optimize the
> composed Akita Metal prover across those two worktrees. The hard bar is at least
> 5x complete
> `jolt_prover::prove` speedup over the optimized CPU backend for BTreeMap,
> Fibonacci, and SHA-2 chain at `T = 2^28`; maximize the worst workload first. Once
> all three clear 5x with a credible margin, continue reducing total wall time while
> preserving that floor, `T = 2^25`--`2^28` performance, exact proof verification,
> and the 90 GiB RSS guard. Also reduce fixed costs and fit public
> geometry/activity-based hybrid switchovers around smaller workloads, including
> `T = 2^20`.
>
> Freeze and use the evaluator commands in the document. In particular, BTreeMap
> `T = 2^28` must include `--target-trace-size 150000000`, routine treatment runs
> must omit `--format`, and every accepted result must print successful proof
> verification. Do not shift witness- or transcript-dependent work outside the
> timed proving boundary.
>
> Start from the existing Perfetto analysis and maintain a disjoint end-to-end
> latency model. The provisional high-activity RAM route has moved BTreeMap from
> 56.34 s to a two-run mean of 51.52 s at 80.08 GiB RSS; fixed hot-message chunks
> are complete, so do not reimplement them. First widen the still-serial compaction
> group to the largest supported SIMD-aligned width capped at 1,024. If it misses
> the document's bar, price the hot-only out-of-place count/prefix/scatter design.
> Run focused cold/hot parity, a T25 sentinel, and only then a T28 treatment if the
> affected-span model is credible. Next consider deferred opening-index and
> coefficient fusion, the Stage 4/5 scheduling discrepancy, the `log_K = 14`
> bytecode carrier, and commit wrapper/traffic residue; rerank after every result.
> Use an analysis-led sequential loop with one scoped change, one prediction and
> falsifier, one focused correctness check, and normally one warm affected-workload
> treatment. Reuse frozen controls. A routine measurement gate must stay under 120
> seconds excluding compilation; do not run repeated baselines, full matrices, or
> traces during ordinary iterations. Repeat only ambiguous, surprising, or
> parent-promoting results.
>
> Keep generic fp128 work in Akita and Jolt-specific geometry and orchestration in
> Jolt. Route by geometry/activity rather than workload name, charge all hybrid CPU
> work and shifted preparation to the proving boundary, and fail closed on qualified
> paths. Avoid protocol changes by default; only a bounded, documented minor change
> with a written analytical need is in scope, and it must update prover and verifier
> together without reducing soundness. Keep accepted changes bisectable, maintain a
> terse ignored analysis/ledger rather than new controller machinery, preserve
> negative results, keep experimental artifacts out of production, and do not push.
> Do not declare completion merely upon crossing 5x: finish the final validation and
> continue until the document's analytical stop condition is met. Use the
> `engineer`, `gpu-kernel-analysis`, `autoresearch-loop`, `experiment-design`,
> `result-validation`, and `coding-standards` skills, with this document's lean gate
> taking precedence over any heavier historical loop.
