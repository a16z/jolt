# Akita Metal evaluation-proof goal

## Objective

Accelerate Akita's fp128 evaluation/opening proof by at least **5x over its CPU
implementation** on every Jolt trace shape we care about, including `T = 2^28`.
The timed unit is the complete `AkitaNativeBatching::prove_batch` call: validation,
transcript binding, every CPU/Metal operation, synchronization and readback, proof
assembly, serialization, and transcript absorption. Setup and the already-required
commit are excluded, but any new per-proof preparation or work shifted into commit
is charged to the integrated result.

This goal ends when the 5x opening floor, correctness, routing, memory, and
production-cleanup gates pass. It reports full Jolt proving time and the remaining
Amdahl gap, but it does not expand into unrelated PIOP optimization. Closing that
gap is a later goal.

Start from:

- Akita fork commit `30e99fed9` on a new `perf/metal-eval-proof` worktree branch;
  it already contains the accepted Metal commit backend and is based on Akita
  `upstream/main` at `2869b67bf`.
- Jolt commit `c6d0b1467` on `feat/akita-metal`, used initially only for the exact
  adapter harness and final integration checks.

Do not push or integrate either branch without approval.

## Current bound

The current opening is still CPU-only: `jolt-akita` constructs a
`UniformProverStack<CpuBackend>` inside native batching. Retained measurements are
provisional anchors, not new experimental evidence:

| Case | CPU opening | 5x ceiling |
|---|---:|---:|
| `T=2^25`, standard workloads | 9.22-10.4 s | 1.84-2.08 s |
| `T=2^28`, BTreeMap 150M target | 20.8 s | 4.16 s |

At `T=2^28`, the 21.9-second opening profile contains approximately 5.15 s of
root one-hot decompose/fold, 4.75 s of root stage-2 sumcheck, 3.06 s of root
stage-1 sumcheck, 1.76 s of ring-switch witness construction, 1.45 s of the next
witness commitment, 1.29 s of root coefficient packing, and 1.32 s of root NTT
preparation. The first three alone are about 59% of the call, so optimizing only
the sparse fold cannot reach 5x. A candidate must satisfy

```text
predicted_open = host_serial + sum(cpu_phase_i / predicted_speedup_i)
                 + transfer_and_sync
```

with `predicted_open <= frozen_cpu_open / 5`. Reject a design analytically if its
uncovered floor already exceeds the target. In practice, reaching 5x requires
roughly 90% coverage at about 10x, or a comparable mix of faster kernels and less
work.

## Architecture

Keep protocol orchestration and the public proof in Akita. Use Akita's existing
four-cluster `ProverComputeStack` and per-level stack selection instead of adding
a Jolt-specific prover:

- **Akita/`akita-metal`:** generic fp128 kernels, persistent per-proof device
  workspaces, setup/matrix residency, CPU-tail policy, metrics, and parity tests.
- **Akita prover:** only the backend seams needed for device-resident root work or
  large sumcheck rounds. Prefer operation-level traits; introduce a fused root
  operation only when materializing an intermediate provably breaks the 5x bound.
- **`jolt-akita`:** expose the packed trace view to those generic kernels, select
  the Metal opening stack, fail closed for qualified shapes, and host the exact
  Jolt-shape acceptance harness.
- **`jolt-prover`:** no algorithm changes during the search; it supplies final
  verified workload sentinels.

First route the current backend through opening to expose any existing digit-row
wins. Then attack, in measured order: root decompose/fold and coefficient packing;
stage-1 and stage-2 sumcheck scans/folds; ring-switch and next-witness commit; NTT
and transfer/residency overhead. Keep large round state device-resident and switch
the shrinking tail to CPU. Do not transpose Jolt's cycle-major trace.

The initial protocol and canonical schedule remain byte-identical. A schedule or
small protocol change is allowed only when a written traffic/compute model shows
the unchanged protocol cannot clear 5x. Such a change must be public-shape-derived
or explicitly encoded, update prover and verifier together, and remain isolated
from kernel commits.

## Fixed harness

Add a non-Criterion, single-shot `akita_eval_proof` bench target under
`crates/jolt-akita/benches/`. It constructs the real `TracePackedOneHot`, canonical
K256 schedule, retained commitment hint, point, evaluation, and transcript used by
Jolt. Fixture construction, setup, and one commit occur outside the timed region
and are reported separately. CPU and Metal consume clones of the same fixture.

Required cases are `T=2^25` with 29 and 30 live columns, and `T=2^28` with 30
columns and the accepted populated-row shape. One command selects CPU, Metal, or
an alternating pair and emits one JSON record containing:

- revisions, fixture and schedule digests, shape, populated rows, and backend;
- complete opening wall time and disjoint subphase times;
- GPU-active, command-wall, transfer/readback, allocation, and peak-RSS metrics;
- selected route, CPU-tail work, and a `fallback=false` assertion;
- claimed evaluation, proof digest/size, CPU parity, and verifier result.

The evaluator rejects missing metrics, a changed fixture/schedule, fallback on a
qualified operation, invalid proof, mismatched evaluation or proof bytes, work moved
outside the boundary, or RSS above the retained 90 GiB max-scale guard.

Freeze one CPU JSON anchor per case after the harness is correct. Reuse it until
CPU code, protocol, schedule, fixture, compiler flags, or timing boundary changes.
Never rerun a control merely because a candidate changed Metal code.

## Lean research loop

Each iteration is one mechanism and one prediction:

1. Name the measured phase, its cost, traffic/operation floor, proposed change,
   predicted complete-opening time, and falsifying observation.
2. Build incrementally. Run one focused CPU/Metal parity test for the changed
   operation.
3. Run one single-shot `T=2^25` treatment. Accept a clear improvement with the
   predicted counters; reject/revert a miss. Repeat once only when the decision is
   within 3% or the observation is invalid.
4. Run `T=2^28` only for a material architectural milestone or final candidate.
5. Keep accepted changes as small bisectable commits and a terse ledger line.

A warm routine gate should take seconds and must stay under two minutes excluding
incremental compilation. No Criterion windows, broad parameter sweeps, repeated
baselines, long polling, full-workspace validation, or new autoresearch-controller
machinery during search. A terse ledger and the harness JSON are enough. Prefer a
short analytical redesign over accumulating micro-optimizations with an inadequate
ceiling.

## Final gate and launch prompt

Final acceptance requires two order-reversed CPU/Metal pairs for each fixed harness
case, with the worst pair at least 5x; all relevant Akita and `jolt-akita` parity,
routing, and verifier tests; clippy and formatting; one verified Metal Jolt proof
for Fibonacci, SHA-2, SHA-3, and BTreeMap at `T=2^25`; and one verified `T=2^28`
BTreeMap proof. The earlier commit phase must remain at least 5x, no PIOP path may
materially regress, and max-scale RSS must remain at most 90 GiB. Remove search
variants, raw evidence, knobs, and dead instrumentation before declaring success.

Copy/paste goal-mode prompt:

> Create and pursue the goal in `specs/akita-metal-eval-proof-goal.md`. Build the
> fixed single-shot harness first, freeze its CPU anchors once, and then develop a
> generic fp128 Metal evaluation/opening backend primarily in the Akita fork. The
> hard acceptance bar is at least 5x complete `AkitaNativeBatching::prove_batch`
> speedup over CPU for both Jolt `T=2^25` trace shapes and the `T=2^28` shape, with
> no fallback, verified proofs, unchanged canonical protocol unless analytically
> necessary, preserved 5x commit performance, and at most 90 GiB max-scale RSS.
> Use an analysis-led, one-change loop: one focused parity test and one single-shot
> treatment per iteration; reuse frozen controls; rerun only ambiguous results;
> reserve max-scale and full Jolt runs for milestones and the final gate. Keep
> accepted work bisectable, keep experimental artifacts out of production, and do
> not push. Once the local 5x gate and production cleanup pass, report the actual
> integrated Jolt timings and a disjoint residual-time budget for the later
> end-to-end goal; do not broaden this goal into unrelated PIOP optimization. Use
> the `gpu-kernel-analysis`, `autoresearch-loop`, `experiment-design`,
> `result-validation`, and `coding-standards` skills, with this document's leaner
> gate taking precedence over any heavier historical harness.
