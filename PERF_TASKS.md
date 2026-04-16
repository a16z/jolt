# Perf Tasks

## Philosophy

Close the perf gap between the rewritten modular stack (`jolt-zkvm` +
`jolt-compiler` + `jolt-cpu` + friends) and legacy `jolt-core` WITHOUT
collapsing the ML-compiler abstraction. Handlers must remain ≤ 30 LOC and
protocol-unaware; performance work either (a) extends the `ComputeBackend`
trait with fused/specialized operations, (b) makes a runtime handler
faster while staying generic, or (c) teaches the compiler to emit tighter
op sequences. It does NOT reintroduce bespoke protocol-aware handlers.

## Loop Protocol (program counter)

Read CLAUDE.md "Perf Loop Protocol" for the full workflow. Summary: one
iteration per loop tick — measure, pick hypothesis, implement, correctness
gate, perf gate, commit (always), schedule next tick. Never exit except
when the Phase 3 stop condition fires.

## Current State

- **Phase**: 1 (small traces for fast iteration)
- **log_t**: 12 (overrides `max_trace_length` to 2^12; actual prover work
  is min(guest cycles padded, 2^12))
- **Program**: `muldiv` (only program supported on the modular stack today)
- **Stall counter**: 1
- **Last green iter**: 1 — P10 `segmented_reduce` parallelize+hoist
  (−72.24% prove_ms: 14607 → 4055 ms, ratio 41.4× → 11.5×)
- **Green streak**: 0 (iter 2 P11 flat +0.5%, reverted)
- **Phase 3 stop condition**: `modular.prove_ms ≤ core.prove_ms` at
  `log_t ∈ {18, 20}`, 3 consecutive green iters. Only this exits the loop.

## Iter 1 Perfetto profile (log_t=12, muldiv, modular-only, 14575 ms)

| Rank | Span | Self ms | % | Calls | Avg µs |
|-----:|------|--------:|---:|------:|-------:|
| 1 | `reduce_dense` | 12717.6 | **78.3%** | 83 428 | 152.4 |
| 2 | `InstanceSegmentedReduce` subtree | 12517.2 | 77.1% | 20 | 625 860 |
| 3 | `Program::build_with_features` | 682.4 | 4.2% | 4 | (guest build) |
| 4 | `G1::msm` | 405.1 | 2.5% | 5 402 | 75.0 |
| 5 | `multi_pair_g2_setup_parallel` | 370.8 | 2.3% | 112 | 3 650 |
| 6 | `Materialize` | 325.1 | 2.0% | 197 | 1 665 |

Trace: `benchmark-runs/perfetto_traces/muldiv_log_t12_iter1.json`.
Command: `cargo run --release -p jolt-bench -- --program muldiv --log-t 12 --stack modular --trace-chrome muldiv_log_t12_iter1`.

**Smoking gun**: `InstanceSegmentedReduce` loops serially over non-zero
`outer_eq` positions and issues one small `self.reduce(...)` (→
`reduce_dense`) per position. For this trace that's ~210 inner reduces
per outer call × 20 outer calls ≈ 4 200 reduces. Each reduce is below
`PAR_THRESHOLD = 1024` so the inner serial path is taken; the outer
loop itself is also serial. Net: single-core, all 83 K reduces executed
sequentially, with per-iteration `data.clone()` wrapping of inputs into
`DeviceBuffer::Field(...)` before each call.

## Phase Graduation

| Phase | log_t | Graduate when... |
|---|---|---|
| 1 | 12 | median(mod/core) ≤ 1.5× across 3 green iters |
| 2 | 14 | median(mod/core) ≤ 1.2× across 3 green iters |
| 3 | 18, then 20 | mod ≤ core on both, 3 green in a row → **STOP** |

## Gates (run on every iteration before accepting a change)

```bash
# Correctness (bench is useless if these fail)
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
cargo nextest run -p jolt-equivalence zkvm_proof_accepted --cargo-quiet
cargo nextest run -p jolt-equivalence --cargo-quiet

# Lint (both feature sets per CLAUDE.md)
cargo clippy -p jolt-compiler -p jolt-compute -p jolt-cpu -p jolt-zkvm -p jolt-dory -p jolt-openings -p jolt-verifier -p jolt-bench --message-format=short -q --all-targets -- -D warnings
```

## Measurement (1 iter for fast loop cadence)

```bash
cargo run --release -p jolt-bench -- \
    --program muldiv --iters 1 --warmup 1 --log-t 12 \
    --json perf/last-iter.json
```

Thresholds for the perf gate (vs `perf/baseline-modular-best.json`):
- **Accept** improvement: ≥5% faster
- **Reject** regression: ≥5% slower
- **Inconclusive band** (±5%): one confirmation rerun; still in band → flat

## Commit discipline

One commit per iteration, always:
- **Improvement**: `perf(<scope>): P<n> <name> (-X% prove_ms on <program> @ log_T=<n>)`
  — code + `PERF_TASKS.md` + `perf/baseline-modular-best.json` + `perf/history.jsonl`
- **Flat / reverted**: `journal: P<n> reverted (<reason>)` — only bookkeeping files

## Hypothesis queue

Each entry: `- [ ] P<n>: <name> — target: <file:line>` followed by:
- **Hypothesis**: what changes and why it should be faster
- **Abstraction risk**: low / med / high and why
- **Expected delta**: rough guess

Seeded from Explore agent findings (ranked by expected delta × low risk).

- [ ] P1: Parallel prefix-mle mask buffer construction — target: `crates/jolt-zkvm/src/runtime/prefix_suffix.rs:59-80`
    - **Hypothesis**: outer rule loop + mask-bit iteration is sequential; wrap in `rayon::par_iter` with per-thread local HashMaps merged at end. Instruction-lookup read-checking is 20-30% of total prover time.
    - **Abstraction risk**: low — belongs in runtime, stays protocol-agnostic.
    - **Expected delta**: 15-30%
    - **Status**: deprioritized after iter 1 profile. `prefix_suffix` does
      NOT show up in the top spans; it's not a current bottleneck at
      log_t=12. Revisit if it surfaces at higher log_t.

- [ ] P2: Fused reduce+bind kernel on ComputeBackend — target: `crates/jolt-zkvm/src/runtime/handlers.rs:629-652` (InstanceReduce) + `:534` (interpolate_evaluate in BatchRoundBegin)
    - **Hypothesis**: InstanceReduce → next-round interpolate_evaluate is effectively a fused bind+eval done sequentially. Add `ComputeBackend::reduce_and_bind(kernel, inputs, challenges, scalar) -> Vec<F>` with default-impl fallback; CPU backend specializes to avoid intermediate roundtrip.
    - **Abstraction risk**: low — extends trait with defaulted method, handlers stay generic.
    - **Expected delta**: 8-15%

- [ ] P3: Batch-reduce fusion at Materialize boundaries — target: `crates/jolt-zkvm/src/runtime/handlers.rs:553-580` + `crates/jolt-zkvm/src/runtime/helpers.rs:84-150`
    - **Hypothesis**: Each `Op::Materialize` calls `backend.upload()` then kernel evaluation separately — for ephemeral tables (EqTable, LtTable, EqProject) this forces an intermediate clone. Add `ComputeBackend::materialize_and_use(binding, fn)` callback that fuses the two.
    - **Abstraction risk**: low — callback closure stays protocol-agnostic.
    - **Expected delta**: 10-20%

- [ ] P4: Lazy instance eval buffer allocation in BatchRoundBegin — target: `crates/jolt-zkvm/src/runtime/handlers.rs:539`
    - **Hypothesis**: `state.last_round_instance_evals = vec![Vec::new(); num_instances]` unconditionally allocates per-instance Vec every batch round, even when the instance immediately becomes inactive. Switch to SmallVec or sparse HashMap keyed by active instance indices.
    - **Abstraction risk**: minimal — purely data-structure swap.
    - **Expected delta**: 5-8%

- [ ] P5: Rayon scheduling across independent Materialize phases — target: `crates/jolt-zkvm/src/runtime/mod.rs:111-160`
    - **Hypothesis**: Schedule walk is single-threaded; multiple independent Materialize ops in one round serialize. Compiler annotates ready sets; runtime parallelizes with thread-safe device_buffer indexing.
    - **Abstraction risk**: medium — needs dependency annotation in compiler, careful sync.
    - **Expected delta**: 8-20%

- [ ] P6: Const-generic specialization for batch instance counts — target: `crates/jolt-zkvm/src/runtime/handlers.rs:520-540`
    - **Hypothesis**: Most batches have 1-4 instances; runtime Vec indexing is dynamic. Bake instance counts as const metadata on CompiledKernel.
    - **Abstraction risk**: medium — requires compiler metadata, trait changes.
    - **Expected delta**: 5-12%

- [ ] P7: Avoid clone of challenge vector in fused_rlc_reduce — target: `crates/jolt-zkvm/src/runtime/helpers.rs:244-246` + `:230-240`
    - **Hypothesis**: `successors().take().collect()` per call allocates fresh `powers` vec; intermediate Cow<'_> vec materialized even on padded-poly hot path.
    - **Abstraction risk**: minimal.
    - **Expected delta**: 3-5%

- [ ] P8: Evaluate op — eliminate interpolate download roundtrip — target: `crates/jolt-zkvm/src/runtime/handlers.rs:70-102`
    - **Hypothesis**: Downloads 1-2 element buffer just to interpolate a scalar. Add `ComputeBackend::evaluate_and_interpolate(buf, scalar) -> F`.
    - **Abstraction risk**: low.
    - **Expected delta**: 2-4%

- [ ] P9: Constant-fold Toom-Cook grid selection — target: `crates/jolt-cpu/src/product_sum.rs:18-46`
    - **Hypothesis**: D/P dispatch at every kernel compilation; bake into kernel spec at schedule generation when statically known.
    - **Abstraction risk**: very low.
    - **Expected delta**: 1-3%

## Notes

Design decisions, dead ends, and stall-mode observations accumulate here.

- **Iter 2 — P11 `reduce_dense` slice-borrow + Dense fast path in
  `segmented_reduce`** (target: `crates/jolt-cpu/src/backend.rs`).
  Result: flat (two runs +0.36%, +0.64% vs 4054.58 ms baseline). Reverted.
  Hypothesis was that the per-iter `.to_vec()` wrapping of outer-indexed
  inputs into `DeviceBuffer::Field(...)` inside `segmented_reduce` was
  eating the remaining ~9% self-time shown in the iter-2 Perfetto trace.
  Changing `reduce_dense{,_fixed,_dynamic}` to take `&[&[F]]` and adding
  a Dense fast path that passes slice refs directly was correct and
  compiled cleanly, but the speedup was swamped — either the `to_vec`
  was already hoisted out of the hot inner by rayon's chunking after
  P10, or reduce_dense's own work (74% self-time) dominates such that
  shaving per-call overhead is lost in the noise. Revisit only if a
  future profile shows the Dense wrapping cost resurfacing after other
  wins bring `reduce_dense` self-time down.

## Done

- **Iter 1 — P10: Parallelize `CpuBackend::segmented_reduce` + hoist
  inner-only clones + eliminate double-copy** (target:
  `crates/jolt-cpu/src/backend.rs:520-562`).
  - Result: prove_ms 14606.88 → 4054.58 on muldiv @ log_t=12 (**−72.24%**).
  - Ratio vs core: 41.4× → 11.5×.
  - Perfetto showed `InstanceSegmentedReduce` subtree at 77.1% self-time,
    driven by a serial outer loop of ~4,200 small reduces/iter with
    per-iter `data.clone()` wrapping of inputs into `DeviceBuffer::Field(...)`.
    Fix parallelizes the non-zero-weight outer positions via rayon,
    clones each inner-only input ONCE up front, and swaps the
    double-copy (`copy_from_slice` + `.clone()`) for a single
    `slice.to_vec()` on outer-indexed inputs.
  - Abstraction risk: low — all changes inside CpuBackend; trait unchanged;
    handler untouched.
