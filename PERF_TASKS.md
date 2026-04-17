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
- **Stall counter**: 4 (iter 26 P35 parallel-over-groups fused_rlc_reduce flat/reverted; iter 25 P33 parallel inner rlc_combine flat/reverted; iter 24 P32 parallel Materialize flat/reverted; iter 23 instrumentation-only; iter 22 instrumentation-only; iter 21 flat; iter 20 green; iter 19 green; iter 18 reverted; iter 17 profiling-only; iter 16 infra; iter 15 infra; iter 14 infra; iter 13 flat; iter 12 green)
- **Last green iter**: 20 — parallelize `Op::Commit` outer loop via rayon.
  42 serial Dory `PCS::commit` calls (508 ms wall, 720 ms CPU, 1.4× effective
  parallelism) → parallel collect of commitments + serial transcript append
  (−22.6% prove_ms: 1867 → 1444 ms best, ratio 5.9× → 4.4×)
- **Green streak**: 3 (iter 2 P11 flat +0.5%; iter 3 P12 flat −2.9%; iter 4 P13 flat −1.9%; iter 5 P14 flat +1.8%; iter 6 P17 regressed +6.4%; iter 7 P18 flat −0.1%; iter 8 P19 flat +0.6%; iter 9 P16 flat −3.45%; iter 10 P20 flat +0.47%; iter 11 instrumentation-only; iter 12 P24 −9.24%; iter 13 P25 flat +0.10%; iter 14 Gruen infra primitive; iter 15 Gruen infra reduce; iter 16 Gruen infra variant; iter 17 post-P24 re-profile; iter 18 Gruen dispatch reverted; iter 19 Gruen end-to-end −49.2%; iter 20 parallel Op::Commit −22.6%; iter 21 P28 parallelize lt_evals + EqPlusOne::evals flat −1.7%; iter 22 instrumentation-only — per-stage CPU vs wall saturation; iter 23 instrumentation-only — per-op-class CPU vs wall saturation + explicit dory `parallel` feature; iter 24 P32 parallel Materialize outer dispatch flat −0.55%, reverted — nested rayon pessimization hypothesis; iter 25 P33 parallel inner rlc_combine flat −0.69%, reverted — only ~11% of ReduceOpenings time is inner-loop parallelizable; iter 26 P35 parallel-over-groups fused_rlc_reduce flat +1.19%, reverted — likely single dominant group at log_t=12 so outer par_iter adds overhead with no parallelism gain)
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

- [ ] P15: Devirtualize `kernel.evaluate` in `reduce_dense_fixed<_, 4, 4>` — target: `crates/jolt-cpu/src/backend.rs:699-750` + `crates/jolt-cpu/src/product_sum.rs:50-60`
    - **Hypothesis**: iter-5 instrumentation showed `reduce_dense_4x4` is
      70.9% self-time (81 933 calls / 197.5 µs avg). Every inner iter goes
      through `(self.eval_fn)(...)` on `Box<dyn Fn>`, blocking the compiler
      from inlining `eval_prod_4_assign` and hoisting loop-invariants.
      Bypass the Box for the known D=4 product-sum shape by holding a
      typed static function pointer (or detecting the shape in
      `CpuKernel::new` and caching an optional `eval_prod4_fn: fn(...)`)
      so reduce_dense_fixed<_,4,4> calls the static function directly.
    - **Abstraction risk**: low — internal to jolt-cpu, trait unchanged.
    - **Expected delta**: 5-12% (single biggest hot-path; gain grows with
      how much the vtable call blocks register scheduling and inlining).

<!-- P16 consumed iter 9; see Notes for result & diagnosis. -->
<!-- P19 consumed iter 8; see Notes for result & diagnosis. -->
<!-- P20 consumed iter 10 (first attempt — fused loop only); see Notes. -->
<!-- P24 consumed iter 12 GREEN (−9.24%); see Notes. -->

- [ ] P25: Extend P24 parallelism unlock to small-chunk-bandwidth kernels that
  are NOT in reduce_dense_fixed's const-generic list — target: add
  const-generic shapes like (5,3), (6,3), (7,4), (9,4), (10,4), (4,5), (5,5)
  covering booleanity+hamming+ram_ra_virtual+inst_ra_virtual+inc/output. This
  moves those from `reduce_dense_dynamic` (heap scratch, indirection) to
  `reduce_dense_fixed` (stack scratch, pointer-array). P24 proved parallelism
  was stuck; now unlock per-iter ILP.
    - **Hypothesis**: dynamic path costs ~10-20% more per iter than fixed
      due to heap Vec scratch + indirect &[&Vec<F>] access. For booleanity
      (NI~10), that's ~40ms saved on stage 5 alone.
    - **Abstraction risk**: low — internal to reduce_dense shape dispatch.
    - **Expected delta**: 2-5%.

- [ ] P21: True Gruen split-eq (tensor decomposition of outer×inner eq) in
  segmented sumcheck — target: `crates/jolt-zkvm/src/runtime/handlers.rs`
  (InstanceSegmentedReduce handler pipeline) + new
  `ComputeBackend::gruen_segmented_reduce` method + possibly a new compiler
  `Op::SegmentedReduceGruen`. Cross-reference:
  `jolt-core/src/zkvm/ram/read_write_checking.rs:256,309-388` (core's
  implementation of the same technique).
    - **Hypothesis**: iter 10 P20 (fused loop, no decomposition) landed
      flat (+0.47%); confirms the structure of segmented_reduce is NOT
      the bottleneck. The real gap is arithmetic intensity per round:
      core emits ONE cubic polynomial per round via `gruen_poly_deg_3`
      using E_out_vec (prefix) × E_in_vec (suffix), folding outer eq
      as a prefix weight multiplied ONCE outside the inner sweep.
      Modular currently computes Toom-Cook points per (outer, inner)
      pair (4 F::mul per inner, D=4), then eq-weights per outer. If
      outer eq is reduced to a prefix-weight scalar across all active
      outer positions at the beginning of each round, the inner sweep
      runs over a SINGLE virtual polynomial — collapsing 4× D field
      mul / outer / inner → 4× field mul / inner (one per Toom-Cook
      point), a ~N_active_outer×D reduction on the inner hot path.
    - **Abstraction risk**: high — requires the compiler to recognize
      segmented-eq patterns (outer × inner tensor) and lower to a
      Gruen-style op; backend needs a specialized primitive; handler
      stays ≤ 30 LOC but the compile-time contract widens.
    - **Expected delta**: 30-50% wall-time on segmented rounds; these
      are ~51% of prove wall, so net ~15-25%. If the compiler lowering
      is clean and the backend primitive well-optimized, this is the
      closest thing to a single structural fix for the remaining gap.

- [ ] P22: Devirtualize `kernel.evaluate` for the reduce_dense_4x4 hot
  path — already queued as P15 but bears re-examination: iter-5
  instrumentation showed `(NI=4, NE=4)` is 70.9% self-time / 81 933
  calls / 197.5 µs avg. The vtable call through `Box<dyn Fn>` in the
  innermost loop blocks inlining `eval_prod_4_assign` and prevents
  loop-invariant hoisting. Compared to P21 (structural): lower expected
  delta (5-12%) but smaller scope and lower abstraction risk. Consider
  as a fallback if P21 takes > 2 iters to land.

- [ ] P29: Parallelize Stage 7 Dory Open op-dispatch — target:
  `crates/jolt-zkvm/src/runtime/handlers.rs` (Op::Open handler) + same
  compiler pattern as iter 20's Op::Commit parallel fan-out.
    - **Hypothesis**: iter 22 saturation instrumentation showed Stage 7
      (23.1% wall / 364 ms) runs at 1.63 cores / 20.4% saturation — the
      Dory open proof. Iter 20 landed a −22.6% win on Op::Commit by
      dispatching independent commits in parallel outer rayon while each
      commit's internal tier-1/tier-2 MSMs still ran internally parallel.
      Open follows the same pattern: the external `dory::prove` call is
      serial-dominated internally, but the MULTIPLE open calls in stage 7
      (one per committed polynomial, ~40+ opens per prove) are
      independent. Fan them out like iter 20.
    - **Abstraction risk**: low — same pattern as P27/iter 20 (parallel
      over Vec of independent PCS ops, serial transcript append after).
      Need to confirm dory::prove's output doesn't feed transcript until
      collection.
    - **Expected delta**: 8-15% total wall (364 ms × (1.63/4-6 cores
      speedup) ≈ 150-250 ms saved; some Dory internal contention
      possible).

- [ ] P30: Parallelize Stage 1 Materialize op-dispatch — target:
  `crates/jolt-zkvm/src/runtime/mod.rs:175` (execute loop) + compiler
  annotation of independent-Materialize groups in stages.
    - **Hypothesis**: iter 22 saturation showed Stage 1 (35.1% wall /
      553 ms) already best-saturated at 3.38 cores (42%). Iter 20's
      Op::Commit parallelism lifted it from ~1.4× to 3.4×. Next lever:
      parallelize Materialize ops (~5 large calls in stage 1 per iter 21
      analysis, +Cloth P27 fan-out). If stage-1 Materializes are
      topologically independent (different PolynomialId outputs, no
      cross-consumption within the stage), batch-dispatch them via
      `par_iter().for_each(|op| run_materialize(...))` with a locked
      HashMap for output buffer insertion.
    - **Abstraction risk**: medium — needs compiler/schedule to annotate
      "independent op groups" or runtime to analyze dependencies at
      dispatch time. Same shape as P5 in the original queue but now
      data-grounded.
    - **Expected delta**: 8-15% total wall (push stage 1 3.4× → 5-6×
      → 553 × 3.4/5.5 = 342 ms = 211 ms saved ≈ 13% wall).

- [x] P31: Finer-grained op-class saturation instrumentation — DONE iter 23.
  See **Notes / Iter 23** for the full op-class breakdown. Key finding:
  Materialize family (Materialize + MaterializeUnlessFresh + ReadCheckingReduce)
  runs at ~1.0 threads totalling ~510 ms wall — near-serial — now the top
  parallelism opportunity.

- [ ] P32: Parallel Materialize-family op dispatch — target:
  `crates/jolt-zkvm/src/runtime/mod.rs` execute loop + compiler-side
  "independent op window" detection (same pattern as iter 20's parallel
  Op::Commit win).
    - **Hypothesis**: iter 23 op-class instrumentation showed
      `Op::Materialize` runs at **1.02 threads / 12.7% sat** across 197
      calls totalling **297 ms wall** (stage 1 + stage 3 dominated), and
      `Op::MaterializeUnlessFresh` at **1.00 threads / 12.5% sat** across
      46 calls totalling **121 ms wall**. Combined: **418 ms wall at
      ~1.0 threads**. The per-op internal backend work is already
      rayon-parallel where applicable (`eq_table` PAR_THRESHOLD gate), so
      the serial residue is either (a) small tables below PAR_THRESHOLD,
      or (b) the outer dispatch loop iterating materializes sequentially.
      Identifying topologically-independent Materialize ops in a stage
      and dispatching them via `into_par_iter` (same pattern as iter 20
      Op::Commit) should lift effective parallelism to 4-6 threads,
      saving ~200-280 ms wall.
    - **Abstraction risk**: medium — requires either (a) compiler to
      annotate independent-op windows in Executable::ops, OR (b) runtime
      to build per-Materialize input/output dependency graph at dispatch
      time. `BufferProvider::Buf` must be Send (already is per iter 20's
      Op::Commit parallelization). Bigger concern: `Materialize` writes
      to `state.buffers: HashMap<BufferId, Buf>`, which needs Mutex for
      parallel insert — unless we switch to `Vec<Option<Buf>>` indexed
      by BufferId for lock-free writes. Handlers stay ≤30 LOC.
    - **Expected delta**: 10-18% wall (418 × (1.0/5) ≈ 84 ms remaining,
      saving ~334 ms ≈ 22%, discount for coordination overhead and the
      subset of Materializes that are already small enough that parallel
      dispatch overhead dominates).

- [x] P33: Parallelize single-call `ReduceOpenings` internals (iter 25,
  flat −0.69%, REVERTED). Only ~11% of the 93 ms was in the Horner
  inner loop; rest is in per-group materialize + combine_hints. See
  Notes.

- [x] P35: Parallelize `fused_rlc_reduce` across groups (iter 26, flat
  +1.19%, REVERTED). Likely single dominant group at log_t=12; outer
  par_iter adds overhead with no parallelism gain. See Notes.

- [ ] P36: Instrument fused_rlc_reduce group structure — target:
  `crates/jolt-zkvm/src/runtime/helpers.rs` `fused_rlc_reduce`.
    - **Hypothesis**: three consecutive parallelism attacks on
      Materialize/ReduceOpenings failed at log_t=12 despite iter 23's
      1.0-thread readings. Need to directly measure group count,
      per-group wall, and per-op materialize wall inside
      `fused_rlc_reduce` to confirm/deny the "single dominant group"
      hypothesis. Emit `tracing::info!` target `perf_rlc` with group
      count, max-group size, per-group (wall, poly_count, total_elems).
    - **Abstraction risk**: none — instrumentation only.
    - **Expected delta**: none; infra to steer next attack.

- [ ] P37: Re-profile at log_t=14 — target: `perf/` measurement only.
    - **Hypothesis**: parallelism opportunities visible in iter 23
      op-class (Materialize 510 ms / 1.0 threads) may be
      log_t-dependent. At log_t=12, polynomials are ≤4096 elements,
      below most rayon granularity thresholds. At log_t=14 (16x work
      per op), the same ops should either (a) rise proportionally in
      wall and remain at 1.0 threads → confirms real serial work
      worth parallelizing, or (b) stay ~constant or drop in wall
      proportion → confirms overhead-dominated reading at log_t=12.
    - **Abstraction risk**: none — measurement only.
    - **Expected delta**: none directly; infra to steer next attack.

- [ ] P34: Investigate Dory Open internal parallelism — target:
  `dory-pcs` crate (external, but we control the fork).
    - **Hypothesis**: iter 23 showed `Op::Open` at **1.87 threads / 23.4%
      sat** in 257 ms / 1 call. `parallel` feature IS enabled (confirmed
      via `cargo tree`, propagated through `cache`). So the internal
      parallelism gap is WITHIN dory — specific phases of `dory::prove`
      are serial (Fiat-Shamir transcript boundaries, non-MSM scalar
      arithmetic, fold steps). Profile `dory::prove` top-down via
      perfetto flamegraph to identify the serial residue, then (a) add
      rayon to identified hot loops upstream in dory-pcs repo, or (b) if
      protocol-bound, accept as a hard cap.
    - **Abstraction risk**: low-medium — upstream code change in dory-pcs
      is outside this repo but user owns the fork; runtime side stays
      untouched.
    - **Expected delta**: 3-8% wall (depends on how much of 257 ms is
      inherently serial). If Dory internal can push from 1.87 → 5 cores,
      save ~160 ms ≈ 10% wall.


## Notes

Design decisions, dead ends, and stall-mode observations accumulate here.

- **Iter 26 — P35 parallel-over-groups in fused_rlc_reduce (flat +1.19%, REVERTED)**
  (target: `crates/jolt-zkvm/src/runtime/helpers.rs` `fused_rlc_reduce`;
  `handlers.rs` dispatch_op signature; `mod.rs` execute; `prove.rs` prove).
  **Hypothesis**: iter 25 post-mortem suggested most of the 93 ms ReduceOpenings wall lives in per-group work (provider.materialize + rlc_combine + combine_hints). Rhos are drawn sequentially through the transcript, but once drawn each group is independent. Restructured fused_rlc_reduce: (1) serial first pass drawing all rhos, (2) `groups.into_par_iter().zip(rhos).map(...)` for per-group combine. Added `P: BufferProvider<F> + Sync` bound through dispatch_op, execute, prove. Expected 5-8% wall.
  **Change (~30 LOC)**: signature changes on fused_rlc_reduce + dispatch_op + execute + prove to thread the `P: BufferProvider<F> + Sync` generic; body restructured to two-pass pattern. Updated call-sites in 4 test files (jolt-zkvm e2e/muldiv_e2e, jolt-bench modular stack, jolt-equivalence muldiv) to add `, _` to the turbofish.
  **Gates**: 41/41 jolt-equivalence green (transcript_divergence + zkvm_proof_accepted + full suite); clippy clean across 8 modular crates + tests.
  **Perf (5 runs at log_t=12)**: cold 1459 / warm 1456.1, 1414.2, 1497.0, 1466.4 ms. Median of warm runs (2-5) = 1461.26 ms vs ratchet 1444.14 = **+1.19%**, inside ±5% band → flat → revert.
  **Diagnosis (why flat/slight regression)**: most likely a single dominant group (or very few groups) at log_t=12. Rationale:
  - Ops with different evaluation points end up in different groups, but the modular prover at log_t=12 has ~40 opening claims that largely evaluate at the SAME point (the sumcheck challenge vector for that stage), so they collapse into a single big group.
  - With 1 dominant group, `into_par_iter` over a length-1 Vec adds rayon task setup overhead (~10s of microseconds) while only one rayon worker does all the work.
  - Combined with the viral `+ Sync` bound (trivial overhead but still), the net is a slight regression.
  **Lessons for next iter**: (1) we've now failed three consecutive parallelism attacks at log_t=12 (P32, P33, P35) despite iter 23's op-class data showing 1.0 threads on 510 ms of Materialize work and 93 ms of ReduceOpenings. The pattern is suspicious — **the 1.0-threads readings may be structurally accurate but the workloads at log_t=12 are too small to reward rayon overhead**. (2) Two complementary moves: (a) **iter 27 INSTRUMENTATION**: add a log of group count + per-group materialize time + per-group rlc_combine time inside fused_rlc_reduce so we can confirm whether there's even distributable work; (b) **iter 28 log_t=14**: re-measure at phase-2 log_t to see if the parallelism opportunities we've been trying to exploit actually scale. If iter 23's 510 ms / 1.0 threads reading holds at log_t=14 with proportionally more work per op, then parallel attacks should start working. If the reading changes shape at log_t=14 (different ops rise to the top), we pivot. log_t=14 is still well under the 1-minute bench limit; total prove time should grow ~4-8×.

- **Iter 25 — P33 parallel inner rlc_combine (flat −0.69%, REVERTED)**
  (target: `crates/jolt-openings/src/reduction.rs` `rlc_combine`).
  **Hypothesis**: iter 23 op-class showed `ReduceOpenings` 93 ms / 1.00 threads / 12.5% sat / 1 call. Internal hot path is `fused_rlc_reduce`'s per-group Horner combine `result[i] = *r * rho + val` over full polynomial slices. Outer Horner loop is sequential (data-dependent accumulator) but inner per-position update is trivially data-parallel. Added `par_iter_mut().zip(p.par_iter()).for_each(...)` with `PAR_THRESHOLD=1024`. Expected 5-6% wall assuming the 93 ms is dominated by inner-loop arithmetic.
  **Change (~12 LOC + rayon dep)**: added `rayon` to jolt-openings deps; introduced `PAR_THRESHOLD=1024` constant; branched `rlc_combine` on `len >= PAR_THRESHOLD` to switch between serial and parallel inner loop.
  **Gates**: 41/41 jolt-equivalence green (transcript_divergence + zkvm_proof_accepted + full suite); clippy clean across 8 modular crates.
  **Perf (5 runs at log_t=12)**: cold 1482 / warm 1446.6, 1397.1, 1435.1, 1433.4 ms. Median of warm runs (2-5) = 1434.24 ms vs ratchet 1444.14 = **−0.69%**, inside ±5% band → flat → revert.
  **Diagnosis (why flat)**: 10 ms wall savings on a 93 ms op = ~11% parallelized. The other 82 ms inside `ReduceOpenings`/`fused_rlc_reduce` is NOT in the inner Horner loop. Most likely:
  - **Per-group `provider.materialize(pi)`** calls inside the group loop — these do the same internal eq/lt/eq_plus_one table builds as standalone `Op::Materialize` (which iter 23 showed at 1.02 threads). Iter 24's attempt to parallelize those via outer-dispatch batching failed due to nested-rayon pessimization; inside `fused_rlc_reduce` they stay serial.
  - **Polynomial sizes** — at log_t=12 many committed polys are well below 2^12 elements. `PAR_THRESHOLD=1024` gate means the inner parallel branch fires only for larger polys (eq/lt tables, advice polys). Small polys stay serial with zero gain.
  - **`combine_hints` call** does an MSM-like combine of PCS hints — this is NOT inside `rlc_combine`. Likely a non-trivial chunk of the 93 ms.
  **Lessons for next iter**: (1) inner-loop attack was the wrong leverage point; real savings are in per-group materialize + combine_hints, not the RLC horner itself; (2) **P35 candidate**: parallelize the group loop in `fused_rlc_reduce` AFTER drawing all rhos sequentially. Draw rhos up front (serial — transcript-ordered), then `.into_par_iter()` over (group, rho) pairs to do materialize + rlc_combine + combine_hints in parallel. All writes go into independently-allocated `Vec`s → no shared mutation. Expected 5-8% wall if group count ≥ 3 and per-group work ≥ 5 ms.

- **Iter 24 — P32 parallel Materialize outer dispatch (flat −0.55%, REVERTED)**
  (target: `crates/jolt-zkvm/src/runtime/mod.rs` main dispatch loop;
  `crates/jolt-zkvm/src/prove.rs` generic signature).
  **Hypothesis**: iter 23 showed Materialize family (Materialize + MaterializeUnlessFresh + ReadCheckingReduce) ≈ 510 ms wall at ~1.0 threads / ~13% sat. Same shape as iter 20's Op::Commit win (1.4× → 6.7× parallelism). Expected 10-18% wall by parallel-dispatching consecutive Materialize ops in a batch via `rayon::into_par_iter`.
  **Change**: converted dispatch for-loop to while-loop that greedily collects consecutive `Op::Materialize` into a batch, then `par_iter().map(|(bind, chs)| materialize_binding(...)).collect()` then serial `provider.insert_binding(buffer)`. Required changing `execute` generic to `P: BufferProvider<F> + Sync` and propagating `+ Sync` bound through `prove.rs` (viral bound). Verified `materialize_binding` pure (`&self` provider read, no shared mutation).
  **Gates**: 41/41 jolt-equivalence green (transcript_divergence + zkvm_proof_accepted + full suite); clippy clean after adding Sync bound.
  **Perf (5 runs at log_t=12)**: cold 1866, then 1548, 1455, 1414, 1418 ms. Median of warm runs (2-5) = 1436 ms = **−0.55% vs ratchet 1444.14**, inside ±5% band → flat.
  **Post-change op-class**: Materialize 284 ms / 1.10 threads / 13.8% sat (was 297/1.02/12.7%). Structural batching confirmed: **24 batches formed, 197 Materialize calls**, with 3 large batches of sizes 33, 41, 81 covering **155/197 calls (79%)**. Despite batching 79% of the calls, effective parallelism only moved from 1.02 → 1.10 threads.
  **Why flat (hypotheses)**:
  - **Nested rayon pessimization**: `materialize_binding` internals (`eq_table`, `lt_table`) already use rayon with `PAR_THRESHOLD=1024`. Outer `par_iter` forks into a pool where inner `par_iter` calls contend for the same work-stealing queue. For small Materialize ops (avg ~1.5 ms), synchronization overhead ≥ useful work.
  - **Allocator lock contention**: each Materialize allocates a large `Vec<F>`. Concurrent `Vec<F>::with_capacity` across 33-81 threads hits the system allocator (glibc malloc arena contention) more than SIMD adds work per thread.
  - **Individual work too small**: Materialize avg 1.5 ms — below rayon's efficient-granularity threshold of ~1 ms. Overhead dominates.
  **Per protocol**: flat (±5% band) → revert. Reverted `crates/jolt-zkvm/src/runtime/mod.rs` + `crates/jolt-zkvm/src/prove.rs`. Bookkeeping-only commit.
  **Lessons for next iter**: (1) when outer dispatch work is already internally rayon-parallel, nested parallelism needs explicit `join` not `par_iter` to avoid queue contention; (2) P33 (ReduceOpenings 93 ms / single call / simple RLC fold) is lower-risk — no nested rayon, single large op to split.

- **Iter 23 — Instrumentation: per-op-class CPU-vs-wall saturation + explicit dory `parallel` feature (infra, no perf claim)**
  (target: `crates/jolt-zkvm/src/runtime/mod.rs` op_class_tag + accumulator;
  `Cargo.toml` workspace dep `dory` features).
  **Motivation**: iter 22 revealed per-stage saturation but stages pack
  heterogeneous ops (Materialize + Commit + Bind + … in stage 1); the next
  attack needs op-class attribution. User also flagged "make sure we are
  propagating the parallel feature for dory!" — on inspection the feature
  WAS transitively enabled via dory-pcs's `cache = ["arkworks", "parallel"]`,
  confirmed via `cargo tree -p jolt-dory -e features | grep parallel`. Made
  the feature flag explicit in the workspace dory dep so future dependency
  pruning can't accidentally disable it.
  **Change (~50 LOC)**: added `op_class_tag(op: &Op) -> Option<&'static str>`
  helper mapping ~35 Op variants to short static tags (microsecond
  orchestration ops return None to avoid getrusage overhead dominating).
  Wrapped each `dispatch_op` call in a `(wall, cpu)` pair and aggregated by
  tag into `HashMap<&'static str, (Duration, Duration, u64)>`. Emitted
  sorted `tracing::info!` events on target `perf_op` at end of execute.
  **Gates**: 41/41 jolt-equivalence green (transcript_divergence + zkvm_proof_accepted); clippy clean. **Perf (3 runs at log_t=12)**: 1457, 1431, 1516 ms;
  median 1457 (+0.9% vs ratchet 1444.14, inside ±5% band). **Ratchet not
  updated** (instrumentation-only). **Stall counter NOT incremented** per
  iter 11/17/22 precedent.

  **Findings (8-core machine, log_t=12, muldiv, modular, 1504 ms wall run)**:

  | Rank | Op | Wall (ms) | CPU (ms) | Calls | threads_avg | Sat % |
  |---:|---|---:|---:|---:|---:|---:|
  | 1 | **Materialize** | **297** | 302 | 197 | **1.02** | **12.7%** |
  | 2 | **Open** | **257** | 482 | 1 | **1.87** | **23.4%** |
  | 3 | InstanceReduce | 248 | 679 | 217 | 2.74 | 34.3% |
  | 4 | InstanceSegmentedReduce | 203 | 1379 | 20 | 6.80 | 85.0% |
  | 5 | InstanceBind | 128 | 656 | 216 | 5.14 | 64.2% |
  | 6 | **MaterializeUnlessFresh** | **121** | 121 | 46 | **1.00** | **12.5%** |
  | 7 | **ReduceOpenings** | **93** | 93 | 1 | **1.00** | **12.5%** |
  | 8 | **ReadCheckingReduce** | **92** | 97 | 128 | **1.06** | **13.3%** |
  | 9 | Commit | 49 | 326 | 3 | 6.66 | 83.3% |
  | rest | (≤17 ms each) | — | — | — | — | — |

  **Key takeaways (rank by leverage × tractability)**:

  1. **Materialize family ≈ 510 ms wall at ~1.0 threads** (Materialize 297 +
     MaterializeUnlessFresh 121 + ReadCheckingReduce 92). This is the new
     #1 parallelism opportunity — single-threaded despite 7 idle cores.
     Per-op internal work is already rayon-parallel where applicable
     (`eq_table` has a PAR_THRESHOLD=1024 gate); the serial residue is
     in the outer dispatch loop. **Attack**: parallel outer dispatch of
     topologically-independent Materialize ops (same pattern as iter 20
     Op::Commit which lifted commit from 1.4× → 6.7×). **P32 added**:
     expected 10-18% wall. Trace: `muldiv_log_t12_iter23_op_class.json`.
  2. **Dory Open 257 ms at 1.87 threads / 23.4% sat**, 1 call. The
     `parallel` feature IS enabled for dory-pcs (confirmed via cargo
     tree), so the remaining gap is inside dory. Protocol has
     Fiat-Shamir transcript boundaries that are inherently serial, but
     the MSM-heavy phases (setup_parallel, fold steps) can fan out
     further. **P34 added**: profile dory::prove top-down, push internal
     rayon; expected 3-8% wall.
  3. **ReduceOpenings 93 ms single-call at 1.0 threads**. Low hanging
     fruit — a single big RLC of ~40 opening claims. Fold+reduce via
     rayon is trivial. **P33 added**: expected 5-6% wall.
  4. **InstanceSegmentedReduce at 6.80 threads / 85% sat** (203 ms) is
     already excellent — the P24/P26 Gruen kernel work made this work
     fully parallel internally. Don't touch.
  5. **Commit at 6.66 threads / 83.3%** — iter 20's P27 parallel
     outer-commit win holding strong. Don't touch.
  6. **InstanceBind at 5.14 threads / 64.2%** (128 ms). Moderate
     saturation but moderate wall; worth looking at later if Materialize
     fam + Dory-open closed.

  **Why `parallel` was already transitively enabled**: dory-pcs v0.3.0
  defines `cache = ["arkworks", "parallel"]`, so including `cache`
  (which we need for dory's setup cache) pulls `parallel` along. Added
  explicit `parallel` to the feature list anyway as defense-in-depth —
  if someone later removes `cache`, parallelism survives.

  **Hypothesis queue updates**: P31 marked DONE. P32 (parallel
  Materialize dispatch) is the top iter-24 candidate. P33 (parallel
  ReduceOpenings) and P34 (dory internal parallel profile) queued as
  follow-ups.

- **Iter 22 — Instrumentation: per-stage CPU-vs-wall saturation (infra, no perf claim)**
  (target: `crates/jolt-zkvm/src/runtime/{cpu_clock.rs,mod.rs}`, +libc dep).
  **Motivation**: user guidance — "instrument further to gather more info about
  parallelism saturation and stage level differences". Iter 21's flat −1.7%
  outcome + the modular stack still being 4.4× off core (1444 vs 330 ms)
  suggested a need for data-grounded hypotheses instead of more hand-picked
  targets. **Change**: 5-line `process_cpu_time()` helper wrapping
  `getrusage(RUSAGE_SELF)` on Unix (user + system time across threads); ~40
  LOC in `runtime/mod.rs` tracking `(wall, cpu)` per stage via the existing
  `BeginStage` op and emitting a `tracing::info!` summary on target
  `perf_stage` at end of `execute`. Overhead: 2 getrusage calls per stage
  (~0.5 µs each) × 8 stages = negligible. **Gates**: 41/41 jolt-equivalence
  green (transcript_divergence + zkvm_proof_accepted included), clippy clean
  across all 8 modular crates. **Perf (5 runs at log_t=12)**: 1450, 1605,
  1724, 1509, 1468 ms; median 1509 ms (+4.5% vs ratchet 1444.14, inside
  ±5% band). Core prove varied 329–345 ms in the same set (~5% ambient
  noise), confirming the measurement is in the noise envelope. **Ratchet
  not updated** (no improvement). **Stall counter NOT incremented**
  (instrumentation-only iter per iter 11/17 precedent).

  **Findings (run with `RUST_LOG=perf_stage=info --trace-chrome`)**:
  8-core machine, modular_prove total 1575 ms wall / 4258 ms CPU =
  **2.70 threads avg / 33.8% saturation**. Per-stage:

  | Stage | Wall (ms) | CPU (ms) | threads_avg | Sat % | Wall share |
  |------:|----------:|---------:|------------:|------:|-----------:|
  |   0   |        8  |      19  |       2.32  |  29.0 |      0.5%  |
  | **1** |  **553**  | **1872** |    **3.38** |  42.3 |    **35.1%** |
  |   2   |        7  |       7  |       1.11  |  13.8 |      0.4%  |
  | **3** |  **181**  |   481    |     2.66    |  33.3 |    **11.5%** |
  |   4   |      130  |     153  |    **1.17** | **14.6** |    8.3%  |
  | **5** |  **270**  |   777    |     2.88    |  36.0 |    **17.1%** |
  |   6   |       12  |      13  |       1.14  |  14.3 |      0.7%  |
  | **7** |  **364**  |   594    |    **1.63** | **20.4** |   **23.1%** |

  **Key takeaways (rank by leverage × tractability)**:

  1. **Stage 7 (364 ms / 23.1% wall) runs at 1.63 cores / 20.4% sat.** This
     is Dory open, consistent with iter 20's 258 ms/16.5% observation.
     External `dory::prove` call is mostly serial. **Attack**: parallel-map
     multiple opens like iter 20 did for commit — if opens are independent
     (per PolynomialId/batch), a parallel outer loop gives ~2-3× wall
     reduction → 150-250 ms saved = **9-15% total wall**.
  2. **Stage 1 (553 ms / 35.1% wall) runs at 3.38 cores / 42.3% sat.**
     Biggest wall share; already best-saturated but still 4.6 idle cores.
     Stage 1 = Commit + Materialize (per iter 20). Iter 20's parallel commit
     lifted this from 1.4× → 3.4×. Next lever: parallelize the Materialize
     ops across the stage, similar to iter 20's pattern. If we can push to
     6 cores avg, stage 1 drops 553 → 310 ms = **~15% total wall**.
  3. **Stage 4 (130 ms / 8.3% wall) runs at 1.17 cores / 14.6% sat.** Nearly
     serial. Probably Spartan inner/outer sumcheck claim reductions — small
     work units, serial logic. Medium leverage (~4-6% potential) but low
     risk, worth investigation.
  4. **Aggregate parallelism gap**: 8 − 2.70 = 5.3 cores idle on average.
     If all 8 stages ran at 6 cores avg, total wall would drop
     1575 × 2.70/6 = 709 ms (~55% reduction). That's the theoretical
     ceiling — realistic target ~4 cores avg = 1050 ms (~33% reduction).

  **Hypothesis queue update**: add P29 (Stage 7 parallel Dory open),
  P30 (Stage 1 parallel Materialize), P31 (Stage 4 parallelism
  investigation via finer-grained op-class instrumentation).

- **Iter 21 — P28 parallelize `lt_evals` + `EqPlusOnePolynomial::evals` (REVERTED, flat −1.7%)**
  (targets: `crates/jolt-poly/src/lt.rs:144-155`, `crates/jolt-poly/src/eq_plus_one.rs:71-118`).
  **Design step**: post-iter-20 Perfetto trace
  (`benchmark-runs/perfetto_traces/muldiv_log_t12_iter21_post_parallel_commit.json`,
  wall 1560 ms) showed "Materialize family" at 408 ms / 26.2% of wall — driven
  by 5 giant eq/lt/eq_plus_one table constructions in stages 1+3 totalling
  ~380 ms. `CpuBackend::eq_table` ALREADY has a parallel path (eq.rs
  `PAR_THRESHOLD=1024` gate). `CpuBackend::lt_table` delegates to
  `LtPolynomial::evaluations` → `lt_evals` which is sequential.
  `CpuBackend::eq_plus_one_table` delegates to `EqPlusOnePolynomial::evals`
  which is also sequential. The doubling pattern in both is cleanly
  parallelizable over disjoint index pairs. Hypothesis: parallelizing both
  unlocks the 5-10× sequential bottleneck on the late rounds (where most
  work concentrates) for ~15-20% wall-time savings on the five hot calls.
  **Implementation**: (a) `lt_evals` — gate `left.iter_mut().zip(right.iter_mut())`
  on `left.len() >= PAR_THRESHOLD=1024`, swap to `par_iter_mut().zip(par_iter_mut())`.
  (b) `EqPlusOnePolynomial::evals` — fused the two while-loops into one per-chunk
  pass (within each `step`-sized chunk at `base = chunk_idx * step`, compute
  `epo[base+half_step] = eq[base] * r_lower_product`, then update `eq[base]`
  and `eq[base+half_step]` from the captured `eq[base]` value). Gate on
  `num_chunks >= PAR_THRESHOLD=1024` via `par_chunks_mut(step)`.
  Both use `crate::polynomial::PAR_THRESHOLD`. **Gates**: 160/160 jolt-poly
  tests green, 41/41 jolt-equivalence green (transcript_divergence +
  zkvm_proof_accepted included), clippy clean across all 8 modular crates +
  jolt-poly + jolt-core (host, host+zk). **Perf**: 2 runs vs ratchet 1444.14 ms —
  `iter21-run1.json` 1397.99 ms (−3.19%, below 5% accept threshold);
  `iter21-run2.json` 1441.28 ms (−0.20%, flat). Both runs in ±5%
  inconclusive band; per protocol → **revert as flat**. **Diagnosis**:
  signal is real and directional (both runs below baseline, avg −1.7%),
  but magnitude insufficient to clear 5% acceptance threshold. Likely causes:
  (a) actual table sizes are smaller than the 2^20+ the trace's cumulative
  wall suggested — at log_t=12, most eq_plus_one/lt tables are ≤2^12, so
  only 2 rounds of 12 cross the PAR_THRESHOLD=1024 gate → parallelism
  covers only ~75% of work per call. (b) rayon fork/join overhead for
  `par_chunks_mut(step)` at small step (step=2, 4 in late rounds) may
  consume a large fraction of the nominal work per item. (c) the 380 ms
  attribution in the iter-21-post-commit trace likely included materialize
  overhead (alloc_zeroed for 2^20-ish tables, memset, other ops) beyond
  just the arithmetic inner loops my change targets. **Takeaway for iter 22**:
  materialize bottleneck needs a different attack — either (i) reduce the
  NUMBER of materializes by caching cross-round, (ii) shrink their size
  (smaller inputs), (iii) parallelize more aggressively with raw pointers
  instead of par_chunks_mut to skip rayon chunk-splitter overhead, or
  (iv) pivot to a different bottleneck (Dory Open at 258 ms / 16.5% or
  kernel 21 InstanceReduce at 205 ms / 13.2%). Raw-pointer approach is
  the lowest-risk retry of this hypothesis.

- **Iter 20 — P27 parallelize `Op::Commit` outer loop via rayon (GREEN −22.6%)**
  (target: `crates/jolt-zkvm/src/runtime/handlers.rs:319-359`). **Design step**:
  fresh Perfetto trace on post-Gruen binary
  (`benchmark-runs/perfetto_traces/muldiv_log_t12_iter20_post_gruen.json`,
  wall 2296 ms) showed Commit at **22.13% wall** (508 ms) = the single largest
  remaining bottleneck after kernel 3's Gruen port. Internal trace breakdown:
  42 serial `DoryScheme::commit` calls × 12 ms avg; 720 ms CPU-time but only
  508 ms wall = **1.4× effective parallelism**. Each individual commit is
  already parallelized internally (tier-1 G1::msm chunks + tier-2 Pedersen via
  rayon), but the outer 42-iteration loop ran serially. Since each commit is
  independent (no transcript append during commit — that's deferred), the
  outer loop is trivially parallelizable. Change: ~18 LOC in handlers.rs —
  (a) seq-materialize polys (BufferProvider's `&mut` receiver prevents sharing;
  sequential Cow::Owned materialization is cheap — ~0.1 ms per poly at log_t=12),
  (b) `data.into_par_iter().map(|(pi, data)| PCS::commit(&data[..], pcs_setup))`
  for the parallel commit stage, (c) sequential transcript append in original
  order for Fiat-Shamir determinism. **Gates**: 41/41 jolt-equivalence green
  (transcript_divergence + zkvm_proof_accepted included), clippy clean across
  all 8 modular crates. **Perf**: 3 runs vs ratchet 1866.86 ms —
  `iter20-run1.json` (pre-change) 1875.99 ms (+0.49%, flat confirm baseline);
  `iter20-run2.json` 1444.14 ms (−22.62%); `iter20-run3.json` 1461.77 ms
  (−21.68%). Both post-change runs clear +5% accept threshold. Best 1444.14,
  median 1453 ms. **Ratchet**: `baseline-modular-best.json` updated to 1444.14 ms.
  Green streak 2 → 3. Ratio vs core: 5.90× → 4.38× (best). **Why this worked**:
  the existing inner rayon fan-out was saturating 1-2 cores per commit; adding
  outer parallelism across 8 cores × 8 commits-in-flight fills the work-steal
  queue depth, collapsing serial wall into ~cores-wide parallel wall. Memory
  cost: 42 × ~128KB (log_t=12) = ~5.4 MB transient alloc for Cow::Owned copies,
  negligible. **Next iter (21)**: post-Gruen + post-parallel-commit trace will
  show new top-3. Candidate targets: Stage 1 Materialize (309 ms wall / 13.5%)
  — fuse materialize-into-kernel-input with bind to eliminate re-materialization
  across rounds. Stage 5 `InstanceReduce` kernel 21 (IncClaimReduction, 315 ms /
  13.7%) — NOT Gruen-compatible (4 independent eq·product terms, no single eq
  factor) per Explore agent iter 20 investigation; attack requires algorithmic
  rework of the claim-batching structure. Stage 7 `DoryScheme::open` (282 ms /
  12.3%) — opening is sequential Dory inner-product loop; structural rework
  would need batched opening.

- **Iter 18 — Gruen runtime dispatch REVERTED (dead-code regression, stall counter → 2)**
  **Hypothesis**: wire `Iteration::Gruen` to `reduce_tensor_gruen_deg2` in
  `CpuBackend::reduce` so the iter 14/15/16 infra could start being exercised.
  No compiler-side emission yet — the dispatch code is dead until a later iter
  teaches a kernel spec to emit `Iteration::Gruen`. Change was purely additive:
  (a) ~10 LOC replacing the panic in the `Iteration::Gruen` arm, (b) ~35 LOC
  of `reduce_tensor_gruen_deg2<F>` helper with `#[tracing::instrument]`,
  (c) ~45 LOC unit test covering 6 shapes × 2 binding orders.

  **Gates**: 41/41 jolt-equivalence green (transcript_divergence +
  zkvm_proof_accepted included); clippy clean jolt-cpu + jolt-core
  (host, host+zk). All correctness gates passed — the perf result was the
  sole reject signal.

  **Perf**: three clean runs against ratchet 3672.07 ms —
  `iter18-run1.json` 4109.18 ms (+11.9%), `iter18-run2.json` 4222.54 ms
  (+15.0%), `iter18-run3.json` 3910.96 ms (+6.5%). All three ≥ 5% slower.
  Revert confirmed at 3769.11 ms (+2.6%, noise band).

  **Diagnosis (tentative)**: change was verified dead — grep of
  `Iteration::Gruen` shows it only in backend consumer sites, no compiler
  emits it. Candidate mechanisms for the measurable regression on
  never-executed code: (a) monomorphization of `reduce_tensor_gruen_deg2<F>`
  in `jolt-cpu` (hot crate) shifts function layout in the release binary
  and perturbs iCache / iTLB for `reduce_dense` / `reduce_tensor`, (b)
  `#[tracing::instrument]` inserts span-registration code at the Monomorphized
  site even when the function is never called, (c) ambient system noise
  amplified by the 2m 35s background build that preceded the first two runs
  (run 3 was at +6.5%, closest to noise band, consistent with cooling off).
  Core prove times were also inconsistent (333 → 388 ms) in run 3,
  supporting a partial ambient-noise contribution, but the 3-run average
  is too far outside noise to call "flat".

  **Takeaway for iter 19**: infra-only iters add dead code in the hot crate
  and apparently can measurably regress the perf gate with no behavioural
  change — the cost of splitting the Gruen port into ≥ 5 infra commits is
  compounding. Switch strategy: iter 19 ports Gruen end-to-end for kernel 3
  in one commit (compiler emission + runtime state tracking + dual-path
  assertion for 1-2 rounds + delete old Toom-Cook path + measure) so the
  perf gate fires on a change that actually exercises the new code. If that
  single commit is too large, split iter 19 = dual-path + iter 20 = delete
  old path + measure, which still only introduces live code in the gate.

  **Bookkeeping**: iter 17's note refers to "outer_remaining (kernel 3)".
  Attributing `InstanceSegmentedReduce` spans by kernel-arg against
  `crates/jolt-compiler/examples/jolt_core_module.rs` shows kernel 3 is
  actually the RAM RW phase 1 kernel (formula
  `eq_cycle · ra · ((1+γ)·val + γ·inc)`, `NI=4`, `NE=4`, LowToHigh, segmented).
  Still degree-2 product inside `l(X)`, so Gruen remains applicable — the
  kernel identity was mislabelled, not the opportunity.

- **Iter 17 — Post-P24 re-profile (profiling-only, stall counter unchanged)**
  (trace: `benchmark-runs/perfetto_traces/muldiv_log_t12_iter17_post_p24.json`).
  **Why**: iter 11's profile (`muldiv_log_t12_iter11_staged.json`) was the basis
  for the Gruen hypothesis queue, but P24's −9.24% win (iter 12) invalidated
  part of it — booleanity reduce dropped out of the top-5, so the queue needed
  to be reconciled against the current top spans before continuing to burn
  iters on the Gruen port. Per user memory on Perfetto-first hypothesis
  selection. **Top 10 spans on post-P24 trace (wall 4352 ms, profile overhead
  included)**:

  | Rank | Span | Total ms | % wall | Calls | Avg µs |
  |-----:|------|---------:|-------:|------:|-------:|
  | 1 | `reduce_dense` (all threads) | 18 962.6 | — | 83 428 | 227.3 |
  | 2 | `modular_prove` (wall) | 4 352.2 | 100.00 | 1 | — |
  | 3 | `stage` (8 stages) | 3 882.6 | 89.2 | 8 | 485 320 |
  | 4 | `InstanceSegmentedReduce` | 2 481.3 | **57.0** | 20 | 124 067 |
  | 5 | `CpuBackend::segmented_reduce` | 2 481.3 | 57.0 | 20 | 124 063 |
  | 6 | `prove` (outer wrappers) | 765.3 | 17.6 | 3 | 255 101 |
  | 7 | `Program::build_with_features` | 733.4 | 16.9 | 4 | 183 348 |
  | 8 | `Commit` | 465.3 | 10.7 | 3 | 155 114 |
  | 9 | `DoryScheme::commit` | 465.0 | 10.7 | 42 | 11 072 |
  | 10 | `BN254::multi_pair_g2_setup` | 429.7 | 9.9 | 112 | 3 836 |

  **Reconciliation vs iter 11**: `InstanceSegmentedReduce` subtree was 77.1%
  of prove wall at iter 11 (12 517 ms / 16 234 ms). Post-P24 it is 57.0%
  (2 481 ms / 4 352 ms). The absolute drop (~10 s saved at profile scale) is
  the P24 booleanity parallelism unlock plus the general wall compression
  from 14.6 s → 4.35 s since iter 0. `ReadCheckingReduce` (booleanity
  cluster) is now 93.1 ms / 128 calls — a 78% reduction vs iter 11's stage 5
  460 ms. Rank 1 on thread-time remains `reduce_dense` at 83 428 calls total,
  227 µs avg (iter 11 had 83 428 calls, 152 µs avg — higher avg now is
  the per-call arithmetic, not more calls).

  **Verdict**: `InstanceSegmentedReduce` is STILL the #1 wall bottleneck at
  57%. The structural remainder after P24 is outer_remaining (kernel 3) +
  other dense kernels inside segmented_reduce. Gruen port hypothesis
  survives — the structural target (collapse per-outer × per-inner Toom-Cook
  to one cubic-per-round) is unchanged. **Second-tier targets** (each 10-18%
  of wall): `Commit` + `DoryScheme::commit` (465 ms, streaming tier-1/tier-2
  Pedersen), `multi_pair_g2_setup_parallel` (429 ms, pairing setup for
  tier-2), `DoryScheme::open` (276 ms, opening proof). These are
  Dory-specific and sit downstream of sumcheck — lower leverage than Gruen
  for Phase 1 unless Gruen lands and leaves them dominant.

  **Hypothesis queue update**: no new entries — existing P21 Gruen split-eq
  remains the highest-leverage target. Dory attack surface (P26+: streaming
  commitment overlap, batched pairings) is queued but deferred until Gruen
  lands or is shown infeasible. **Gates**: no code changes, so no gates run.
  **Perf**: no measurement (profiling iter). **Next iter (18)**: resume
  Gruen port — wire `Iteration::Gruen` runtime dispatch. Plan: teach the
  outer_remaining kernel spec to emit `Iteration::Gruen`; add a
  `ComputeBackend::gruen_reduce` trait method returning `(q_const, q_quad)`;
  handler assembles the cubic via `gruen_cubic_evals` and stores all 4
  evals in `last_round_coeffs`. Dual-path assertion for 1-2 rounds before
  iter 19 deletes the old Toom-Cook path.

- **Iter 16 — Gruen infrastructure: `Iteration::Gruen` enum variant
  (infrastructure, no perf claim)** (targets:
  `crates/jolt-compiler/src/kernel_spec.rs` +4 lines,
  `crates/jolt-cpu/src/backend.rs` +2 match arms,
  `crates/jolt-metal/src/{backend.rs,msl_reduce.rs}` +3 match arms).
  **Scope decision**: land the variant and make every exhaustive `match`
  still compile, without any site actually *emitting* Gruen. The CPU
  reduce arm panics `"Iteration::Gruen runtime dispatch lands in iter
  17"`; Metal arms panic `"Gruen iteration not yet supported on Metal —
  use CpuBackend"`; `bind` treats Gruen identically to `Dense`/`DenseTensor`
  (the buffers are bound the same way, only the round-poly assembly
  changes). Two extra inputs follow formula columns (same shape as
  `DenseTensor`: outer_eq then inner_eq). **Result**: new variant exists,
  all existing emission sites unchanged, zero runtime behavior change.
  **Gates**: 41/41 jolt-equivalence green, transcript_divergence +
  zkvm_proof_accepted green, clippy clean across 8 modular crates
  (compiler/compute/cpu/zkvm/dory/openings/verifier/bench), clippy clean
  jolt-core (host, host+zk). (Metal backend has pre-existing
  ChallengeIdx indexing errors on `refactor/crates` that predate my
  changes — confirmed via `git stash`.) **Perf**: 3689.12 ms (+0.46% vs
  3672.07 baseline), flat as expected (variant unused). **Next iter (17)**:
  wire runtime handler — teach `reduce` to dispatch `Iteration::Gruen`
  to `reduce_dense_gruen_deg2` and assemble the cubic via
  `gruen_cubic_evals`. Also need to change caller's cubic-assembly shape:
  the runtime handler currently calls `kernel.evaluate` which returns
  4 Toom-Cook-grid evals; Gruen variant needs to receive `(q_const,
  q_quad, prev_claim, current_scalar, w_current)` and emit the cubic at
  {0,1,2,3}. Will likely add a new `CpuKernel` path or a sibling reduce
  entry-point. Dual-path assertion for 1-2 rounds before iter 18 deletes
  old path and ratchets.

- **Iter 15 — Gruen infrastructure: `reduce_dense_gruen_deg2` free function
  (infrastructure, no perf claim)** (target: `crates/jolt-cpu/src/gruen.rs`).
  **Scope decision**: narrowed from "Iteration::Gruen enum + trait method"
  to a pure free function on jolt-cpu with `BindingOrder` handling and two
  unit tests. No enum variant, no trait plumbing — that lives in iter 16
  when compiler emission needs to pick the path. Keeps iter 15's blast
  radius zero (new file, dead code). **Change**: 54-line addition —
  `reduce_dense_gruen_deg2(e_active, factor_a, factor_b, scalar, order) →
  (q_const, q_quad)` plus ~90 lines of tests. Accumulates
  `Σ e_active[i] × a_lo × b_lo` and `Σ e_active[i] × (a_hi-a_lo) × (b_hi-b_lo)`,
  each scaled by `scalar`. Both `LowToHigh` (pair at `buf[2i], buf[2i+1]`)
  and `HighToLow` (pair at `buf[i], buf[i+half]`) supported. **Tests**:
  (a) bit-exact parity vs naive reference across half ∈ {1, 3, 8, 17}
  for both binding orders; (b) end-to-end Gruen flow — feed reduce output
  through iter 14's `gruen_cubic_evals`, verify the 4 cubic evals match
  direct per-pair accumulation `Σ e_active[i] × l(X) × a_i(X) × b_i(X)` at
  X ∈ {0,1,2,3}. This second test pins that iter 14 + iter 15 compose
  correctly — the exact property iter 17's runtime wiring needs.
  **Gates**: 4/4 gruen tests green, 41/41 jolt-equivalence green,
  clippy clean both host and host+zk. **Perf**: 3665.31 ms (−0.18% vs
  3672.07 baseline), flat as expected (dead code). **Next iter (16)**:
  add `Iteration::Gruen` variant to jolt-compiler and teach the
  outer_remaining kernel spec in `crates/jolt-compiler/examples/jolt_core_module.rs`
  to emit it. Handler stays ≤30 LOC, protocol-unaware. Dual-path
  assertion: runtime runs both Toom-Cook (current) and Gruen (new) and
  asserts the cubic evals are equal for 1-2 rounds before deleting the
  old path in iter 17.

- **Iter 14 — Gruen infrastructure: `gruen_cubic_evals` scalar primitive
  (infrastructure, no perf claim)** (target: new file
  `crates/jolt-cpu/src/gruen.rs` + `pub mod gruen;` export in lib.rs).
  **Design step**: pivoting away from micro-opts (e.g., P25 const-generic
  shape arms) per user's "order-of-magnitude gain" bar while modular is
  >2× off core. The 53% concentration on outer_remaining (kernel 3,
  iter 11 trace) remains the dominant structural target. Rather than
  land Gruen as one 200-LOC iter, split into 4 bounded iters:
  (14) port scalar gruen_poly_deg_3 + unit test; (15) add
  Iteration::Gruen + reduce_gruen backend method; (16) wire compiler
  emission for outer_remaining with dual-path assertion; (17) delete
  old path, measure. Iter 14 is step 1. **Change**: 125-line new
  module with `gruen_cubic_evals(current_scalar, w_current, q_const,
  q_quad, s_0_plus_s_1) → [F; 4]`. Callers supply `w_current` already
  resolved from the binding order (LowToHigh uses `w[current_index-1]`,
  HighToLow uses `w[current_index]`). Two unit tests: (a) bit-exact
  match vs direct `(a+bX)(c+dX+eX²)` multiplication at X ∈ {0,1,2,3}
  for 32 random inputs; (b) Lagrange-interpolation through evals
  reproduces direct evaluation at X=4, verifying the 4 evals describe
  a true cubic. **Gates**: 72/72 jolt-cpu tests green, 41/41
  jolt-equivalence green, clippy clean both host and host+zk.
  **Perf sanity**: 3654.95 ms (−0.47% vs 3672.07 baseline), flat as
  expected since the module is unused dead code today. **Why this is
  valuable even at 0% perf**: it de-risks the arithmetic. Once iter 15
  plumbs a backend reduce that emits (q_const, q_quad), iter 16 can
  assemble the cubic without rediscovering Gruen's formula. Unit tests
  pin the exact arithmetic, so any downstream wiring bug surfaces as
  an integration issue, not an arithmetic one. **Next iter (15)**: add
  `Iteration::Gruen` enum variant to jolt-compiler + new
  `ComputeBackend::reduce_gruen_deg2(kernel, inputs) → (q_const, q_quad)`
  method. The reduce fuses two dot products over pairs into one parallel
  pass: Σ e_active[i] × q_pair_const(lo[i]) for q_const and
  Σ e_active[i] × q_pair_quad(hi[i]-lo[i]) for q_quad. Signature-only
  step with unit tests against the pure function from iter 14. Still no
  compiler/runtime wiring.

- **Iter 13 — P25 extend `with_min_len` lowering to reduce_dense_fixed +
  reduce_sparse (REVERTED, flat)** (targets: `crates/jolt-cpu/src/backend.rs`
  lines 693 dense-fixed 2048→1024 and 1124 sparse 4096→1024). **Design
  hypothesis**: mirror iter 12 P24's win — if the dynamic path was
  under-parallelized, maybe fixed and sparse paths have similar late-round
  or medium-shape blind spots. **Gates**: 41/41 jolt-equivalence green,
  transcript_divergence + zkvm_proof_accepted green, clippy clean both
  host and host+zk. **Result**: run 1 3667.57 ms (−0.12%), run 2 3684.26 ms
  (+0.33%) vs 3672.07 ms baseline. Both in ±5% band; per protocol flat →
  revert. **Diagnosis**: (a) reduce_dense_fixed's dominant consumer is
  kernel 3 (NI=4, NE=4) where half starts at ≈2M and halves per round;
  late rounds drop below PAR_THRESHOLD=1024 entirely (sequential path).
  The 1024-2048 transition band is visited for ~1 round per sumcheck —
  insufficient population to move the needle. (b) reduce_sparse pairs.len()
  distribution is bimodal — either tiny (below PAR_THRESHOLD, sequential)
  or huge (>>4096, saturated under either threshold); the 1024-4096 band
  is thinly populated. **Generalized lesson**: the iter 12 win was
  shape-specific — dynamic path has expensive per-pair work (NI=10 field
  loads + NE=4 evals + dynamic Vec bookkeeping) where 1024-pair tasks
  amortize thread fork cost; smaller-NI fixed workloads either already
  have enough parallelism (large halves) or are below threshold entirely
  (small halves). Don't blanket-extend parallel tuning — profile the
  specific workload first. **Next iter**: P25 (extend const-generic shapes
  for booleanity/hamming — stack-allocated scratch) or P26 (Gruen
  split-eq port for kernel 3, the remaining 53% concentration).

- **Iter 12 — P24 lower reduce_dense_dynamic `with_min_len` (4096 → 1024)**
  (target: `crates/jolt-cpu/src/backend.rs:775`). **Design step**: iter 11
  instrumentation attributed stage 5's 515 ms wall to a single kernel —
  kernel 21 (Booleanity, batch=4 instance=1) at 424 ms / 14 calls / 30 ms
  avg, with first call at 142 ms. Booleanity shape is (NI=1+total_d, NE=4)
  where total_d ≈ instruction_d + bytecode_d + ram_d ≈ 9-10 — not in the
  const-generic list — so it falls to `reduce_dense_dynamic` which used
  `with_min_len(4096)`. For typical booleanity first-round halves (8K-32K),
  that produces only 2-8 rayon tasks; with `with_min_len(1024)` rayon can
  split into 8-32 tasks, unlocking multi-core parallelism that was capped.
  Change: one-char edit (`4096` → `1024`). **Gates**: 41/41 jolt-equivalence
  green, clippy clean across jolt-{cpu,compiler,compute,zkvm,dory,openings,
  verifier,bench}. **Result**: run 1 3672.07 ms (−9.43%), run 2 3687.88 ms
  (−9.05%) vs 4054.58 ms baseline. Avg −9.24%, both runs past 5% accept
  threshold. Ratio vs core: 12.1× → 11.0×. **Why this worked after iter 5
  P14 (which lowered PAR_THRESHOLD and flatlined)**: iter 5 targeted the
  FIXED path (NI=4, NE=4, dominating by call count but at already-good
  parallelism). Iter 12 targets the DYNAMIC path (booleanity-shape, small
  call count but massive per-call work × 1-2× parallelism). Different
  regime, different bottleneck. **Ratchet**: baseline-modular-best.json now
  at 3672.07 ms. Stall counter reset to 0. Green streak 1. **Next iter**:
  P25 — add more const-generic shape arms to `reduce_dense_fixed` for
  booleanity/hamming/ram_ra_virtual shapes; expected to compound P24 gain
  with ILP/register allocation wins on top of parallelism.

- **Iter 11 — per-stage instrumentation (infrastructure, no perf claim)**
  (targets: `crates/jolt-zkvm/src/runtime/mod.rs` — added `_stage_span`
  swap on `BeginStage` so every subsequent op nests under a `stage(index=N)`
  span; `crates/jolt-zkvm/src/runtime/handlers.rs::op_span` — tagged
  `InstanceSegmentedReduce`/`InstanceReduce`/`InstanceBind`/`BatchRoundBegin`
  spans with `batch`, `instance`, `kernel`, `round` fields). Goal: pivot
  from guessing at aggregate `reduce_dense` self-time to attributing
  every microsecond to a specific sumcheck instance and round.
  **Gates**: clippy clean jolt-zkvm + jolt-compiler + jolt-cpu + jolt-compute;
  41/41 jolt-equivalence green. **Perf check (not a hypothesis)**:
  3954.3 ms vs 4054.58 ms baseline = −2.47%, within noise — instrumentation
  has no material cost.
  **Key finding**:
    - **Stage 1 outer_remaining sumcheck** (batch=0, instance=0, kernel=3)
      alone = 53% of total prove wall (1993 ms / 3774 ms) with 7.5×
      effective parallelism (15591 ms thread-time). 10 rounds, per-round
      wall decays classically: 989 → 493 → 246 → 125 → 63 → 33 → 19 →
      12 → 7 → 4 ms.
    - Stage 5 `InstanceReduce` (register RW + RAM val check) = 460 ms
      at 1× parallelism. Secondary target.
    - Stage 7 `DoryScheme::open` = 416 ms (opening proof; not sumcheck,
      not addressable by Gruen).
  **Implication for iter 12**: the Gruen split-eq port must target
  exactly this one sumcheck — outer_remaining, kernel 3. A correct port
  should cut its per-round work by a factor dependent on the outer/inner
  split (likely 2–4×), collapsing ~1000+ ms from stage 1 wall. That
  single fix is worth more than the next 3 hypotheses combined.

- **Iter 10 — P20 fused segmented_reduce (no decomposition)**
  (target: `crates/jolt-cpu/src/backend.rs` — replaced `segmented_reduce`
  with const-generic `segmented_fused_fixed<NI, NE>` + dynamic fallback).
  **Design step**: iter-10 Perfetto showed `InstanceSegmentedReduce`
  subtree at 51.3% wall (2139 ms / 20 calls), combined `reduce_dense`
  variants 62.8% wall. Hypothesis: fuse outer×inner iteration into a
  single parallel fold, eliminate Vec<&Buf> allocations, collapse ~83k
  rayon fork/join into one, and merge per-outer eq weight into fmadd.
  **Gates**: clippy clean jolt-cpu + jolt-zkvm + jolt-compiler + jolt-field
  + jolt-compute; 41/41 jolt-equivalence green.
  **Result**: two runs 4070.50 ms (+0.39%) and 4077.08 ms (+0.54%) vs
  4054.58 ms baseline; avg +0.47%. Flat within ±5% band. Reverted.
  **Diagnosis**: the "wins" I targeted weren't load-bearing.
  (a) Vec<&Buf> allocations = 16 ptrs × 84k calls = 1.3 MB metadata,
      not 5 GB data — I confused Vec-of-refs with data clone.
  (b) The outer rayon par_iter in the old `segmented_reduce` was
      already collapsing most fork/join overhead — "83k rayon calls"
      was misleading; only 20 calls created outer forks, each
      fanning out across ~4200 positions with stolen work.
  (c) Per-outer eq mul folded into fmadd saves ~84k × 4 = 336k
      F::muls ≈ 6 ms on a single core. Rounding error vs 4000 ms.
  **Conclusion**: shape/structure of segmented_reduce is NOT the
  bottleneck. The real gap is per-iter arithmetic intensity (D field
  muls per Toom-Cook eval × active_outer × half_inner). Only a TRUE
  Gruen-style tensor decomposition (outer eq reduced to a prefix
  scalar fold BEFORE the inner sweep, like core's `gruen_poly_deg_3`)
  cuts the inner arithmetic; P20's fused loop did not change inner
  arithmetic at all — it still does outer × inner Toom-Cook work.
  **Next iter pivot**: step back and INSTRUMENT. Current profile
  reports `segmented_reduce` wrapper + `reduce_dense` but does not
  show WHICH of the 20 segmented sumcheck rounds (RAM read-write,
  instruction RA virtual, instruction lookups, bytecode, register
  read-write, etc.) are eating the 2139 ms. Without that, picking
  a specific target for Gruen port is guesswork. Iter 11 plan: add
  per-stage/per-sumcheck-instance spans, re-capture trace, identify
  the top-2 sumcheck instances, then target P21 Gruen port at those.

- **Iter 9 — P16 defer Montgomery reduction through `eval_prod_D_assign`
  via accumulator-writing kernel path** (targets:
  `crates/jolt-cpu/src/toom_cook.rs` — added
  `accumulate_linear_prod` + `accumulate_prod_{2..=8,16,32}` variants;
  `crates/jolt-cpu/src/backend.rs` — added `AccEvalFn<F>` +
  `acc_eval_fn` field on `CpuKernel`, `with_acc_eval`/`has_acc_eval`/
  `evaluate_to_accs`, branched `reduce_dense_fixed` to skip
  `evaluate → acc_add` when acc path is compiled;
  `crates/jolt-cpu/src/product_sum.rs` — added `compile_acc_fn`
  covering D ∈ {2,3,4,5,6,7,8,16,32} at P=1;
  `crates/jolt-cpu/src/lib.rs` — wired acc_fn attachment in `compile()`
  for product-sum specs with P=1 and supported D). **Design step**:
  iter 8's diagnosis pointed to `F::mul` inside `eval_prod_4_assign`
  as the actual dominant cost in `reduce_dense_fixed<_,4,4>` — each
  inner iter pays 4× Montgomery reductions for `outputs[k] = a_k*b_k`,
  which are then trivially acc_add'd. Deferring those reductions by
  writing `accs[k].fmadd(a_k, b_k)` directly eliminates O(D × N_iters ×
  N_calls) Montgomery reductions (~80M total for a muldiv @ log_T=12
  proof). Hypothesis: ~20-25% wall-time savings if Mont reduce is
  30-50 cycles and fills the lion's share of inner-loop work.
  **Gates**: 70/70 jolt-cpu tests, 41/41 jolt-equivalence green,
  clippy clean across jolt-cpu + jolt-core (host + host,zk) +
  jolt-compiler + jolt-field + jolt-zkvm. **Result**: run 1 3870.07 ms
  (−4.55%), run 2 3959.53 ms (−2.34%) vs 4054.58 ms baseline.
  Both in ±5% inconclusive band; consistent improvement signal but
  below the 5% acceptance threshold (avg −3.45%). Per protocol:
  flat → reverted. **Diagnosis**: Montgomery reduction on arkworks
  BN254 `Fr` is faster than the pre-experiment 50-80 cycle estimate
  — the `from_montgomery_reduce` helper likely runs ~15-25 cycles on
  modern x86 with BMI2 (mulx/adx); at 4 reductions per inner iter ×
  ~246 iters × ~84k calls that's only ~80M × 20 cycles ≈ 1.6 Gcycles
  = ~0.5 s on a single core, amortized by 8-way rayon fan-out in
  `reduce_dense_fixed` to ≈ 60-80 ms. The accumulator path also
  introduces new costs: `fmadd` in `Limbs<9>` runs a 4×4 schoolbook
  scatter that is comparable in cost to `F::mul + acc_add` for small
  accumulators, so the savings are partly consumed by the new fmadd
  overhead. Real net gain is the observed 2-5%. **Next structural
  target**: stop optimizing the per-iter arithmetic and attack the
  O(N_calls) dispatch overhead — port core's `GruenSplitEqPolynomial`
  so segmented sumchecks emit ONE cubic per round instead of N
  per-outer reduces (~84k → ~20 calls). That collapses
  `reduce_dense_fixed` calls by 4000×, not their per-call cost.

- **Iter 8 — P19 port folded-slot `[u128; 9]` representation to
  `WideAccumulator`** (target: `crates/jolt-field/src/arkworks/wide_accumulator.rs`).
  **Design step**: research via Explore agent pointed to BN254 Fr
  `WideAccumulator` carry-propagation in `fmadd`/`acc_add`/`merge`
  as a potential structural inefficiency — the existing `Limbs<9>`
  (`[u64; 9]`) path serializes each row's carry through remaining
  limbs on every op; jolt-core's `folded_accum.rs` uses `[u128; N]`
  positional slots with inter-slot carries deferred to a single
  `normalize()` right before Montgomery reduction, making each
  accumulation an independent `u128 += u128` the CPU can issue in
  parallel via ILP. Expected savings on the hot reduce_dense path:
  ~2× on `acc_add`, ~5× on `fmadd`, across an estimated ~80M per-proof
  calls. Target 10-20% wall-time. Implementation: replaced the
  accumulator body with `slots: [u128; 9]`; `fmadd` does a 4×4
  schoolbook scattering into adjacent slots without carries, `acc_add`
  adds the 4 `val` limbs directly into slots 4..=7, `merge` is a
  9-way independent u128 sum, `reduce` runs `normalize()` (single
  carry pass) followed by the existing `bn254_ops::from_montgomery_reduce::<9>`.
  **Gates**: 166/166 jolt-field tests green, 41/41 jolt-equivalence
  green, clippy clean across jolt-{field,compiler,compute,cpu,zkvm,dory,openings,verifier,bench}.
  **Result**: two runs 4121.31 ms (+1.65%) and 4036.40 ms (−0.45%)
  vs 4054.58 ms baseline; avg 4078.85 ms (+0.60%). Flat within ±5%
  band. Reverted. **Diagnosis**: the accumulator arithmetic is not
  the reduce_dense bottleneck. Per-iter cost decomposition inside
  `reduce_dense_fixed<_, 4, 4>`:
  - `kernel.evaluate` runs `toom_cook::eval_prod_4_assign` → 4× `F::mul`
    (Montgomery-reducing) → ~200+ cycles / iter
  - `acc_add` × 4 lanes → 4× ~4 cycles = ~16 cycles / iter
  So the accumulator is <10% of inner work; halving it would yield
  ~5% wall-time improvement at best — masked by variance here.
  Also: rustc/LLVM lower `u64 + carry` chains on BN254's
  `Limbs<N>::fmadd` into `adc`/`addcarry` sequences that pipeline
  well on modern CPUs — the expected serialization penalty was
  overstated in the pre-experiment analysis. **Next structural
  target**: attack `F::mul` inside `eval_prod_4_assign` (P16 in
  queue) — defer Montgomery reduction for the pointwise product
  outputs so they accumulate as unreduced wide integers, dropping
  ~4 Mont-reductions per inner iter × ~80M iters of the reduce_dense
  hot loop. That attacks the actual dominant cost rather than
  the accumulator wrapper.

- **Iter 7 — P18 shape-aware fused segmented_reduce**
  (target: `crates/jolt-cpu/src/backend.rs:520-599`). **Design step**:
  Iter 7 Perfetto trace (generated in stall mode at counter=5) showed
  `reduce_dense` 75.8% self-time (16 222 ms) and `segmented_reduce`
  wrapper 9.4% self-time (2 014 ms) with ~84k reduce_one calls.
  Re-attempted P17 fusion but preserved inner parallelism via the same
  `if inner_half >= PAR_THRESHOLD` rayon gate reduce_dense_fixed uses.
  Structure: const-generic `segmented_reduce_fixed<NI, NE>` for (2,2),
  (3,3), (4,4), (8,4), (8,8), (16,16), (32,32) + dynamic fallback. Each
  reduce_one inlines the per-outer pair loop with a base-offset load
  (`if io[k] { 0 } else { a * inner_size }`), matching both LowToHigh
  and HighToLow binding. Outer par_iter fires when `active.len() >= 2`,
  inner par_iter when `inner_half >= PAR_THRESHOLD` — so nested rayon
  is possible when both axes are large, matching pre-fusion behavior.
  **Gates**: 41/41 equivalence tests green, clippy clean across
  jolt-{zkvm,compute,cpu,compiler,dory,verifier,bench}. **Result**: two
  runs 3981.43 ms (−1.80%) and 4122.88 ms (+1.68%) vs 4054.58 ms
  baseline; avg −0.06%. Flat within ±5% band. Reverted.
  **Diagnosis**: fusion eliminated the 9.4% thread-time wrapper work
  but its contribution to wall time was masked by the outer par_iter
  already running per-outer wrapper work in parallel across threads.
  Per-call overhead at ~24 µs (2 014 ms / 84 k calls) is genuine but
  amortized away when 8-way parallel — net wall savings ≈ wrapper /
  effective_parallelism, which for this workload is within run-to-run
  noise. The big structural opportunity remains: port core's
  `GruenSplitEqPolynomial` approach so segmented sumchecks emit ONE
  cubic per round instead of N per-outer reduces, eliminating the
  segmentation entirely rather than optimizing per-outer dispatch.

- **Iter 6 — P17 fuse segmented_reduce**
  (target: `crates/jolt-cpu/src/backend.rs:520-599`). **Design step**:
  Explore agent comparison confirmed the 11.5× gap is STRUCTURAL — core
  uses Gruen's `GruenSplitEqPolynomial` (jolt-core
  `zkvm/ram/read_write_checking.rs:256, 309-388`) to pre-decompose
  outer×inner eq and emit ONE cubic per round; modular stack's
  `segmented_reduce` iterates `outer_eq.filter_map(non-zero)` and
  issues a separate `self.reduce(kernel, col_refs, challenges)` per
  active outer position — ~4200 tiny `reduce_dense` dispatches per
  InstanceSegmentedReduce × 20 calls = ~84k dispatches. Hypothesis:
  fuse into a const-generic single loop that inlines the inner pair
  iteration across active positions, eliminating Vec<&Buf> allocation,
  outer-slice clones, and reduce_dense dispatch overhead. Implemented
  `segmented_reduce_fixed<NI, NE>` for shapes (2,2), (3,3), (4,4),
  (8,4), (8,8), (16,16), (32,32) plus dynamic fallback, with stack
  `[F; NI]` lo/hi scratch and `[F::Accumulator; NE]` local inner
  accumulator. Gates: 41/41 equivalence tests green, clippy clean.
  **Result**: two runs 4174 ms (+2.95%) and 4457 ms (+9.88%) vs 4055 ms
  baseline — past the +5% regression threshold on rerun. Reverted.
  **Diagnosis**: the fused version serializes the inner pair loop and
  parallelizes ONLY across outer positions. `reduce_dense_fixed` has an
  inner PAR_THRESHOLD gate — when inner_half ≥ 1024, it parallelizes
  pair iterations. Early segmented-sumcheck rounds have large inner_size
  and few active outer positions; the old per-outer path got inner
  parallelism for free there. My fused path lost that, so early-round
  throughput dropped more than the saved per-call overhead on late
  rounds gained. **Fix**: iter 7 candidate P17' — shape-aware dispatch
  that preserves inner parallelism when `inner_half >= PAR_THRESHOLD`,
  or uses nested rayon. Also consider whether the correct answer is a
  Gruen-style precompute + single non-segmented reduce (porting core's
  algorithm rather than optimizing the per-outer iteration). That is
  a larger rewrite but matches core's structural advantage directly.

- **Iter 5 — P14 lower rayon thresholds in `reduce_dense_fixed`**
  (target: `crates/jolt-cpu/src/backend.rs:14 + :693-731`). Motivation:
  per-shape instrumentation of `reduce_dense` (added then reverted)
  showed the `(NI=4, NE=4)` shape dominates at **70.9% self-time /
  81 933 calls / 197.5 µs avg**; all other shapes (2x2, 3x3, 8x4, 8x8,
  16x16, 32x32) together contribute <0.5%. With current
  `PAR_THRESHOLD = 1024` and `with_min_len = 2048`, the average call
  (half ≈ 246) falls fully on the serial inner loop, and even
  half = 2048 gets only a single rayon task. Hypothesis: introduce
  `REDUCE_DENSE_FIXED_MIN_LEN = 256` and gate both the parallel-path
  entry and `with_min_len` on it, giving mid-size reduces real
  fan-out. Result: two runs 4118.15 ms (+1.57%) and 4137.32 ms
  (+2.04%) vs 4054.58 ms baseline — consistently slightly slower,
  both in the ±5% inconclusive band. Reverted. Explanation: at D=4
  the inner work per rayon task (~256 iters × ~8 field mults ≈
  0.2 ms) is just above the rayon fan-out + merge overhead, so the
  extra task-creation cost eats the parallelism gain. Kernels where
  the work/task is larger (higher D, bigger halves at log_T ≥ 14)
  are the regime where this would likely turn positive; revisit at
  Phase 2.

- **Iter 4 — P13 combine P12 parallel `bind` + new
  `interpolate_inplace_many` trait method parallelizing the 16K serial
  InstanceBind `interpolate_inplace` calls** (targets:
  `crates/jolt-compute/src/traits.rs`, `crates/jolt-cpu/src/backend.rs`,
  `crates/jolt-zkvm/src/runtime/handlers.rs`). Result: flat
  (3944.09 ms −2.72%, rerun 4010.05 ms −1.10% vs 4054.58 ms baseline;
  mean −1.91%). Reverted — inconclusive band confirmed on rerun.
  Pattern: filter pids first, then `device_buffers.iter_mut().filter_map`
  to obtain `&mut [&mut Buf]`, pass to new trait method that rayon
  `par_iter_mut`'s. Compiled clean, correctness suite 41/41 green.
  The real bottleneck is not the per-input parallelism at this scale —
  per-call BN254 mul cost dominates per-buffer work, and buffer sizes
  at log_t=12 are small enough that rayon fan-out overhead ≈ speedup.
  This avenue is likely more productive at higher log_t once inner
  buffers are larger. Mark "revisit at Phase 2/3 (log_t=14+)" if
  stalls re-emerge in a regime where InstanceBind per-buffer work
  exceeds rayon overhead.

- **Iter 3 — P12 parallelize `CpuBackend::bind` across inputs via
  `par_iter_mut`** (target: `crates/jolt-cpu/src/backend.rs:169-184`).
  Result: consistent −2.89% (3934.58 ms + 3940.83 ms vs 4054.58 ms
  baseline, both runs). Reverted per 5% threshold despite reproducible
  signal. Backend::bind is called from `bind_kernel_inputs` helper in
  the runtime; InstanceBind handler calls `backend.interpolate_inplace`
  per input (one at a time) so those 16K calls weren't touched. The
  signal was real but small because `bind_kernel_inputs` is called
  rarely compared to the InstanceBind per-input path. Stacking P12
  with a parallelization of the InstanceBind handler's 16K serial
  interpolate_inplace calls (structural handler change) could push
  past 5%. File under "combine with InstanceBind parallelism" in a
  later iter if compound wins needed.

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
