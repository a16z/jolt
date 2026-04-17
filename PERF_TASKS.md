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
- **Stall counter**: 7
- **Last green iter**: 1 — P10 `segmented_reduce` parallelize+hoist
  (−72.24% prove_ms: 14607 → 4055 ms, ratio 41.4× → 11.5×)
- **Green streak**: 0 (iter 2 P11 flat +0.5%; iter 3 P12 flat −2.9%; iter 4 P13 flat −1.9%; iter 5 P14 flat +1.8%; iter 6 P17 regressed +6.4%; iter 7 P18 flat −0.1%; iter 8 P19 flat +0.6%, all reverted)
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

- [ ] P16: Defer Montgomery reduction through the 4-output pointwise mul
  block of `eval_prod_4_assign` — target:
  `crates/jolt-cpu/src/toom_cook.rs:365-374`
    - **Hypothesis**: the last 4 multiplications produce `outputs[0..4]`
      that are *immediately* added into the per-thread accumulator. Using
      BN254 unreduced / non-reduced multiplication primitives (returning
      a 512-bit limb pair) for those 4 mults and letting `acc_add`
      merge in non-reduced form would cut ~4 Montgomery reductions per
      inner iter. ~20% of arithmetic savings on the D=4 hot path.
    - **Abstraction risk**: medium — need a non-reduced mul API on
      `Field` and an `Accumulator` path that consumes it.
    - **Expected delta**: 5-10% (contingent on wiring + field support).

<!-- P19 consumed iter 8; see Notes for result & diagnosis. -->


## Notes

Design decisions, dead ends, and stall-mode observations accumulate here.

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
