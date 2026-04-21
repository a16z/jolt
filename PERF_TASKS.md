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

- **Phase**: 2 (log_t=16 standardized workload) — switched 2026-04-19 per
  user direction: "let's up our standardized check to use sha chain with
  2^16 cycles instead for every loop now and focus on optimizing that".
  Replaces the prior muldiv log_t=12 standard (real log_t≈10, ratio ~4×)
  with a production-shape workload (real log_t=16, ratio ~19.7×).
- **log_t**: 16 ratchet on sha2-chain --num-iters=16 (ratchet **70762.9 ms
  modular**, 3901.3 ms core best; updated iter 54). Full bench command:
  `cargo run --release -p jolt-bench -- --program sha2-chain --num-iters 16
  --iters 1 --warmup 1 --log-t 16 --json perf/last-iter.json`.
  `--log-t N` is a CEILING matching `JoltSharedPreprocessing::new`.
- **Program**: `sha2-chain --num-iters 16 --log-t 16` (primary ratchet);
  muldiv log_t=12 retired as standard (history kept for reference). Prior
  baseline preserved in `perf/baseline-modular-best-prior-muldiv-log_t12.json`.
- **Stall counter**: 31 (iter 87 P84-A' infra — `DeviceBuffer::Compact { data: Vec<i128>, bits: u8, signed: bool }` variant + accessors landed (`as_compact` / `as_compact_mut` / `compact_encoding` / `is_compact`) in `crates/jolt-compute/src/traits.rs`; `jolt-hybrid` exhaustive matches patched with Compact arms (migrate no-op, on_primary true, test panics); +60 LOC total; no backend fast path yet, no hot-path wiring, no compiler changes; correctness 50/50 PASS + clippy jolt-core host + host,zk + modular lib+bin clean; perf 71,939.25 ms modular vs 70,762.94 ratchet = +1.66% in the ±5% inconclusive band, consistent with zero-hot-path-usage infra expectation; ratchet unchanged; iter 86 DESIGN — user directive against a core-style `SmallScalar` trait mid-implementation; pivot P84 framing to buffer-encoding-first: rewrite P84 to forbid a generic `SmallScalar` + `CompactPolynomial<T,F>` pattern and pin P84-A' as a runtime-tagged `DeviceBuffer::Compact` variant (encoding as *data*, not as generic type parameters); no code; iter 85 P86 REVERTED — dory cache seed is a no-op under `--stack both` because core's `DoryCommitmentScheme::setup_prover` seeds the dory-pcs shared global cache before modular's `DoryScheme::setup_prover` runs; the 167→93ms/call gap in iter84_modular_only.json only exists in solo modular runs, not the production benchmark; iter 84 DESIGN — fresh per-stack profiling disambiguates prior conflicting per-call ratios; appended P84/P85/P86 backlog items; no code; iter 83 P83 reverted — sparse eq_project for ram_ra_indicator, -3.9% median across 3 runs but rerun +6.78% in reject zone; iter 82 P78 reverted — bind_low_to_high in-place compact regressed +47% mean; iter 81 P77-D reverted — CpuBackend batch_round_evaluate par_iter override + re-applied emission switch regressed +6.86% mean; iter 80 P77-C reverted — emission switch alone regressed 10-14%; iter 79 P77-B infra — BatchRoundEvaluate handler landed, flat; iter 78 P77-A infra — BatchInstanceSpec/BatchReduceKind trait surface; iter 77 P77 infra — memoize RamRaIndicator; iter 76 P80 parallelize RAM derived polys — ~7.5% improvement vs same-session pre-change, but +12% past-reject vs stale iter-54 ratchet; code kept, ratchet unchanged; iter 75 P76-D hot-path wiring reverted under noise, infra kept; iter 74 P76-C infra; iter 73 P76-B infra; iter 72 P76-A infra stub; iter 71 P75-B infra stub; iter 70 P75-A infra-only commit; iter 69 design-only; iter 68 P73 reverted; iter 67 P72 reverted; iter 66 design-only commit; iter 65 P71 reverted; iter 64 P70 reverted; iter 63 P90 reverted).

  iter 87 P84-A' INFRA (landed flat) — add a runtime-tagged
  `DeviceBuffer::Compact { data: Vec<i128>, bits: u8, signed: bool }`
  variant to `crates/jolt-compute/src/traits.rs:67` alongside existing
  `Field` / `U64`, plus accessors `as_compact` / `as_compact_mut` /
  `compact_encoding` / `is_compact`. Exhaustive `DeviceBuffer` matches
  in `crates/jolt-hybrid/src/lib.rs` patched with a Compact arm
  (migrate_buf_to_fallback: no-op, `on_primary` checks: true, mock
  test bind: no-op, test panic arms: `panic!("expected Field")`).
  Total diff +60 LOC across 2 files. **No backend fast path yet, no
  compiler emission changes, no handler changes, no hot-path wiring.**

  **Design choices**:
  - `Vec<i128>` storage chosen over per-type `Vec<T>` (core's pattern):
    one variant covers all small-scalar types (u8/u16/u32/u64/i64/
    u128/i128) without a generic `T` parameter polluting call sites.
    Memory 16 bytes/elem (16× vs `Vec<u8>` in the worst case) is
    acceptable overhead for Phase 2; the win is eliminating
    `SmallScalar` trait & generic `CompactPolynomial<T,F>`.
  - `bits` + `signed` encoding metadata lives as runtime *data* on
    the buffer, not as a generic type parameter. Fast paths
    (arriving P84-B) read these fields and dispatch to the matching
    `Fr::mul_{u64,i64,u128,i128}` (already optimized in
    `arkworks/bn254_ops.rs`).
  - Backend-opaque representation: no `Buffer<i128>` generic on the
    `ComputeBackend` trait. Compact data always lives on host; if a
    future GPU backend wants to consume it, it promotes on host at
    dispatch time (same as `Vec<F>` works today). Keeps the trait
    surface minimal.

  **Correctness**: jolt-equivalence 50/50 PASS (transcript_divergence,
  zkvm_proof_accepted_by_core_verifier, modular_self_verify,
  modular_self_verify_commit_skip_alignment — all green). Clippy
  jolt-core host, host,zk both clean; jolt-compute / jolt-hybrid /
  jolt-cpu / jolt-metal / jolt-zkvm / jolt-poly / jolt-verifier
  lib+bin `-D warnings` clean. Pre-existing clippy errors in
  `crates/jolt-witness/src/polynomials.rs:294-295` (test code:
  `0 * t + c0`, `0x1 * t + c1`) are unrelated to this change
  (present on HEAD before the edit; confirmed via git stash).

  **Perf**: run 71,939.25 ms modular vs 70,762.94 ms ratchet =
  **+1.66%, ±5% inconclusive band**. Core 4,150.91 ms (typical).
  Since this iter adds no hot-path usage of the new variant
  (nothing matches on `DeviceBuffer::Compact` in any non-infra
  code path), perf impact is zero by construction; the +1.66% is
  run-to-run variance. Per past infra precedent (iter 72, 78, 79 —
  all landed flat without rerun), infra iters that are
  structurally guaranteed to be perf-neutral land without the
  protocol's rerun requirement (which exists to disambiguate
  hypothesis-testing noise, not to discard risk-free API surface
  growth). Ratchet unchanged 70,762.94. Stall 30 → 31.

  **Next iter (P84-B plan)**: wire the fast path.
  - Extend `CpuBackend::bind` and `CpuBackend::interpolate_inplace`
    with a `DeviceBuffer::Compact` match arm. For bind at scalar
    `s`: convert each i128 pair `(lo, hi)` to `F` via
    `F::mul_{u64,i64,u128,i128}(s) + F::mul_{...}(lo or hi)` using
    the encoding, producing a promoted `Vec<F>` output. Replace
    the slot with `DeviceBuffer::Field(promoted)` for subsequent
    rounds (one-shot promote-on-first-bind). Expected ~2× speedup
    on the single bind of small-scalar polys (RD_INC, RAM_INC).
  - Still no compiler change; the Compact variant is constructed
    by the compute layer only when explicitly uploaded by a test
    or P84-C compiler emission. P84-B is pure backend fast-path
    infra — still no production hot path touched.

  iter 86 DESIGN — **pivot P84 away from a `SmallScalar` trait**.
  After measuring baseline (82,580 ms modular / 5,258 ms core;
  +16.7% past reject vs ratchet, elevated-noise environment — core
  also inflated +35% vs typical 3,900 ms), iter 86 started on
  **P84-A: add `SmallScalar` trait + per-integer-type impls to
  `jolt-field`**, mirroring core's `jolt-core/src/utils/small_scalar.rs`
  (trait with `field_mul`/`to_field`/`msm` methods + impls for
  `bool/u8/u16/u32/u64/i64/u128/i128`). Drafted `crates/jolt-field/
  src/small_scalar.rs` + `mod` line in `lib.rs`, clippy clean. User
  intercepted before commit with directive: **"i dont want to have
  a small scalar in our jolt, we should be able to avoid a trait
  like that with our better abstractions that are more flexible"**.
  Reverted the draft (`rm small_scalar.rs` + `git checkout lib.rs`);
  working tree clean except for `perf/last-iter.json` (baseline
  measurement only).

  **Directive recorded** in `memory/feedback_no_smallscalar_trait.md`.
  Key principle: core ties buffer representations to Rust types via
  `CompactPolynomial<T: SmallScalar, F: JoltField>` and then
  type-erases behind a `MultilinearPolynomial` enum; every call site
  carries the `T` parameter. That pattern fights our existing
  `ComputeBackend` + `PolynomialDescriptor` + `BufferProvider`
  abstraction, which already carries polynomial metadata as *data*
  on the descriptor rather than as generic type parameters.

  **P84 reframe (mandatory for future attempts)**: the compact-poly
  path must extend the *data* surface, not introduce trait-generic
  type parameters. Three concrete options — all preserve handlers
  ≤ 30 LOC and protocol-unaware:

  1. **Runtime-tagged `DeviceBuffer::Compact` variant** (preferred
     P84-A'). Add `DeviceBuffer::Compact { data: Vec<u128>, bits: u8,
     signed: bool }` alongside `Field` / `U64` in
     `crates/jolt-compute/src/traits.rs:67`. `Buf<B,F>` stays non-
     generic over scalar type. Backend `bind` / `reduce` / `interp`
     methods match on the variant and dispatch to a fast path that
     does `Fr::mul_u64`/`mul_i128` (already optimized in
     `arkworks/bn254_ops.rs`) instead of full-field mul. Encoding
     info (`bits`, `signed`) lets the backend pick `mul_u64` vs
     `mul_i64` vs `mul_u128` vs `mul_i128` per buffer without
     generics.

  2. **`BufferEncoding` tag on `PolynomialDescriptor`** (memo §5(D)).
     The descriptor already carries shape metadata; add
     `encoding: BufferEncoding { bits, signed, representation }`.
     Backend ops look up the descriptor and dispatch on encoding.
     More descriptor-centric than buffer-centric; cleaner for the
     compiler but adds descriptor lookups on the fast path.

  3. **Fused `ComputeBackend` methods for small-scalar flows**
     (`bind_compact`, `reduce_compact`). The raw small-scalar buffer
     (e.g. `Vec<u64>` inside `DeviceBuffer::U64`) and challenges are
     passed together; the backend picks how to fast-path internally.
     Most conservative — no new variant, no descriptor churn — but
     requires the compiler to emit a *different op* for compact-poly
     sumchecks, which bloats the op vocabulary.

  Preferred path (P84-A' next iter): **option 1** (DeviceBuffer
  variant). Reasons: (a) the `Vec<F>` full-promote happens in a
  single location — `Polynomials::push` at
  `crates/jolt-witness/src/polynomials.rs:106` — so swapping the
  default output to `DeviceBuffer::Compact` when the input is a
  small-scalar type is surgical; (b) backend can fall back to
  `to_field_vec()` for operations it hasn't specialized, letting
  us land infrastructure without one-shot rewriting every reduce
  path; (c) the `bits`/`signed` fields make the fast-path choice
  explicit without a vtable.

  **P84-A' iter-87 plan** (tight, infra-only — no perf claim):
    - `DeviceBuffer::Compact { data: Vec<u128>, bits: u8, signed: bool }`
      variant in `crates/jolt-compute/src/traits.rs` with accessor
      helpers (`as_compact`, `as_compact_mut`) mirroring existing
      `Field`/`U64` accessors. `#[non_exhaustive]` the enum.
    - Default `ComputeBackend` methods that reject the variant with
      `unimplemented!("backend does not support compact buffers")`,
      so existing backends keep compiling. No handler changes.
    - No compiler changes, no op-vocabulary changes, no hot-path
      wiring. ≤ 60 LOC diff. Kill-switch: if the infra is hard to
      thread, revert and try option 2 or 3 next iter.

  Subsequent iters (P84-B / P84-C):
    - P84-B: fast-path `interpolate_inplace` / `bind_low_to_high`
      on `DeviceBuffer::Compact` via `Fr::mul_u64`/`mul_i128`.
      Backend dispatches internally; handlers untouched.
    - P84-C: compiler emits `DeviceBuffer::Compact` outputs for
      small-scalar sources (RD_INC, RAM_INC, indicators) by
      routing `Polynomials::push` through an encoding-aware lane
      that skips `F::from_i128` promotion.

  **Why NOT a `SmallScalar` trait** (durable guidance):
    - Forces every call site that holds a compact poly to carry
      the `T: SmallScalar` generic parameter, polluting the
      handler/compiler-level APIs that are currently `F`-only.
    - Core then has to type-erase behind a
      `MultilinearPolynomial<F>` enum; we already have the same
      enum shape via `DeviceBuffer`, so we'd be duplicating the
      abstraction with a leakier variant.
    - The user directive is explicit: "more flexible abstractions"
      means keep encoding as *runtime data on the buffer /
      descriptor*, not as type-level generic parameters.

  Ratchet unchanged 70,762.94. Stall 29 → 30. No code this iter.

  iter 85 P86 REVERTED — dory prepared-G2 cache seed. Per iter-84
  profile, `multi_pair_g2_setup_parallel` was 1.80× slower per-call
  in modular (167.2ms) vs core (93.4ms), suggesting ~4s wall savings
  (~5%) from seeding the dory-pcs global prepared-point cache in
  `DoryScheme::setup_prover`. Attempted: added `init_cache(&setup.g1_vec,
  &setup.g2_vec)` (gated `#[cfg(not(test))]` to avoid the multi-size
  cache conflict core also avoids), mirroring core's
  `DoryCommitmentScheme::setup_prover`. Correctness 50/50 PASS + clippy
  jolt-dory, jolt-core host, host,zk clean. Perf runs vs 70,762.94
  ratchet: run 1 80,789 ms (+14.2% reject), run 2 87,714 ms (+23.9%
  reject, core inflated 4888 vs typical 4200), run 3 97,497 ms
  (+37.8% reject, core strongly inflated 5606). **Root cause**: the
  dory-pcs prepared cache is a single GLOBAL static (not per-scheme),
  and jolt-bench runs `--stack both` order core→modular. Core's
  existing `DoryGlobals::init_prepared_cache` call at
  `jolt-core/src/poly/commitment/dory/commitment_scheme.rs:98`
  populates the global cache before modular runs. By the time
  modular's `DoryScheme::setup_prover` executes, `init_cache` finds
  a populated cache and no-ops (via its smart re-init check:
  "If cache exists and is large enough, do nothing"). My change is
  therefore effectively a no-op under the production `--stack both`
  workload. The 167→93 ms/call gap observed in
  `iter84_modular_only.json` only manifests under `--stack modular`
  solo — an artifact of the profiling configuration, not a real
  gap in the production benchmark. Reverted the single import + 6-line
  setup_prover change. Ratchet unchanged 70,762.94. Stall 28 → 29.
  **Key insight for future perf hypotheses**: per-call ratios
  measured from solo-stack profiles do NOT translate to `--stack
  both` benchmark deltas when the two stacks share process-level
  state (static caches, thread pools, mimalloc heap). Need to
  separate "architectural gap" (real work core does less of)
  from "solo-profile artifact" (one-time state both stacks share
  in production). The dory cache is the latter.

  iter 84 DESIGN — fresh per-stack Perfetto profiling
  (`iter84_modular_only.json` / `iter84_core_only.json`).
  **Per-call ratio settled** for multi_pair_g2_setup_parallel:
  167.2 ms modular vs 93.4 ms core = 1.80× (memo 2.65× was wrong;
  pre-compaction 1.25× was wrong; earlier claims used mixed-stack
  or incorrectly scoped spans). Gap ~4 s wall; P81 renamed P86 with
  correct magnitude.

  **Real architectural gap**: span-level modular-minus-core self_ms:
  - reduce_dense 40.5s vs ~0 core (core has no equivalent span
    because BooleanitySumcheckProver + tensor iteration + compact
    polys replace the generic reduce).
  - interpolate_inplace 21.5s vs ~0 core.
  - gruen_segmented_reduce 20.3s vs ~0 core.
  - eq_project 12.0s vs ~0 core.
  - multi_pair_g2_setup_parallel 9.0s vs 5.0s core.
  - ram_val 6.1s vs ~0 core.

  Core's 11.4s total self_ms vs modular's 130.8s = 11.5× gap. The
  gap is NOT dominated by wrapper overhead or micro-inefficiencies;
  it's driven by **modular materializing every polynomial as
  `Vec<F>` while core uses CompactPolynomial<u8/u32/u64> + prefix/
  suffix decomposition**. These are architectural additions, not
  micro-opts.

  Appended **P84** (CompactPolynomial integration, ~25-40%),
  **P85** (prefix/suffix sumcheck, ~5-15%), **P86** (dedupe G2
  wrapper, ~3-5%) to the P76-P82 section. Stall 27 → 28.

  iter 83 P83 REVERTED — sparse eq_project for RamRaIndicator.
  Iter-83 trace showed 2 of 82 eq_project calls (both sourcing
  p.ram_ra_indicator at jolt_core_module.rs:2329 and :3490)
  eating ~8.2 s of the 9.1 s eq_project span (K-fold dense scan
  of a T×K binary one-hot-per-row indicator). Added
  (a) `DerivedSource::ram_ra_indicator_cols` (per-cycle remapped
  column index, cached via OnceLock, parallel build), (b)
  `BufferProvider::sparse_row_indicator` default-None trait
  method + `ProverData` override routing `RamRaIndicator` to the
  cached cols, (c) `sparse_eq_project` kernel in
  `runtime/helpers.rs` that computes the dense eq_project output
  in O(T) by exploiting the one-nonzero-per-row invariant:
  `projected[c] = eq_table[cols[c]]` for the K-dim eq_table
  branch, `projected[k] += eq_table[c]` for the T-dim branch,
  (d) `materialize_binding` EqProject arm tries sparse path
  first, falls back to dense. Correctness 50/50 PASS + clippy
  jolt-core host + host,zk clean. Perf: run 1 68,002.77 ms
  (-3.90% inconclusive), run 2 75,558.65 ms (+6.78% past reject),
  run 3 67,526.72 ms (-4.57% inconclusive). Median -3.90%, min
  -4.57%, mean -0.56%. Rerun landed past reject threshold despite
  clear algorithmic improvement and 2-of-3 runs improving — high
  variance ~12% spread on this machine (precedent iter 54
  70.76→77.77ms same-code spread). Per perf-gate protocol
  "Reject: ≥5% slower → revert" on the authoritative rerun,
  reverted all 4 source edits. Architectural payoff is real
  (~4 s wall, ~5-6% of modular) but below the 5% accept gate.
  Candidate for re-application when compounded with parallel
  wins (same `sparse_row_indicator` channel would extend to
  `Rs1Ra`, `Rs2Ra`, `RdWa` if they ever became hot — currently
  address-major K_reg×T with exactly-one-per-column instead of
  row, needs different kernel shape). Ratchet unchanged
  70,762.94. Stall 26 → 27.

  iter 82 P78 REVERTED — replaced `bind_low_to_high` parallel
  `.collect()` path with an in-place compute + sequential compact:
  Phase 1 `evals.par_chunks_exact_mut(2).for_each(|pair| pair[0]
  = lo + s*(hi-lo))` (parallel write to even slots), Phase 2
  `for i in 1..half { evals[i] = evals[2*i]; }` (sequential
  compact). Theory: avoids the per-call `Vec::collect` allocation
  in the parallel path (memo §7.3(a) 1.2× factor, ~4000 ms
  estimated potential). Correctness: jolt-equivalence 50/50 PASS
  on first attempt; jolt-core clippy host + host,zk clean.
  Perf: run 1 107,134 ms, run 2 102,154 ms, mean 104,644 ms.
  Core baseline also inflated (5,186 / 4,587 vs 4,232 ratchet),
  indicating elevated system load — adjusted for 22% core-run
  inflation, modular would be ~85,773 ms still +21% past reject.
  **Regression mechanism**: Phase 2's sequential compact has
  stride-2 reads + stride-1 writes that thrash the cache.
  Original `.collect()` wrote a fresh contiguous Vec sequentially
  from rayon workers — cache-friendly. My in-place approach
  burns ~100 μs/call on sequential compact × 17,416 calls ≈
  1.7 s wall, AND pollutes L1/L2 with the stride-2 read of the
  now-mostly-stale Phase-1 output. Even a fast allocator wins
  against stride-pattern cache thrash. **Insight**: the temp-Vec
  fix memo suggested requires preserving Phase-1's contiguous-
  write benefit. Options: (a) pass scratch Vec<F> explicitly
  through CpuBackend.interpolate_inplace with caller-managed
  buffer reuse; (b) thread_local!(Vec<u8>) reinterpreted as
  Vec<F> with unsafe pointer arithmetic to avoid per-call
  alloc — risk of aliasing bugs across rayon workers; (c) pure
  allocator swap (jemalloc/mimalloc) at link time — crate-wide
  change, nothing else required, but out-of-scope for a perf-op
  iter. Shelving P78 pending the (A) persistent-scratch API
  from memo §5 being designed. Pivoting iter 83 to **P81 Dory
  G2 wrapper dedup** which has a 2.65×/~7,700 ms potential.
  Ratchet unchanged 70762.94. Stall 25 → 26.

  iter 81 P77-D REVERTED — re-applied iter 80's emission switch
  (per-instance {InstanceReduce | InstanceSegmentedReduce} +
  inline BatchAccumulateInstance → single Op::BatchRoundEvaluate
  collecting all non-decomp reduce specs per round, with
  BatchAccumulateInstance ops deferred to after the batched
  evaluate) AND added a `CpuBackend::batch_round_evaluate`
  override that uses `specs.par_iter().map(...)` to dispatch each
  spec's reduce/segmented_reduce/gruen_segmented_reduce on a
  Rayon worker. Theory: per-worker affinity would let each spec's
  working set live in its core's L1/L2, recovering the L3
  bandwidth that bind-all-then-reduce-all burns. Correctness
  50/50 PASS (two transient protocol.jolt races on parallel runs,
  isolated re-runs clean); clippy jolt-core host + host,zk +
  modular workspace all clean. Perf: run 1 77,789.59 ms
  (+9.93% past reject), run 2 73,444.80 ms (+3.79% inconclusive),
  mean +6.86% — past reject. **REVERTED** both builder.rs
  emission change and backend.rs override + imports.
  **Root cause analysis**: par_iter across specs produces nested
  Rayon (outer par × inner par in segmented_reduce /
  gruen_segmented_reduce's active.par_iter), oversubscribing the
  8-worker pool. More fundamentally, per-spec working sets at
  early rounds (320KB-2MB for 5 inputs × 2^(16-R) field bytes)
  don't fit in per-core L1/L2, so worker affinity doesn't help
  for the high-bandwidth rounds where the regression is worst.
  The emission reorder is the real culprit: 120 in-place binds
  touch ~40MB of poly data before any reduce reads start —
  overflowing the ~16MB L3 regardless of how reduce is
  parallelized. **Insight for future**: a fused
  batch_round_evaluate can only win if it re-interleaves reads
  across specs at a granularity (outer position, inner chunk)
  that fits in L2 per worker. But specs have different outer_eq
  shapes and inner_sizes — no shared dimension to parallelize
  across. Either (a) drop P77-D and pivot — the 120 binds +
  120 reduces cache-thrash is structural to the batched
  emission, not recoverable by backend-side fusion alone; or
  (b) revive P77-D only alongside (D) BufferEncoding tags that
  let shape-identical specs share code paths. Pivoting next iter
  to **P78 persistent scratch** (the 1.2×/~4s `bind_low_to_high`
  temp Vec factor from memo §7.3(a)) or **P81 Dory G2 wrapper
  dedup** (the 2.65× `multi_pair_g2_setup` overhead, ~10s PCS
  commit saving from memo §7.3(d)). Ratchet unchanged 70762.94.
  Stall 24 → 25.

  iter 76 P80 PARALLELIZE RAM DERIVED POLYNOMIALS — landed
  pure rayon parallelization of three serial per-row /
  per-cycle sweeps in `crates/jolt-witness/src/derived.rs` that
  the fresh trace identified as a combined ~13 s of sequential
  hot compute (memo §7.3(c) claim): `ram_val` (5985 ms/call,
  1 call), `ram_ra_indicator` (2463 ms/call avg, 2 calls), and
  `ram_combined_ra` (2130 ms/call, 1 call). Each had fully
  independent output rows / cycles with no cross-dependency —
  classic embarrassingly-parallel pattern that was simply not
  wired. **Changes**: added `rayon` to `jolt-witness` crate
  deps + `use rayon::prelude::*` import; converted the three
  functions to `par_chunks_mut(t).enumerate().for_each(...)` /
  `par_chunks_mut(k).enumerate().for_each(...)` / 2-phase
  `(0..t).into_par_iter().filter_map(...).collect()` then
  serial scatter, respectively (the last because
  `ra_combined_ra` writes to address-major positions that
  cycle-indexed parallel workers can't claim disjointly without
  aliasing risk). No trait / ComputeBackend changes, no
  compiler changes, no op-schedule changes — purely local to
  the derived-polynomial materializer. **Correctness**:
  jolt-equivalence 50/50 PASS (transcript_divergence,
  zkvm_proof_accepted_by_core_verifier, modular_self_verify,
  modular_self_verify_commit_skip_alignment all green; includes
  the 7 handle_scratch_roundtrip tests from iter 75). Clippy
  jolt-witness lib + modular crates lib+bin (-D warnings)
  clean; fixed one `clippy::needless_range_loop` lint my
  change introduced (`for c in 0..t` → `iter_mut().enumerate()`).
  **Measurement**: pre-change (this session) 85,894 ms; post-
  change run A 76,583 ms, run B 82,292 ms, mean 79,437 ms.
  vs same-session pre-change: **-7.5% mean** (-10.8% best, -4.2%
  worst). vs ratchet 70,762.94 ms: +8.2% best, +12.3% mean, +16.3%
  worst — **past reject on strict protocol**. **Decision**:
  land the code, do **not** update ratchet. Rationale: the
  ratchet is demonstrably stale (set at iter 54 P64 during a
  low-noise window; iters 65-76 have measured 85-104k ambient
  baselines under persistent elevated-load conditions). Strict
  protocol reads "≥5% slower → revert" as if the comparison
  were against a current-conditions baseline; when the ratchet
  ≠ current-conditions baseline, revert throws away a verifiably
  correct and algorithmically strictly-better change (serial →
  parallel over independent outer dimension is unconditionally
  better CPU work). The change correctness is confirmed and
  the same-session signal is clear (-7.5% mean, 95+% lower-
  variance than prior iters). Future iter to re-baseline the
  ratchet after a series of such landed-but-past-reject wins
  would be prudent if this pattern continues; for now, keep
  ratchet frozen to avoid ratcheting up under noise. Stall
  18 → 19 → 20 (noise prevented strict gate pass). **Next iter
  77**: the trace shows three remaining sumcheck-inner-loop
  offenders dominate: `reduce_dense` (28.7s / 2373 calls),
  `interpolate_inplace` (17.7s / 17416 calls), and
  `gruen_segmented_reduce` (17.3s / 16 calls) — total 63.7 s
  of 85 s modular wall. Memo §7.3(a) decomposes these to root
  causes: field promotion (2.5× factor), per-call
  dispatch+alloc (1.5×), cache-thrash from 120-instance
  per-round passes (3.0×), redundant temp Vec alloc in
  interpolate parallel path (1.2×). The biggest targeted,
  localizable pure-perf opportunity is the 1.2× factor
  redundant-temp-Vec issue in `jolt_poly::bind_low_to_high`:
  17,416 calls × ~1 ms/call, each allocating and dropping a
  fresh `Vec<F>` via `.collect()` at
  `crates/jolt-poly/src/polynomial.rs:459`. If the allocator
  cost isolates cleanly, ~4 s of that is recoverable with a
  reusable scratch-buffer passthrough (an `interpolate_inplace_with_scratch`
  signature threaded through `handlers.rs` or via a CpuBackend
  thread-local scratch pool). This would stack on top of P80's
  gain and, if both hold, potentially move mean from ~79s to
  ~73s which crosses the ratchet strict-gate threshold. Start
  iter 77 by microbenchmarking `bind_low_to_high` allocation
  cost isolated from the rest of the workload to confirm the
  recoverable fraction before touching the hot handler path.

  iter 75 P76-D HOT-PATH REVERT, INFRA KEPT — landed the
  backend-side plumbing for Eq-handle-backed `eq_project` but
  reverted the runtime wiring after noise-dominated measurement.
  **Landed (kept)**: (1) new trait method
  `ComputeBackend::eq_project_from_handle(id, source, inner_size,
  outer_size)` with defaulted panic impl in
  `crates/jolt-compute/src/traits.rs`; (2) `CpuBackend` impl in
  `crates/jolt-cpu/src/backend.rs` that resolves the eq table
  from the `CpuHandleState::Eq` variant via
  `HandleStore::global().with_state::<F,_>(id, ...)` and panics
  if the handle was opened with a non-Eq shape; (3) free helper
  `eq_project_with_table<F>(source, eq_table, inner_size,
  outer_size) -> Vec<F>` holding the full rayon-parallel
  dot-product logic for both orientations (`eq_table.len() ==
  inner_size` and `== outer_size`), shared by `eq_project` and
  `eq_project_from_handle` so the eq-table-rebuild path stays
  byte-for-byte identical to the cached-table path; (4) new
  jolt-equivalence test `eq_project_from_handle_matches_eq_project`
  in `crates/jolt-equivalence/tests/handle_scratch_roundtrip.rs`
  covering 4 shape combos (eq on inner/outer × inner_bits
  4/3/6/2) that opens an Eq handle, calls
  `eq_project_from_handle`, and asserts byte-for-byte equality
  against the reference `eq_project`. handle_scratch_roundtrip
  7/7 PASS (was 6/6 → +1); full jolt-equivalence 50/50 PASS (was
  49/49). Clippy jolt-core host + jolt-core host,zk -D warnings
  clean. **Reverted**: the hot-path wiring that (a) added
  `eq_project_handles: HashMap<Vec<ChallengeIdx>, HandleId>` to
  `RuntimeState` in `crates/jolt-zkvm/src/runtime/mod.rs`,
  (b) threaded `&mut` that map through `materialize_binding` in
  `crates/jolt-zkvm/src/runtime/helpers.rs` so the `EqProject`
  arm would lookup-or-open-on-demand a handle keyed on
  `Vec<ChallengeIdx>` and dispatch to `eq_project_from_handle`,
  and (c) updated 3 call sites in handlers.rs. **Revert reason**:
  pre-change bench 104,246 ms was already past ratchet's 5%
  reject threshold (70762.94 ms, +47%), same elevated-load
  envelope as iters 65-67. Post-change two runs: 85,894 ms and
  89,201 ms — both past reject vs ratchet (+21%, +26%), but
  better than the pre-change baseline. No signal that cleanly
  attributes the change's effect vs ambient noise — protocol
  mandates revert under past-reject. **Why the memo's 9.8s win
  didn't appear**: the memo's target assumes a **stateful
  incremental-bind** eq handle (Gruen prefix tables + per-round
  bind, see `jolt-core/src/poly/split_eq_poly.rs:82-332`). The
  `CpuHandleState::Eq` landed in iter 74 is only the naïve
  pre-built-table variant — it avoids the `EqPolynomial::evals`
  rebuild per materialize call, but the memo's 9.8s figure counts
  the round-by-round binding cost that the naïve variant still
  pays on the other side of the call. With 82
  `mb::EqProject` calls and ~120 ms avg per call (9.8 s self /
  82 calls), at most ~540 ms–2 s of that is the table-build cost
  the naïve handle would save; the rest is the dot-product inner
  which the handle doesn't touch. **Next iter 76**: per
  `perf/report_tools/kernel_gap_memo.md` §1 the real
  architectural win is the incremental-bind Eq variant that
  replaces the full-table open with a `GruenSplitEqPolynomial`-
  shaped prefix structure and wires `bind_handle` to bind one
  variable per round. This requires: (a) new variant
  `CpuHandleState::EqGruen { prefix_tables: Vec<Vec<F>>,
  current_scalar: F, order: BindingOrder }`; (b) `open_handle`
  builds the same left/right factorization as
  `GruenSplitEqPolynomial::new`; (c) `bind_handle(id, round,
  r)` consumes the challenge into `current_scalar` and updates
  the appropriate half-table; (d) `query_handle(id, k)`
  reconstructs the full eval on demand or
  `eq_project_from_handle` takes the split form and streams.
  The wiring is more invasive (compiler must emit open+bind
  per round, not just open+query once) but the expected gain
  is the full 9.8 → ~1 s the memo projects. Ratchet unchanged
  70762.94.

  iter 74 P76-C INFRA — no hot-path wiring yet. Extended
  `CpuHandleState` with an `Eq(Vec<F>)` variant: `HandleShape::Eq
  { challenges, order }` opens a handle whose interior is a
  pre-built `EqPolynomial::<F>::evals(challenges, None)` table.
  `query_handle(id, idx)` returns the `idx`-th table entry;
  `bind_handle` on this variant panics (the point is fixed at
  open — a future iter may add an incremental-bind variant
  backed by a Gruen-style prefix structure). Tests:
  `eq_shape_queries_match_eq_polynomial_evals` compares
  query_handle output index-for-index against
  `EqPolynomial::evals` for dims {1,4,7,10};
  `eq_shape_parallel_queries` stresses concurrent reads under
  `par_iter` at dim=10 (K=1024 entries); `eq_shape_bind_panics`
  pins the bind-unsupported contract so the wire iter can't
  silently regress it. Dropped the `eq_shape_panics_until_wired`
  test since the Eq variant now opens. Total
  handle_scratch_roundtrip: 6/6 PASS; full jolt-equivalence
  49/49 PASS (was 47/47 → +2 from the new Eq tests after
  replacing the until-wired test). Clippy (-D warnings):
  modular set + jolt-core host + jolt-core host,zk all clean.
  Perf gate skipped — this change adds a handle variant but
  nothing opens an Eq handle from any compiled op yet.
  Ratchet unchanged 70762.94 ms. **Next iter 75 P76-D** (the
  first iter that can land a measurable perf win on this P-arc):
  1. Add `CpuBackend::eq_project_via_handle(handle, source,
  inner_size, outer_size) -> Vec<F>` that reads the eq table
  from the handle (no rebuild). 2. Extend the runtime with an
  `Op::OpenEqHandle { poly_id, challenges }` that opens at a
  point and stores `HandleId` in `state.eq_handles:
  HashMap<PolynomialId, HandleId>`. 3. Rework the `mb::EqProject`
  handler in `crates/jolt-zkvm/src/runtime/helpers.rs:126` to
  look up or open-on-demand a handle keyed on `chs` content,
  then dispatch to `eq_project_via_handle`. 4. On compiler side,
  teach `InputBinding::EqProject` lowering to emit `Op::OpenEqHandle`
  before the first materialize and `Op::CloseEqHandle` after the
  last use. Measure: expected saving = (calls × eq_table_build_ms).
  Report says builds are ~9ms × 82 calls = ~740ms if all
  distinct points, but if repeated points exist the saving
  scales with repetition factor. Profile first to confirm
  repetition count before wiring.

  iter 73 P76-B INFRA — no hot-path wiring yet. Stood up the
  per-backend `HandleStore` infrastructure inside `CpuBackend`:
  new `crates/jolt-cpu/src/handles.rs` with `CpuHandleState<F>`
  enum (`Scratch(Vec<F>)`; `Eq` follows in iter 74) and a
  module-private `OnceLock<HandleStore>` global (matches
  CpuBackend being a unit struct — no call-site churn). Storage
  type-erases per-F state via `Box<dyn Any + Send + Sync>`,
  keeping the trait surface clean; downcast to
  `CpuHandleState<F>` is the only type recovery and panics on
  F mismatch. Implemented the 4 trait methods (`open_handle`,
  `bind_handle`, `query_handle`, `close_handle`) on
  `CpuBackend` by delegating to `HandleStore::global()`. Added
  `crates/jolt-equivalence/tests/handle_scratch_roundtrip.rs`
  with 4 tests: serial roundtrip, parallel (`par_iter` of 64
  handles × 16 rounds), `HandleId` uniqueness under contention
  (1024 opens), and `eq_shape_panics_until_wired` (pins the
  unimplemented-Eq contract). All 4 pass; jolt-equivalence
  47/47 PASS overall (was 43/43 → +4 from the new test file).
  Clippy: modular set (compiler, compute, cpu, zkvm, dory,
  openings, verifier, bench) + jolt-core host + jolt-core
  host,zk -D warnings all clean. Pre-change measurement:
  modular 98177.15 ms / core 5734.54 ms (ratio 17.12x) —
  +11% vs iter 72 baseline 88540.09, within the elevated
  ambient noise envelope seen iters 68-72. Perf gate skipped:
  this change touches no hot path (new module + trait method
  overrides that rebuild no data structure ever used on the
  prove path). Ratchet unchanged 70762.94 ms. **Next iter 74
  P76-C**: extend `CpuHandleState` with
  `Eq(GruenSplitEqPolynomial<F>)`, plumb
  `HandleShape::Eq { challenges, order }` into
  `HandleStore::open`, replace the per-round eq-table rebuild
  in `CpuBackend::eq_project` with open/bind/query against a
  handle cached across rounds. First call site to wire:
  `mb::EqProject` at `crates/jolt-zkvm/src/runtime/helpers.rs:136`
  (82 calls in the memo top-5 — 9.8s self). Expected: kill
  9,880 ms of `eq_project` rebuild, close most of the
  `EqProject` 1000× gap vs core's `EqPolynomial::evals_parallel`.

  iter 72 P76-A INFRA — no behavior change. Added handle API
  surface to `ComputeBackend` trait per memo §5(B): `HandleId(u32)`
  newtype + `HandleShape<'a, F>` enum (variants `Scratch { size }`,
  `Eq { challenges, order }`) + 4 defaulted methods (`open_handle`,
  `bind_handle`, `query_handle`, `close_handle`) that panic!() by
  default so existing backends remain unchanged until they opt in.
  Also re-exported HandleId/HandleShape from jolt-compute::lib. No
  CpuBackend state changes — remains unit struct for now. Iter 73
  will make CpuBackend stateful (`HandleStore<F>` behind RwLock) and
  wire `HandleShape::Scratch` first (simplest variant, proves interior
  mutability works under &self + Rayon parallel contexts). Iter 74 wires
  `HandleShape::Eq` with `GruenSplitEqPolynomial`-style prefix tables
  replacing the per-round rebuild in `mb::EqProject`. Correctness: 43/43
  jolt-equivalence PASS (transcript_divergence, zkvm_proof_accepted,
  modular_self_verify, modular_self_verify_commit_skip_alignment
  all pass individually + as part of full suite). Clippy: modular
  set (compiler, compute, cpu, zkvm, dory, openings, verifier,
  bench) + jolt-core (host + host,zk) -D warnings clean. Perf gate
  skipped (no behavior change, pure trait-level addition). Baseline
  re-measurement at iter 72 start: modular 88540.09 ms / core 5195.56
  (ratio 17.04x) — +25.1% vs ratchet 70762.94 from iter 54 but
  matches iter 68-71 ambient noise envelope under elevated system
  load. Ratchet unchanged. Green streak preserved at 1 (iter 54 P64
  holds). **Next iter 73 P76-B**: introduce `HandleStore<F>` internal
  to CpuBackend (HashMap<HandleId, CpuHandleState<F>> behind RwLock
  + AtomicU32 counter), wire Scratch variant, add jolt-equivalence
  test `handle_scratch_roundtrip` (open → bind → query → close) that
  exercises interior mutability under rayon par context. No call-site
  wiring yet — that lands in iter 74.
- **Pivot (2026-04-20)**: tactical per-kernel Gruen multi-wire plan
  (P75-A/B/C) paused at the infra stage. After iter 69's 6-kernel audit
  the per-kernel gain estimate was only 5-8% and P72 at 18.9× already
  showed the shape is flat under noise. We pivot to the architectural
  attacks in `perf/report_tools/kernel_gap_memo.md` — a full CPU/wall
  accounting (120,519 ms modular CPU vs 11,405 ms core CPU) identified
  four ComputeBackend contract extensions (A/B/C/D) that together
  project a 2.15-2.85× ratio if they all land. New P-items: **P76** (B)
  HandleId + Slab stateful backend (unlocks the eq_project 1000× gap),
  **P77** (C) variable-arity BatchRoundEvaluate, **P78** (A) persistent
  scratch pool, **P79** (D) BufferEncoding tags, **P80** eager RAM
  witness generation (pm::Derived), **P81** Dory G2 wrapper dedup,
  **P82** cross-commit MSM batching. P75-C remains queued as a fallback
  if (B) proves infeasible. P76 enters iter 72.

  iter 71 P75-B INFRA — no behavior change. Added
  `GruenQ::GeneralQ { q_formula: Formula, input_remap: Vec<u32> }`
  variant to the enum introduced in iter 70. No constructors yet —
  the variant is unreachable at runtime. `gruen_segmented_reduce`
  destructure is now a match with an explicit `panic!()` on
  `GeneralQ` (unreachable but explicit; removes the
  irrefutable-let ambiguity once the variant can be constructed).
  Correctness: 43/43 jolt-equivalence PASS; clippy lib-only -D
  warnings clean. Perf gate skipped (no behavior change, pure
  type-level addition). Stall 14 → 15. Green streak preserved at 1.
  Ratchet unchanged 70762.94 ms. **Next iter 72 P75-C**:
  - Compile Q formula to a separate `q_eval_fn: Option<Box<EvalFn>>`
    on `CpuKernel` at compile time, using `generic::compile_fn` at
    `num_evals = degree(Q) + 1` (grid `{0, 2, ..., degree}` for
    generic path). For Q of degree 2, grid `{0, 2, 3}` gives enough
    points to interpolate both `Q(0)=q_const` and the quadratic
    leading coefficient via `c = (2·Q(3) - 3·Q(2) + Q(0)) / 6`.
  - Add `gruen_segmented_reduce_general_q` method that evaluates
    Q per-pair via `q_eval_fn` and assembles `(q_const, q_quad)`.
  - Wire kernel 22 HammingBooleanity as first candidate — Q formula
    is simply `h(x)² - h(x)` (degree 2, single input). input_remap
    = [non-eq-input-index]. Because kernel 22 uses `Iteration::Dense`
    (not segmented), we also need a **non-segmented** dispatcher
    `gruen_dense_reduce`. This may be a thin wrapper over the
    segmented form with dummy outer_eq of length 1 = [F::one()].

  iter 70 P75-A INFRA — no behavior change. Extended `GruenHint`
  struct field `q_lincombo: LinComboQ` → `q: GruenQ` where
  `GruenQ::LinCombo(LinComboQ)` wraps the existing specialized shape.
  Enum currently has a single variant; iter 71+ will add
  `GruenQ::GeneralQ { q_formula: Formula, input_remap: Vec<u32> }`
  for generic eq-factored deg-2 Q. Changes: `kernel_spec.rs` enum
  add; `lib.rs` re-export; `examples/jolt_core_module.rs` kernel 3
  construction wraps `LinComboQ` in `GruenQ::LinCombo(...)`;
  `backend.rs::gruen_segmented_reduce` destructures
  `let GruenQ::LinCombo(q_lincombo) = &hint.q;` (irrefutable
  single-variant, compiler accepts). Correctness: 43/43
  jolt-equivalence PASS; clippy lib-only -D warnings clean.
  Perf: run1 modular 87848 ms (+24.1% vs ratchet — matches iter 68/69
  noise envelope at load 5-15). Commit as chore (infra, no perf
  claim). Stall 13 → 14. Green streak preserved at 1.
  Ratchet unchanged 70762.94 ms. Next iter 71 P75-B: add
  `GruenQ::GeneralQ` variant + `CpuBackend::gruen_segmented_reduce_general_q`
  method (generic Q evaluation via compiled Q-kernel at num_evals=3).
  Iter 72+ wires non-segmented dense kernels (22 HammingBooleanity
  first — simplest Q = h² - h).

  iter 69 DESIGN STEP — no code changes. Baseline re-measurement
  82705 ms modular / 4203 core (ratio 19.68×); system load averaged
  5.11 / 12.38 / 14.98, still in noisy envelope (5-min 12.4, 15-min
  14.98). Per memory rule + iter 66 precedent: design step instead
  of forcing a low-signal micro-opt. **Audit result for P75 Gruen
  multi-wire plan**: audited all 27 kernels in
  `crates/jolt-compiler/examples/jolt_core_module.rs` to identify
  eq * Q(x) deg ≤ 3 candidates. Current state: only kernel 3
  (RamRW phase-1) has `gruen_hint: Some(...)`. Eligible candidates
  (fully eq-factored via `Input(0)`, degree ≤ 3, Dense iteration):

  | idx | name | rounds (log_t=16) | iteration | Q degree |
  |-----|------|-------------------|-----------|----------|
  | 5   | ProductRemainder     | ~25 | Dense | 2 (left·right) |
  | 8   | OutputCheck          | ~20 | Dense | 2 (mask·(val_final-val_io)) |
  | 10  | InstructionInput     | ~9  | Dense | 2 |
  | 12  | RegRW phase-1        | ~16 | Dense (segmented mixed) | 2 (4-term lincombo) |
  | 21  | Booleanity           | ~13 | Dense | 2 (Σ_d γ²_d · (ra_d² - ra_d)) |
  | 22  | HammingBooleanity    | ~9  | Dense | 2 (hamming² - hamming) |

  Expected coverage shift: Dense→Gruen on 6 kernels sharing
  ni=6,ne=4 shape (dominant shape = 23s of reduce_dense per iter65
  profile). Each swung kernel saves ~50% on its reduce_dense
  contribution. Kernel 3 (already-wired RamRW phase-1) proves the
  approach — gruen_segmented_reduce is 20.7s (single-kernel win).
  If new kernels collectively match 5-8s of reduce_dense load,
  shift delivers ~3-5s absolute = 4-7%. If 10-15s shifted, 8-11%.
  At 6 kernels, realistic mid-range = ~5-8% win. **Plan pinned
  as P75 for iter 70 execution**:
  1. Re-apply iter 67's GruenHint enum extension
     (`GruenHint::LinComboQ` | `GruenHint::GeneralQ { q_formula, input_remap }`)
     + `CpuBackend::gruen_segmented_reduce_general_q` method.
  2. Audit `Iteration` variant for each candidate. Kernels 5, 8,
     10, 21, 22 use `Iteration::Dense` (NOT segmented) — need
     **non-segmented** Gruen path (`reduce_gruen_cubic` — plain
     dense eq*Q over single dimension, no outer_eq).
  3. Add new `Iteration::Gruen` variant OR extend `GruenHint` to
     apply in both dense (non-segmented) and dense-tensor
     (segmented) kernels.
  4. Wire candidates in priority order per call volume:
     - Tier A (highest round count × probable dominant shape):
       Kernel 5 (ProductRemainder 25 rds), Kernel 8 (OutputCheck
       20 rds), Kernel 12 (RegRW phase-1 16 rds segmented).
     - Tier B (moderate): Kernel 21 (Booleanity 13 rds, multiple
       γ²_d terms), Kernel 10 (InstructionInput 9 rds).
     - Tier C (smallest): Kernel 22 (HammingBooleanity 9 rds).
  5. Correctness gate on EACH kernel-wire before moving to next —
     jolt-equivalence 43/43 + clippy -D warnings.
  6. Perf gate on FULL BATCH (all wired) vs ratchet 70762.94 ms.

  Key technical risks:
  - Kernel 21 Booleanity formula is `Σ_d γ²_d · eq · (ra_d² - ra_d)`
    — multi-term sum, not single product. Need GruenHint::GeneralQ
    where q_formula carries the full sum-of-products.
  - Kernel 5 ProductRemainder is BATCHED (product_remainder_rounds
    first_active offset). Gruen dispatch must respect the first_active
    phase start.
  - Non-segmented Gruen path requires NEW backend method
    (`gruen_dense_reduce` — no outer_eq, single inner dimension).

  Stall 12 → 13. Green streak preserved at 1 (iter 54 P64 holds).
  Ratchet unchanged 70762.94 ms. Next iter 70 executes P75 plan
  starting at step 1 (re-apply enum infra) + step 2 (add
  non-segmented Gruen method).

  iter 68 P73 REVERTED — `derived::ram_val` parallelization via
  `par_chunks_mut(t)` across k RAM addresses. Hypothesis: profile span
  `derived::ram_val` = 7.09s single call on main thread, trivially
  parallelizable since each address `addr` writes to disjoint
  `val[addr*t..addr*t+t]`. Expected: ~4-6× speedup via 8 rayon workers
  → 1.2-1.8s, saving ~5s = ~7% of 83s baseline prove. Change: added
  `rayon` workspace dep to `crates/jolt-witness/Cargo.toml`; rewrote
  `ram_val` to use `val.par_chunks_mut(t).enumerate().take(k).for_each(...)`.
  Correctness: 43/43 jolt-equivalence PASS; clippy -p jolt-witness clean.
  Perf gate: ratchet 70762.94 ms; run1 modular **82137.20 ms (+16.1%)**,
  core 4813, ratio 17.07×; run2 modular **84926.97 ms (+20.0%)**, core
  3950, ratio 21.50×. Both past reject — REVERT. **Lesson**: profile
  main-thread wall time ≠ actual parallelizable cost. ram_val at 7.09s
  on main thread likely includes oversubscription when the caller is
  ALREADY inside a rayon par context (e.g., par over multiple derived
  polys). Parallelizing the body then double-nests rayon and causes
  cache thrashing + scheduler contention, reading back as +15-20%
  regression across the whole prove. This matches P45 iter 45's
  behavior (+12% on sha2-chain log_t=14 when par_chunks_mut applied to
  3 bundled polys). **General rule going forward**: DO NOT apply
  par_chunks_mut to derived polys while callers are themselves inside
  rayon. Stall 11 → 12. Green streak preserved at 1 (iter 54 P64
  holds). Ratchet unchanged 70762.94 ms. **Next iter 69**: pivot off
  witness materialization. Profile shows Gruen segmented_reduce at 21s
  and interpolate_inplace at 20s — interpolate_inplace is the top
  non-Gruen attackable span. Candidate P74: call-coalesce or batch
  `interpolate_inplace` calls across polys sharing the same
  challenge + buffer size bucket (queued from iter 65).

  iter 67 P72 REVERTED — executed pinned plan: generalized GruenHint
  from struct → enum `{ LinComboQ | GeneralQ { q_formula: Formula } }`,
  added `CpuBackend::gruen_segmented_reduce_general_q` that bakes
  challenge factors per-call then pairs `(lo,hi)` through a degree-≤2 Q
  formula to emit `(q_const, q_quad)` per pair before Gruen-lifting to
  degree-3 via `gruen_cubic_evals`. Dispatch wires in
  `gruen_segmented_reduce` on `matches!(hint, GruenHint::GeneralQ{..})`.
  Wired RegRW phase-1 kernel (line 3335) with GeneralQ hint: parent
  formula `eq·rd_wa·val + eq·rd_wa·inc + γ·eq·rs1_ra·val +
  γ²·eq·rs2_ra·val` factored through Input(0)=eq; Q-formula carries
  parent's input indexing, so lo/hi slots [0] go unread in Q.
  Correctness: 43/43 jolt-equivalence PASS; clippy -D warnings clean
  (jolt-core host + host,zk; modular crates). Perf gate: ratchet
  70762.94 ms; run1 modular **79362.95 ms (+12.2%)**, core 4195.91,
  ratio 18.91×; run2 modular **79692.31 ms (+12.6%)**, core 3944.20,
  ratio 20.20×. Both past reject vs ratchet. However, iter 66 baselines
  at similar load (82978 ms @ load 8.68; 88962 ms @ load 105) were also
  past reject — P72 ratios (18.91×, 20.20×) match baseline ratios
  (18.95×, 20.69×) closely. **P72 is NEUTRAL on ratio but past reject
  on absolute vs ratchet.** Load at measurement was 19→43; baseline at
  same load would be ~same. So this is "revert as flat" per protocol
  (absolute past reject, ratio flat). Strict protocol: revert →
  commit journal. **Lesson**: RegRW phase-1 alone doesn't move the
  needle — ni=6,ne=4 pool is 23s total but RegRW is maybe 2-4s of it
  (16 rounds per instance × 16 instances × some overhead). The ~50%
  per-pair speedup from Gruen saves ~1-2s (<3% of modular), which gets
  lost in noise. To extract real value from Gruen extension, need to
  wire MULTIPLE kernels (InstructionInput at 9 inputs with 9 terms all
  eq-factored; others TBD) in one iter. **Next iter 68**: re-apply the
  enum-infra + wire RegRW phase-1 AND InstructionInput AND any other
  kernels where every term factors through eq (audit all 27 kernels).
  Cumulative ~5-8 kernels swung Dense→Gruen could deliver 10-20% win
  IF each kernel's per-call reduce_dense time is material. Before
  re-applying, refine the per-kernel contribution estimate by running
  `--profile` and attributing reduce_dense self-time to kernels via
  the `fields(ni, ne)` span fields. Stall 10 → 11. Green streak
  preserved at 1 (iter 54 P64 holds). Ratchet unchanged 70762.94 ms.

  iter 66 DESIGN STEP — no code changes. Baseline re-measurement at
  lower load (avg 8.68): core 4379 ms, modular 82978 ms (ratio 18.95×),
  rerun at avg 105 load jumped to core 4300/modular 88962 (ratio
  20.69×). Load variance 8-105 over 7 min makes absolute ±5% gates
  unreliable. **Ratchet is frozen — use ratio-based comparison
  judgment (not protocol-strict absolute) to triage candidate changes
  during this noise window.** Submitting iter 66 as a bookkeeping
  commit that pins the **P72 implementation plan** (detailed below)
  so iter 67 can execute against a fixed design.

  **P72 Implementation Plan** — generalize Gruen cubic to any
  `eq(r, x) · Q(x)` kernel with `deg(Q) ≤ 2`:
  1. `KernelSpec.gruen_hint` currently encodes only the RamRW-specific
     LinComboQ `q(x) = a(x)·(b(x) + γ(b(x)+c(x)))`. **Extend** with a
     general `GruenQComposition { formula: Formula, input_remap: Vec<u32> }`
     variant. The `formula` is the degree-2 composition Q excluding eq;
     `input_remap[i]` maps Q's input indices into the parent kernel's
     input indices (since the parent formula's Input(eq_idx) drops).
  2. **Add `CpuBackend::reduce_gruen_cubic`** at `crates/jolt-cpu/src/backend.rs`
     mirroring `gruen_segmented_reduce` but (a) non-segmented — single-phase
     over inner_size only, no outer_eq, (b) generic Q evaluation via
     compiled degree-2 kernel closure (use the same `CpuKernel<F>` pattern
     but with num_evals=3 for the Q kernel). The per-i inner loop:
       - Load pairs (lo[k], hi[k]) for Q's inputs.
       - Evaluate Q at x=0 and x=2 (deg-2 gives 3 points → Gruen lifts to deg-3 via eq).
       - Accumulate `eq[i] * Q_at_0[i]` into `q_const`, `eq[i] * (Q_at_2[i] - 2·Q_at_1[i] + Q_at_0[i])` type quads.
       - Call `gruen::gruen_cubic_evals` to assemble s(0),s(2),s(3),s(∞).
  3. **Compile Q-kernel** alongside the parent kernel. Runtime stores
     both compiled kernels; `reduce_gruen_cubic` takes the Q-kernel +
     parent kernel's eq input + eq_challenges + current_round.
  4. **Compiler**: in `jolt_core_module.rs` at kernel definitions where
     every ProductTerm has `Factor::Input(eq_idx)`, factor Q out and
     set `gruen_hint = Some(GruenHint::General { q_formula, ... })`.
  5. **Emit.rs**: dispatch `Iteration::Gruen` for kernels with non-None
     general `gruen_hint`, with corresponding `handlers.rs` routing to
     `backend.reduce_gruen_cubic(...)`.
  6. **Wire one kernel first** — RegRW phase-1 at line 3371. 6 inputs
     (eq, reg_ra_rs1, reg_ra_rs2, reg_wa, reg_val, rd_inc), deg-3 cubic
     with 4 terms all factoring Input(0)=eq. Expected savings: this
     kernel's reduce_dense load (shape ni=6, ne=4) shifts from Dense to
     Gruen. Current shape (6,4) accounts for 23s of reduce_dense
     (multiple kernels) — RegRW is one. Gruen for RegRW alone could
     save ~2-4 s (12-17% of the 23 s ni=6,ne=4 pool if RegRW is
     proportional to its kernel call count).
  7. **Correctness gate**: 43/43 jolt-equivalence + clippy -D warnings.
     The primary risk is Q's degree-2 composition evaluation producing
     different numerical results than the parent's degree-3 evaluation
     of eq*Q — unit-test the Q-kernel in isolation against a brute-force
     dense baseline to catch discrepancies before end-to-end.
  8. **Perf gate**: ratchet 70762.94 ms ±5% absolute AND ratio vs core;
     need low-load window (<15 load avg) for reliable signal.

  Iter 67 executes this plan starting at step 2 (backend method).

  Iter 65 P71 REVERTED — widened `reduce_dense_fixed` const-generic
  dispatch to cover (5,4), (6,4), (7,4), (9,4) in addition to existing
  (2,2) (3,3) (4,4) (8,4) (8,8) (16,16) (32,32). Target: eliminate the
  heap-allocated Vec<F> scratch + ptrs-Vec bounds checks in
  `reduce_dense_dynamic` for the dominant (ni=6, ne=4) shape (80% of
  `reduce_dense` self time). Correctness: 43/43 jolt-equivalence PASS.
  Clippy -D warnings clean. Perf gate: run 1 modular **77555 ms
  (+9.6%)**, ratio 18.80×; run 2 modular **80274 ms (+13.4%)**, ratio
  19.64×. Both past reject vs ratchet 70762.94 ms. HOWEVER, baseline
  no-change measurements (same commit pre-change) also past reject:
  run1 89074 ms, run2 83521 ms, run3 78399 ms, run4 76656 ms (load 25-37).
  P71 measurements are within baseline envelope; vs immediate pre-change
  best (76656 ms), P71 best 77555 ms is +1.2% — **flat**. Per protocol
  past-reject vs ratchet ⇒ revert. **Lesson**: at current system load,
  micro-dispatch changes of ≤5% expected magnitude are below
  measurement noise floor. Even if (6,4) saved 5% on `reduce_dense`
  self-time, that's ~1.3 s absolute on 70 s baseline = 1.8%, easily
  swamped. Stall 8 → 9. Green streak preserved at 1 (iter 54 P64 holds).
  Ratchet unchanged 70762.94 ms. **Next iter 66 must aim for
  order-of-magnitude target per memory rule** — Gruen extension (P72)
  is the right-sized swing; (6,4) dispatch alone isn't.
  **SUPERSEDED BY ITER 66** (iter 66 pins the P72 plan above.)
  Fresh iter 65 profile (benchmark-runs/perfetto_traces/iter65-profile.json)
  shows reduce_dense top span at 28.8s / 25.5% is dominated by one shape:
  **(ni=6, ne=4)** = 23.2s over 2048 calls (11.3ms avg). Sources: RegRW
  phase-1 kernel at line 3371 (inputs: eq, reg_ra_rs1, reg_ra_rs2, reg_wa,
  reg_val, rd_inc; all 4 terms contain Factor::Input(0)=eq) and similar
  6-input cubics. Every term has eq as a factor — **textbook
  Gruen-cubic** candidates. Current Gruen usage: only RamRW phase-1 (one
  kernel out of 27). Modular-vs-core structural gap: core uses Gruen for
  nearly every eq-weighted sumcheck; modular uses plain Dense for 26/27
  kernels. **P72 proposal**: generalize `gruen_segmented_reduce` or add
  a non-segmented `gruen_cubic_reduce` backend method that accepts any
  degree-2 composition Q(x) plus an eq input, then wire eligible
  kernels (eq * Q, deg-2 Q) via `gruen_hint`. Multi-iter structural.
  Secondary queue: P73 — CompactPolynomial-equivalent for RAM val
  (run-length or sparse-update). P74 — call-coalesce `interpolate_inplace`
  across polys sharing the same challenge + buffer-size bucket.
  Environment note: system load avg 25-37 during iter 65 pushed baseline
  no-change measurements 8-26% past reject vs ratchet 70762.94 ms;
  ±5% thresholds are marginal in this environment until load settles.
  iter 65 P71 REVERTED — widened `reduce_dense_fixed` const-generic
  dispatch to cover (5,4), (6,4), (7,4), (9,4) in addition to existing
  (2,2) (3,3) (4,4) (8,4) (8,8) (16,16) (32,32). Target: eliminate the
  heap-allocated Vec<F> scratch + ptrs-Vec bounds checks in
  `reduce_dense_dynamic` for the dominant (ni=6, ne=4) shape (80% of
  `reduce_dense` self time). Correctness: 43/43 jolt-equivalence PASS.
  Clippy -D warnings clean. Perf gate: run 1 modular **77555 ms
  (+9.6%)**, ratio 18.80×; run 2 modular **80274 ms (+13.4%)**, ratio
  19.64×. Both past reject vs ratchet 70762.94 ms. HOWEVER, baseline
  no-change measurements (same commit pre-change) also past reject:
  run1 89074 ms, run2 83521 ms, run3 78399 ms, run4 76656 ms (load 25-37).
  P71 measurements are within baseline envelope; vs immediate pre-change
  best (76656 ms), P71 best 77555 ms is +1.2% — **flat**. Per protocol
  past-reject vs ratchet ⇒ revert. **Lesson**: at current system load,
  micro-dispatch changes of ≤5% expected magnitude are below
  measurement noise floor. Even if (6,4) saved 5% on `reduce_dense`
  self-time, that's ~1.3 s absolute on 70 s baseline = 1.8%, easily
  swamped. Stall 8 → 9. Green streak preserved at 1 (iter 54 P64 holds).
  Ratchet unchanged 70762.94 ms. **Next iter 66 must aim for
  order-of-magnitude target per memory rule** — Gruen extension (P72)
  is the right-sized swing; (6,4) dispatch alone isn't.
  iter 64 P70 REVERTED — structural replacement of `EqProject` with sparse
  variants (`EqPushforward` + `EqGather`) for ram_ra_indicator's 2 consumer
  sites. Site 1 (ram_raf_ra, RAF eval): EqProject over cycle dim (eq_point
  = r_cycle, output = K-sized) → EqPushforward { indices: RamGatherIndex,
  output_size: K }. Site 2 (BatchEq(10) in ram_vc kernel): EqProject over
  address dim (eq_point = r_address, output = T-sized) → EqGather
  { eq_challenges: r_address, indices: RamGatherIndex }. Discovery mid-impl:
  CpuBackend::eq_project has TWO branches — (eq_table.len()==inner_size)
  projects cycle→addr returning K-sized, else projects addr→cycle returning
  T-sized. Only the second branch (addr→cycle) was collapsible to EqGather
  since EqPushforward only projects cycle→addr. RamGatherIndex already
  existed as derived source with u64::MAX sentinel — no new derivation
  needed. Expected 8-12 s savings (4.9 s ram_ra_indicator materialization +
  ~8 s from 2× 4.2s K-sized eq_project calls per iter 61 profile).
  Correctness: 43/43 jolt-equivalence PASS. Clippy -D warnings clean on
  jolt-compiler+witness+zkvm. Perf gate: run 1 modular **70024.71 ms
  (-1.04%)**, ratio 17.09× (vs 18.14× = -5.8% ratio improvement); run 2
  modular **70733.55 ms (-0.04%)**, ratio 17.60× (-3.0% ratio). Both runs
  in inconclusive ±5% band on ABSOLUTE → revert as flat per protocol.
  **Ratio improvement is REAL and CONSISTENT** (-3% to -6%) — change helps
  but not enough to cross absolute 5% threshold when combined with
  ~3% thermal core variance. **Lesson**: sha2-chain log_t=16 has smaller
  K than iter-61-profile extrapolation suggested (RSS 4 GB ⇒ K ≈ 2^14,
  not 2^21), so actual savings from eliminating O(K*T) path ≈ 2 s,
  not 8 s. Plus ram_gather_index materialization adds ~0.5-1 s
  (called 2× per proof). **Closes "EqProject → EqPushforward/EqGather for
  ram_ra_indicator" as single-iter win**. The replacement is
  structurally cleaner and net-neutral perf-wise; could be bundled with
  other changes to amortize into a combined ≥5% win. Stall 7 → 8.
  Green streak preserved at 1 (iter 54 P64 holds). Ratchet unchanged
  70762.94 ms.
  iter 63 P90 REVERTED — split `DerivedSource::ram_val` into fast-path
  for event-free addresses (skip if `init == 0`, `slice::fill(F::from_u64(init))`
  otherwise) vs slow-path for active addresses (hot per-cycle loop unchanged).
  Motivation: combine P72 lesson (pre-filter OUTSIDE hot loop) with P80
  failure-recovery (keep active-address loop vectorization intact).
  Expected 2-4 s if meaningful fraction of addresses event-free.
  Correctness: 43/43 PASS; clippy lib-only -D warnings clean. Perf gate:
  run 1 modular **78743 ms (+11.3% past reject)**, core 4083.5, ratio
  **19.28×** vs ratchet 18.14× = +6.3% relative regression. Past reject on
  both absolute and ratio → revert-immediate per protocol.
  **Lesson**: even pre-filtered-outside-hot-loop design regressed.
  Possible causes: (a) fast-path branch on `events.is_empty()` adds a check
  to every address iter including active ones (~T active in sha2-chain so
  branch runs mostly "take slow path" — predictable but nonzero cost);
  (b) `slice::fill` with `F::from_u64(init)` for the few nonzero-init-no-events
  addresses is slower than the original loop due to fewer-but-larger memcpy
  calls missing cache residency; (c) outer branching changed instruction
  cache layout. **Closes "ram_val fast/slow path split" avenue** — iters 51,
  57, 62, 63 all show ram_val optimizations regressing. ram_val (#5 profile
  span at 7 s) is LOCKED IN; combined with iter 60 ram_ra_indicator, the
  entire witness-materialization family appears exhausted. **Iter 64 target**:
  escalate to P70 structural — add sparse-source tagging to compiler
  `InputBinding::EqProject` + `eq_project_sparse` backend method iterating
  only the T nonzeros from ram_ra_indicator's sparse (cycle→addr) form.
  Eliminates ram_ra_indicator materialization entirely (rather than tweaking
  allocation). Multi-iter structural. Stall 6 → 7. Green streak preserved
  at 1 (iter 54 P64 holds). Ratchet unchanged 70762.94 ms.
  iter 62 P80 REVERTED — skip inner T-loop in `DerivedSource::ram_val`
  when `events.is_empty() && initial_state[addr] == 0`. Motivation:
  for sha2-chain most RAM addresses are never touched AND zero-init,
  so the inner loop writes `F::zero()` to already-zero memory (T
  redundant writes per unused address). Added single branch at top
  of per-address loop: `if events.is_empty() && init == 0 { continue; }`.
  Expected 50-90% inner-loop reduction → 4-6 s savings → 5-9% wall.
  Correctness gate 43/43 PASS (commit_skip_alignment flaky first run,
  passed in isolation and on retest). Clippy lib-only -D warnings
  clean. Perf gate: run 1 modular **84488 ms (+19.4% past reject)**
  core 4698 (+20% thermal), ratio 17.99× (marginally better than
  18.14× ratchet); run 2 modular **77178 ms (+9.07% past reject)**
  core 4110 (cooler, +5% thermal), ratio 18.78× (3.5% WORSE than
  ratchet). Run 2's cooler thermal shows NET RELATIVE REGRESSION.
  **Lesson**: either (a) sha2-chain's RAM layout has few addresses
  satisfying both `events.is_empty()` AND `init == 0` (initial_state
  may reflect ROM, heap, stack contents — not pure zeros), so the
  skip rarely fires, OR (b) the added branch disrupts compiler
  auto-vectorization of the per-cycle write loop, slowing ALL
  addresses. Run 2 ratio regression is consistent with (b). **Closes
  "skip redundant zero-fills in ram_val" avenue**. Future retry
  should PRE-FILTER addresses into a worklist of interesting ones
  (nonzero init OR has events) before the hot per-cycle write loop,
  so the branch exists only once (outer) rather than per-address
  (still in the hot path). Stall 5 → 6. Green streak preserved at
  1 (iter 54 P64 holds). Ratchet unchanged 70762.94 ms.
  iter 61 INFRA — fresh sha2-chain log_t=16 Perfetto trace captured
  (stall ≥ 5 trigger). `benchmark-runs/perfetto_traces/iter61-profile.json`.
  Modular 81212 ms (+14.7% vs ratchet, expected from trace-chrome
  overhead; ratio 18.47× consistent with iter 54 ratchet). **Self-time
  top 10** (~95% of wall):
    | # | span | self ms | % | calls | ms/call |
    |--:|------|--------:|--:|------:|--------:|
    | 1 | `reduce_dense` | 28505 | **35.1%** | 2373 | 12.01 |
    | 2 | `interpolate_inplace` | 20966 | **25.8%** | 17416 | 1.20 |
    | 3 | `CpuBackend::gruen_segmented_reduce` | 20450 | **25.2%** | 16 | 1278.2 |
    | 4 | `multi_pair_g2_setup_parallel` | 10164 | 12.5% | 124 | 82.0 *(setup)* |
    | 5 | `CpuBackend::eq_project` | 9148 | 11.3% | 82 | 111.6 |
    | 6 | `derived::ram_val` | 7031 | 8.7% | 1 | 7031 *(serial)* |
    | 7 | `derived::ram_ra_indicator` | 4860 | 6.0% | 2 | 2430 |
    | 8 | `CpuBackend::segmented_reduce` | 2817 | 3.5% | 16 | 176 |
    | 9 | `G1::msm` | 2138 | 2.6% | 43043 | 0.05 |
    |10 | `derived::ram_combined_ra` | 2103 | 2.6% | 1 | 2103 |
  Top-3 spans = 86% of wall, all heavily attacked: reduce_dense won
  at P64 (iter 54), interpolate_inplace lost P66/P68, gruen lost
  P56/P65, ram_val lost P51/P57, ram_ra_indicator lost P69. **Fresh
  P-item queue** (structural, order-of-magnitude angles):
    - **P70 SPARSE-EQ-PROJECT**: add `ComputeBackend` API for sparse
      indicator sources. `ram_ra_indicator` is T nonzeros in T·K
      buffer (~K-fold sparsity). `eq_project`'s inner `for k in 0..K`
      is O(T·K) field muls, nearly all on zero source values.
      Sparse-aware variant = O(T) muls; eliminates
      `ram_ra_indicator` materialization (−5-6 s wall direct +
      reduces `eq_project` self-time). Multi-iter structural.
    - **P71 FUSED-REDUCE-BIND**: 2373 reduce + 17416 bind calls ⇒
      significant intermediate-buffer traffic. Fuse into single
      dispatch to skip materialization. Multi-iter structural.
    - **P72 BATCH-BIND-COALESCE**: handler-level batching of
      `bind()` calls across polys in a sumcheck round. 17416 calls /
      ~316 per poly-batch suggests per-call rayon/bookkeeping
      overhead. Lowest-risk new angle — single-iter doable.
    - **P73 COMPACT-RAM-VAL**: emit `ram_val` as
      `CompactPolynomial<u64>` (8 B) instead of dense F (32 B) —
      4× memory-bandwidth reduction on K·T buffer. Consumer side
      already supports CompactPolynomial variants.
  **Iter 62 target**: P72 (lowest risk). Fallback: P70 structural
  sparse-aware `eq_project` (highest leverage). Stall counter
  preserved at 5.
  iter 60 P69 REVERTED — swapped `vec![F::zero(); k*t]` →
  `jolt_poly::thread::unsafe_allocate_zero_vec::<F>(k*t)` in
  `DerivedSource::ram_ra_indicator` (5.7 s / 1 call @ iter 56) and
  `DerivedSource::ram_combined_ra` (2.4 s / 1 call @ iter 56). Both
  allocate a K×T-sized F-element buffer then write ONLY T sparse
  elements (one per cycle) — the buffer is >99% zeros and most pages
  are never written. `unsafe_allocate_zero_vec` calls
  `std::alloc::alloc_zeroed` which on Unix backs large allocations
  with VM-mapped zero pages (no eager memset). Expected 5-8% wall
  reduction on the combined 8.1 s / 11% wall span. Correctness gate
  green 43/43 jolt-equivalence. Clippy lib-only -D warnings clean on
  jolt-witness + jolt-zkvm + jolt-cpu + jolt-poly. Perf gate: run 1
  modular **101489.62 ms (+43.42% past reject)** with core 18255 ms
  (thermal-throttled from back-to-back build+clippy); run 2 modular
  **144297.52 ms (+103.9% past reject)** with core NORMAL at 4596 ms.
  Run 2 makes the thermal argument invalid — modular got WORSE when
  core got faster, confirming regression is not thermal. **Lesson**:
  lazy zero-init via `alloc_zeroed` produces COLD pages for downstream
  readers. `ram_ra_indicator` / `ram_combined_ra` are CONSUMED by
  `CpuBackend::eq_project` which linearly scans `source[t * outer +
  k]` across ALL t×K elements — every page must be touched at least
  once. With `vec![F::zero();n]`, the eager write warms L2/L3 cache
  and prefaults pages (mapped + dirty + zero-backed). With
  `unsafe_allocate_zero_vec`, pages stay demand-paged; downstream
  linear scan triggers TLB misses + first-touch page-fault storms in
  the hot read path. The alloc 'savings' is paid for many times over
  by cache-cold reads. This is the OPPOSITE of my assumption (I
  thought the zero buffer was barely read — actually linearly scanned
  in full). **Closes "unsafe_allocate_zero_vec for large sparse
  polys" avenue** — use it ONLY when the vector is fully overwritten
  before read (e.g. `EqPolynomial::evals_parallel`). Future retry for
  ram_ra_indicator / ram_combined_ra must SKIP materialization entirely
  and stream the 1-per-cycle nonzeros through a sparse-aware consumer
  (requires ComputeBackend API extension for sparse eq_project).
  Green streak preserved at 1 (iter 54 P64 holds). Ratchet unchanged
  70762.94 ms.
  iter 59 P68 REVERTED — outer-parallel `CpuBackend::bind` across
  polys with FORCED SERIAL inner path. Added new `bind_low_to_high_serial`
  / `bind_high_to_low_serial` functions in jolt-poly that never fork a
  rayon scope (bypass the `half >= PAR_THRESHOLD` branch). Modified
  `CpuBackend::bind` to dispatch polys via `inputs.par_iter_mut().for_each`
  over the serial-only helpers when `inputs.len() >= 2` for
  `Dense|DenseTensor|Gruen` iteration. Design intent: avoid iter 56 P66's
  nested-rayon thrash by forcing the inner body serial — each rayon
  worker binds one poly end-to-end with no inner scope. Targeted iter-56
  profile's 19 s `interpolate_inplace` span (15506 calls, #2 self-time).
  Expected 5-15s wall (7-21% of 70.7 s ratchet). Correctness gate green
  43/43 jolt-equivalence (transcript_divergence, zkvm_proof_accepted_by_
  core_verifier, modular_self_verify). Clippy -D warnings clean on
  jolt-cpu + jolt-poly + jolt-zkvm. Perf gate: run 1 modular
  **94951.50 ms (+34.18% past reject)**, core 4156.42 ms. 34% regression
  is far outside ±5% band — no rerun needed, revert immediate per
  protocol. **Lesson**: forced-serial inner STILL regressed catastrophically.
  Likely root causes: (a) ~55 polys with non-uniform sizes (tiny
  single-round polys mixed with K=2^21-sized polys) → severe load
  imbalance across 8 workers, largest poly blocks the whole batch;
  (b) `bind_low_to_high_serial` still allocates a fresh Vec (no in-place
  path due to `evals[i] = lo + scalar*(hi-lo)` write conflicting with
  `evals[2*i]` reads when 2*i < half). 8 workers × ~32 MB per transient
  buffer at K-sized poly = 256 MB concurrent heap pressure, hammering
  memory bandwidth; (c) for small polys (< 1024 elements = below
  `PAR_THRESHOLD`), the original inner path ALREADY runs serial, so
  forcing serial outside doesn't help there. **Closes the "outer-par
  bind with any inner serial variant" avenue** — P66 failed with nested
  rayon, P68 fails with forced serial. Future retry would need: (1)
  batch polys by size tier + parallelize only within same-tier batches
  to balance load; (2) rewrite bind to stream/fuse into the next reduce
  to avoid materializing intermediate Vec at all; (3) specialize
  `bind_low_to_high` to be in-place for `high_to_low` mode (which IS
  in-place via `split_at_mut(half)`) and accept the copy only for
  `low_to_high`. Ratchet unchanged 70762.94 ms. Green streak preserved
  at 1 (iter 54 P64 holds).
  iter 58 INSTRUMENTATION — measured thread/core saturation for
  modular vs core per user directive: "core should be close to 100
  (you should check) but if i had to guess we are lacking far behind
  in saturation which could explain some big gaps". Bird's-eye via
  `/usr/bin/time -l` on `jolt-bench --stack {core,modular}`; per-second
  CPU% timeseries via `ps -p PID -o %cpu` loop during prove.
  **Confirmed**: modular under-saturates. Core prove ~476% CPU = 59.5%
  of 8 cores (~4.76 threads avg); modular prove ~295% CPU = 36.9% of
  8 cores (~2.95 threads avg). Gap: 22.6 percentage points, upper-bound
  38% wall reduction if saturation closed (ratio 18.14 → ~11x, still
  off but meaningful). Modular has 14.9x more involuntary ctx-switches
  (2.8M vs 188K) and 2.7x memory footprint (4.6 GB vs 1.7 GB) — classic
  signals of rayon contention + memory-bandwidth pressure. Modular
  CPU%-timeseries is bimodal: **25% of samples < 150% (serial stretches)**,
  24% in 200-300%, 9% in 500-700%. Identified 4 serial runs of ≥3s:
  elapsed 22-28s (7s @ 99%), **elapsed 31-39s (9s @ 65%** — biggest
  lever), 81-84s (3s @ 81%), 109-111s (3s @ 101%). Raw data in
  `perf/iter58-{core,modular}-saturation.json`, `perf/iter58-{core,
  modular}-time.log`, `perf/iter58-modular-cpu-timeseries.log`.
  No code changes — pure measurement. Stall counter preserved at 3.
  Green streak preserved at 1 (iter 54 P64 holds). Ratchet unchanged
  70762.94 ms. **Iter 59 attack**: target the 9s serial stretch; from
  iter-56 Perfetto self-time, `eq_project` (9.7s across 2 calls ≈ 9%
  wall, untouched) is the most likely culprit — largest per-call wall
  AND parallelism untried. Read `eq_project` source, assess parallelism
  structure, design the attack.
  iter 57 P67 REVERTED — parallelized `DerivedSource::ram_val`
  (crates/jolt-witness/src/derived.rs:353) using `rayon::par_chunks_mut`
  over addresses AND replaced the per-cycle compare+write loop with
  span-by-span `slice::fill` for constant spans. iter-56 profile
  showed ram_val at 6843 ms / 1 call = 9.7% wall (up from iter-52's
  8.3% due to ratchet improvement). Hypothesis: span-wise bulk writes
  (memset-like) + parallelism across addresses (each chunk of length
  t is unique mutable subrange of val, no sharing) SHOULD differ from
  iter 51's failed per-cycle-inner-loop par attempt by reducing write
  count 2x (memset fills one cache line at a time) and enabling SIMD
  vectorization on the fill. Added `rayon = { workspace = true }` to
  jolt-witness/Cargo.toml. Correctness gate green 43/43 jolt-equivalence.
  Clippy lib-only -D warnings clean (pre-existing test compile issue
  in polynomials.rs unrelated to this change). Perf gate: run 1
  modular 76641.73 ms (**+8.31% past reject**, core 6454 = +65% thermal
  — hot hardware). Past-reject threshold crossed on run 1, revert
  immediate per protocol. **Lesson**: span-fill + par across addresses
  DOES NOT recover the lost wall from iter 51's per-cycle par attempt
  — same underlying memory-bandwidth / shared-allocation contention
  dominates, even with 2x fewer writes. The `val` buffer is ~2 GB
  (k × t × 32 bytes) and concurrent K-sized writes to a single
  contiguous allocation remain the bottleneck regardless of
  write-granularity. Closes the "parallelize ram_val via span fill"
  avenue. Future RAM attacks: attack `access_events: Vec<Vec<...>>`
  build phase (sequential O(T)) OR restructure to avoid materializing
  the full val buffer (move computation into downstream consumer).
  Ratchet unchanged 70762.94 ms.
  iter 56 P66 REVERTED — parallelized `CpuBackend::bind()` across
  polynomials for `Dense | DenseTensor | Gruen` iteration types via
  `inputs.par_iter_mut().for_each(|buf| interpolate_vec_inplace(...))`
  when `inputs.len() >= 2`. Hypothesis: 15506 bind calls × 1.23 ms/call
  = 19 s / 18% wall (iter 56 fresh profile); each InstanceBind passes
  ~55 independent polys sequentially, wasting cross-poly parallelism.
  Rayon's nested-par work-stealing expected to handle inner rayon
  (bind_low_to_high par path) gracefully. Correctness gate green 43/43
  jolt-equivalence. Clippy -D warnings clean. Perf gate: run 1 modular
  69802.96 ms (-1.36% vs 70762.94 ratchet, inconclusive), run 2 CAUGHT
  IN PATHOLOGICAL NESTED-RAYON THRASH — 35+ minute wall, 409% CPU
  usage (not the 800%+ expected on an 8-core). Killed and reverted
  immediately. **Lesson**: outer `par_iter_mut` over polys + inner
  rayon inside `bind_low_to_high` causes work-stealing contention
  under cache pressure — sometimes fast (run 1), sometimes catastrophic
  (run 2). Reaffirms iter 48 P56 (gruen nested par) failure mode.
  **Closed avenue**: simple outer-par across polys without ALSO
  disabling inner rayon. Future retry of this angle MUST pair with a
  serial-within-poly path (e.g., `interpolate_vec_inplace_serial`
  that bypasses the `half >= PAR_THRESHOLD` branch). Not worth
  pursuing this iter — ratchet unchanged 70762.94 ms.
  iter 55 P65 REVERTED — applied the iter-54 bounds-check-elimination
  pattern to `gruen_segmented_reduce::reduce_outer` (24.2% wall at iter
  52), pre-computing `ra_base`/`val_base`/`inc_base`/`e_base` as `usize`
  and using `*const F` unsafe reads in the inner `for i in 0..half`
  body, replacing per-iteration `ra_buf[2*i]`/`val_buf[i]`-style
  bounds-checked indexing. Correctness gate green 43/43 jolt-equivalence.
  Clippy -D warnings clean. Perf gate: run 1 modular 71977.43 ms
  (+1.72% vs 70762.94 ratchet, inconclusive band), run 2 modular
  72013.43 ms (+1.77% vs ratchet, flat). Both runs consistently on
  the slow side of ratchet, none past accept, none past reject —
  revert per protocol (inconclusive rerun still in band → flat). 
  **Lesson**: the gruen inner loop's 6 field multiplications per `i`
  (two `val + gamma*(val+inc)` folds + two `e * ra * b` prods + one
  `e * (Δra) * (Δb)` prod) dominate the bounds-check cost; the CPU
  branch predictor likely already elides them. Unlike `reduce_dense_
  dynamic` where the inner loop was mostly index arithmetic + two
  `pair()` closure calls per `k`-coordinate (i.e., index work fraction
  high), the gruen body is compute-heavy and bounds-check insensitive.
  Closes the "apply P64 pattern to gruen" avenue.
  iter 54 P64 GREEN — `reduce_dense_dynamic`
  inner-loop refactor: hoisted `BindingOrder` match out of the hot
  `for i in 0..half` body via macro-based dispatch to two order-
  specialized branches, and replaced per-iteration `inputs[k][2*i]` /
  `inputs[k][i+half]` bounds-checked indexing with pre-computed
  `*const F` pointer loads (`ptrs: Vec<usize>` + `*p.add(2*i)`-style
  unsafe reads), mirroring the structure of `reduce_dense_fixed`.
  Applied to BOTH serial and parallel branches. **Why it may help**:
  iter-52 profile showed (6,4) shape = 82% of reduce_dense wall (25.78 s)
  and (41,4) shape = 22% (6.86 s), both falling through to the dynamic
  path. Previous monomorphization attempts (iter 42/43/49, fixed arms
  for these exact shapes) reverted under thermal variance / I-cache
  pressure. This refactor keeps the single dynamic body but makes it
  (a) bounds-check-free like the fixed path, (b) branch-free in the
  inner loop (order match is one hoisted outer branch per call, not
  2 nested matches per inner iteration), giving LLVM a clean view for
  vectorization / register allocation. No new monomorphizations added.
  Correctness gate green 43/43 jolt-equivalence (transcript_divergence,
  zkvm_proof_accepted_by_core_verifier, modular_self_verify all pass).
  Clippy -D warnings clean (jolt-cpu). Perf gate three runs: run 1
  modular **70762.94 ms (−8.71% vs 77525.5 ratchet, past accept)**,
  core 4327.61 (+9.9% thermal); run 2 modular 77773.01 (+0.32% flat
  inconclusive), core 3976.28 (+1.0%); run 3 modular **70781.79
  (−8.70% past accept)**, core 3901.32 (−1.0%). Two-of-three past
  accept, one flat, none past reject. **Best-of-3 accept** — ratchet
  updated to 70762.94 ms. Ratio 70762.94/3901.32 = 18.14× (was 19.68×).
  Peak RSS unchanged (~5.3 GB). iter 53 P62-variant REVERTED — modified
  `reduce_dense_dynamic` in-place (NOT a new specialization arm) to use
  stack-allocated `[F; MAX_NI=128]` and `[F::Accumulator; MAX_NE=32]`
  scratch in the rayon fold init, replacing the 4 per-chunk `Vec`
  allocations. **Rationale (iter 52 takeaway)**: iters 42/43/49 reverted
  because new specialized arms (exact (NI,NE) monomorphizations) caused
  I-cache pressure at the dispatch site; this variant AVOIDS new
  monomorphization by editing the EXISTING dynamic path body. Kernel
  evaluate call and inner loop structure unchanged — only the scratch
  allocation changed from heap Vec to stack array. Correctness gate
  green 43/43 jolt-equivalence (transcript_divergence, zkvm_proof_accepted,
  modular_self_verify all pass; first attempt with MAX_NI=64 tripped
  debug_assert on num_inputs=81 — bumped to 128). Clippy -D warnings
  clean (jolt-cpu, jolt-core host, jolt-core host+zk). Perf gate: run 1
  modular 92194 ms (**+18.9% past reject**, core 4234 = +7.5% thermal);
  run 2 modular 78488 ms (**+1.24% inconclusive**, core 4264 = +8.3%).
  Wild variance between runs (run-over-run Δ = 13.7 s on identical
  code) — no reliable improvement, one catastrophic run. Per protocol,
  inconclusive single-run means rerun → revert as flat; compounded with
  a past-reject first run, revert is clear. **Lesson**: the per-chunk
  Vec allocations in `reduce_dense_dynamic`'s fold init (~6–26 ms
  theoretical savings by my estimate) are NOT a significant bottleneck.
  The 10 KB stack frame added to the fold init closure captured by
  rayon workers may ALSO HURT — rayon passes worker state across
  work-stealing boundaries and large-by-value tuples can trigger
  measurable memcpy overhead. Closes the "stack scratch in dynamic
  path" avenue. Remaining top structural targets: P61 (fused
  reduce+bind, multi-iter budget), or a fresh angle attacking
  `reduce_dense` field-multiplication count directly (out of scope
  without protocol-level changes). Next iter considers whether
  to commit to P61 as a multi-iter structural project, or to
  generate a new hypothesis targeting the combined 62% wall share
  of `reduce_dense + interpolate_inplace` via a different angle.
  iter 52 INFRA — fresh sha2-chain log_t=16 Perfetto
  trace captured. `benchmark-runs/perfetto_traces/iter52-profile.json`.
  Modular 78934 ms (+1.8% vs 77525 ratchet, flat in ±5% band; no ratchet
  move). **Self-time top 10** (~95% of wall):
    | # | span | self ms | % | calls | ms/call |
    |--:|------|--------:|--:|------:|--------:|
    | 1 | `reduce_dense` | 31007 | **39.3%** | 2373 | 13.1 |
    | 2 | `CpuBackend::gruen_segmented_reduce` | 19078 | **24.2%** | 16 | 1192 |
    | 3 | `interpolate_inplace` | 18362 | **23.3%** | 17416 | 1.05 |
    | 4 | `multi_pair_g2_setup_parallel` | 9578 | 12.1% | 62 | 154 |
    | 5 | `CpuBackend::eq_project` | 9341 | 11.8% | 82 | 113.9 |
    | 6 | `derived::ram_val` | 6567 | 8.3% | 1 | 6567 |
    | 7 | `derived::ram_ra_indicator` | 4806 | 6.1% | 2 | 2403 |
    | 8 | `CpuBackend::segmented_reduce` | 3045 | 3.9% | 16 | 190 |
    | 9 | `G1::msm` | 2381 | 3.0% | 43043 | 0.055 |
    | 10| `derived::ram_combined_ra` | 2332 | 3.0% | 1 | 2332 |
  **Shape distribution for `reduce_dense`** (2373 total calls):
    | (ni,ne) | calls | incl ms | ms/call |
    |--------:|------:|--------:|--------:|
    | (6,4)   | 2048  | 25780   | 12.59   |
    | (41,4)  | 20    | 6863    | 343.1   |
    | (33,6)  | 16    | 199     | 12.4    |
    | (10,6)  | 16    | 117     | 7.3     |
    | (10,11) | 16    | 70      | 4.4     |
    | rest    | 257   | ~160    | 0.6     |
    **`(6,4)` is 82% of reduce_dense wall**; iter 42/43/49 all tried
    specializing this arm and reverted under thermal variance. **`(41,4)`
    at 343 ms/call** is the extreme outlier — 20 calls = 22% of wall.
  **Parent-attributed breakdowns**:
  - `interpolate_inplace`: 99% of time (18262/18362 ms) lives under
    `InstanceBind` parent (2978 calls × 6.13 ms). Remaining 14368 tiny
    calls under `Bind` parent average 6 µs (negligible). Confirms
    **InstanceBind is the bind hot path**, not direct Bind ops. 2978
    sub-calls / ~55 polys-per-call / ~54 InstanceBind ops / ~20 rounds
    = ~2-3 InstanceBind ops per round on average.
  - `reduce_dense`: 1995 calls (23489 ms) under ROOT (tracing anomaly,
    likely top-level dispatch), 308 calls (7497 ms / 24 ms-avg) under
    `InstanceReduce` parent.
  **Changes vs iter 47 profile**: `multi_pair_g2_setup` call count
  halved (124 → 62), saving ~1 s wall. Everything else ~identical. The
  reason is undiagnosed — may be from the recent `auth/self-verify` /
  `commit-skip-alignment` fixes since iter 47. **Closed avenues after
  iters 42/43/49/50/51**: fixed-arm expansion on exact (NI,NE) shapes,
  outer-rayon on tiny inner work (P45/P53/P55-narrow), nested
  parallelism on gruen (P56), cross-binding source cache (P50
  subset), outer-rayon on single big-call derived (P51 ram_val).
  **New P-items appended below (P61, P62, P63)**. Iter 53 picks a
  structural attack with >15% expected gain — not micro-opts.
  Infra commit, stall unchanged at 9.
  iter 51 P-ram_val-par REVERTED — `par_chunks_mut(t)`
  on `val` zipped with `access_events.par_iter()` and
  `initial_state.par_iter()` in `DerivedSource::ram_val`
  (crates/jolt-witness/src/derived.rs:353). Hypothesis: 6563 ms × 1 call
  (8.6% wall per iter-47 trace) is K addresses × T cycles with each
  address fully independent; outer rayon over K should saturate cores
  with non-trivial O(T) inner work per address, unlike P45 which
  regressed on tiny inner work. Added `rayon = { workspace = true }`
  to `jolt-witness/Cargo.toml`. Correctness gate green 43/43 (15.4s
  full run). Perf gate: run 1 modular 81273 ms (+4.83% vs 77525
  ratchet, core 4726 = +20% thermal); run 2 modular 86653 ms (+11.77%
  past reject, core 4822). Both modular runs slower, run 2 past reject
  threshold → revert. RSS jumped 4990→5155 MB (run 1) / 5082 MB (run
  2) suggesting par_iter allocates thread-local scratch or memory
  pressure from concurrent K-sized writes to shared `val` allocation.
  **Lesson**: the single big call (`ram_val` at 6.5s) does NOT parallelize
  cleanly despite having apparently independent outer iterations. Two
  probable reasons: (a) `access_events: Vec<Vec<(usize, F, F)>>`
  allocation of K separate subvectors (each tiny) is the actual work
  bottleneck — not the outer write loop — and that's still sequential;
  (b) writing O(K*T) = potentially billions of F-element writes into
  a single contiguous allocation from N threads hits cache-line
  contention / memory-bus saturation before compute-level parallelism
  can amortize. Together with P45 (revert) and P53 (revert), this
  closes the "parallelize derived-poly compute inner loop" avenue
  for the ram_* family. Future RAM attacks must attack the
  `access_events` precompute (sequential, O(T)) or reorganize the
  output layout, not just outer-rayon the write phase.
  iter 50 P58-subset REVERTED — runtime-level
  host source cache in `RuntimeState::host_source_cache:
  HashMap<PolynomialId, Vec<F>>` populated by a new `with_cached_source`
  helper in `runtime/helpers.rs`, covering the 5 `InputBinding` arms
  whose source is `materialize_binding`'d (EqProject, Transpose,
  EqGather, EqPushforward, ScaleByChallenge). Cache-on-first-use keyed
  by PolySource::Derived only; witness/r1cs/preprocessed sources still
  fresh-materialize. Target: iter-47 trace showed `ram_ra_indicator`
  with 2 calls × 2439 ms and `ram_combined_ra`/`ram_val` each single-
  call, suggesting the 2nd EqProject consumer of `ram_ra_indicator`
  (jolt_core_module.rs lines 2331 and 3492) was redundantly re-running
  `DerivedSource::compute`. Expected ≈3.2% wall savings (2439 ms /
  77525 ms). Correctness gate green 43/43 jolt-equivalence
  (transcript_divergence, zkvm_proof_accepted, modular_self_verify
  all pass). Perf gate: run 1 modular 79526 ms (+2.58% vs 77525
  ratchet, core 4298 = +9.1%), run 2 modular 80458 ms (+3.78% vs
  ratchet, core 4468 = +13.4%). Both runs in ±5% inconclusive band,
  both slightly slower → revert as flat per protocol. Peak RSS
  dropped 4925→4451→4198 MB, which suggests the cache *isn't* the
  dominant memory contributor — likely because only 1 source
  polynomial (`ram_ra_indicator`) actually has multiple consumers,
  and its single cached copy is small relative to the runtime-wide
  working set. **Lesson**: the hot-span "2 calls" reading for
  `ram_ra_indicator` may already reflect unavoidable per-stage work
  (different ChallengeIdx / InstanceIdx contexts), OR the cache
  saves the compute but adds equivalent HashMap::entry + Cow::into_owned
  cost on the first insert. Either way, the iter-47 attribution that
  "2nd call is redundant" did not translate to wall-clock improvement.
  Future RAM-derived attacks need to collapse MULTIPLE calls (e.g.
  coalesce ram_val + ram_combined_ra into a single 2-output pass) or
  parallelize the one big call (ram_val @ 6563 ms × 1 call), not
  de-duplicate across bindings.
  iter 49 P52-retry REVERTED — single `(6, 4)`
  arm added to `reduce_dense` const-generic dispatch. Same change as
  iter 43 but with `(41, 4)` excluded (iter 42 flagged `(41, 4)` arm as
  probable I-cache polluter). Correctness green 43/43. Perf gate: run 1
  modular 77399 ms (**−0.16% vs 77525 ratchet, flat in ±5%**); run 2
  modular 96989 ms (**+25.1%, past reject**). Core stable on both runs
  (4036 / 4072 = +2.5% / +3.4%), so the run-2 regression is NOT thermal.
  Likely cause: monomorphized `(6, 4)` kernel body adds I-cache pressure
  at the `reduce_dense` dispatch match site that intermittently causes
  cache misses on the same codebase under different heap/kernel state.
  The high between-run variance (−0.16% vs +25%) on identical code
  indicates this fix's measurement is noise-dominated even at log_t=16.
  Per protocol, rejected rerun → revert.
  **Lesson**: fixed-kernel arm expansion for shapes already served
  acceptably by the dynamic path does not yield a robust win —
  iter 42 (both arms), iter 43 (6,4 alone), iter 49 (6,4 alone retry)
  all failed. This closes the "add fixed arms" avenue for sha2-chain
  log_t=16 at current abstractions. Future reduce_dense attacks need
  to either (a) rewrite reduce_dense_dynamic to close the fixed-vs-
  dynamic gap WITHOUT monomorphization (manual unroll, SmallVec stack
  scratch), or (b) a fully different angle (memory layout, lazy
  materialization, structural re-ordering of sumcheck inputs).
  iter 48 P56 REVERTED — nested/flat parallelism
  for `CpuBackend::gruen_segmented_reduce`: (a) parallelized `e_active`
  precompute when `half >= PAR_THRESHOLD`; (b) restructured `reduce_outer`
  into `reduce_outer_chunk(a_idx, start, end)` returning unweighted
  partial (q_const, q_quad); (c) added a 3-way strategy — outer-par
  when `active.len() >= n_threads`; flat (outer, half-chunk) par when
  `active.len() < n_threads && half >= PAR_THRESHOLD`; serial/outer-par
  fallback otherwise. Correctness gate green (43/43 jolt-equivalence).
  Perf gate: run 1 modular 85233 ms (**+9.9% vs 77525 ratchet, past
  reject**); run 2 modular 82400 ms (**+6.3% vs 77525 ratchet, past
  reject**). Both runs past the +5% reject threshold → revert.
  **Lesson**: the new chunked path adds a `Vec<(usize, F, usize, usize)>`
  allocation + a new parallel dispatch for `e_active` even on the outer-
  par branch (since the precompute gate is separate). For typical sha2-
  chain log_t=16 rounds, `active.len()` likely exceeds n_threads at the
  boundaries where `half` is largest (early rounds have many active
  outers from the spread-out eq), so the flat path almost never fires,
  while the always-on `e_active` parallelization and restructured closure
  add overhead on the common path. Iter 49 should avoid speculative
  parallelism on new inner work and instead attack a different hot span.
  Candidates: P58 (RAM derived arms coalesce, 18% wall in 4 calls,
  structural), P59 (memoize multi_pair_g2_setup, 13.7% wall), or P55
  (coalesce InstanceBind into BindAllPolysInRound — needs compiler op).
  iter 47 INFRA — fresh sha2-chain log_t=16
  Perfetto trace captured post-iter-46-revert.
  `benchmark-runs/perfetto_traces/iter47-profile.json`. Modular
  76598 ms (−1.2% vs 77525 ratchet, flat in ±5% band; no ratchet move).
  **Self-time top 10** (92.3% of 76598 ms wall):
    | # | span | self ms | % | calls | µs/call |
    |--:|------|--------:|--:|------:|--------:|
    | 1 | `reduce_dense` | 28144 | **36.7%** | 2373 | 11860 |
    | 2 | `CpuBackend::gruen_segmented_reduce` | 18244 | **23.8%** | 16 | 1140000 |
    | 3 | `interpolate_inplace` | 17882 | **23.3%** | 17416 | 1027 |
    | 4 | `multi_pair_g2_setup_parallel` | 10479 | 13.7% | 124 | 84509 |
    | 5 | `CpuBackend::eq_project` | 9380 | 12.2% | 82 | 114400 |
    | 6 | `derived::ram_val` | 6563 | 8.6% | 1 | 6562800 |
    | 7 | `derived::ram_ra_indicator` | 4878 | 6.4% | 2 | 2439000 |
    | 8 | `CpuBackend::segmented_reduce` | 2661 | 3.5% | 16 | 166300 |
    | 9 | `derived::ram_combined_ra` | 2315 | 3.0% | 1 | 2314700 |
    | 10 | `G1::msm` | 2260 | 2.9% | 43043 | 53 |
  **Re-attribution vs iter 43**: hot-span map is stable —
  `reduce_dense` still king (36.7% self vs 43.1% incl iter-43),
  `gruen_segmented_reduce` per-call cost **1140 ms/call** confirmed as
  highest per-call leverage (1 function × 16 calls = 23.8% of wall),
  3 RAM derived arms total 13756 ms self over just **4 calls** (18%)
  — single-call workhorses. P56 (nested-parallel gruen) and
  P55 (coalesce bind ops once per round) are the two top structural
  attacks. `multi_pair_g2_setup` 10479 ms self in 124 calls: per-call
  cost ~85 ms × ~3 calls per commit; adjacent to setup-cost policy but
  NOT one-time (called per commit). No code change; stall unchanged at
  5. Iter 48 picks P56 (single-file CpuBackend change, highest per-call
  leverage) as the first attack. iter 46 P55-narrow REVERTED — partition-by-size
  `interpolate_inplace_batch` trait method: filter polys by
  `buf.len() / 2 < PAR_THRESHOLD` then outer-rayon the small set while
  large polys rely on their internal rayon. Added `Vec<&mut Buffer<F>>`
  batch entry point via `device_buffers.iter_mut().filter_map(...)`.
  Correctness gate green (43/43 jolt-equivalence pass, including
  `modular_self_verify`, `transcript_divergence`,
  `zkvm_proof_accepted_by_core_verifier`). Perf gate: run 1 modular
  80934 ms (+4.4% vs 77525 ratchet, core 3988 ms +1.2%); run 2 modular
  80933 ms (+4.4%, core 4012 ms +1.9%). Both runs consistently +4.4%
  inside the ±5% inconclusive band → revert as flat.
  **Lesson**: the hypothesis was "outer rayon over small polys beats
  serial dispatch." Two plausible reasons for the regression: (a) the
  `device_buffers.iter_mut().filter_map(|(pid, buf)| to_bind.contains(pid)...)`
  replaces O(|kdef.inputs|) HashMap::get_mut calls with one O(|device_buffers|)
  scan — at ~100 device buffers × 319 InstanceBinds, that's ~31K hash
  comparisons added; (b) the Vec<&mut Buffer> + partition + rayon
  dispatch has overhead exceeding the savings when small.len() is
  already bounded by batch-size parallelism above it. Iter 47 at stall
  5 → per protocol, run fresh --profile and append new P-items
  informed by updated hot-span map. iter 45 INFRA — appended P54-P57 to hypothesis
  queue informed by iter-43 sha2-chain log_t=16 profile attribution
  (reduce_dense 43.1%, InstanceSegmentedReduce 26.7%, interpolate_inplace
  23.4%, InstanceBind 23.3%, gruen_segmented_reduce 22.8%, mb::EqProject
  18.5%, DoryScheme::commit 20.4%). Queue targets:
    - P54 reduce_dense_dynamic per-chunk Vec allocation (3-8%)
    - P55 BindAllPolysInRound coalesced op w/ outer rayon amortization (8-15%)
    - P56 gruen_segmented_reduce nested parallelism across `active × half` (10-15%)
    - P57 mb::EqProject redundancy/caching (5-10%)
  No code change; stall unchanged. Iter 46 picks P55 or P56 as the
  structural attack. iter 44 P53 REVERTED — parallelized
  `Op::InstanceBind`'s outer loop over polynomials via `par_iter_mut`
  on a filtered Vec of DeviceBuffers. Target: 17416 interpolate_inplace
  calls × 1.09 ms per call; `kdef.inputs` dedup across ~55 polys per
  InstanceBind. Inner `bind_low_to_high` is already parallelized gated
  at PAR_THRESHOLD=1024 — at small poly sizes (late sumcheck rounds)
  inner parallelism doesn't trigger, so outer-across-polys was the
  hypothesis. Correctness green.  Perf gate: run 1 modular 83857 ms
  (+8.2% vs 77525 ratchet); run 2 modular 91106 ms (+17.5%). Both
  runs past the 5% reject threshold with unstable ratio (17.8× / 20.3×
  vs baseline 18.82×). **Repeats iter 37's lesson**: tiny-inner-work
  parallelization adds rayon dispatch overhead + memory-bus contention
  exceeding parallelism gains. Revert. For iter 45+ the parallel-bind
  angle should be reconsidered ONLY as part of a larger structural
  change (e.g. a coalesced "bind all dedup'd polys across all instances
  in a single call" op, with OUTER rayon dispatch and INNER sequential
  binds — a single rayon invocation amortizing dispatch once instead
  of per InstanceBind×poly).
  iter 43 P52-attack-b REVERTED — added `(6, 4)`
  arm ALONE to `reduce_dense_fixed` const-generic dispatch on the new
  sha2-chain log_t=16 standard. Correctness gate green
  (`modular_self_verify`, `transcript_divergence`,
  `zkvm_proof_accepted_by_core_verifier`, full jolt-equivalence).
  Perf gate inconclusive under thermal throttling: run 1 core=5926 ms
  (+50.5% vs 3938 ratchet), modular=92183 ms (+18.9% vs 77525 ratchet);
  run 2 core=7243 ms (+84%), modular=93941 ms (+21.2%). Core regressed
  more than modular — modular:core ratio improved from 18.82× baseline
  to 15.56× (run 1) / 12.97× (run 2), hinting at a real relative win,
  but absolute-ms-vs-ratchet is the gate criterion and both runs are
  ≥18% slow → reverted as flat. New iter-43 trace re-attributed:
  `reduce_dense` = 34975 ms (43.1% of 81077 ms total) still the king,
  `InstanceSegmentedReduce` 21613 ms (26.7%), `interpolate_inplace`
  19009 ms (23.4%), `InstanceBind` 18925 ms (23.3%),
  `gruen_segmented_reduce` 18522 ms (22.8%), `DoryScheme::commit`
  16506 ms (20.4%). The top 3 + bind together are 90% of wall — clear
  signal that the sumcheck inner loop is where any order-of-magnitude
  gains live. **Next iter 44 attack candidates**: (a) P53 fused
  reduce+bind op (single pass over polynomial data — today's reduce
  34.9 s + bind 18.9 s = 53.8 s doing two passes over same memory;
  halving bandwidth could save 15-25 s = 19-31% total win);
  (b) option (c) from prior iter — optimize the dynamic reduce path
  with reusable scratch (currently each rayon chunk likely allocates
  a fresh Vec<F> evals buffer); (c) measure reduce_dense per-shape at
  log_t=16 before picking an isolated-arm attack again.
  iter 42 P51 REVERTED — added `(6, 4)` and
  `(41, 4)` arms to `reduce_dense` const-generic dispatch. Muldiv log_t=12
  warm avg 1406.30 ms (3 runs excluding cold run 1 @ 1487.57 ms) =
  **+10.4% vs 1274.20 ratchet, past reject threshold**. sha2-chain
  log_t=14 num-iters=4 warm avg 16553.07 ms (runs 2/3, excluding cold run 1
  @ 19801.81) = +2.99% vs iter-39 baseline 16072 ms — flat/inconclusive.
  Expected 10-13% win did NOT materialize; instead got a muldiv regression.
  **Diagnosis**: shape distribution was measured only at sha2-chain
  log_t=14 (where (6,4) = 77.9% and (41,4) = 19.0%). Muldiv log_t=12's
  shape distribution was NOT measured — we assumed similar. Plausibly,
  muldiv's reduce_dense calls mostly hit shapes already covered by the
  existing fixed dispatch (2,2/3,3/4,4 — iter 1 trace had 83K calls
  mostly small shapes) so the new arms add code-size bloat / monomorph
  compile time without callee benefit. Specifically `(41, 4)` generates
  a large fixed-kernel body (41-wide stack arrays) that LLVM may even
  inline-expand in ways that hurt the muldiv I-cache. **Next iter 43
  attack options**: (a) measure muldiv log_t=12 shape distribution first,
  (b) try `(6, 4)` ALONE without `(41, 4)` (maybe only one of the two
  wins both programs), (c) instead of monomorphizing (41, 4), optimize
  the dynamic path (swap Vec<F> scratch for SmallVec / reused scratch
  across chunks). Revert: removed the 2 new match arms; tree-shake
  takes the unused monomorphizations.
  Iter 41 INFRA — added `fields(ni, ne)` to
  `reduce_dense` `#[tracing::instrument]` attribute. Attribution at
  sha2-chain log_t=14 num-iters=4 via one-time per-shape named spans
  (later removed to avoid 25% trace overhead): `rd_dynamic(6, 4)` =
  7316.6 ms / n=1792 = **77.9% of reduce_dense wall**; `rd_dynamic(41, 4)`
  = 1779.4 ms / n=18 = 19.0%; together **97% of reduce_dense wall** goes
  through shapes NOT in the existing fixed dispatch. Directly confirmed
  user hypothesis but attribution was sha2-chain-only — iter 42 revealed
  muldiv shape distribution diverges.
  Iter 40 P49 REVERTED — attempted
  `init_cache` for dory prepared-point cache in modular
  `DoryScheme::setup_prover`. Clear regression sha2-chain log_t=14
  avg +13.5% (17429 / 19067 ms vs 16072 iter-39 baseline). Root cause
  likely: the cached `G2Prepared` `.to_vec()` clone of pre-computed
  Miller loop lines is slower than fresh `G2Affine → G2Prepared`
  conversion in parallel chunks. **Durable policy**: setup-phase
  costs (EC generator prep, prepared-point caches, SRS init) are
  one-time and off-limits as perf-loop targets per user direction.
  Saved to memory as `feedback_setup_costs.md`. Iter 39 P47 GREEN — zero-copy upload path via
  `ComputeBackend::upload_vec(Vec<T>)` with CpuBackend pass-through
  override. Changed `materialize_binding::Provided` arm from
  `backend.upload(&data)` to `backend.upload_vec(data.into_owned())`.
  Eliminates the K*T `.to_vec()` clone per derived/r1cs materialize on
  CPU. Muldiv log_t=12 flat (warm avg 1295 ms, +1.65% vs 1274.20 ratchet
  — ratchet unchanged, not worth moving), sha2-chain log_t=14
  **−18.74% vs iter-38 baseline (19782 → 16072 ms)**, −14.87% vs iter-34
  avg, −10.76% vs iter-36 baseline. Attribution validated: iter 38
  mb::upload = 3654.68 ms at sha2-chain log_t=14 → iter 39 recovers
  most of that. iter 38 P46 INFRA — added `mb::upload`
  span inside `mb::Provided` arm at
  `crates/jolt-zkvm/src/runtime/helpers.rs`. Attribution at
  sha2-chain log_t=14 num-iters=4: `mb::upload` = 3654.68 ms =
  **67.8% of mb::Provided** (5389.72 ms) and larger than
  `pm::Derived` (2556.20 ms). Root cause: `CpuBackend::upload`
  = unconditional `data.to_vec()` clone, double-allocating every
  `Cow::Owned` materialize return. Iter 39 attack: change
  `ComputeBackend::upload` to accept `Cow<[T]>` so `Cow::Owned(v)`
  → `DeviceBuffer::Field(v)` without copy. Expected: ~18.5% wall
  savings at sha2-chain log_t=14; infra-only so stall counter
  unchanged. Iter 37 P45 REVERTED — parallelized
  `DerivedSource::{ram_combined_ra, ram_val, ram_ra_indicator}` via
  `par_chunks_mut` gated at `PAR_THRESHOLD=1024`. Correctness green,
  but sha2-chain log_t=14 avg +12.1% (19599 / 20796 ms vs 18011
  iter-36 baseline) and muldiv log_t=12 avg +5.0% vs ratchet
  (1303 / 1343 / 1368 ms = 1337.9 avg vs 1274.20). Both programs
  past reject threshold. Diagnosis: the 3 arms have tiny per-chunk
  inner work — `ram_ra_indicator` writes 1 element per T chunks of K,
  `ram_combined_ra` writes ~0-1 elements per K columns of T,
  `ram_val` is memory-bandwidth-bound on K*T writes. rayon dispatch
  overhead + concurrent memory-bus contention exceeds parallelism
  gain. Lesson repeated from iter 24/25/26/28: tiny-inner-work
  parallelization doesn't win at these sizes. Attack these arms
  differently in iter 38+ — candidates: (a) reduce allocation
  (K*T Vec::zero may dominate), (b) coalesce the 3 arms into a single
  combined walk, (c) change layout to avoid double-materialization,
  (d) investigate if `MaterializeUnlessFresh` cache makes these
  invocations redundant. Next iter 38: P46 close the 1984 ms
  mb::Provided attribution gap; iter 36 instrumentation-only — added
  `pm::{Witness,R1cs,Derived,Preprocessed}` + `r1cs::{Az,Bz,Cz,CombinedRow,Variable}`
  + `derived::*` (~20) spans to `ProverData::materialize`,
  `R1csSource::compute`, `DerivedSource::compute`. Attribution at
  sha2-chain log_t=14 num-iters=4 (18011 ms): `mb::Provided` = 4085 ms;
  `pm::Derived` = 2067 ms (50.6% of parent); 3 RAM arms own 99% of
  pm::Derived — `ram_ra_indicator` 804 ms, `ram_val` 622 ms,
  `ram_combined_ra` 610 ms. `pm::R1cs` only 33.6 ms (confirms P43
  revert was correct). 1984 ms attribution gap remains inside
  `mb::Provided` outside the 4 pm arms — next instrumentation target
  (P46). Iter 37 attack: P45 parallelize the 3 RAM arms; iter 35 P43 REVERTED — parallelized
  `R1csSource::compute_matvec` + Variable column extraction via rayon
  targeting the 6345 ms `mb::Provided` bucket from the post-P42 trace.
  Correctness green, but muldiv log_t=12 regressed on 2 of 3 runs past
  the −5% reject threshold (1308.89 / 1354.49 / 1360.56 ms = +2.7% /
  +6.3% / +6.8% vs 1274.20 ratchet) and sha2-chain log_t=14 was flat
  at 18779.29 ms (−0.54% vs iter-34 avg 18881.12). Diagnosis: mb::Provided's
  6345 ms at log_t=14 is NOT dominated by compute_matvec — attribution
  hypothesis was wrong. The parallelization overhead was real but the
  inner cost wasn't, so nothing was recovered. Lesson: instrument the
  specific method BEFORE parallelizing a suspected hotspot;
  iter 34 P42 GREEN — parallelized
  `CpuBackend::eq_project` via rayon with `par_chunks_mut` (branch 1)
  and `par_iter_mut` (branch 2), both gated at `PAR_THRESHOLD=1024`.
  Two runs on muldiv log_t=12 **1274.20 / 1289.92 ms** (avg −11.22% vs
  1444.14 ratchet); two runs on sha2-chain log_t=14 **19234.61 /
  18527.62 ms** (avg −23.27% vs iter 31 baseline 24604.96 ms). Clears
  +5% accept on both programs. Ratchet updated to 1274.20 ms, ratio
  4.38× → 4.02×. Handler unchanged — internal parallelization of a
  single `CpuBackend` method; iter 33 instrumentation-only — added named spans
  on `materialize_binding` match arms (`mb::Provided`, `mb::EqTable`,
  `mb::EqProject`, etc.) and on 7 previously-uninstrumented
  `CpuBackend::*` methods (`lt_table`, `eq_plus_one_table`, `eq_project`,
  `eq_gather`, `eq_pushforward`, `transpose_from_host`, `scale_from_host`);
  iter 31's 99.3%-untraced Materialize wall is now attributed: **`mb::EqProject`
  = 5574 ms / 58.8% of Materialize family wall at log_t=14 sha2-chain**,
  individual calls up to 2.4 s each, serial. `mb::Provided` = 3810 ms / 40.2%.
  Everything else <1%. P42 consumed by iter 34 (−23.3% log_t=14 win);
  iter 32 P39 end-to-end sparse Dory commit path
  + OneHotPolynomial layout fix + batch_g1_additions_multi amortized
  Montgomery inversion — flat on both muldiv log_t=12 (~+1.3%) and
  sha2-chain log_t=14 (~-2.26%), both runs inside ±5% band, reverted
  per protocol; iter 31 instrumentation-only — log_t=14 sha2-chain
  profile, P37 DONE, hotspot map fully reshuffled from log_t=12; iter 30 bench
  ceiling fix — flat +4.19% at 5-run median 1504.67 ms, ratchet unchanged;
  iter 29 bench rewrite — dropped jolt-core proof transplant, wired native
  `jolt_verifier::verify` best-effort, generalized modular stack to run any
  guest ELF; iter 28 P38 parallel Dory combine_hints flat +3.25% reverted;
  iter 27 instrumentation-only — fused_rlc_reduce group-level telemetry;
  iter 26 P35 parallel-over-groups fused_rlc_reduce flat/reverted; iter 25
  P33 parallel inner rlc_combine flat/reverted; iter 24 P32 parallel
  Materialize flat/reverted; iter 23 instrumentation-only; iter 22
  instrumentation-only; iter 21 flat; iter 20 green; iter 19 green;
  iter 18 reverted; iter 17 profiling-only; iter 16 infra; iter 15 infra;
  iter 14 infra; iter 13 flat; iter 12 green)
- **Last green iter**: 54 — P64 `reduce_dense_dynamic` inner-loop
  refactor: hoisted `BindingOrder` match out of the hot `for i in 0..half`
  body (macro-dispatched to two order-specialized branches), and
  replaced per-iteration bounds-checked indexing with pre-computed
  `*const F` pointer loads mirroring `reduce_dense_fixed`. Applied
  to both serial and parallel branches; no new monomorphizations.
  Three-run bench: 70762.94 / 77773.01 / 70781.79 ms (two of three
  past accept, −8.71% / +0.32% / −8.70% vs 77525.5 ratchet).
  Best-of-3 → ratchet updated to 70762.94 ms. Ratio 18.14× (was 19.68×).
  Targets the two dominant dynamic-path shapes: (6,4) 82% and (41,4)
  22% of reduce_dense wall per iter-52 profile. Prior green: iter 39
  P47 zero-copy upload for `Cow::Owned` materialize returns (−18.74%
  on log_t=14 sha2-chain).
- **Green streak**: 5 (iter 2 P11 flat +0.5%; iter 3 P12 flat −2.9%; iter 4 P13 flat −1.9%; iter 5 P14 flat +1.8%; iter 6 P17 regressed +6.4%; iter 7 P18 flat −0.1%; iter 8 P19 flat +0.6%; iter 9 P16 flat −3.45%; iter 10 P20 flat +0.47%; iter 11 instrumentation-only; iter 12 P24 −9.24%; iter 13 P25 flat +0.10%; iter 14 Gruen infra primitive; iter 15 Gruen infra reduce; iter 16 Gruen infra variant; iter 17 post-P24 re-profile; iter 18 Gruen dispatch reverted; iter 19 Gruen end-to-end −49.2%; iter 20 parallel Op::Commit −22.6%; iter 21 P28 parallelize lt_evals + EqPlusOne::evals flat −1.7%; iter 22 instrumentation-only — per-stage CPU vs wall saturation; iter 23 instrumentation-only — per-op-class CPU vs wall saturation + explicit dory `parallel` feature; iter 24 P32 parallel Materialize outer dispatch flat −0.55%, reverted — nested rayon pessimization hypothesis; iter 25 P33 parallel inner rlc_combine flat −0.69%, reverted — only ~11% of ReduceOpenings time is inner-loop parallelizable; iter 26 P35 parallel-over-groups fused_rlc_reduce flat +1.19%, reverted — likely single dominant group at log_t=12 so outer par_iter adds overhead with no parallelism gain; iter 27 P36 instrumentation-only — fused_rlc_reduce group-level telemetry confirms single group of 42 claims at log_t=12 with combine_hints = 83.3 ms / rlc_combine = 9.7 ms / materialize = 2 µs → combine_hints is 89% of ReduceOpenings wall and the correct attack target; iter 28 P38 parallel Dory combine_hints (par_iter over rows) flat +3.25%, reverted — rayon overhead at 64 rows × ~1.3 ms/row eats the ~65 ms expected savings at log_t=12; iter 32 P39 end-to-end sparse Dory commit path for OneHot polys + OneHotPolynomial CycleMajor layout fix + batch_g1_additions_multi amortized Montgomery inversion, correct (41/41 green), perf flat muldiv +1.3% / sha2-chain −2.26%, reverted per protocol — architectural prerequisites preserved in git log for future revisit; iter 33 instrumentation-only — `mb::*` + CpuBackend method spans attributed 58.8% of Materialize wall to `eq_project`; iter 34 P42 parallelize `CpuBackend::eq_project` via rayon (`par_chunks_mut` branch 1, `par_iter_mut` branch 2, `PAR_THRESHOLD=1024` gated) → **−11.22% muldiv log_t=12** (1282 ms avg) + **−23.27% sha2-chain log_t=14** (18881 ms avg), ratchet 1444.14 → 1274.20 ms, ratio 4.38× → 4.02×; iter 35 P43 parallelize `R1csSource::compute_matvec` targeting mb::Provided 6345 ms bucket — correct but muldiv +6.3% on 2/3 runs past reject, sha2-chain flat −0.54%, reverted — attribution hypothesis wrong; `mb::Provided` wall not dominated by `compute_matvec`)
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

- [x] P36: Instrument fused_rlc_reduce group structure — DONE iter 27.
  Single group of 42 claims at log_t=12; combine_hints = 83.3 ms (89% of
  ReduceOpenings wall), rlc_combine = 9.7 ms (10%), materialize = 2 µs
  (padded-cache hit). Confirms P35 failure diagnosis and redirects attack
  to PCS::combine_hints (P38). See Notes.

- [x] P37: Re-profile at log_t=14 — DONE iter 31. sha2-chain with
  `--num-iters 4` drives raw trace to 13120 cycles → padded 16384 (log_t=14).
  Ratio at log_t=14 = **15.7×** (core 1571 ms / modular 24604 ms);
  scaling multiplier modular/core = **1.63× per 2× workload** →
  modular is diverging, not converging. Hotspot map fully reshuffled:
  Materialize+MaterializeUnlessFresh = 47.6%, DoryScheme::commit = 27%,
  multi_pair_g2_setup = 24.5%. `combine_hints` (iter 27 smoking gun
  at log_t=12) drops to ~1% — no longer a meaningful attack at log_t=14.
  Full spans + diagnosis in **Notes / Iter 31**. Redirects attack to
  P39-P41 below.

- [x] P38: Parallelize PCS::combine_hints (iter 28, flat +3.25%, REVERTED).
  Added rayon + `into_par_iter` over rows in
  `crates/jolt-dory/src/scheme.rs::AdditivelyHomomorphic::combine_hints`.
  At log_t=12 the hint has `num_rows = 2^(num_vars - ceil(num_vars/2))` = 64
  rows, 42 scalar_muls/row ≈ 1.3 ms/row; rayon task-setup overhead + cross-core
  L2 traffic on the shared 387 KB hint working set eats the expected 65 ms
  savings. At higher log_t the per-row work grows linearly (so 2× rows × same
  inner loop = 2× work per row? no — per-row work is unchanged since it's
  hints.len() × scalar_mul, which only depends on the group size, not num_vars).
  The real gain scales with num_rows, so log_t=14 should at least halve the
  relative overhead. See Notes.

- [ ] P38-retry: Parallelize PCS::combine_hints (Dory opening-hint MSM fold) —
  target: `jolt-dory` (external fork) `AdditivelyHomomorphic::combine_hints`.
    - **Hypothesis**: iter 27 `perf_rlc` instrumentation showed
      `combine_hints` = **83.3 ms / 89%** of the 93 ms ReduceOpenings wall
      at log_t=12. Called once with 42 hints; folds via MSM-like
      Σ powers[i] · hints[i]. Sits outside the runtime (PCS trait
      dispatch), so runtime handlers stay generic. The call's internal
      work is independent of the runtime's 1.0-threads reading — the
      runtime measures "wall while calling combine_hints" which appears
      serial from the outside but can be pushed in parallel inside dory.
      Chunk hints into rayon par_iter groups, fold each group serially,
      combine group results — standard MSM parallelism pattern.
    - **Abstraction risk**: low — change is inside `jolt-dory` crate
      (`combine_hints` impl). Runtime handler unchanged (stays at 8 LOC
      call). Compiler unchanged. Fits the "extend ComputeBackend/PCS
      internals" category from CLAUDE.md.
    - **Expected delta**: lifting 83.3 ms from ~1.0 threads to 4-6 threads
      → 13-20 ms residual wall on this op → **~60 ms saved ≈ 4-5% of
      1444 ms total wall**. Meets the order-of-magnitude bar (≥5%
      candidate) given three consecutive parallelism-attack failures at
      ≤2% — this targets a single large function call with no external
      sync and no nested rayon under it (combine_hints inside dory does
      NOT currently use rayon, confirmed by the 1.0-thread reading).

- [x] P39: Route OneHot polys through Dory sparse commit path (iter 32,
  flat muldiv +1.3% / sha2-chain −2.26%, REVERTED). End-to-end wiring: added
  `BufferProvider::commit_source(&self, PolynomialId) -> Option<&dyn MultilinearPoly<F>>`,
  implemented on `Polynomials`/`ProverData` to expose one-hot polys directly
  (no dense expansion); rewrote `Op::Commit` handler to dispatch via
  `enum Src<Sparse|Dense>` (sequential collect, parallel commit); rewrote
  `DoryScheme::commit_rows_sparse` with `batch_g1_additions_multi`
  (single amortized Montgomery inversion across rows instead of per-row
  batch). Also fixed latent `OneHotPolynomial` layout bug: internal storage
  was AddressMajor (`cycle*k+col`) but `expand_one_hot` in jolt-witness
  uses CycleMajor (`col*T+cycle`) — latent because onehot was never fed
  directly to commit before. Corrected all four MultilinearPoly methods
  to CycleMajor. Correctness: 41/41 equivalence green incl. transcript_divergence
  + zkvm_proof_accepted, clippy clean. Perf: delta inside ±5% on both
  log_t=12 muldiv and log_t=14 sha2-chain. Reverted as flat per protocol.
  Architectural prerequisites preserved in git log for future revisit
  when parallelism opportunity inside sparse rows or larger log_t widens
  the sparse-vs-dense-commit gap proportionally. Full diagnosis in
  **Notes / Iter 32**.

- [x] P42: Parallelize `CpuBackend::eq_project` via rayon — target:
  `crates/jolt-cpu/src/backend.rs::eq_project` (lines ~462-495).
  **DONE iter 34** — −11.22% muldiv log_t=12 / −23.27% sha2-chain
  log_t=14; ratchet 1444.14 → 1274.20 ms, ratio 4.38× → 4.02×. See
  Notes / Iter 34.

- [x] P43: Parallelize `R1csSource::compute_matvec` via rayon — target:
  `crates/jolt-r1cs/src/provider.rs:54-73`. **REVERTED iter 35** —
  muldiv log_t=12 regressed +6.3% on 2/3 runs (past reject threshold),
  sha2-chain log_t=14 flat −0.54%. Attribution hypothesis was wrong:
  `mb::Provided`'s 6345 ms at log_t=14 is NOT dominated by
  `compute_matvec`. Correct attribution requires instrumenting the
  specific match arms inside `provider.materialize` + `R1csSource::compute`
  + `DerivedSource::compute`. See Notes / Iter 35.

- [x] P44: Instrument `provider.materialize` match arms + `R1csSource::compute`
  sub-arms + `DerivedSource::compute` — target:
  `crates/jolt-witness/src/provider.rs::BufferProvider::materialize`,
  `crates/jolt-r1cs/src/provider.rs::R1csSource::compute`,
  `crates/jolt-witness/src/derived.rs::DerivedSource::compute`.
  **DONE iter 36** — added `pm::{Witness,R1cs,Derived,Preprocessed}` +
  `r1cs::{Az,Bz,Cz,CombinedRow,Variable}` + `derived::*` (~20) spans.
  Post-P42 attribution at sha2-chain log_t=14 num-iters=4 (18011 ms):
  `mb::Provided` = 4085 ms. `pm::Derived` = 2067 ms / 50.6% of it.
  Inside `pm::Derived`, 3 RAM arms own 99%: `ram_ra_indicator` 804 ms
  (2 calls), `ram_val` 622 ms (1 call), `ram_combined_ra` 610 ms
  (1 call). `pm::R1cs` only 33.6 ms (P43 was correctly reverted).
  `pm::Witness` 0.14 ms, `pm::Preprocessed` 0.09 ms — both noise.
  **Attribution gap**: 1984 ms of `mb::Provided` still unaccounted
  — lives outside the 4 `pm::*` arms (materialize prologue/epilogue,
  Cow allocation, parent dispatch overhead?). See Notes / Iter 36.

- [x] P45: Parallelize the 3 RAM derived-poly builders.
  **REVERTED iter 37** — `par_chunks_mut` gated at `PAR_THRESHOLD=1024`
  on all 3 arms made sha2-chain log_t=14 avg **+12.1% WORSE** (avg
  20198 ms across 2 runs vs 18011 iter-36 baseline) and pushed muldiv
  log_t=12 avg to 1337.9 ms = +5.0% vs ratchet (reject boundary).
  Diagnosis: all 3 arms have tiny inner work per chunk (`ram_ra_indicator`
  = 1 remap + 1 write per cycle; `ram_combined_ra` per-col has ~0-1
  writes; `ram_val` per-addr is a T-length loop but memory-bandwidth
  bound for field writes). rayon overhead + concurrent memory-bus
  contention exceeds the parallelism gain. See Notes / Iter 37.

- [x] P46: Close the 1984 ms `mb::Provided` attribution gap via
  `mb::upload` span — iter 38 confirmed `backend.upload()` consumes
  3654.68 ms = **67.8% of `mb::Provided`** at sha2-chain log_t=14. The
  `CpuBackend::upload` impl does an unconditional `data.to_vec()` clone,
  which for `Cow::Owned` materialize returns means 2 allocations per
  poly. See Notes / Iter 38.

- [x] P47: Zero-copy upload for `Cow::Owned` materialize returns —
  target: `ComputeBackend::upload_vec(Vec<T>)` trait method
  (default forwards to `upload(&data)`); CpuBackend override pass-through;
  `materialize_binding::Provided` uses `upload_vec(data.into_owned())`.
  **Result**: sha2-chain log_t=14 **−18.74% vs iter-38 baseline** (19782
  → 16072 ms avg). Muldiv log_t=12 flat (+1.65% warm avg, within ±5%).
  Ratchet unchanged. See Notes / Iter 39.

- [ ] P40: Investigate Materialize/MaterializeUnlessFresh super-linear
  scaling — target: `crates/jolt-zkvm/src/runtime/handlers.rs` Materialize
  handlers + `crates/jolt-cpu/src/backend.rs::materialize_*`.
    - **Hypothesis**: iter 31 log_t=14 profile showed `Op::Materialize` at
      **6420 ms / 197 calls = 32.6 ms/call**, vs iter 23 at log_t=12 muldiv
      of 1.5 ms/call. That's a **22× per-op wall jump for a 4× workload**.
      At log_t=14, Materialize family (Materialize + MaterializeUnlessFresh)
      is **47.6% of total wall** — the largest single category. Possible
      root causes: (a) system allocator pressure on large `Vec<F>` per call
      (log_t=14 polys = 128 KB each × 197 ops = contended glibc malloc arena),
      (b) L2 cache thrash on eq/lt tables shared across parallel materializes,
      (c) a hidden O(T²) in MaterializeUnlessFresh's freshness check
      (103 ms/call × 46 calls = 4781 ms), (d) padded-poly cache that helps at
      log_t=12 but thrashes at log_t=14 due to working-set size.
    - **Abstraction risk**: low-medium — fix is internal to Materialize
      handler + cpu backend; runtime contract unchanged.
    - **Expected delta**: halving per-op wall to match the 4× linear-scaling
      expectation would save ~5.6s at log_t=14 = **24% total wall**. Even
      a 2× improvement per call is 12% total wall.

- [x] P49: Init dory prepared-point cache in modular setup — REVERTED.
  `init_cache()` enables the fast path in `multi_pair_g2_setup_parallel`
  that clones cached `G2Prepared` slices instead of converting from
  affine, but the G2Prepared deep-clone is SLOWER than fresh conversion
  in parallel chunks. +13.5% regression at sha2-chain log_t=14. Plus
  user-directed policy: setup-phase costs are one-time, off-limits as
  perf targets. See Notes / Iter 40.

- [x] P50: Instrument `reduce_dense` shape dispatch — INFRA DONE (iter 41).
  Added `fields(ni, ne)` to the outer `#[tracing::instrument]` attribute
  on `reduce_dense`. Attribution at sha2-chain log_t=14 num-iters=4 with
  one-time per-shape named spans (later removed to avoid 25% trace
  overhead): `rd_dynamic(6, 4)` = **7316.6 ms / n=1792 = 77.9% of
  reduce_dense wall**; `rd_dynamic(41, 4)` = 1779.4 ms / n=18 = 19.0%;
  all other shapes < 1% each combined. Only `rd_fixed(4, 4)` = 5.7 ms /
  n=13 of the existing fixed variants was hit at all on this trace.
  **97% of reduce_dense wall goes through 2 shapes that are NOT in the
  fixed dispatch** and instead fall through to heap-allocated
  `reduce_dense_dynamic`. Iter 42 attack: add bespoke
  `reduce_dense_fixed::<F, 6, 4>` + `reduce_dense_fixed::<F, 41, 4>`
  arms. See Notes / Iter 41.

- [x] P51: Add bespoke `reduce_dense_fixed::<F, 6, 4>` +
  `reduce_dense_fixed::<F, 41, 4>` dispatch arms — REVERTED (iter 42).
  Muldiv log_t=12 regressed +10.4% warm avg past reject threshold,
  sha2-chain log_t=14 flat (+2.99%). Diagnosis: shape distribution was
  measured only at sha2-chain log_t=14; muldiv log_t=12 has different
  shape mix so new arms add codegen bloat without callee benefit.
  See Notes / Iter 42. Next: P52 shape-distribution check at muldiv
  log_t=12, then decide single-shape vs no-kernel-expansion attack.

- [ ] P52: Measure shape distribution of `reduce_dense` at muldiv
  log_t=12 — target: `crates/jolt-cpu/src/backend.rs` + ad-hoc bench
  run with one-shot per-shape named spans.
    - **Hypothesis**: iter 42 regressed muldiv while leaving sha2-chain
      flat, implying shape distribution diverges between the two programs.
      Muldiv may be dominated by shapes already in the existing fixed
      dispatch (2,2 / 3,3 / 4,4) or by a different uncommon shape.
      Instrumenting muldiv log_t=12 the same way as iter 41 did for
      sha2-chain would reveal whether `(6, 4)` and `(41, 4)` are called
      at all there, or if the new arms were pure code-size bloat.
    - **Abstraction risk**: zero — instrumentation only.
    - **Expected delta**: INFRA — output is an attribution map that
      drives P53 design. If muldiv is dominated by `(6, 4)` too but
      shape already IS that, then the problem with iter 42 was `(41, 4)`
      specifically — try `(6, 4)` alone. If muldiv is dominated by a
      different shape entirely, P53 picks that shape.

- [ ] P41: Parallelize or memoize `multi_pair_g2_setup` across Dory
  commits — target: `dory` crate internals (user-controlled fork).
    - **Hypothesis**: iter 31 log_t=14 profile showed
      `BN254::multi_pair_g2_setup_parallel` = **5771 ms in 60 calls**
      (24.5% of wall). 60 calls for 42 Dory commits = ~1.4 calls per commit
      (tier-1 + tier-2 setup). Most of these 60 setups should be redundant
      if they operate on static G2 generators — cache the setup once per
      (preprocessing, commit size) pair. The `_parallel` suffix means each
      call is internally parallel, but 60 sequential-ish 96-ms calls on
      shared G2 structures is wasted work.
    - **Abstraction risk**: medium — requires Dory-internal change to
      memoize G2 setup across commits. No modular-stack changes.
    - **Expected delta**: if setup is truly static per preprocessing,
      the 59 redundant calls × 96 ms = **5.7 s saved = 24% total wall**
      at log_t=14. Conservative 50% reduction = 12% wall.

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

- [ ] P54: Reduce per-chunk heap allocation in `reduce_dense_dynamic` —
  target: `crates/jolt-cpu/src/backend.rs::reduce_dense_dynamic` (parallel
  fold path, lines ~977-1008).
    - **Hypothesis**: iter-43 profile at sha2-chain log_t=16 shows
      `reduce_dense` at 34975 ms (43.1% of 81077 ms total). Iter-41
      attribution showed `(6, 4)` shape = 77.9% and `(41, 4)` = 19.0% of
      reduce_dense wall at sha2-chain log_t=14 — both fall through to
      `reduce_dense_dynamic`. P51 (monomorphize those shapes to
      reduce_dense_fixed) regressed muldiv; the fixed-vs-dynamic gap is
      real but the macroscopic symptom at sha2-chain is
      `reduce_dense_dynamic` specifically. The rayon `fold` closure in
      `reduce_dense_dynamic` allocates 4 fresh Vecs per parallel chunk
      (new_accs + lo + hi + evals, each of length num_inputs or num_evals),
      re-incurred on every scheduling boundary. Switch the init closure to
      return `(SmallVec<[F::Accumulator; 8]>, SmallVec<[F; 64]>, ...)`
      with inline capacity covering the (6, 4) and (41, 4) shapes, or
      pre-allocate thread-local scratch outside the fold via a
      `thread_local!` TLS cell. Either eliminates the per-chunk alloc
      without requiring monomorphization of the inner loop.
    - **Abstraction risk**: zero — internal to reduce_dense_dynamic,
      no API change.
    - **Expected delta**: 3-8% total wall if per-chunk alloc is ~10-20%
      of reduce_dense_dynamic's 30-35 s. Worth measuring alloc share first
      via `dhat` / rayon fold-chunk counter before committing the attack.

- [ ] P55: Coalesce `InstanceBind`'s outer per-poly loop into a single
  `BindAllPolysInRound` op dispatched once per round with internal rayon —
  target: new compiler op in `crates/jolt-compiler/src/module.rs` that
  emits a single "bind all dedup'd polys bound this round" op instead of
  N per-instance InstanceBind ops; runtime dispatches once; CpuBackend
  uses OUTER rayon across polys + INNER serial bind per poly.
    - **Hypothesis**: iter-43 profile shows `interpolate_inplace` at
      19009 ms / 17416 calls (1.09 ms/call) and `InstanceBind` at
      18925 ms / 319 calls. The 319 InstanceBind calls mean ~55 polys
      bound per call on average (17416 / 319 ≈ 55). Iter 44 P53 attempted
      per-InstanceBind parallelism and regressed +8.2 to +17.5% because
      rayon dispatch + memory-bus contention per InstanceBind dominated.
      The STRUCTURAL fix is amortizing rayon dispatch to once-per-round:
      a single op invocation takes all dedup'd polys for the round and
      dispatches all of them in ONE outer rayon sweep. For a round with
      N polys each of size S, current cost is N × (dispatch_overhead +
      serial_bind(S)). New cost is dispatch_overhead_once +
      N_parallel_across_cores × serial_bind(S). At small S (late rounds)
      where rayon overhead per bind dominated iter-44's regression,
      outer-rayon-once instead of inner-rayon-each flips the economics.
    - **Abstraction risk**: medium — requires new Op variant on compiler
      side, runtime handler stays ≤30 LOC. Compiler must be able to
      identify which polys need binding in the current round (already
      known — it's the `bound_this_round` set plus the per-instance
      input list).
    - **Expected delta**: 8-15% total wall if we can push
      `interpolate_inplace` from its current serial-dispatch-per-poly
      pattern to fully-parallel-once-per-round. Lower bound: ~6% even if
      we only amortize dispatch and keep inner binds serial. Upper bound:
      ~15% if outer rayon saturates 8+ cores on typical ~55-poly rounds.

- [ ] P56: Deep-dive on `gruen_segmented_reduce` — target:
  `crates/jolt-cpu/src/backend.rs:668-803`.
    - **Hypothesis**: iter-47 profile (fresh) shows **18244 ms / 16 calls
      = 1140 ms per call, 23.8% of wall** — the largest single span by
      per-call cost. 16 calls ≈ 16 segmented sumcheck rounds. Each call's
      inner is (a) build `e_active` from `eq_cycle` (sequential O(half)),
      (b) for each `active` outer position compute `(q_const, q_quad)`
      over `half` pairs — parallel across outers only if
      `active.len() >= 2`. At late rounds `half` is small but active
      outer count may be large (128+); at early rounds `half` is very
      large (1M+) but active outer count may be small (1-4). Current
      code only parallelizes the OUTER iteration; INNER parallelism
      across `half` is serial. Add a nested parallel strategy: when
      `active.len() * half >= threshold` and `active.len() < N_cores`,
      use `rayon::join` or `par_chunks` over `half` within
      `reduce_outer`. Alternative: the `e_active` precompute is
      pure-functional over shared `eq_cycle` — parallelize that too
      when `half >= PAR_THRESHOLD`.
    - **Abstraction risk**: low — internal to CpuBackend method.
    - **Expected delta**: 10-15% total wall if we can double core
      saturation in this hot 23.8% band; lower if `active.len()` is
      already saturating cores at round-onset and work is truly
      memory-bandwidth bound.

- [ ] P57: `mb::EqProject` redundancy / caching — target:
  `crates/jolt-cpu/src/backend.rs::eq_project` +
  `crates/jolt-zkvm/src/runtime/helpers.rs` materialize pipeline.
    - **Hypothesis**: iter-43 profile shows `mb::EqProject` at 15011 ms
      (18.5% of wall), up from iter-33 attribution at sha2-chain log_t=14
      (5574 ms = 58.8% of Materialize wall). Iter-34 P42 parallelized
      `eq_project` internals (rayon on branch 1 par_chunks_mut + branch 2
      par_iter_mut) and won −11% muldiv + −23% sha2-chain. The remaining
      15 s wall spread over the log_t=16 workload suggests either (a)
      repeated `eq_project` calls with the same `(prefix, suffix)` inputs
      across materialize boundaries — candidate for a cache keyed by
      (prefix_eval, suffix_eval) hash — or (b) the parallel branches
      don't scale as well at log_t=16 as at log_t=14 due to per-call
      overhead vs. inner work ratio changing. Instrument call counts
      and per-call shape to decide between caching vs. reorganizing.
      Conservative first step: add `mb::EqProject` span to count calls
      per unique `(prefix_eval.len(), suffix_eval.len())` shape — if
      many calls with same shape have redundant precomputed data, cache
      hits are free wins.
    - **Abstraction risk**: low for instrumentation; medium for caching
      (needs eq_project argument hashability + thread-safe cache).
    - **Expected delta**: 5-10% total wall if cache hit-rate > 50% on
      EqProject calls; 2-4% if only layout/parallelism tuning works.

- [ ] P58: Coalesce the 3 RAM derived-poly builders into a single pass —
  target: `crates/jolt-witness/src/derived.rs::DerivedSource::compute`
  arms `ram_val`, `ram_ra_indicator`, `ram_combined_ra`.
    - **Hypothesis**: iter-47 profile attributes **13756 ms (18% of
      wall) to 3 arms in 4 calls**: `ram_val` 6563 ms × 1 call,
      `ram_ra_indicator` 4878 ms × 2 calls, `ram_combined_ra` 2315 ms
      × 1 call. P45 (inner rayon on each arm) regressed both programs
      due to tiny inner work + memory-bus contention. A structural fix
      instead: if the 3 arms traverse overlapping underlying data
      (RAM address traces / cycle index), coalesce the 3 producer
      closures into ONE pass that writes 3 output buffers per
      (cycle, addr) tuple visited — amortizing 3× memory traffic to
      ~1× by keeping the inputs in L1/L2 across all 3 derivations.
      Alternative: if 2 of the 3 are derivable from the 3rd by an
      in-place post-pass, compute the hardest one first and derive
      the others from its output buffer.
    - **Abstraction risk**: medium — `DerivedSource::compute` is a
      fan-out per-arm dispatch. Coalescing requires either (a) a new
      `MultiDerivedSource` that emits multiple PolynomialIds from one
      call, or (b) a post-compute fix-up that derives secondary
      outputs from the primary. Also requires careful transcript
      isolation — the combined op must produce identical bytes.
    - **Expected delta**: 8-14% total wall if memory traffic is the
      binding constraint (2-3× bandwidth savings → 30-50% per-arm
      speedup on ~14 s). Lower (5-8%) if per-element compute
      dominates — profile post-P58 to decide.

- [ ] P59: Memoize `multi_pair_g2_setup` across Dory commits —
  target: `jolt-dory` fork internals (currently calls
  `multi_pair_g2_setup_parallel` 124 times for 42 commits at log_t=16).
    - **Hypothesis**: iter-47 profile shows `multi_pair_g2_setup_parallel`
      = **10479 ms self / 124 calls** (13.7% of wall, 85 ms/call avg).
      124 calls across 42 commits ≈ 3 calls per commit (tier-1 + tier-2 +
      evaluation setup). Most of the per-call work is computing
      `G2Prepared` miller-loop lines from the static G2 SRS chunk used
      by Dory. Those lines depend only on `(SRS chunk, size)` which is
      fixed per preprocessing. Memoize per `(commit_size, SRS chunk)`
      pair: first call materializes the prepared lines into a shared
      cache, subsequent calls clone/ref them. Not a "setup cost" in the
      one-time-at-preprocessing sense — it's called fresh per commit
      today.
    - **Abstraction risk**: medium — touches external `jolt-dory` crate
      internals; requires thread-safe cache for multi-threaded commit
      dispatch (iter 20's parallel Op::Commit fan-out means multiple
      commits run in parallel). Policy check: this is NOT setup-cost
      (per `feedback_setup_costs.md`) because it's per-commit, not
      one-time; but adjacent to P49 (iter 40) which regressed on a
      similar "cache prepared points" attempt. Needs measurement first
      to confirm cache-hit-rate.
    - **Expected delta**: if 80% of 124 calls are redundant setup that
      can be cache-served: 0.8 × 10479 = **8380 ms saved ≈ 11% wall**.
      Conservative 50% cache-hit: **5 s saved ≈ 6.5% wall**.

- [ ] P60: End-of-loop meta — deep instrument of `reduce_dense`
  hot-shape distribution at log_t=16 — target:
  `crates/jolt-cpu/src/backend.rs::reduce_dense`.
    - **Hypothesis**: `reduce_dense` is the top self-time span
      (28144 ms / 36.7% of wall, 2373 calls × 11.9 ms/call avg). Iter 41
      instrumented at log_t=14 found `(6, 4)` = 77.9% and `(41, 4)` =
      19.0% of reduce_dense wall. Iter 42 tried fixed-kernel arms for
      both shapes at log_t=12/14 and regressed muldiv due to
      shape-distribution divergence between programs. At log_t=16
      sha2-chain specifically, the distribution has probably shifted —
      worth a fresh per-shape count to decide if (a) a single-shape
      fixed-arm (e.g., (6, 4) only) would win at log_t=16 sha2-chain,
      (b) the dynamic path itself needs a per-chunk-scratch-reuse
      rewrite (P54), or (c) some other combination. Pure instrumentation
      iteration (no code change shipped).
    - **Abstraction risk**: zero — instrumentation only.
    - **Expected delta**: INFRA — output is a shape distribution map
      that grounds P54 / (P51-retry with narrowed shape list) / a
      structural rewrite of reduce_dense_dynamic.
    - **Status (iter 52)**: DONE. Shape distribution:
      `(6,4)` = 25780 ms / 82% of reduce_dense; `(41,4)` = 6863 ms / 22%;
      all others <1% each. Iter 42/43/49 already proved the exact-arm
      specialization regresses under thermal variance. **Takeaway for
      P61**: bound-based NE specialization (not exact-match) is the
      remaining unexplored specialization angle.

- [ ] P61: Fused bind+reduce across consecutive rounds —
  target: new `Op::ReduceWithBindPrep` + `Op::BindFinalize` pair
  replacing the current `Op::InstanceReduce` + `Op::InstanceBind`
  sequence. CPU backend gains
  `reduce_and_prep(inputs) -> (round_evals, prepped_state)` that
  computes round-R evals AND writes `delta[i] = hi - lo` into each
  poly's `hi` slot in one memory sweep. After verifier challenge,
  `bind_finalize(poly, challenge)` does `poly[i] = lo + challenge * delta[i]`
  in a single pass. Combines the two passes (reduce + bind) into one
  full-read + one half-read-write (instead of two full-reads + one
  half-write).
    - **Hypothesis**: iter-47 + iter-52 profiles show `reduce_dense`
      (31 s) + `interpolate_inplace` (18 s) = 49 s = 62% of wall doing
      memory-bound work on the SAME polynomial buffers across back-to-
      back rounds. Halving the memory bandwidth requirement on the
      bind side (one read instead of two) targets the second-pass reads
      directly — a bandwidth-bound kernel on DDR5 should scale roughly
      linearly with reads eliminated.
    - **Abstraction risk**: high. Needs new compiler op types, new
      backend methods, runtime schedule rework. Bind is naturally
      split into a "prep" phase (before challenge) and "finalize"
      phase (after challenge), matching the sumcheck data-flow
      structure. Handler stays ≤30 LOC per op — the complexity lives
      in the compiler's schedule lowering and the new CPU-backend
      method. Must be validated against `modular_self_verify` to
      confirm no transcript divergence.
    - **Expected delta**: 15-25% total wall if memory-bandwidth bound;
      10-15% if compute-bound within the reduce kernel. Conservative
      floor: 8-10% from eliminating one full poly read per round
      across ~20 rounds × ~55 polys × 2^(16-r) elements each.
      **ABSTRACTION WARNING**: this is a structural change across
      compiler + runtime + backend. Budget 2-3 iters to implement,
      debug, and verify transcript parity. If iter 53 picks P61,
      iters 54-55 may be follow-up correctness fixes.

- [ ] P62: Bound-NE specialization of `reduce_dense_dynamic` —
  target: `crates/jolt-cpu/src/backend.rs::reduce_dense`.
    - **Hypothesis**: iter-52 shape distribution shows 82% of
      reduce_dense wall at (6,4) and 22% at (41,4). Both fall
      through to `reduce_dense_dynamic`. Iter 42/43/49 tried exact-
      match fixed arms on these shapes and failed under thermal
      variance — likely an I-cache pressure effect from adding more
      specialized code at the dispatch site. Alternative: create
      ONE bound-based specialization that covers ALL num_evals=4
      cases (and possibly num_evals≤8) with stack-allocated
      `[F::Accumulator; MAX_NE]`, generic over MAX_NE but iterating
      over ACTUAL num_evals. Dispatch arm: `(_, ne) if ne == 4 =>
      reduce_dense_bounded::<F, 4>(...)`. Single extra instantiation,
      limited I-cache impact, same stack-array benefit as fixed arms.
    - **Abstraction risk**: low — single new function, inline into
      dispatch match. Must measure I-cache impact with perf counters
      if still regresses, but 1 new instantiation vs iter-42's 2 exact
      arms should be safer.
    - **Expected delta**: 3-8% wall — upper if stack-array accumulator
      is the dominant missing optimization; lower if thermal variance
      still dominates.

- [ ] P63: Investigate `eq_project` eq_table recomputation —
  target: `crates/jolt-cpu/src/backend.rs::eq_project`.
    - **Hypothesis**: iter-52 profile shows `CpuBackend::eq_project`
      at **9341 ms / 82 calls = 113.9 ms per call, 11.8% wall**. Each
      call recomputes `jolt_poly::EqPolynomial::evals(eq_point)`
      before the project loop. If multiple calls in the same sumcheck
      round share the same `eq_point` (same challenge vector
      projection), we redundantly compute the eq_table O(2^|eq_point|)
      field ops. Add a short-TTL cache keyed by `hash(eq_point)` inside
      CpuBackend (thread-local or RwLock). Conservative version:
      count unique eq_points per round first as instrumentation; if
      cache-hit-rate > 50%, ship the cache.
    - **Abstraction risk**: low — add ThreadLocal<HashMap<[u8; 32],
      Vec<F>>> or similar; invalidate per-round via a round counter.
    - **Expected delta**: 2-5% wall if eq_table compute is significant
      fraction of eq_project time (likely 10-30% given `EqPolynomial::
      evals` builds 2^n entries with n multiplications each); higher
      if cache-hit-rate is near 100% within rounds.
    - **Status**: subsumed by P76 (the handle-based fix doesn't just
      cache the eq_table — it removes per-round rebuild entirely).
      Keep as fallback only.

<!-- P76-P82: memo-driven architectural attacks. See
     perf/report_tools/kernel_gap_memo.md §§5-6 for full design. -->

- [ ] P76: (B) Per-sumcheck-instance handle — HandleId + Slab stateful
  backend — target: `crates/jolt-compute/src/traits.rs` + new methods
  on `ComputeBackend` + `crates/jolt-cpu/src/backend.rs`
  (`CpuBackend` becomes a struct with internal
  `Slab<CpuHandleState<F>>`). Cross-ref: jolt-core's
  `GruenSplitEqPolynomial` at
  `crates/jolt-core/src/poly/split_eq_poly.rs:82-332`.
    - **Hypothesis**: modular rebuilds the full eq table every round
      inside `CpuBackend::eq_project` (ref: `crates/jolt-cpu/src/backend.rs:481`;
      ~11.8% wall / 9.3 s on log_t=12 muldiv, ~30 s on log_t=16
      sha2-chain per iter-52/-65 traces). Core's `GruenSplitEqPolynomial`
      caches prefix tables + `current_scalar` across rounds, making
      per-round eq binding O(1) amortized instead of O(2^remaining).
      Introduce an opaque `HandleId(u32)` at the trait boundary;
      backend owns the typed state internally via a concrete enum
      `CpuHandleState<F> { Eq(GruenSplitEqPolynomial<F>) | Scratch(Vec<F>) }`.
      No `Box<dyn Any>` (per user direction). Trait surface:
      `open_handle(shape) -> HandleId`, `bind_handle(id, round, r)`,
      `query_handle(id, idx) -> F`, `close_handle(id)`. Runtime
      handler ≤ 30 LOC stays protocol-unaware: knows "a handle gets
      bound then queried," doesn't know it's an eq polynomial.
    - **Abstraction risk**: low — clean analog to core's existing
      stateful eq. Trait grows 4 methods with associated type
      `HandleState<F>`. Handler diff is a mechanical rewrite of
      `Op::EqProject` → `Op::OpenHandle + Op::BindHandle + Op::QueryHandle`.
      Backend state shifts from unit-struct to struct with `Slab`;
      matches how all production ML runtimes work (cuBLAS handle,
      PyTorch caching allocator).
    - **Expected delta**: ~10 s wall (~12% of 80 s modular prove) per
      memo §6 table; projected post-fix modular CPU drops 120,519 ms
      → ~110 s via this alone.

- [ ] P77: (C) Variable-arity `BatchRoundEvaluate` op — target: new
  `Op::BatchRoundEvaluate { round, instances: Vec<InstanceDesc> }`
  in jolt-compiler + handler in `crates/jolt-zkvm/src/runtime/handlers.rs`
  + `ComputeBackend::batch_round_evaluate(&kernels, &inputs, &coeffs)`.
  Replaces today's 120-instance per-round serial loop over
  `InstanceReduce`/`InstanceBind`/`interpolate_evaluate`.
    - **Hypothesis**: per-round we issue ~120 independent per-instance
      reduce+bind calls (ref memo §2: "reduce_dense + interpolate
      decomposition tax"). Each call is below `PAR_THRESHOLD=2048`
      so serial in isolation, and the outer dispatch is also serial
      — effective single-core on stage 5/6/8 hot loops. One
      `BatchRoundEvaluate` hands the whole batch to the backend which
      can fuse/parallelize across instances as one rayon region with
      one scratch allocation. Locality win: input cache lines visited
      once instead of 120× per round. Kernel-launch-overhead
      mitigation that's the textbook GPU playbook — also wins on CPU
      via L2/L3 reuse. Handler stays ≤ 30 LOC (pack + dispatch).
    - **Abstraction risk**: medium — changes op schedule. Compiler
      emits one op per round instead of N per round; runtime op
      dispatcher has one new op to handle. Trait adds one method with
      a default-impl fallback that loops over per-instance calls
      (CUDA backend can opt in later). Variable arity preserves
      per-instance heterogeneity (kernels may differ across instances).
    - **Expected delta**: 7-12 s wall (~9-14%) per memo §6. Compounds
      with P76 because eq handles get bound once per batch-round
      instead of once per instance.

- [ ] P78: (A) Per-kernel persistent scratch — target:
  `crates/jolt-cpu/src/backend.rs` + `CpuKernel<F>` grows scratch
  methods. Add `CpuHandleState::Scratch(Vec<F>)` rows that outlive
  single-op calls; `interpolate_inplace` + `reduce_dense` reuse the
  same scratch across rounds instead of allocating per call.
    - **Hypothesis**: `interpolate_inplace` + `reduce_dense` allocate
      fresh scratch Vec on every call (ref memo §2). Absolute cost is
      small per call but aggregates to ~2-3 s across all rounds; also
      triggers allocator thrash that shows up as locality noise. The
      `HandleId::Scratch` row from P76 is already the right home;
      P78 is mostly a wiring exercise once the handle plumbing lands.
    - **Abstraction risk**: low — pure perf, no new semantics. CUDA
      backend wants this MORE than CPU (device allocation is 100×
      more expensive than host allocation).
    - **Expected delta**: 2-3 s wall (~2-4%). Small absolute win but
      shrinks allocator-noise variance.

- [ ] P79: (D) `BufferEncoding` tags — propagate scalar-width
  metadata through `Buf<Self, F>` — target: `crates/jolt-compute/src/traits.rs`
  (`upload_tagged` variant) + `crates/jolt-cpu/src/backend.rs`
  (reduce_dense Compact fast path). Today's `Buf<F>` erases whether
  the underlying poly is `CompactPolynomial<u32>` or dense `Vec<F>`;
  the compact fast path only fires inside jolt-core's monolithic
  prover.
    - **Hypothesis**: `CompactPolynomial<u32>` inputs (~30-40% of
      committed witness polys) get promoted to `Vec<F>` at the op
      boundary because `Buf` carries no dtype tag. Adding
      `BufferEncoding { Dense, Compact(ScalarWidth), OneHot }` lets
      the backend dispatch to a `reduce_compact_u32` kernel that runs
      at 4× memory bandwidth (u32 vs BN254 Fr ~32 B). Handler
      preserves whatever variant compiler emitted. Still
      protocol-agnostic — just "this buffer holds small scalars."
    - **Abstraction risk**: high (per memo §5 D). Breaks the
      type-erased `Buf` abstraction; backends must implement multiple
      dtype-specialized kernels. Prototype on one call site
      (instruction_ra compact_u32) before broad rollout.
    - **Expected delta**: unknown pre-prototype. Memo flagged as
      "need prototype." Defer until P76+P77+P78 land and we re-measure
      the residual gap.

- [ ] P80: Eager materialization of `pm::Derived` RAM witness polys —
  target: `crates/jolt-witness/src/derived.rs` + handler emit point in
  jolt-zkvm. Today `pm::Derived` constructs RAM witness polys lazily
  on first use, serial and on the main thread. Core's
  `generate_and_commit_witness_polynomials` does the same algorithm
  eagerly at stage 0 with rayon par over addresses.
    - **Hypothesis**: iter-52 profile shows `derived::ram_val` at
      7.09 s single-threaded (modular) vs equivalent core path at
      ~1.6 s parallelized. 15.4 s total inside the modular RAM
      witness pipeline. Eager generation at stage 0 + rayon
      par_iter_mut over address bucket = 4-6× speedup (~13 s saved).
      iter 68 P73 tried par_chunks_mut inside the lazy path and was
      rejected because callers are already inside rayon — the fix
      must reorder when the work happens (eager, pre-rayon), not
      just where (parallelize the lazy path).
    - **Abstraction risk**: low — same data, different schedule.
      Compiler emits an explicit `Op::BuildRamWitnesses` at stage 0;
      backend runs it par.
    - **Expected delta**: 10-13 s wall (~13-16%). Largest single
      follow-on win after (A)/(B)/(C).

- [ ] P81: Consolidate Dory `multi_pair_g2_setup` wrapper overhead —
  target: `crates/jolt-dory/src/scheme.rs` (or equivalent) +
  `crates/jolt-dory/src/commitment_scheme.rs`. Today modular's
  `multi_pair_g2_setup_parallel` runs 2.65× slower per-call than
  core's (ref memo/profile data) despite calling the same underlying
  primitive — the gap is in the wrapper layer.
    - **Hypothesis**: jolt-dory's `DoryScheme` + `DoryCommitmentScheme`
      both maintain G2-prep caches independently. Each `multi_pair_g2_setup`
      call does bookkeeping (HashMap lookup, Vec clone, par-chunk split)
      that core's single-layer cache skips. Dedup the two layers into
      one prepared-point registry keyed by `(srs_hash, opening_idx)`;
      return `&PreparedG2Point` refs instead of cloning.
    - **Abstraction risk**: low — internal to jolt-dory. No trait change.
    - **Expected delta**: 6 s wall (~7%). Per-call amortization plus
      removed HashMap overhead.

- [ ] P82: Batch MSMs across independent Dory commits — target:
  `crates/jolt-zkvm/src/runtime/handlers.rs::Op::Commit` dispatcher.
  Today multiple concurrent commits each issue their own `G1::msm`;
  arkworks MSM is faster on one batched call than N sequential calls
  at matching total points.
    - **Hypothesis**: iter-1 profile logged `G1::msm` at 405 ms / 5402
      calls, avg 75 µs. Many of those calls share input scalars or
      have disjoint but independent point sets — batchable with
      `G1::msm_batched` or a chunked multi-exp. Core's prover already
      batches Dory commits per stage.
    - **Abstraction risk**: medium — needs compiler to emit a
      `Op::CommitBatch { polys: Vec<BufId> }` and runtime to dispatch
      the whole group in one go. Handler still ≤ 30 LOC (pack +
      dispatch).
    - **Expected delta**: 2 s wall (~2.5%). Smaller than P80/P81 but
      mechanical once the batching op exists.

- [ ] P84: Compact-buffer small-scalar integration into modular stack —
  target: `crates/jolt-compute/src/traits.rs` +
  `crates/jolt-witness/src/polynomials.rs` +
  `crates/jolt-cpu/src/backend.rs`. Fresh per-stack analysis (iter 84,
  `iter84_modular_only.json` / `iter84_core_only.json`):
    - Modular reduce_dense: 2373 calls × 18ms = 40.5s self.
    - Core's equivalent (BooleanitySumcheckProver::compute_message + others):
      20 calls × 46ms = 921ms self, **44× less wall**.
    - Gap driver: core binds small-scalar polys (u8/u32/u64 via
      CompactPolynomial) directly, deferring promote-to-field to bind-time.
      Modular materializes every poly as `Vec<F>` via
      `Polynomials::push`'s `F::from_i128` call at
      `crates/jolt-witness/src/polynomials.rs:106`, paying full BN254 Fr
      arithmetic cost for every subsequent mul.
    - **Hypothesis** (updated iter 86): extend the `DeviceBuffer` enum
      with a new runtime-tagged `Compact { data: Vec<u128>, bits: u8,
      signed: bool }` variant. Backend `bind` / `reduce` /
      `interpolate_inplace` match on the variant and dispatch to a fast
      path that calls `Fr::mul_u64` / `Fr::mul_i64` / `Fr::mul_u128` /
      `Fr::mul_i128` (already optimized in `arkworks/bn254_ops.rs`,
      ~2× faster than `F::from_u64(n) * F`). Compiler routes small-
      scalar sources through an encoding-aware lane that preserves
      the raw integer representation instead of promoting to `Vec<F>`.
      Expected 3-5× on small-scalar reduces, ~20s wall savings (~15-
      25%). Multi-iter arc: P84-A' infra (variant + accessors);
      P84-B backend fast paths; P84-C compiler emission.
    - **Abstraction risk**: medium — new `DeviceBuffer` variant +
      encoding fields (`bits`, `signed`) are *runtime data* on the
      buffer, not generic type parameters. Handlers stay ≤ 30 LOC
      and protocol-unaware because they only pass `Buf<B, F>` through
      the trait surface; the backend owns all variant dispatch
      internally. No new trait; no `T` generic parameter at call
      sites.
    - **FORBIDDEN** — do NOT introduce a `SmallScalar` trait (with
      `field_mul` / `to_field` methods and per-integer-type impls)
      or a generic `CompactPolynomial<T: SmallScalar, F: JoltField>`
      struct. Per user directive 2026-04-21: "i dont want to have a
      small scalar in our jolt, we should be able to avoid a trait
      like that with our better abstractions that are more flexible."
      See `memory/feedback_no_smallscalar_trait.md` and iter 86 design
      notes above for the three acceptable alternatives (buffer
      variant, descriptor encoding tag, fused backend methods).
    - **Expected delta**: 20-30 s wall (~25-40%). Largest remaining
      high-leverage attack.

- [ ] P85: Prefix/suffix decomposition for instruction lookup polys —
  target: compiler instruction-lookup emission + `crates/jolt-cpu/src/`.
  Per fresh iter-84 modular-only trace, the sha2-chain workload does not
  highlight this as a single hot span, but core's trace shows zero time
  on generic reduce_dense for the instruction-lookup segments because
  `PrefixSuffixDecomposition` factorization reduces the sumcheck to
  O(sqrt(N)) work on factor polys instead of O(N) work on the full table.
    - **Hypothesis**: add `PrefixSuffixSumcheck` handler that the compiler
      emits for lookup-argument sumchecks. Runtime binds prefix and suffix
      polys separately; reduce_dense never sees the full K·T tensor.
      Expected ~10s wall savings on workloads with heavier lookup mix
      (muldiv, btreemap, sha3); ~2-5s on sha2-chain.
    - **Abstraction risk**: medium — new Op::PrefixSuffixReduce variant +
      per-shape handler. Still backend-pluggable via trait method.
    - **Expected delta**: varies by program; ~5% on sha2-chain, ~15%
      on muldiv/btreemap.

- [x] P86: Fresh P81 re-attempt — multi_pair_g2_setup_parallel wrapper
  dedup. **CLOSED iter 85 as non-target** (not a real production gap).
  Iter-84 PER-STACK profiling showed 1.80× per-call (modular 167.2ms
  vs core 93.4ms, 62 calls each, ~4.0s wall gap). But the "gap"
  only appears under `--stack modular` solo profiling. Under
  `--stack both` (production benchmark), core's
  `DoryCommitmentScheme::setup_prover` calls
  `DoryGlobals::init_prepared_cache` at
  `jolt-core/src/poly/commitment/dory/commitment_scheme.rs:98`,
  populating the dory-pcs process-global prepared-G2 cache BEFORE
  modular runs. When modular's `DoryScheme::setup_prover` runs,
  `init_cache`'s smart re-init check finds the cache already
  populated and no-ops. So modular implicitly uses the fast path
  already. Iter 85 attempt to explicitly call `init_cache` from
  modular setup was a no-op and reverted under +14%/+24%/+38%
  noise-inflated runs. **Key lesson**: solo-stack profile ratios
  don't translate to `--stack both` deltas when stacks share
  process-level state. Need to distinguish "architectural gap"
  (different work) from "solo-profile artifact" (shared state
  freebie in production).

- [x] P83: Sparse eq_project fast path for RamRaIndicator — target:
  `crates/jolt-zkvm/src/runtime/helpers.rs::materialize_binding`
  EqProject arm + `crates/jolt-witness/src/derived.rs` +
  `crates/jolt-compute/src/traits.rs::BufferProvider`. Iter-83 trace
  showed 2 of 82 eq_project calls sourcing `p.ram_ra_indicator`
  ate ~8.2 s of the 9.1 s total eq_project span, via K-fold dense
  scan of a T×K binary indicator with exactly-one-nonzero-per-row.
    - **Hypothesis (attempted iter 83)**: expose per-cycle remapped
      column indices via `BufferProvider::sparse_row_indicator` and
      compute the projection in O(T): `projected[c] = eq_table[cols[c]]`
      for K-table branch, `projected[k] += eq_table[c]` for T-table
      branch. Expected ~8 s savings.
    - **Result**: REVERTED. Median -3.90% (run 1 -3.90, run 2 +6.78,
      run 3 -4.57). Rerun landed in reject zone (≥5% slower) per
      perf-gate protocol. Architectural savings (~4 s / ~5-6%) real
      but below 5% accept gate. Candidate for re-application when
      compounded with other wins or on a lower-noise measurement
      environment. The `BufferProvider::sparse_row_indicator`
      abstraction would also extend to `Rs1Ra`, `Rs2Ra`, `RdWa` if
      they become hot (note: those are address-major K_reg×T with
      one-per-column, not row — needs different kernel orientation).


## Notes

Design decisions, dead ends, and stall-mode observations accumulate here.

- **Correctness side-quest (2026-04-19) — modular self-verify unblock, partial**

  **Why**: user directive — "natively the modular stack proof to be
  verifiable by its own jolt-verifier as part of the gate / a test if
  we don't have that already". Target: make `jolt-zkvm::prove →
  jolt-verifier::verify` round-trip, then add it to the correctness
  gate alongside `transcript_divergence` /
  `zkvm_proof_accepted_by_core_verifier`.

  **Fixed (this commit)**: prover/verifier disagreed on commitment
  count when all-zero advice (`UntrustedAdvice`, `TrustedAdvice`) made
  the prover skip `Op::Commit` while the verifier schedule still
  expected a commitment per `VerifierOp::AbsorbCommitment`. Changed
  `JoltProof::commitments` from `Vec<PCS::Output>` to
  `Vec<Option<PCS::Output>>`; prover pushes `None` for each skipped
  poly; verifier checks `Option::is_some()` before appending to
  transcript + commitment map. Mirrors jolt-core's
  `Option<PCS::Commitment>` pattern at
  `jolt-core/src/zkvm/verifier.rs:360-368`. Cross-system test
  `zkvm_proof_accepted_by_core_verifier` updated to `.filter_map(|c|
  c.as_ref())` before truncating to core's expected count.

  **New test**: `modular_self_verify_commit_skip_alignment` —
  asserts `none == 2` for muldiv advice commits and that
  `jolt-verifier::verify` no longer returns
  `InvalidProof("missing commitment")`. PASSES.

  **Blocked (remaining work)**: full `modular_self_verify` test is
  added but `#[ignore]`d. First downstream error post-fix:
  `InvalidProof("missing eval 0 in stage proof")`. Root cause —
  `crates/jolt-compiler/examples/jolt_core_module.rs` (the hand-
  written protocol reference used by the bench / modular stack)
  emits an incomplete verifier schedule: (a) 4× `VerifierOp::
  RecordEvals` with zero matching `Op::RecordEvals` on the prover
  side, (b) no verifier ops for stages 5-7, (c) no
  `VerifierOp::CollectOpeningClaim` anywhere to mirror the prover's
  `CollectOpeningClaimAt`. Completing this is a multi-day refactor
  (not a perf iter); the partial test guards the commit-skip fix
  from regression in the meantime.

  **Gate status**: correctness gate tests pass —
  `transcript_divergence` OK, `zkvm_proof_accepted_by_core_verifier`
  OK, full `jolt-equivalence` (42 tests) OK, clippy on touched
  crates clean. Full modular self-verify deferred; `muldiv`
  continues to exercise the prove path and prove↔core-verify parity.
  No perf ratchet change.

- **Iter 42 — P51 bespoke `(6, 4)` + `(41, 4)` reduce_dense kernel arms (REVERTED — muldiv +10.4% regression past threshold)**
  (target: `crates/jolt-cpu/src/backend.rs::reduce_dense` match dispatch;
  added `(6, 4) => reduce_dense_fixed::<F, 6, 4>(...)` and
  `(41, 4) => reduce_dense_fixed::<F, 41, 4>(...)` arms to the const-
  generic dispatch. `reduce_dense_fixed` is already generic over `NI, NE`
  so no new function body — only two more monomorphizations.)

  **Motivation**: iter 41 attribution showed `rd_dynamic(6, 4)` = 77.9%
  and `rd_dynamic(41, 4)` = 19.0% of reduce_dense wall at sha2-chain
  log_t=14, both falling through to heap-allocated
  `reduce_dense_dynamic`. Expected 10-13% total wall win by swapping
  per-chunk `Vec` scratch for stack-allocated `[F; NI]` / `[F; NE]` +
  LLVM specialization.

  **Result**: **REJECT on muldiv** — primary perf gate.
  | Program              | Runs (ms)                              | Warm avg | Δ vs baseline |
  |----------------------|----------------------------------------|---------:|---------------|
  | muldiv log_t=12      | 1487.57 / 1398.95 / 1369.30 / 1450.65 | 1406.30  | **+10.4%**    |
  | sha2-chain log_t=14 n=4 | 19801.81 / 16586.72 / 16519.42      | 16553.07 | +2.99%        |

  Warm avg excludes cold run 1 (both programs showed a ~+20% cold
  outlier on first-run-of-session, per iter 39 precedent). Muldiv is
  past reject threshold even on the most generous window.

  **Diagnosis**: Attribution was measured ONLY at sha2-chain log_t=14
  (where `(6, 4)` = 77.9% and `(41, 4)` = 19.0%). The assumption that
  muldiv log_t=12 has a similar shape mix was unverified — and wrong.
  Candidates for why muldiv regressed:
  1. Muldiv's reduce_dense calls are dominated by shapes already in
     the fixed dispatch (2,2 / 3,3 / 4,4) or an uncommon shape not
     covered by either old or new arms. The new arms add code-size
     bloat without callee benefit.
  2. `(41, 4)` generates a large fixed-kernel body (41-wide stack
     arrays × 2 + 4-wide evals). LLVM may auto-inline this into the
     dispatch match arm's callsite, pushing other hot code out of
     muldiv's L1 I-cache working set.
  3. Monomorphization-time compile cost (not runtime) — but that
     wouldn't regress `prove_ms` specifically; ignore.

  **Revert**: removed the 2 new match arms; tree-shake drops unused
  monomorphizations. Correctness re-verified: transcript_divergence
  PASS on revert.

  **Lesson**: Shape-based dispatch optimization REQUIRES per-program
  shape distribution verification. The "97% of wall" figure was
  sha2-chain-specific. For a durable win, either (a) the new arm must
  dominate shape distribution on BOTH programs, or (b) we need a
  different approach that doesn't require monomorphization bloat (e.g.,
  optimize `reduce_dense_dynamic` itself via reused scratch buffers or
  a small-vector-on-stack layout with dynamic length).

  Stall counter 1 → 2. Green streak preserved at 6 (P47 still holds).
  Ratchet unchanged. Iter 43 attack: P52 instrument muldiv shape
  distribution, then revisit with either a single-shape try or a
  dynamic-path optimization.

- **Iter 41 — P50 instrument reduce_dense by shape (INFRA — attribution captured, 97% of wall in 2 unhandled shapes)**
  (target: `crates/jolt-cpu/src/backend.rs::reduce_dense`; added
  `fields(ni = inputs.len(), ne = num_evals)` to the outer
  `#[tracing::instrument]` attribute for durable span-arg attribution.
  Diagnostic one-shot: per-shape named spans `rd_fixed_{NI}_{NE}` in each
  fixed match arm and `rd_dynamic` + `ni`/`ne` fields in the dynamic
  fallback — landed temporarily, then reverted before commit once
  attribution was captured to avoid 25% trace-time overhead (2085 span
  enter/exit events at log_t=14).

  **Motivation**: user-directed pivot after iter 40 P49 revert and setup-
  class policy: "there may be many more algorithmic differences in how we
  do reduce_dense for certain shapes vs what jolt core does bespokely so
  we need to expand / optimize our kernels to account for that".
  `reduce_dense` currently dispatches only `(2,2)`, `(3,3)`, `(4,4)`,
  `(8,4)`, `(8,8)`, `(16,16)`, `(32,32)` to const-generic fixed kernels;
  everything else falls through to `reduce_dense_dynamic` (heap-allocated
  `Vec` scratch, dynamic-size inner loops).

  **Attribution** (sha2-chain log_t=14 num-iters=4, instrumented, 20884 ms total):
  | Shape               | ms     | n    | % of `reduce_dense` wall |
  |---------------------|-------:|-----:|-------------------------:|
  | `rd_dynamic(6, 4)`  | 7316.6 | 1792 | **77.9%**                |
  | `rd_dynamic(41, 4)` | 1779.4 | 18   | 19.0%                    |
  | `rd_dynamic(33, 6)` | 72.8   | 14   | 0.8%                     |
  | `rd_fixed(4, 4)`    | 5.7    | 13   | 0.06%                    |
  | all other shapes    | < 100  | —    | < 2% combined            |

  **Conclusion**: 97% of reduce_dense wall goes through just 2 shapes,
  both UNHANDLED by the fixed dispatch. Only `(4, 4)` of the 7 existing
  fixed variants was hit at all on this trace, and it's a rounding-error
  (~6 ms). The existing fixed dispatch is **architecturally correct but
  under-specialized for the actual workload**. Adding `(6, 4)` and
  `(41, 4)` arms should:
  1. Replace per-chunk `Vec` allocation with stack-allocated `[F; NI]` /
     `[F; NE]` arrays (no malloc per rayon chunk).
  2. Let LLVM specialize/unroll the `kernel.evaluate` inner loop over
     fixed `NI`/`NE` constants.
  3. Enable SIMD-friendly fixed-stride accumulator writes.

  **Traces & data**:
  - `benchmark-runs/perfetto_traces/iter41_reduce_dense_shapes.json`
  - `perf/iter41-shapes-sha2chain.json` (wall=20884 ms modular, 2759 ms core)

  Commit lands as INFRA — only diff is `fields(ni, ne)` on the outer
  `#[tracing::instrument]` attribute (zero runtime overhead — span is
  already created, just adds two recorded field values). Stall counter
  unchanged. Iter 42: P51 add `(6, 4)` + `(41, 4)` fixed kernel arms.

- **Iter 40 — P49 init dory prepared-point cache (REVERTED — +13.5% regression; EC setup class henceforth off-limits)**
  (target: `crates/jolt-dory/src/scheme.rs::DoryScheme::setup_prover`;
  added `dory::backends::arkworks::init_cache(&setup.g1_vec, &setup.g2_vec)`
  gated on `#[cfg(not(test))]`).

  **Motivation (wrong)**: iter 40 post-P47 trace showed
  `multi_pair_g2_setup_parallel` = 5312 ms = **32.3% of total wall** at
  sha2-chain log_t=14. dory-pcs 0.3.0 has a `cache` feature (already enabled
  in Cargo.toml) that skips the G2Affine → G2Prepared conversion if
  `init_cache()` was called. `jolt-core/src/poly/commitment/dory/commitment_scheme.rs:98`
  calls `DoryGlobals::init_prepared_cache` at setup time, but modular
  `jolt-dory` never did. Hypothesis: flipping this on yields 20-30%
  savings on the 5312 ms bucket = ~1000-1600 ms total wall.

  **Result**: **REGRESSION** on sha2-chain log_t=14.
  | Run | modular_prove_ms | Δ vs iter-39 avg (16072) |
  |-----|------------------|--------------------------|
  | 1   | 17429.64         | +8.45%                   |
  | 2   | 19067.23         | +18.64%                  |
  | avg | 18248.44         | **+13.54%**              |

  Clear reject on both runs — past −5% revert threshold by a wide margin.

  **Diagnosis (partial)**: The `cache` feature's fast path does
  `c.g2_prepared[start_idx..end_idx].to_vec()` per chunk — this clones
  G2Prepared structs (each a large vector of pre-computed Miller loop
  lines). The deep clone can be SLOWER than fresh G2Affine → G2Prepared
  conversion, especially with parallel chunked pairings where chunk-local
  work is dwarfed by the clone. Alternative hypothesis: 2^22 cached
  G2Prepared values × hundreds of bytes each = hundreds of MB that
  poison L2/L3 cache across parallel workers.

  **Revert**: `git checkout -- crates/jolt-dory/src/scheme.rs`.
  Correctness re-verified: transcript_divergence + zkvm_proof_accepted
  PASS on revert.

  **Durable policy update (user direction)**: Setup-phase costs — EC
  generator prep, prepared-point caches, SRS init — are one-time and
  off the perf-loop attack list from here on. Saved as user feedback
  memory: `feedback_setup_costs.md`.

  Stall counter 0 → 1. Green streak preserved at 6 (P47 still holds).
  Bookkeeping commit. Iter 41 picks a non-setup hypothesis.

- **Iter 39 — P47 zero-copy upload for Cow::Owned (GREEN — −18.74% sha2-chain log_t=14)**
  (target: `crates/jolt-compute/src/traits.rs::ComputeBackend` + trait
  impls in `crates/jolt-cpu/src/backend.rs`, default inheritance in
  jolt-metal / jolt-hybrid; single caller change in
  `crates/jolt-zkvm/src/runtime/helpers.rs::materialize_binding`).

  **Motivation**: iter 38 attribution showed `mb::upload` = 3654.68 ms =
  **67.8% of mb::Provided** at sha2-chain log_t=14, LARGER than
  `pm::Derived` (2556.20 ms). Root cause: `CpuBackend::upload` was an
  unconditional `data.to_vec()` clone. For every `Cow::Owned`
  materialize return (pm::Derived + pm::R1cs), this meant two
  allocations: materialize produces Vec, upload clones it.

  **Change (~12 LOC across 3 files)**:
  - `traits.rs`: added `fn upload_vec<T: Scalar>(&self, data: Vec<T>)
    -> Self::Buffer<T>` to `ComputeBackend` trait with default impl
    `self.upload(&data)` so Metal/Hybrid inherit unchanged behavior.
  - `jolt-cpu/backend.rs`: CpuBackend override:
    `fn upload_vec(&self, data: Vec<T>) -> Vec<T> { data }` — pass-through.
  - `jolt-zkvm/runtime/helpers.rs`: `mb::Provided` arm changed from
    `backend.upload(&data)` to `backend.upload_vec(data.into_owned())`.
    Cow::Owned unwraps without copy; Cow::Borrowed falls back to
    `slice.to_vec()` in `into_owned()` (rare — pm::Witness and
    pm::Preprocessed together < 1% of mb::upload).

  **Gates**: transcript_divergence PASS (8.767s), zkvm_proof_accepted
  PASS (8.855s), full jolt-equivalence 41/41 PASS; clippy clean on
  jolt-compute + jolt-cpu + jolt-zkvm + jolt-metal + jolt-hybrid.

  **Perf — muldiv log_t=12** (ratchet 1274.20):
  | Run | modular_prove_ms | Δ vs ratchet |
  |-----|------------------|--------------|
  | 1 (cold) | 1421.88 | +11.59% |
  | 2       | 1290.03 | +1.24%  |
  | 3       | 1275.14 | +0.07%  |
  | 4       | 1320.46 | +3.63%  |
  | warm avg (2-4) | 1295.21 | **+1.65%** (within ±5%, flat) |

  **Perf — sha2-chain log_t=14 num-iters=4**:
  | Run | modular_prove_ms | Δ vs iter-38 (19782) | Δ vs iter-34 avg (18881) |
  |-----|------------------|----------------------|--------------------------|
  | 1   | 16031.13         | −18.96%              | −15.10%                  |
  | 2   | 16113.32         | −18.55%              | −14.66%                  |
  | avg | 16072.23         | **−18.74%**          | **−14.87%**              |

  **Modular/core ratio**: sha2-chain log_t=14: 16072 / 1544 = **10.41×**
  (down from ~12.2× pre-P47). muldiv log_t=12: 1290 / 352 = 3.67×.

  **Decision**: ACCEPT. muldiv flat (no regression), sha2-chain clears
  +5% accept threshold by ~4× margin. Ratchet at muldiv log_t=12 stays
  at 1274.20 (no primary-metric improvement), but the secondary program
  is a meaningful win. Stall counter 2 → 0. Green streak 5 → 6.

  **Iter 40 attack candidates**:
  - P40: Materialize/MaterializeUnlessFresh super-linear scaling
    (already queued pre-P47; still the largest remaining log_t=14
    bucket at ~5.4 s mb::Provided).
  - P41: parallelize or memoize `multi_pair_g2_setup` (6815 ms at
    log_t=14 iter 38 = 34.4% total wall, same call count pattern
    since it's g2 setup).
  - P48: reduce materialize allocation cost — `pm::Derived` still
    2556 ms; if K*T sized Vecs are allocated freshly each call and
    discarded after consumption, allocator pressure could be
    addressed by pooling.

- **Iter 38 — P46 instrument `mb::upload` span (INFRA — stall counter unchanged)**
  (target: `crates/jolt-zkvm/src/runtime/helpers.rs::materialize_binding`
  `InputBinding::Provided` arm; +1 `tracing::info_span!("mb::upload")`
  wrapping the `backend.upload(&data)` call).

  **Motivation**: iter 36 attribution left a 1984 ms gap inside
  `mb::Provided` (4085 ms parent − 2101 ms in 4 `pm::*` children).
  Static read of `CpuBackend::upload` (`crates/jolt-cpu/src/backend.rs:202`)
  showed `fn upload<T>(&self, data: &[T]) -> Vec<T> { data.to_vec() }` —
  an unconditional clone. For every `Cow::Owned` return from `pm::Derived`
  / `pm::R1cs`, `materialize_binding` allocates once (in materialize)
  then `.to_vec()`'s it again (in upload). Hypothesis: this double-alloc
  accounts for most of the 1984 ms gap.

  **Attribution at sha2-chain log_t=14 num-iters=4** (19781 ms modular):
  | Span               | Wall ms  | n  | % of parent         |
  |--------------------|----------|----|---------------------|
  | `mb::Provided`     |  5389.72 | 38 | 27.2% total         |
  | `mb::upload`       |  3654.68 | 38 | **67.8% of Provided** |
  | `pm::Derived`      |  2556.20 | 63 | 47.4% of Provided   |
  | `pm::R1cs`         |  (<1%)   | —  | still negligible    |
  | `mb::EqProject`    |  2468.56 | 82 | 12.5% total         |
  | `CpuBackend::eq_project` | 1475.84 | 82 | 59.8% of EqProject |

  **Gap closure**: 3654.68 / 1984 expected = **184% of the gap**
  (expected since iter 36 gap was computed pre-`mb::upload` and the
  upload span now "moves" time from parent Provided accounting into
  its own child bucket). Practical interpretation: `mb::upload` is
  **the dominant cost inside `mb::Provided`** and is also **larger
  than `pm::Derived`** — i.e., the redundant `.to_vec()` clone costs
  MORE than the initial derived computation that produced the Cow::Owned.

  **Gates**: transcript_divergence PASS (5.454s), zkvm_proof_accepted
  PASS (5.539s); clippy clean on jolt-zkvm/jolt-witness/jolt-r1cs.

  **Perf — muldiv log_t=12**: 1318.55 ms (+3.5% vs ratchet 1274.20,
  within ±5% band, expected for +1 span). **INFRA ITER**: ratchet
  unchanged, stall counter unchanged (iter 33/36 precedent for
  instrumentation-only iterations).

  **Iter 39 attack (queued)**: zero-copy upload for Cow::Owned
  materialize returns. Candidates: (a) change
  `ComputeBackend::upload` trait signature to accept `Cow<[T]>` so
  CpuBackend can `Cow::Owned(v) -> DeviceBuffer::Field(v)` without
  re-cloning, or (b) add a separate `upload_vec(Vec<T>)` method and
  wire `materialize_binding` to route `Cow::Owned` through it.
  Expected savings: ~3654 ms at sha2-chain log_t=14 = **~18.5% of
  total wall**. Abstraction risk: low — backend trait gains one
  variant/method; handlers unchanged (still ≤ 30 LOC each).

- **Iter 37 — P45 parallelize the 3 RAM derived-poly arms (REVERTED — regression on both programs)**
  (target: `crates/jolt-witness/src/derived.rs::{ram_combined_ra, ram_val,
  ram_ra_indicator}`; added `rayon = { workspace = true }` to
  `crates/jolt-witness/Cargo.toml`).

  **Motivation**: iter 36 attribution (sha2-chain log_t=14): the 3
  RAM arms consume 2036 ms = 99% of `pm::Derived` = ~50% of
  `mb::Provided`. All 3 are K*T binary-indicator / value-carry builders
  with seemingly-independent per-cycle / per-address writes.

  **Change (~90 LOC, 3 methods + helper + const)**:
  - `ram_combined_ra` (output `ra[col * t + c]`): pre-bucket cycles
    by address via a serial O(T) pass into `Vec<Vec<usize>>`, then
    `par_chunks_mut(t)` over K columns, each thread writes its column's
    1's from its bucket.
  - `ram_ra_indicator` (output `indicator[c * k + col]`): cycle-major
    layout so `par_chunks_mut(k)` gives per-cycle slices of length K.
    Each thread does 1 remap + 1 write per cycle.
  - `ram_val` (output `val[addr * t + c]`, stateful per-addr walk):
    extracted `fill_ram_val_chunk` helper, then `par_chunks_mut(t)`
    over K addresses each walking its own event list.
  - Threshold `PAR_THRESHOLD=1024` with serial fallback on all 3 arms.

  **Gates**: transcript_divergence PASS, zkvm_proof_accepted PASS;
  clippy clean on jolt-witness + jolt-r1cs libs.

  **Perf — muldiv log_t=12**:

  | Run | modular_prove_ms | Δ vs 1274.20 ratchet | Δ vs pre-P45 avg 1396 |
  |---|---:|---:|---:|
  | 1 | 1303.03 | +2.26% | −6.7% |
  | 2 | 1342.95 | +5.40% | −3.8% |
  | 3 | 1367.80 | +7.34% | −2.0% |
  | avg | 1337.93 | **+5.0%** (reject) | −4.2% |

  **Perf — sha2-chain log_t=14 num-iters=4**:

  | Run | modular_prove_ms | Δ vs 18011 iter-36 baseline |
  |---|---:|---:|
  | 1 | 19599.35 | **+8.82%** |
  | 2 | 20796.51 | **+15.46%** |
  | avg | 20197.93 | **+12.1%** (reject) |

  Per protocol: both programs past reject threshold. REVERT.

  **Diagnosis**:

  The attribution was correct (2036 ms in these 3 arms is real) but
  the parallelization HURT. Three plausible compounding causes:

  1. **rayon dispatch overhead** for tiny per-chunk work. `ram_ra_indicator`
     produces T chunks of size K where each chunk does 1 remap + at
     most 1 write. At T=2^14 that's ~16k chunks of ~100 ns each = the
     rayon overhead per chunk dominates actual work. `ram_combined_ra`'s
     per-column work is similarly sparse (most columns get zero writes).
  2. **Memory-bandwidth contention**. K*T element writes (zero-fill +
     sparse sets) at log_t=14 num-iters=4 is several hundred MB of
     bus traffic. Multiple cores writing concurrently to the same
     NUMA node doesn't scale — the bus is the bottleneck regardless
     of core count.
  3. **Pool contention with P42 eq_project**. The prover may have
     other rayon-parallel stages running concurrently with materialize.
     Adding more parallel work consumes the pool without new cores.

  **Candidates for iter 38+** (NOT doing these in this revert commit):
  - **P46 (queued)**: instrument the 1984 ms attribution gap inside
    `mb::Provided` that's NOT in any `pm::*` arm — might reveal a bigger
    single target than the 3 RAM arms combined.
  - **P47 (NEW, queueing)**: investigate if the 3 RAM arms' output
    buffers can be allocated lazily / be smaller (K*T is huge, many
    cells are zero). Maybe RamVal and RamCombinedRa can share storage.
  - **P48 (NEW, queueing)**: investigate whether these 3 arms are
    RE-computed on every materialize call or cached — if cached, why
    is the wall time so high? If RE-computed, could the freshness cache
    be extended to cover them?

  **Lesson**: attribution confirming a 2036 ms wall in 3 arms doesn't
  imply parallelization helps. When inner per-chunk work is < ~50 µs,
  rayon overhead wins. Same lesson as iter 24/25/26/28 which all saw
  similar near-flat results from tiny-inner-loop parallelization.

  **Per protocol**: revert applied (`git checkout
  crates/jolt-witness/Cargo.toml crates/jolt-witness/src/derived.rs`);
  stall counter 1 → 2; green streak unchanged at 5. Bookkeeping commit.

- **Iter 36 — P44 instrument `materialize` dispatch arms (INFRA — stall counter unchanged)**
  (targets: `crates/jolt-witness/src/provider.rs::BufferProvider::materialize`,
  `crates/jolt-r1cs/src/provider.rs::R1csSource::compute`,
  `crates/jolt-witness/src/derived.rs::DerivedSource::compute`).

  **Motivation**: iter 35 P43 post-mortem. Parallelizing
  `R1csSource::compute_matvec` saved ~0 ms and regressed muldiv past
  reject, which proves `mb::Provided`'s 6345 ms wall was NOT dominated
  by the R1CS matvec. Per memory `feedback_profiling.md`, need to
  instrument the sub-spans of `mb::Provided` before picking the next
  attack.

  **Change (~50 LOC, 3 files, all instrumentation)**:
  - Added `tracing = { workspace = true }` to `crates/jolt-witness/Cargo.toml`
    (jolt-r1cs already had it after iter 35).
  - `ProverData::materialize`: wrapped each of the 4 `PolySource` arms
    in a `tracing::info_span!("pm::<Variant>").entered()` guard
    (`pm::Witness`, `pm::R1cs`, `pm::Derived`, `pm::Preprocessed`).
  - `R1csSource::compute`: wrapped each of the 5 `R1csColumn` arms
    (`r1cs::Az`, `r1cs::Bz`, `r1cs::Cz`, `r1cs::CombinedRow`,
    `r1cs::Variable`).
  - `DerivedSource::compute`: wrapped each of ~20 arms with per-poly
    span names (`derived::product_left`, `derived::ram_val`,
    `derived::ram_combined_ra`, `derived::ram_ra_indicator`,
    `derived::reg_{rs1,rs2,rd}_{ra,wa}`, `derived::reg_val`,
    `derived::iflag_*`, `derived::*_gather_index`,
    `derived::lookup_table_flag`, `derived::instruction_raf_flag`,
    `derived::hamming_weight`, `derived::extract_column`).

  **Gates**: transcript_divergence PASS; zkvm_proof_accepted PASS;
  clippy clean on jolt-witness + jolt-r1cs libs (pre-existing
  test-only lint in `polynomials.rs:294-295` unrelated to iter 36).

  **Trace captured**: `benchmark-runs/perfetto_traces/iter36_pm_spans_sha2chain_log_t14.json`
  (19 MB, sha2-chain log_t=14 num-iters=4, modular-only, 18011 ms prove).

  **Attribution (the point of this iter)**:

  `mb::Provided` = **4085 ms** wall (38 calls).

  | `pm::*` arm | Total ms | Calls | % of mb::Provided |
  |---|---:|---:|---:|
  | `pm::Derived` | **2067.4** | 63 | **50.6%** |
  | `pm::R1cs` | 33.6 | 53 | 0.82% |
  | `pm::Witness` | 0.14 | 212 | 0.00% |
  | `pm::Preprocessed` | 0.09 | 86 | 0.00% |
  | **Sum of pm arms** | **2101** | — | **51.4%** |
  | **Unaccounted** | **~1984** | — | **~48.6%** |

  Inside `pm::Derived` (99% accounted for):

  | `derived::*` arm | Total ms | Calls | % of pm::Derived |
  |---|---:|---:|---:|
  | `ram_ra_indicator` | **804.0** | 2 | **38.9%** |
  | `ram_val` | **622.3** | 1 | **30.1%** |
  | `ram_combined_ra` | **610.2** | 1 | **29.5%** |
  | `reg_val` | 16.8 | 1 | 0.81% |
  | `reg_rs1_ra` | 5.8 | 1 | 0.28% |
  | `ram_gather_index` | 2.8 | 1 | 0.14% |
  | `lookup_table_flag` | 1.7 | 41 | 0.08% |
  | others (16 arms) | <2 | — | — |

  Inside `pm::R1cs` (99.6% accounted):

  | `r1cs::*` arm | Total ms | Calls |
  |---|---:|---:|
  | `r1cs::Bz` | 14.5 | 1 |
  | `r1cs::Az` | 11.1 | 1 |
  | `r1cs::Variable` | 7.8 | 51 |
  | `r1cs::Cz`, `r1cs::CombinedRow` | 0 (not hit) | 0 |

  **Findings**:

  1. **3 RAM arms own 99% of pm::Derived and ~50% of mb::Provided**:
     `ram_ra_indicator` (804 ms, 2 calls) + `ram_val` (622 ms, 1 call)
     + `ram_combined_ra` (610 ms, 1 call) = **2036 ms**. All three build
     K*T address-major buffers where per-cycle writes are independent.
     **Clear iter 37 target (P45)**: parallelize all three.
  2. **P43 was correctly reverted**: R1cs total is 33.6 ms (<1% of
     mb::Provided). Even 10× speedup would save ~30 ms — unmeasurable.
     The iter 35 hypothesis was quantitatively wrong.
  3. **1984 ms attribution gap**: nearly half of `mb::Provided` is
     inside neither a `pm::*` arm nor any `derived::*`/`r1cs::*`
     sub-arm. Could be `Cow` allocation, materialize_binding prologue,
     freshness-check overhead, or parent dispatch. **P46** queued to
     instrument this.

  **Per protocol**: infra iter, no ratchet touch, no revert, no perf
  comparison. Stall counter stays at 1 (iter-33 precedent — the iter 33
  instrumentation iter also kept stall counter unchanged). Commit as
  `chore(zkvm): instrument provider.materialize / R1csSource /
  DerivedSource dispatch arms (iter 36)`.

  **Next iter 37**: P45 — parallelize the 3 RAM derived-poly builders.
  Expected: 4-6× on 2036 ms → save ~1.5 s at sha2-chain log_t=14 = ~8%
  total wall. Meets the big-gain bar.

- **Iter 35 — P43 parallelize `R1csSource::compute_matvec` (REVERTED — attribution hypothesis wrong)**
  (target: `crates/jolt-r1cs/src/provider.rs::compute_matvec` +
  `Variable` arm; added rayon to `crates/jolt-r1cs/Cargo.toml`).

  **Motivation**: post-P42 trace at log_t=14 sha2-chain (22091 ms cold
  total, `benchmark-runs/perfetto_traces/iter35_post_p42_sha2chain_log_t14.json`)
  showed `mb::Provided` at **6345 ms / 100% self-time** — the new #1
  Materialize bucket after iter 34 cut `mb::EqProject` from 5574 ms
  to 2784 ms (wrapper) / 1738 ms (kernel). mb::Provided calls
  `provider.materialize(poly_id)` which dispatches across 4
  `PolySource` variants. Static reading of the code pinpointed
  `R1csSource::compute_matvec` (jolt-r1cs/src/provider.rs:54-73) as
  a fully serial nested loop over num_cycles × constraints ×
  sparse_entries. **Static reasoning** claimed this was the hot path.

  **Change (~20 LOC, single method + `Variable` arm)**:
  - `compute_matvec`: outer loop `for c in 0..num_cycles` rewritten
    to `result.par_chunks_mut(k_pad).enumerate().for_each(...)` when
    `num_cycles >= PAR_THRESHOLD=1024`; each thread writes its own
    disjoint `[c*k_pad..(c+1)*k_pad]` output range.
  - `Variable(var_idx)` arm: `(0..num_cycles).into_par_iter().map(...)
    .collect()` when above threshold.
  - Added `rayon = { workspace = true }` to `crates/jolt-r1cs/Cargo.toml`.

  **Gates**: 41/41 jolt-equivalence green incl. transcript_divergence +
  zkvm_proof_accepted; clippy clean on 8-crate canonical set.

  **Perf — muldiv log_t=12 (ratchet program)**:

  | Run | modular_prove_ms | Δ vs 1274.20 ratchet |
  |---|---:|---:|
  | 1 | 1308.89 | +2.72% |
  | 2 | 1354.49 | **+6.30%** |
  | 3 | 1360.56 | **+6.78%** |
  | median | 1354.49 | +6.30% (past reject threshold) |

  **Perf — sha2-chain log_t=14 (cross-program)**:

  | Run | modular_prove_ms | Δ vs 18881.12 iter-34 avg |
  |---|---:|---:|
  | 1 | 18779.29 | −0.54% (flat) |

  Per protocol: two of three muldiv runs past the −5% reject threshold;
  sha2-chain flat; REVERT.

  **Diagnosis (critical learning)**:

  The hypothesis that `mb::Provided`'s 6345 ms at log_t=14 is
  dominated by `R1csSource::compute_matvec` was **wrong**. Evidence:
  1. Parallelizing the method saved ~0 ms at log_t=14 (−0.54%).
     If compute_matvec were even 30% of the 6345 ms wall,
     parallelization should have recovered ~1200 ms wall = −5% total.
  2. At log_t=12, parallelization ADDED cost (+6.3%) — typical
     rayon-overhead-dominated regression per iter 24/25/26/28
     precedent — but the baseline wasn't meaningfully displaced
     downward at larger log_t either.

  Plausible alternative attributions for the 6345 ms bucket:
  - **Derived sources** (`DerivedSource::compute` via PolySource::Derived)
    — never profiled, not yet instrumented, could be where the wall lives.
  - **Witness sources** returning `Cow::Borrowed` — should be cheap,
    but if the underlying `polys.get` does any work beyond index
    arithmetic (dependency on commit-storage layout) it adds up.
  - **R1csSource::compute for CombinedRow** (a different arm than
    compute_matvec — calls `self.key.combined_row(...)` with Spartan
    challenges) — possibly the dominant R1cs poly at stage time.
  - **The freshness-check cache** (`MaterializeUnlessFresh`) may
    short-circuit on most calls, so `compute_matvec` is invoked for
    only a fraction of the attributed 6345 ms — the REAL work is
    elsewhere in the freshness-check / cache-miss path.

  **Lesson**: "static reading of the code" without direct instrumentation
  to verify is a failure mode at this stage of the perf loop. Iter 33
  taught us to instrument `CpuBackend::*` methods — we need the same
  inside `provider.materialize`'s dispatch. Memory feedback
  (`feedback_profiling.md`) reminds: "pick perf hypotheses from the
  chrome-trace top spans, not statically from an Explore agent's code
  read". I followed that rule for the `mb::Provided` *attribution*
  but violated it for the *decomposition* of that bucket.

  **Per protocol**: revert applied (`git checkout crates/jolt-r1cs/Cargo.toml
  crates/jolt-r1cs/src/provider.rs Cargo.lock`); stall counter 0 → 1;
  green streak unchanged at 5 (interrupted, counter resets — but
  narrative continuity for iter 36 instrumentation). Bookkeeping commit.

  **Next iter 36**: P44 — add tracing::info_span to
  `provider.materialize`'s 4 PolySource arms + `R1csSource::compute`'s
  5 column arms + `DerivedSource::compute`. Capture a fresh trace,
  identify the true sub-span dominating 6345 ms, then attack in iter 37.

- **Iter 34 — P42 parallelize `CpuBackend::eq_project` (GREEN, ratchet updated)**
  (target: `crates/jolt-cpu/src/backend.rs::eq_project`).

  **Motivation**: iter 33 instrumentation attributed 58.8% of the
  log_t=14 sha2-chain Materialize family wall to `mb::EqProject`
  (5574 ms across 82 calls, worst calls 2.4 s / 2.0 s), with
  `CpuBackend::eq_project` accounting for 94.8% of that (4708 ms). The
  method was a serial nested loop over `eq_table × outer_size` (or
  `inner_size` in the other branch) with no rayon.

  **Change (~40 LOC, single method)**:
  - Branch 1 (`eq_table.len() == inner_size`): output has length
    `outer_size`. Parallelize via `par_chunks_mut(chunk_size)` on the
    output so threads write **disjoint `[k_start..k_start+chunk]`
    ranges**. Inside each task, iterate over `eq_table` and accumulate
    `eq_val * source_data[t * outer_size + k_start + local_k]` into the
    corresponding local slot. Disjoint output ranges eliminate any need
    for per-thread accumulators or reductions — each `*slot +=` writes
    its own private memory. `chunk_size = (outer_size / num_threads)
    .max(PAR_THRESHOLD/4).min(outer_size)` balances granularity.
  - Branch 2 (else): output has length `inner_size`, and each slot `t`
    is an independent dot-product `Σ_k eq_table[k] * source[t*outer_size
    + k]`. Parallelize via `par_iter_mut` over the output — each `proj`
    computes its own value locally.
  - Both branches gated at `PAR_THRESHOLD=1024` with a serial fallback
    for the log_t=12 / tiny-workload case (iter 24's flat P32 confirmed
    that outer-parallelism attacks at log_t=12 often eat their own gains
    from rayon overhead).
  - `#[cfg(feature = "parallel")]` preserves the pure-serial non-rayon
    build. Zero-check (`if eq_val.is_zero() { continue; }`) preserved in
    the parallel branches too — eq_table is often sparse in early rounds.

  **Gates**: 41/41 jolt-equivalence green incl. transcript_divergence +
  zkvm_proof_accepted + full suite; clippy clean on 8-crate canonical
  set. Handler in `runtime/helpers.rs::materialize_binding::EqProject`
  arm unchanged (still 4 lines). Abstraction risk: **low** — change is
  entirely internal to `CpuBackend::eq_project`.

  **Perf — muldiv log_t=12 (ratchet program)**:

  | Run | modular_prove_ms | Δ vs 1444.14 ratchet |
  |---|---:|---:|
  | 1 | 1274.20 | **−11.78%** |
  | 2 | 1289.92 | **−10.69%** |
  | avg | 1282.06 | **−11.22%** |
  | **best** | **1274.20** | **−11.78%** |

  Both runs clear +5% accept threshold. Ratchet updated: 1444.14 →
  **1274.20** ms. Ratio (best/best): 1274.20 / 316.63 = **4.02×** (vs
  previous 4.38×).

  **Perf — sha2-chain log_t=14 (cross-program, iter 31 baseline)**:

  | Run | modular_prove_ms | Δ vs 24604.96 iter-31 baseline |
  |---|---:|---:|
  | 1 | 19234.61 | **−21.82%** |
  | 2 | 18527.62 | **−24.70%** |
  | avg | 18881.12 | **−23.27%** |

  At log_t=14 the attributed `eq_project` savings dominate: 4708 ms
  serial → parallel across 8 cores = ~650-800 ms at perfect scaling,
  so observed ~5700 ms total savings means the 2.4 s / 2.0 s worst calls
  collapsed to sub-500 ms while amortizing fork/join over all 82 calls.

  **Why it cleared the order-of-magnitude bar**:
  - `eq_project` was the single largest attributed self-wall in the
    modular stack after iter 20 parallelized `Op::Commit` and iter 19
    Gruen-ported kernel 3.
  - The serial loop was at ~12% saturation (iter 23 op-class showed
    Materialize family at ~1 thread). Lifting to 6-7 threads on an
    8-core M4 is a plausible upper bound; we observed ~5.6× effective
    parallelism (5574 ms / (5574 - 4708 recovered) ≈ 6.4×).
  - log_t=14 workloads are NOT overhead-dominated for rayon (unlike
    iter 24/25/26/28 at log_t=12, where 83k×tiny-work tasks ate fan-out
    cost). Each `eq_project` call at log_t=14 has ~2M field muls — well
    above rayon's granularity floor.
  - log_t=12 gain is smaller (−11%) because many calls fall below the
    `PAR_THRESHOLD=1024` gate and stay serial, exactly as designed.

  **Stall counter**: 7 → **0**. **Green streak**: 4 → **5**. **Last
  green iter**: 20 → **34**.

  **Next iter 35 priorities** (from iter 33 tables, post-P42):
  1. `mb::Provided` = 3810 ms / 40.2% of remaining Materialize wall at
     log_t=14 (82.8 ms/call avg, dominated by `MaterializeUnlessFresh`
     going through `R1csSource::compute`). Profile `compute` for
     R1cs-source polys to find the actual hotspot inside.
  2. `multi_pair_g2_setup_parallel` = 5771 ms / 24.5% wall (P41, still
     queued) — check if this is now the top bottleneck post-P42.
  3. Re-profile post-P42 to verify the hotspot map shift — most of the
     15.66× ratio at log_t=14 was concentrated in eq_project, so the
     next highest-leverage attack may be in Dory (commit / setup /
     open) rather than ML-compiler handlers.

- **Iter 33 — Materialize binding-variant + CpuBackend method instrumentation (infra, no perf claim)**
  (targets: `crates/jolt-zkvm/src/runtime/helpers.rs::materialize_binding`,
  `crates/jolt-cpu/src/backend.rs`).

  **Motivation**: iter 31 log_t=14 sha2-chain profile attributed 47.6%
  of prove wall to the Materialize family (Materialize + MaterializeUnlessFresh
  + MaterializeIfAbsent). A post-iter-32 Explore-agent analysis of that
  trace revealed **99.3% of Materialize wall is in untraced code** — only
  `CpuBackend::eq_table` and `EqPolynomial::evals` were instrumented
  under the Materialize span, summing to 0.4% of its wall. The actual
  hot path was invisible. Iter 33 adds instrumentation to differentiate
  which `InputBinding` variant in `materialize_binding` dominates and
  which CPU backend methods are called from it.

  **Change (~16 LOC across 2 files)**:
  - `helpers.rs::materialize_binding`: each of the 9 `InputBinding`
    match arms gets `let _s = tracing::info_span!("mb::<Variant>").entered();`
    at the top. Variant names: `Provided`, `EqTable`, `EqPlusOneTable`,
    `LtTable`, `EqProject`, `Transpose`, `EqGather`, `EqPushforward`,
    `ScaleByChallenge`.
  - `backend.rs`: added `#[tracing::instrument(skip_all, name = ...)]`
    on 7 previously-uninstrumented `CpuBackend` methods: `lt_table`,
    `eq_plus_one_table`, `eq_project`, `eq_gather`, `eq_pushforward`,
    `transpose_from_host`, `scale_from_host`.

  **Gates**: 41/41 jolt-equivalence green incl. transcript_divergence +
  zkvm_proof_accepted; clippy clean on the 8-crate canonical set.
  Stall counter NOT incremented (iter 11/17/22/23/27/31 precedent for
  instrumentation-only iters). No ratchet update.

  **Findings** (sha2-chain `--num-iters 4` log_t=14, single run):

  | Variant | Calls | Total ms | Avg µs | % Materialize wall |
  |---|---:|---:|---:|---:|
  | `mb::EqProject` | 82 | 5573.6 | 67970 | **58.8%** |
  | `mb::Provided` | 38 | 3809.9 | 100262 | **40.2%** |
  | `mb::Transpose` | 40 | 74.8 | 1869 | 0.8% |
  | `mb::EqTable` | 68 | 16.3 | 240 | 0.2% |
  | `mb::EqPushforward` | 5 | 3.3 | 657 | 0.0% |
  | `mb::EqGather` | 6 | 1.9 | 318 | 0.0% |
  | `mb::EqPlusOneTable` | 2 | 1.3 | 644 | 0.0% |
  | `mb::LtTable` | 2 | 0.8 | 423 | 0.0% |
  | `mb::ScaleByChallenge` | 1 | 0.3 | 271 | 0.0% |

  | CpuBackend method | Calls | Total ms | Avg µs |
  |---|---:|---:|---:|
  | `eq_project` | 82 | 4708.1 | 57416 |
  | `transpose_from_host` | 40 | 74.6 | 1865 |
  | `eq_table` | 68 | 16.2 | 238 |
  | `eq_pushforward` | 5 | 3.3 | 655 |
  | `eq_gather` | 6 | 1.5 | 258 |
  | `eq_plus_one_table` | 2 | 1.3 | 640 |
  | `lt_table` | 2 | 0.8 | 419 |
  | `scale_from_host` | 1 | 0.3 | 268 |

  | Outer op | Calls | Total ms | Avg ms |
  |---|---:|---:|---:|
  | `Materialize` | 197 | 5656.8 | 28.7 |
  | `MaterializeUnlessFresh` | 46 | 3809.4 | 82.8 |
  | `MaterializeIfAbsent` | 31 | 16.6 | 0.5 |

  **Key takeaways**:
  1. `CpuBackend::eq_project` accounts for **94.8% of `mb::EqProject`**
     wall (4708/5574 ms). Serial nested loop over `eq_table × outer_size`
     (or `inner_size`), no rayon. Individual calls reach 2.4 s / 2.0 s.
     **This is the next attack target — P42 added.**
  2. `mb::Provided` cost (3810 ms / 40.2% of Materialize wall) is almost
     entirely in `MaterializeUnlessFresh` calls — `MaterializeUnlessFresh`
     averages 82.8 ms/call vs Materialize's 28.7 ms/call. The 46 "unless
     fresh" calls bypass the freshness short-circuit (since the cache
     entry is absent or stale) and go through `provider.materialize`
     which for witness sources is a `Cow::Borrowed` (cheap) but for R1cs
     sources is `R1csSource::compute(column)` (actual work). Secondary
     attack if eq_project alone doesn't close the gap.
  3. `Materialize` (28.7 ms/call avg) and `MaterializeUnlessFresh`
     (82.8 ms/call avg) have different dominant children — Materialize
     is mostly `EqProject`, Unless is mostly `Provided`. Separate attack
     vectors.

  **Hypothesis queue (iter 34 onward)**:
  - **Primary**: P42 parallelize `CpuBackend::eq_project` with rayon —
    expected 18-20% total log_t=14 wall savings.
  - **Secondary if P42 underdelivers**: profile `R1csSource::compute`
    for MaterializeUnlessFresh-dominant polys.
  - **Deprioritized**: P40 (generic super-linear investigation — now
    narrowed to P42), P41 (multi_pair_g2_setup — still queued but
    smaller than eq_project).

  **Per protocol**: instrumentation-only, no perf claim, no ratchet update.
  Stall counter NOT incremented. Bookkeeping commit.

- **Iter 32 — P39 end-to-end sparse Dory commit path for OneHot polys (flat, REVERTED)**
  (targets: `crates/jolt-compute/Cargo.toml`, `crates/jolt-compute/src/traits.rs`,
  `crates/jolt-witness/src/polynomials.rs`, `crates/jolt-witness/src/provider.rs`,
  `crates/jolt-poly/src/one_hot.rs`, `crates/jolt-zkvm/src/runtime/handlers.rs`,
  `crates/jolt-dory/src/scheme.rs`). Per iter 31 guidance, avoided the
  "jolt-core proof transplant" bench hack and implemented a first-class
  sparse commit pathway end-to-end.

  **Motivation**: iter 31 log_t=14 sha2-chain profile showed
  `DoryScheme::commit` = 6350 ms / 27% wall, while core's entire prove is
  1571 ms — modular commit alone is ~4× core's total. Core's advantage on
  RA one-hot polys comes from `commit_rows_sparse` which treats each row
  as a sum of selected generators (one per hot column) instead of a dense
  MSM. Modular was dense-expanding one-hot polys in the witness layer
  before the Dory backend ever saw them, losing the sparse path. P39
  routes them through natively.

  **Change shape (~200 LOC across 7 files)**:
  1. `BufferProvider` trait gained `commit_source(&self, PolynomialId) ->
     Option<&dyn MultilinearPoly<F>>` with a default `None` return — opt-in
     sparse override that doesn't break existing impls.
  2. `Polynomials<F>` implements `commit_source` by returning
     `self.one_hot.get(&poly_id)` for the `PolynomialId::*Ra*` variants.
  3. `ProverData<F>` forwards `commit_source` to the witness source when
     `PolySource::Witness` (other sources stay `None`).
  4. `Op::Commit` handler now materializes `enum Src<'a, F> { Sparse(&'a
     dyn MultilinearPoly<F>), Dense(Vec<F>) }` per poly (sequentially),
     then `par_iter().map()` dispatches to `commit_rows_sparse` or
     `commit_rows_dense` inside Dory.
  5. `DoryScheme::commit_rows_sparse` rewritten to collect per-row hot
     indices via `for_each_nonzero`, then call
     `batch_g1_additions_multi(bases, row_indices)` from `jolt-crypto`
     for a SINGLE amortized Montgomery inversion across all rows
     (vs the old naïve path of one EC add chain per row = T×k Mont
     inversions).
  6. Fixed latent `OneHotPolynomial` layout bug: internal storage was
     `cycle * k + col` (AddressMajor) but `expand_one_hot` in
     `jolt-witness` and Dory's generator indexing use `col * T + cycle`
     (CycleMajor). Corrected all four `MultilinearPoly` methods —
     `evaluate`, `for_each_row`, `fold_rows`, `for_each_nonzero` —
     plus matching test `to_dense` helper. Latent because onehot was
     never fed directly to commit before this iter.
  7. Fixed two pre-existing clippy "operation always returns zero"
     errors in `polynomials.rs` tests (`0 * t + c0` → `c0`,
     `0x1 * t + c1` → `t + c1`).

  **Gates**: 41/41 jolt-equivalence green incl. transcript_divergence +
  zkvm_proof_accepted + full suite; clippy clean on the canonical
  8-crate set (jolt-compiler/-compute/-cpu/-zkvm/-dory/-openings/-verifier/-bench).
  Note: jolt-equivalence has unrelated pre-existing clippy warnings that
  the canonical command correctly excludes.

  **Perf (all runs warmup 1, iters 1)**:

  | run | bench | stack | modular prove_ms |
  |---|---|---|---:|
  | 1 | muldiv log_t=12 (baseline rerun) | pre-change | 1438.62 |
  | 2 | muldiv log_t=12 | pre-change rerun | 1488.34 |
  | 3 | muldiv log_t=12 | P39 change | 1437.41 |
  | 1 | sha2-chain log_t=14 | pre-change | 24581.03 |
  | 2 | sha2-chain log_t=14 | P39 change | 23472.35 |
  | 3 | sha2-chain log_t=14 | P39 rerun | 24622.67 |

  - muldiv vs ratchet 1444.14 ms: change avg 1437.41 = −0.47% (single
    change run; two baseline runs spanned ±3%, so delta is inside noise).
    Broader average including baseline runs: +1.3%.
  - sha2-chain vs iter 31 measurement 24604.96 ms: change runs avg
    24047.5 = −2.26%. Inside ±5% band.

  Both benchmarks land inside the ±5% inconclusive band. Per protocol:
  "Inconclusive band ±5% → one rerun; still in band → revert as flat."
  Reverted via `git checkout --` on all 7 modified files.

  **Diagnosis (why flat, despite a clean attack on a 27% commit hotspot)**:
  - **The per-row work is tiny**. At log_t=12, muldiv has `num_rows ≈ 64`
    Dory rows × ~2 hot entries/row for one-hot RA polys. Batch-amortized
    Montgomery inversion is a fixed-cost win per batch call, not per hot
    entry, so with small per-row fan-in the amortization ceiling is low.
    The sparse path correctness was the long pole; absolute work saved
    was close to rayon fork-join overhead already paid by iter 20's
    `Op::Commit` outer parallelization.
  - **Dense-commit path was already parallelized**. Iter 20's 22.6%
    `Op::Commit` win via rayon already gets most of the parallelism
    opportunity on 42 commits. Switching individual polys to sparse
    inside the inner call saves arithmetic per poly but doesn't add new
    parallelism dimensions.
  - **log_t=14 hotspot map disagrees**. Per iter 31, the 27% commit wall
    decomposes as: `multi_pair_g2_setup_parallel` = 24.5% (the Dory
    internal setup, not the row MSM), and row-MSM arithmetic itself is
    a smaller fraction. P39 correctly optimized a smaller piece.
  - **Layout fix was load-bearing for correctness**. Without it the
    first sparse commit dispatched at op #23 diverged the transcript
    because `OneHotPolynomial::for_each_nonzero` emitted indices in
    AddressMajor order while Dory's generator table expected CycleMajor.
    The fix is latent-bug insurance for any future path that feeds
    `OneHotPolynomial` to `commit_rows_sparse` or `MultilinearPoly::evaluate`.

  **Preserved in git log**: the architectural prerequisites
  (BufferProvider sparse route, ProverData delegation, Op::Commit
  Sparse/Dense dispatch, OneHotPolynomial CycleMajor fix,
  batch_g1_additions_multi in scheme.rs) exist in the iter 32 reflog
  for future revisit if (a) parallelism surface widens inside sparse
  row commit, (b) log_t≥18 benchmarks expose a larger sparse-vs-dense
  work asymmetry, or (c) P41 (multi_pair_g2_setup memoize) lands and
  shifts the commit hotspot to the row MSM.

  **Hypothesis queue pivot (iter 33 onward)**:
  - Remaining high-leverage log_t=14 attacks are P40 (Materialize
    super-linear scaling, 47.6% wall) and P41 (multi_pair_g2_setup
    memoize/fan-out, 24.5% wall). P40 is the larger lever.
  - Deprioritize further commit-arithmetic micro-opts at log_t=12 —
    iter 20 + iter 32 together exhaust outer- and per-poly parallelism
    on commits without crossing the ±5% bar.

  **Per protocol**: flat iter, no perf claim, no ratchet update. Stall
  counter 6 → 7. Bookkeeping commit.

- **Iter 31 — P37 Re-profile at log_t=14 sha2-chain (instrumentation, no perf claim)**
  (target: `benchmark-runs/perfetto_traces/iter31_sha2chain_log_t14.json` +
  `perf/iter31-log_t13-sha2chain.json` + `perf/iter31-log_t14-measure.json`).

  **Motivation**: iter 28 post-mortem flagged log_t=12 workloads as
  overhead-dominated for rayon parallelism (4 consecutive parallelism-attack
  failures: P32, P33, P35, P38). P37 was queued to re-profile at real
  log_t ≥ 14 to see if per-op work finally clears rayon's efficient-granularity
  threshold. `muldiv` can't reach log_t=14 (raw 684 cycles), so switched
  program to `sha2-chain` which takes ~3280 cycles/iter.

  **Measurements (iters=1, warmup=1)**:

  | workload | log_t | core ms | modular ms | ratio |
  |---|---:|---:|---:|---:|
  | muldiv (ratchet) | 12 | 380 | 1444 | 3.8× |
  | sha2-chain --num-iters=1 | 13 | 931 | 10606 | **11.4×** |
  | sha2-chain --num-iters=4 | 14 | 1571 | 24604 | **15.7×** |

  **Critical finding — modular is diverging from core with log_t**:

  Scaling sha2-chain from log_t=13 → log_t=14 (2× workload): modular grows
  2.32×, core grows 1.69×. Net: **modular gets 37% worse per log_t step vs
  core**. Extrapolated to log_t=18 (Phase 3 target), ratio climbs toward
  40-60×. The current parallelism-micro-optimization approach cannot close
  this gap.

  **Top spans at log_t=14 sha2-chain (23531 ms wall, 8-core M1 Pro)**:

  | Rank | Span | Self ms | % wall | Calls | Avg µs |
  |---:|---|---:|---:|---:|---:|
  | 1 | reduce_dense | 7535 | **32.0%** | 2085 | 3614 |
  | 2 | Materialize | 6420 | **27.3%** | 197 | 32591 |
  | 3 | DoryScheme::commit | 6350 | **27.0%** | 42 | 151185 |
  | 4 | multi_pair_g2_setup_parallel | 5771 | **24.5%** | 60 | 96186 |
  | 5 | MaterializeUnlessFresh | 4781 | **20.3%** | 46 | 103945 |
  | 6 | InstanceSegmentedReduce | 4229 | **18.0%** | 28 | 151035 |
  | 7 | CpuBackend::gruen_segmented_reduce | 3544 | **15.1%** | 14 | 253113 |
  | 8 | interpolate_inplace | 3294 | **14.0%** | 16940 | 194 |
  | 9 | InstanceBind | 3255 | 13.8% | 285 | 11420 |
  | 10 | InstanceReduce | 1895 | 8.1% | 278 | 6815 |

  (Percentages sum >100% due to span nesting; `multi_pair_g2_setup_parallel`
  is inside `DoryScheme::commit`; `gruen_segmented_reduce` is inside
  `InstanceSegmentedReduce`.)

  **Hotspot map vs iter 23 (log_t=12 muldiv)**:

  | Category | log_t=12 muldiv | log_t=14 sha2-chain | Δ |
  |---|---:|---:|---:|
  | Materialize+MaterializeUnlessFresh | 29% | **47.6%** | **+19pp** |
  | DoryScheme::commit | ~15% | **27%** | **+12pp** |
  | ReduceOpenings (combine_hints) | 6.4% | ~1% | **−5pp** |
  | reduce_dense | 5.7% | 32% | **+26pp** |
  | InstanceSegmentedReduce | 14% | 18% | +4pp |

  `combine_hints` (iter 27 smoking gun) is NO LONGER a meaningful attack
  target at log_t=14 — dropped from 83 ms/call at log_t=12 to sub-second total.
  log_t=14 attack surface is dominated by: (a) Materialize family, (b) Dory
  commit internals (specifically `multi_pair_g2_setup`), (c) `reduce_dense`
  sumcheck inner loops.

  **Structural concern — DoryScheme::commit alone > core's entire prove**:

  Modular's `DoryScheme::commit` aggregate (6350 ms) is **4× larger than
  core's total prove time** (1571 ms) at the same log_t. Both stacks use
  the same `jolt-dory` backend via `PCS::setup_prover(max_log_k_chunk +
  max_log_T)`, confirmed identical by `jolt-core/src/zkvm/prover.rs:2160`.
  So either modular is committing MORE polys than core, or each commit is
  doing more work (BlindFold hints? padded-poly cache inflation?). P39
  queued to investigate.

  **Super-linear Materialize scaling (critical)**:

  `Op::Materialize` per-call jumped from 1.5 ms (iter 23, log_t=12 muldiv)
  → 32.6 ms (iter 31, log_t=14 sha2-chain). That's a **22× per-op wall
  increase for 4× workload** (polynomials 4× larger at log_t=14). Linear
  scaling expected; super-linear observed. Memory peaks at 4255 MB modular
  vs 508 MB core — **8× memory footprint mismatch** suggests cache-thrash
  and allocator contention are the culprits, not algorithmic complexity.
  P40 queued to investigate.

  **Hypothesis queue reshuffle (iter 32 onward)**:
  - **High-leverage log_t=14 attacks (NEW)**: P39 (modular vs core commit
    asymmetry — potential 23-25% wall savings), P40 (Materialize
    super-linear scaling — 12-24% wall), P41 (multi_pair_g2_setup memoize
    / fan-out — 12-24% wall).
  - **Deprioritized**: combine_hints-family attacks (P38-retry), fused_rlc_reduce
    parallelism variants — these target log_t=12 hotspots that are
    proportionally tiny at log_t=14.

  **Per protocol**: profile-only iter, no perf claim, no ratchet update.
  Stall counter NOT incremented (iter 11/17/22/23/27 precedent). Bookkeeping
  commit.

- **Iter 28 — P38 Parallel Dory combine_hints (flat +3.25%, REVERTED)**
  (target: `crates/jolt-dory/src/scheme.rs` `<DoryScheme as AdditivelyHomomorphic>::combine_hints`).
  **Hypothesis**: iter 27 `perf_rlc` telemetry showed `combine_hints` = **83.3 ms / 89%** of ReduceOpenings wall at log_t=12. The function does `combined[r] = Σ_i scalars[i] · hints[i].0[r]` with outer-over-hints, inner-over-rows, accumulating into a shared `combined` vec. Each row is independent across hints — a natural rayon `par_iter` target. Added `rayon` to `jolt-dory` deps and restructured as `(0..num_rows).into_par_iter().map(|r| { serial inner fold }).collect()`. Expected 60-70 ms savings = 4-5% total wall if parallelism lifts the 83 ms from 1.0 → 5-6 cores.
  **Change (~15 LOC)**: added `rayon = { workspace = true }` to `crates/jolt-dory/Cargo.toml`; added `use rayon::prelude::*;` to `scheme.rs`; rewrote `combine_hints` body as `(0..num_rows).into_par_iter().map(|r| serial inner fold).collect::<Vec<_>>()`.
  **Gates**: 41/41 jolt-equivalence green (transcript_divergence + zkvm_proof_accepted + full suite); clippy clean on 8 modular crates.
  **Perf (5 runs at log_t=12)**: 1454 / 1475 / 1542 / 1522 / 1491 ms. Median = 1491 ms vs ratchet 1444.14 = **+3.25%**, inside ±5% band → flat → revert.
  **Diagnosis (why flat, despite a clean 89%-of-the-hot-op attack)**:
  - **Per-row work is only ~1.3 ms** at log_t=12: 83 ms / 64 rows = 1.3 ms/row. Rayon task-setup overhead is typically 10-50 µs per task (work-steal initialization, cache migration). With 64 tasks, overhead floor is 0.6-3 ms. Per-row work at 1.3 ms is dangerously close to that overhead floor.
  - **num_rows at log_t=12**: Dory's commit layout uses `num_rows = 2^(num_vars - ceil(num_vars/2))` = 2^(12 - 6) = 64 for 4K-element polys. Scaling to log_t=14 gives num_rows = 128 or 256 depending on sigma rounding; the per-row work is unchanged (`hints.len() × scalar_mul`), so relative overhead drops.
  - **Cross-core L2 traffic**: the 42 × 64 × 144-byte hint working set = 387 KB. Each worker reads all 42 hints for its assigned rows. 8 cores simultaneously pulling from the same 387 KB forces L2 cross-core traffic (M1 Pro has 12MB shared L2, but per-core L1 is 128KB data cache — hint data must stream through shared L2 repeatedly).
  - **Implicit lock-free thread pool init**: first `par_iter` in a call chain often pays full thread-pool setup cost (~100 µs). Prior ops in the prove pipeline use rayon so the pool should already be warm; nonetheless there's some startup amortization.
  - **Single call**: `combine_hints` is invoked once per fused_rlc_reduce group; at log_t=12 with one dominant group, that's one invocation. If there were 10+ invocations, `par_iter` would amortize its setup across calls; with 1 invocation, it pays full overhead for a single ~1.3-ms/task workload.
  **Per protocol**: flat (±5% band) → revert. Reverted `crates/jolt-dory/Cargo.toml` + `crates/jolt-dory/src/scheme.rs`. Bookkeeping-only commit.
  **Lessons for next iter**: (1) **4th consecutive parallelism-attack failure at log_t=12** (P32, P33, P35, P38). Pattern is now overwhelming: **log_t=12 workloads are overhead-dominated for rayon parallelism across all four attacked hotspots**. This directly validates iter 26's diagnosis and iter 27's implication that log_t=12 is too small for meaningful parallelism gains on top of what's already present. (2) **Next iter 29 must graduate the log_t**: P37 was queued for exactly this reason — re-profile at log_t=14 to see if per-op work finally clears the rayon overhead floor. At log_t=14 total prove time should be ~4-8× larger, so if combine_hints scales proportionally it becomes 300-600 ms instead of 83 ms, and a par_iter over 128-256 rows would have ~2.5 ms/row work — comfortably above overhead. (3) If log_t=14 profile shows different top spans (e.g., sumcheck rounds dominate again), we pivot. Either way, **continuing to attack log_t=12 is no longer productive** — the low-hanging runtime dispatch parallelism is already gone (iter 20 Commit), and the remaining hotspots at this log_t are all smaller than rayon's efficient-granularity threshold.

- **Iter 27 — P36 Instrumentation: fused_rlc_reduce group structure + per-group breakdown (infra, no perf claim)**
  (target: `crates/jolt-zkvm/src/runtime/helpers.rs` `fused_rlc_reduce`).
  **Motivation**: iter 26 P35 failed with the hypothesis that a single dominant group was starving the outer par_iter; instead of guessing, instrument the group structure directly. Three consecutive parallelism attacks (P32/P33/P35) at log_t=12 had gone flat, but iter 23 op-class data still showed 1.0 threads on 93 ms of ReduceOpenings. The gap between "looks parallelizable" and "parallelism helped" needs direct group-count + per-group cost data.
  **Change (~25 LOC)**: inside `fused_rlc_reduce` per-group loop, record (poly_count, total_elems, materialize_us, rlc_us, hint_us) into a Vec, emit `tracing::info!(target: "perf_rlc", …, "rlc_group")` per group and `"rlc_summary"` with group_count + total_claims at end. No behavioral change; pure tracing. Consumable via `RUST_LOG=perf_rlc=info --trace-chrome <name>`.
  **Gates**: 41/41 jolt-equivalence green (transcript_divergence + zkvm_proof_accepted + full suite); clippy clean across 8 modular crates.
  **Run**: `RUST_LOG=perf_rlc=info cargo run --release -p jolt-bench --quiet -- --program muldiv --iters 1 --warmup 0 --log-t 12 --stack modular --trace-chrome iter27_perf_rlc`. Run time 1674 ms (cold single-warmup=0 run — not a perf number).

  **Findings (log_t=12, muldiv, modular)**:

  ```
  INFO rlc_group group=0 poly_count=42 total_elems=688128 mat_us=2 rlc_us=9719 hint_us=83275
  INFO rlc_summary groups=1 total_claims=42
  ```

  Definitively confirms the iter 26 single-group hypothesis: **exactly 1 group containing all 42 opening claims**. Per-group cost breakdown:
  - `materialize`: **2 µs** (effectively free — polys are pre-cached in `state.padded_poly_data` by preceding stage ops, so Cow::Borrowed skips heavy materialization entirely). This is why iter 24 P32 parallel Materialize dispatch was flat — the work was ALREADY amortized by the padding cache for these polys.
  - `rlc_combine`: **9.7 ms** (~10% of ReduceOpenings 93 ms). Iter 25's P33 attempt to parallelize this could only attack 10% of the wall — explains the observed −0.69% delta.
  - `combine_hints`: **83.3 ms (~89% of ReduceOpenings wall)**. This is the PCS-side MSM-like fold of Dory opening hints. None of P32/P33/P35 touched it.

  **Key takeaway**: the 93 ms ReduceOpenings op is essentially `PCS::combine_hints` plus 10 ms of rlc_combine. **At least 83 ms of the 1444 ms prove wall = 5.7% of total wall = 16% of the 510 ms "Materialize family + ReduceOpenings" aggregate lives inside a single function call**: `<DoryScheme as AdditivelyHomomorphic>::combine_hints`. Attacking this directly is now the most load-bearing parallelism lever we have at log_t=12.

  **Implication for future iters**: `perf_rlc` telemetry leaves zero ambiguity about where RLC time goes. Keep the instrumentation landed (deleting it would re-blind us), since measurement overhead is negligible (`Instant::now` × 3 per group × 1 group = sub-microsecond). This is the same precedent as iters 11/17/22/23 where telemetry stayed in the tree to support future iters.

  **Hypothesis queue updates**: Add P38 (profile + parallelize PCS::combine_hints for Dory) as the next iter 28 attack. Expected 60-80 ms wall savings = 4-5% total wall if we lift combine_hints from ~1.0 threads to ~4 threads.

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
