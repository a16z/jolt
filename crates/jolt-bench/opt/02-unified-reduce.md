# Ticket 2 — Unified reduce surface: one Op, one backend method, one dispatch

**Status:** proposed — supersedes Ticket 1's fragmented reduce Ops. Port
strictly; no backwards compatibility required once parity is proven.
**Est. perf win:** primary target is the ~22 s of `reduce_dense` time
spent inside `segmented_reduce`'s nested-rayon fan-out (diagnosed in
Ticket 1 post-mortem). Projected ≥20% prove_ms on sha2-chain log_T=16
by flattening the outer×inner nest into a single Rayon region.
**Est. effort:** medium-large (~800–1000 LOC + handler simplification +
compiler emission rewrite). Migration is mechanical once the shape is
right.
**Philosophy check:** the handler for the new unified op is ≤ 30 LOC.
Handlers for the four current reduce Ops are deleted. Runtime becomes
protocol-unaware over the reduce family.

## Why this exists

Ticket 1's `BatchRoundEvaluate` fusion underdelivered (flat vs fresh
baseline, ±5% noise band) because it only addressed **13% of
`reduce_dense` calls** — the per-instance batch-round reduces. The
other **87%** come from `CpuBackend::segmented_reduce`'s internal
`active.par_iter().map(|&(a,w)| self.reduce(...))` fan-out (backend.rs:600).
That call pattern can't be attacked from the Op level today because
`InstanceSegmentedReduce` is a distinct Op that routes through a
distinct backend method — the fusion pass never sees the work inside it.

The root cause is architectural, not algorithmic: **two orthogonal axes
(Op shape × kernel iteration shape) are tangled into four Op variants
and four backend methods.** Every perf hypothesis about the reduce path
has to navigate which combination is hot and what's reachable from which
dispatch point. Unifying the surface makes the whole reduce workload
one optimization problem instead of four.

### Current fragmentation (what we're collapsing)

| Op (compiler emits) | Handler calls | Inner kernel | Notes |
|---|---|---|---|
| `Op::SumcheckRound` | `backend.reduce` | one of `reduce_dense` / `reduce_tensor` / `reduce_sparse` / `reduce_domain` via `kernel.iteration` | 19 calls / proof |
| `Op::InstanceReduce` | `backend.reduce` | same as above | **308 calls** / proof — addressable today |
| `Op::InstanceSegmentedReduce` | `backend.segmented_reduce` OR `backend.gruen_segmented_reduce` (handler branches on `kernel.gruen_hint`) | `reduce_dense` (via `self.reduce` fan-out, ~130× per call) OR fused Gruen cubic | 32 calls that spawn ~2046 internal `reduce_dense` calls |
| `Op::BatchRoundEvaluate` (from Ticket 1) | `backend.batch_round_evaluate` | CPU override calls `reduce` per singleton Dense group, `reduce_dense_group` for multi-Dense, falls back per-instance for Segmented/Gruen | Only addresses Dense path |

**Four Ops, four backend methods, one real job.** Each perf win requires
coordinating across all of them (e.g., flattening segmented requires a
new Op *or* an override inside `segmented_reduce`, neither of which is
the clean shape).

## Target design

### One Op

```rust
// crates/jolt-compiler/src/module.rs
Op::Reduce { specs: Vec<ReduceSpec> }
```

Every reduction the runtime dispatches is one `Op::Reduce` carrying the
whole batch. Non-batched sumchecks emit a length-1 `specs`. Batch-round
windows emit a length-N `specs`. No other reduce Ops exist.

### One spec shape

```rust
pub struct ReduceSpec {
    /// Compiled kernel id (indexes into the module's kernel table).
    pub kernel: KernelId,
    /// Input buffer references — order matches the kernel's expected
    /// input layout (value columns first, then axis-specific extras).
    pub inputs: Vec<BufferRef>,
    /// Index-space shape to sweep over.
    pub axes: ReduceAxes,
    /// Where the resulting `Vec<F>` evals write back into runtime state.
    pub destination: ReduceDestination,
}

pub enum ReduceAxes {
    /// Single-axis flat sweep over `0..n` cycles.
    /// Corresponds to today's `Op::SumcheckRound` + `Op::InstanceReduce`
    /// with `Iteration::Dense | DenseTensor | Sparse`.
    Flat,

    /// Outer × inner product sweep.
    /// Corresponds to today's `Op::InstanceSegmentedReduce`. The backend
    /// reads `kernel.iteration` (`DenseTensor` / `Gruen`) to pick the
    /// fused inner kernel — the axes only describe the sweep bounds.
    Product {
        outer_eq: BufferRef,
        inner_only: BitVec,
        inner_size: usize,
        /// Previous-round claim + current round index for Gruen cubic
        /// assembly. `None` for non-Gruen kernels. Backend ignores these
        /// unless `kernel.iteration == Iteration::Gruen`.
        gruen_context: Option<GruenContext>,
    },

    /// Domain-indexed Lagrange sweep (univariate skip).
    Domain {
        domain_size: usize,
        stride: usize,
        domain_start: i64,
        domain_indexed: BitVec,
    },

    /// Sparse merge-join over sorted u64 keys.
    Sparse {
        keys: BufferRef,
    },
}

pub enum ReduceDestination {
    /// Write to `state.last_round_instance_evals[instance.0]`.
    Instance { batch: BatchIdx, instance: InstanceIdx },
    /// Write to the current sumcheck's round-polynomial slot.
    SumcheckRound { sumcheck: SumcheckIdx, round: usize },
}

pub struct GruenContext {
    pub prev_claim_slot: ChallengeIdx,
    pub current_round: usize,
}
```

The critical property: **`axes` describes the index space, `kernel.iteration`
describes the inner kernel.** These are independent — today they're
conflated because the outer segmented/gruen/batch distinction is forced
into the Op variant.

### One backend method

```rust
// crates/jolt-compute/src/traits.rs
pub trait ComputeBackend: Send + Sync + 'static {
    // ... existing buffer / eq / commit / etc.

    /// Evaluate a batch of kernel reductions.
    ///
    /// Replaces `reduce`, `segmented_reduce`, `gruen_segmented_reduce`,
    /// and `batch_round_evaluate`. Length-1 specs correspond to
    /// today's single-instance calls; length-N specs replace batch
    /// round evaluations. Backend is free to group by `(axes variant,
    /// binding order, inner_size)` and dispatch one fused loop per
    /// group.
    fn reduce<F: Field>(
        &self,
        specs: &[ReduceSpec],
        inputs: &ReduceInputs<'_, Self, F>,
        challenges: &[F],
    ) -> Vec<Vec<F>>;
}
```

`ReduceInputs` is a thin borrow-view into the runtime state's buffer
arena so the backend doesn't have to reach into the runtime for each
`BufferRef`. Signature to be pinned down in Phase A; a sketch:

```rust
pub struct ReduceInputs<'a, B, F> where B: ComputeBackend, F: Field {
    buffers: &'a [Buf<B, F>],
    kernels: &'a [B::CompiledKernel<F>],
}
```

### One handler

```rust
// crates/jolt-zkvm/src/runtime/handlers.rs
Op::Reduce { specs } => {
    let results = backend.reduce(specs, state.reduce_inputs(), &state.challenges);
    for (spec, evals) in specs.iter().zip(results) {
        match &spec.destination {
            ReduceDestination::Instance { batch, instance } => {
                state.write_instance_evals(*batch, *instance, evals);
            }
            ReduceDestination::SumcheckRound { sumcheck, round } => {
                state.write_sumcheck_round_evals(*sumcheck, *round, evals);
            }
        }
    }
}
```

**~15 LOC real logic.** Everything else — the grouping, the rayon
dispatch, the kernel selection, the buffer prefetch — lives in the
backend.

### One grouping/dispatch location inside the backend

```rust
// crates/jolt-cpu/src/backend.rs
impl ComputeBackend for CpuBackend {
    fn reduce<F: Field>(
        &self,
        specs: &[ReduceSpec],
        inputs: &ReduceInputs<'_, Self, F>,
        challenges: &[F],
    ) -> Vec<Vec<F>> {
        // 1. Partition specs by (axes variant, binding order, inner_size, kernel signature).
        // 2. Within each partition, pick a dispatch strategy:
        //    - Flat × single spec     -> reduce_dense_fixed fast path
        //    - Flat × multi spec      -> reduce_dense_group (batched inner loop)
        //    - Product × Gruen kernel -> fused cubic assembly over (outer, inner)
        //    - Product × Dense kernel -> fused (outer, inner) flat loop
        //      (replaces the nested segmented_reduce fan-out)
        //    - Domain / Sparse        -> existing leaf kernels
        // 3. One rayon region per partition — no nested par_iter.
        //    The outer×inner Product case uses flat_map over the product
        //    range, not par_iter-over-outer with inner par_iter.
        // 4. Write results into a pre-allocated `Vec<Vec<F>>` indexed
        //    by spec position.
    }
}
```

This is the **single optimization surface** for all reduce work. Future
perf wins are isolated experiments inside `reduce` — no new Ops, no
trait churn.

## Migration phases

No backwards compatibility: once parity is proven, the old Ops and
methods are deleted. Dual-path validation bridges the switchover.

### Phase A — land the new types + new backend method alongside the old ones

**Files:**
- `crates/jolt-compiler/src/module.rs` — add `Op::Reduce { specs }`,
  `ReduceSpec`, `ReduceAxes`, `ReduceDestination`, `GruenContext`.
  Leave the four existing reduce Ops in place.
- `crates/jolt-compute/src/traits.rs` — add new `reduce` method
  (rename the old `reduce` to `reduce_single` temporarily; keep
  `segmented_reduce` / `gruen_segmented_reduce` / `batch_round_evaluate`
  as-is). Add `ReduceInputs<'_, B, F>` borrow view.
- `crates/jolt-compute/src/traits.rs` — add
  `per_instance_reference_reduce` reference implementation that
  byte-identically matches the old paths by routing each spec through
  the corresponding old method (`reduce_single` / `segmented_reduce` /
  `gruen_segmented_reduce` based on `axes + kernel.iteration`). This
  is the dual-path oracle, analogous to `per_instance_batch_evaluate`.
- `crates/jolt-cpu/src/backend.rs` — implement the new `reduce` as a
  thin wrapper over `per_instance_reference_reduce`. Not optimized yet
  — only proving the shape compiles and round-trips.

**Exit:** `cargo clippy -p jolt-compute -p jolt-cpu` green. No runtime
changes yet — new surface exists but is unused.

### Phase B — compiler emits `Op::Reduce` dual-form

Update every emission site in the compiler to emit `Op::Reduce {
specs: vec![ONE_SPEC] }` **in addition to** the old Op. The runtime
ignores the new Op during Phase B (dead code), but equivalence tests
serialize + deserialize the full stream, catching shape bugs early.

**Files:**
- `crates/jolt-compiler/src/compiler/emit.rs` — every call site that
  emits `Op::SumcheckRound`, `Op::InstanceReduce`, or
  `Op::InstanceSegmentedReduce` also emits the equivalent
  `Op::Reduce`. (Guarded by `cfg(debug_assertions)` or a feature flag
  during development to keep the output stream small.)
- `crates/jolt-cpu/src/fuse.rs` — temporarily extend the fuse pass to
  recognize and collapse adjacent `Op::Reduce` emissions within a
  batch-round window. (This is the new home of Ticket 1's fusion
  logic, in its final shape.)

**Exit:** compiler emits the new Ops; tests still pass because the
runtime skips them. Grep confirms every old-Op emission has a paired
new-Op emission.

### Phase C — runtime flips to the new path behind a dual-path harness

Add a new env toggle `JOLT_UNIFIED_REDUCE=1` (default off).

**When off:** runtime executes old Ops, skips `Op::Reduce`. (Phase B
behavior.)

**When on:** runtime executes `Op::Reduce` only. Skips the old
reduce Ops. Shadow harness (like `JOLT_FUSE_DEBUG`) runs both paths
and asserts byte-identical output per op before writing results.

**Files:**
- `crates/jolt-zkvm/src/runtime/handlers.rs` — add the `Op::Reduce`
  handler arm (15 LOC per sketch above). Wire it to
  `state.reduce_inputs()`.
- `crates/jolt-zkvm/src/runtime/state.rs` — add `reduce_inputs()`
  accessor returning `ReduceInputs<'_, B, F>`. Also add
  `write_instance_evals` / `write_sumcheck_round_evals` helpers that
  today's handlers inline — centralizing the writeback.
- `crates/jolt-compute/src/linker.rs` — add `ReduceDebugMode` +
  `JOLT_UNIFIED_REDUCE=1` env toggle, parallel to `FuseDebugMode`.
- `crates/jolt-equivalence/tests/unified_reduce.rs` — new test module
  proving `Op::Reduce` path equals the old-Ops path for every sumcheck
  stage. Run with `JOLT_UNIFIED_REDUCE=1 JOLT_FUSE_DEBUG=1` both on.

**Exit:** `JOLT_UNIFIED_REDUCE=1 cargo nextest run -p jolt-equivalence`
green across `transcript_divergence`, `zkvm_proof_accepted`,
`modular_self_verify`, and the new `unified_reduce` suite.

### Phase D — delete the old surface

After Phase C is green on CI and the perf gate in Phase E is met:

**Files:**
- `crates/jolt-compiler/src/module.rs` — delete `Op::SumcheckRound`,
  `Op::InstanceReduce`, `Op::InstanceSegmentedReduce`,
  `Op::BatchRoundEvaluate`, `InstanceEvalKind`.
- `crates/jolt-compiler/src/compiler/emit.rs` — delete old emissions;
  keep only `Op::Reduce`.
- `crates/jolt-compute/src/traits.rs` — delete `reduce_single`,
  `segmented_reduce`, `gruen_segmented_reduce`,
  `batch_round_evaluate`, `per_instance_batch_evaluate`,
  `BatchReduceKind`, `BatchInstanceSpec`.
- `crates/jolt-cpu/src/backend.rs` — delete the old trait impl
  methods. Inline `segmented_reduce`'s outer loop and
  `gruen_segmented_reduce`'s cubic assembly into the unified
  `reduce`'s `ReduceAxes::Product` branch.
- `crates/jolt-cpu/src/fuse.rs` — delete; the grouping logic moved
  into `backend.reduce` itself.
- `crates/jolt-zkvm/src/runtime/handlers.rs` — delete the four old
  handler arms. `Op::Reduce` handler is the only reduce-family arm.
- `crates/jolt-compute/src/linker.rs` — delete `FuseDebugMode` +
  `ReduceDebugMode` dual-path plumbing (no longer needed once the old
  paths are gone).

**Exit:** repo has a single reduce Op, a single backend method, a
single handler arm. `wc -l crates/jolt-zkvm/src/runtime/handlers.rs`
drops by ~200 LOC. Perf gate must still pass after deletion.

### Phase E — optimize inside `backend.reduce`

Now that all reduce traffic funnels through one method, the real perf
work starts. **Primary attack: flatten the `Product` axes into a
single-rayon-region loop** over active (outer × inner) positions,
eliminating the nested-rayon fan-out that `segmented_reduce` has today.

Concretely, the current shape:
```rust
// Current — 2 rayon regions, ~130 reduce calls per segmented call
active.par_iter().map(|&(a, w)| {
    let col_refs = /* slice inputs at outer a */;
    self.reduce(kernel, &col_refs, challenges)  // ← nested region
})
```

Target shape (for Dense kernel on `Product` axes):
```rust
// Target — 1 rayon region, 1 leaf kernel invocation
active.par_iter()
    .with_min_len(PAR_THRESHOLD / inner_size)
    .map(|&(a, w)| {
        let mut local_evals = [F::Accumulator::zero(); NE];
        for i in 0..half {
            // Hand-rolled inner reduce — no self.reduce dispatch.
            let (lo_refs, hi_refs) = slice_at(a, i);
            kernel.eval_fn(&mut local_evals, lo_refs, hi_refs, challenges);
        }
        local_evals.map(|acc| weight * F::from_accumulator(acc))
    })
    .reduce(|| [F::zero(); NE], combine_evals)
```

Same shape as `gruen_segmented_reduce` already uses. **The point of
Phases A–D is to make this rewrite a localized backend concern rather
than a cross-crate refactor.**

Further optimizations enabled by the unified surface:
- Grouping multi-spec `Product` partitions that share `outer_eq` —
  one outer loop, many instance accumulators — matching the jolt-core
  booleanity shape.
- Kernel JIT / monomorphization: once the backend owns the dispatch,
  it can specialize on `(axes variant, num_evals, kernel shape)` at
  compile time rather than trait-dispatch per call.
- Shared eq table prefetch across specs in the same batch-round.

Each of these is a **single-file experiment inside `backend.rs`**, not
a cross-crate refactor.

## Validation strategy

### Dual-path harness (bridge through Phase C)

Same pattern Ticket 0 established: `JOLT_UNIFIED_REDUCE=1
JOLT_FUSE_DEBUG=1` runs both the new `Op::Reduce` handler AND the old
reduce Ops over the same state, and asserts byte-identical
`last_round_instance_evals` / `sumcheck_round_evals` after each op.
Any reordering or axis-mapping bug surfaces at the first divergence.

Infra to add:
- `crates/jolt-compute/src/linker.rs` — `ReduceDebugMode::Shadow`
  variant. When on, the linker keeps both the old-Op stream AND the
  new-Op stream cached on the `Executable`; the runtime executes both
  in lockstep.
- `crates/jolt-zkvm/src/runtime/handlers.rs` — add shadow assertion
  at the end of the `Op::Reduce` handler (mirrors the
  `BatchRoundEvaluate` shadow).
- `crates/jolt-equivalence/tests/unified_reduce.rs` — new test file:
  - `unified_reduce_flat_matches_legacy` — flat sumcheck rounds.
  - `unified_reduce_product_matches_legacy` — segmented paths.
  - `unified_reduce_product_gruen_matches_legacy` — Gruen path.
  - `unified_reduce_batch_matches_legacy` — batch-round windows.
  - `unified_reduce_domain_matches_legacy` — uniskip domain path.
  - `unified_reduce_sparse_matches_legacy` — memory-checking sparse
    path.

Each test runs a small sha2-chain program with both toggles on and
asserts the shadow harness fires zero divergence assertions.

### Correctness gate (required before each phase commit)

```bash
JOLT_UNIFIED_REDUCE=1 JOLT_FUSE_DEBUG=1 cargo nextest run -p jolt-equivalence \
  transcript_divergence zkvm_proof_accepted modular_self_verify --cargo-quiet
JOLT_UNIFIED_REDUCE=1 cargo nextest run -p jolt-equivalence --cargo-quiet
JOLT_UNIFIED_REDUCE=0 cargo nextest run -p jolt-equivalence --cargo-quiet   # regression guard on old path

cargo clippy -p jolt-core --features host         --all-targets -- -D warnings
cargo clippy -p jolt-core --features host,zk      --all-targets -- -D warnings
cargo clippy -p jolt-cpu                          --all-targets -- -D warnings
cargo clippy -p jolt-compute                      --all-targets -- -D warnings
cargo clippy -p jolt-compiler                     --all-targets -- -D warnings
cargo clippy -p jolt-zkvm                         --all-targets -- -D warnings
```

### After Phase D deletion

The `JOLT_UNIFIED_REDUCE` toggle goes away (only one path remains).
The gate collapses to:

```bash
cargo nextest run -p jolt-equivalence --cargo-quiet
cargo nextest run -p jolt-core muldiv --features host --cargo-quiet
cargo nextest run -p jolt-core muldiv --features host,zk --cargo-quiet
```

## Perf gate

Phases A–D are **correctness-only**: no perf delta expected
(Phase A–C are refactors, Phase D is deletion). The perf hypothesis
lives in **Phase E**.

Phase E sub-iter: flatten `Product` × Dense path per the sketch above.

```bash
cargo run --release -p jolt-bench -- --program sha2-chain \
  --num-iters 16 --log-t 16 --iters 1 --warmup 1 \
  --json perf/last-iter.json
```

**Target:** ≥20% reduction in modular `prove_ms` on sha2-chain
log_T=16 (address the 22 s of `reduce_dense` currently buried inside
`segmented_reduce`'s fan-out). Current clean baseline ~77 s modular.
Target ≤62 s.

**Accept:** ≥5% reduction → update `perf/baseline-modular-best.json`.
**Target:** ≥20% reduction.
**Reject:** <5% → revert only the Phase E kernel rewrite; Phases A–D
stay landed because they're prerequisite infra regardless of the
specific flatten attempt.

## File-by-file work list

| Phase | File | Change |
|---|---|---|
| A | `crates/jolt-compiler/src/module.rs` | `+` new types: `Op::Reduce`, `ReduceSpec`, `ReduceAxes`, `ReduceDestination`, `GruenContext` |
| A | `crates/jolt-compute/src/traits.rs` | `+` `fn reduce(&[ReduceSpec], …)`, `+` `per_instance_reference_reduce`, `+` `ReduceInputs<'_, B, F>`; rename old `fn reduce` → `fn reduce_single` |
| A | `crates/jolt-cpu/src/backend.rs` | `+` impl new `reduce` forwarding to reference impl |
| B | `crates/jolt-compiler/src/compiler/emit.rs` | `+` emit `Op::Reduce` alongside old Ops at every reduce emission site |
| B | `crates/jolt-cpu/src/fuse.rs` | `~` fuse collapses adjacent `Op::Reduce` emissions into one (grouping logic moves here) |
| C | `crates/jolt-zkvm/src/runtime/handlers.rs` | `+` `Op::Reduce` handler arm (15 LOC); `~` old arms emit deprecation warning when executed with `JOLT_UNIFIED_REDUCE=1` |
| C | `crates/jolt-zkvm/src/runtime/state.rs` | `+` `reduce_inputs()`, `write_instance_evals()`, `write_sumcheck_round_evals()` accessors |
| C | `crates/jolt-compute/src/linker.rs` | `+` `ReduceDebugMode::{Off, Shadow}` + env toggle; wire shadow stream caching |
| C | `crates/jolt-equivalence/tests/unified_reduce.rs` | `+` 6 tests per above |
| D | all above | `~` delete old Ops, methods, handler arms, shadow infra |
| D | `crates/jolt-cpu/src/backend.rs` | `~` inline old `segmented_reduce` + `gruen_segmented_reduce` logic into unified `reduce` |
| E | `crates/jolt-cpu/src/backend.rs` | `~` flatten `Product` × Dense into single-rayon-region loop (perf attack) |

## Ordering safety — what the runtime must preserve

Any reordering of reduces must preserve the same ordering invariants
Ticket 1 documented for `fuse_batch_round_reduces`:

- **Intra-window:** `BatchAccumulateInstance { instance }` reads
  `state.last_round_instance_evals[instance.0]`; the corresponding
  `ReduceSpec { destination: Instance { instance, .. } }` writes that
  slot. The batch-round window must run all destination writes before
  any accumulate-block reads — the new handler preserves this because
  the full `Vec<Vec<F>>` result materializes before the per-spec
  writeback loop.
- **Cross-window:** the compiler still emits `BatchRoundBegin
  .. BatchRoundFinalize` windows; `Op::Reduce` slots into the same
  place the old reduce ops occupied.
- **Challenge state:** unchanged — `Op::Reduce` reads `state.challenges`,
  doesn't mutate it.

## Open design decisions

- **Q1: Should `kernel.iteration` also collapse into `ReduceAxes`?**
  Today `Iteration::Dense | DenseTensor | Sparse | Domain | Gruen`
  lives on the compiled kernel. The new `ReduceAxes` describes the
  *sweep*; `kernel.iteration` describes the *inner kernel*. Arguments
  for keeping them separate:
  - `Iteration::Gruen` controls a specialized inner kernel (cubic
    assembly with prev_claim), not a different sweep shape.
  - Multiple axes can reuse the same inner kernel (e.g., `Flat` +
    `Iteration::DenseTensor` makes sense in principle — the eq is
    baked into the kernel, the sweep is flat).
  - Merging them doubles the enum cardinality for no dispatch
    savings.

  Recommendation: keep separate. Revisit in Phase E if the split
  causes dispatch friction.

- **Q2: Per-call kernel table lookup cost.** Today `spec.kernel:
  usize` indexes into the module's kernel vec; the runtime resolves
  once per op. With a length-N `Vec<ReduceSpec>` we resolve N kernel
  handles. For N=130 (segmented fan-out scale) this is ~130 pointer
  chases — negligible vs ~100µs of actual work per spec. No
  pre-resolution needed.

- **Q3: `ReduceInputs` borrow pattern.** The backend needs to resolve
  `BufferRef` → `&Buf<Self, F>`. Two options:
  - Hand the backend an accessor closure: `|r: BufferRef| -> &Buf`.
  - Hand the backend a slice + lookup fn as a struct with &mut
    borrow. Struct is more rustic; closure is one fewer type. Pin
    down in Phase A prototype.

- **Q4: Does this make the fusion pass obsolete?** Yes — in Phase D.
  The compiler emits `Op::Reduce { specs }` directly with the full
  batch-round window already in `specs`. No separate pass required.
  The fuse pass module (`crates/jolt-cpu/src/fuse.rs`) gets deleted.

- **Q5: GPU backend readiness.** The unified surface matches GPU
  compute-pipeline dispatch shape well — one kernel launch per
  partition, with `specs` as the per-instance dispatch parameter
  buffer. No GPU-hostile design choices here. If/when a GPU backend
  lands, it implements `fn reduce(&[ReduceSpec], …)` and inherits all
  the grouping logic from CPU or provides its own.

## Rollback

- **During Phase A–C:** dual-path is always live. Revert by turning
  off `JOLT_UNIFIED_REDUCE` (runtime default). No commit revert
  needed; just restore the default env.
- **After Phase D deletion:** no rollback. The old Ops are gone. If
  Phase E perf rewrite regresses, revert just the `backend.reduce`
  body — the Op surface and handler stay.

## Dependencies

- Ticket 0 (dual-path validation infra, commit `8befd6613`) — reuse
  the `FuseDebugMode` + shadow-stream pattern for the new
  `ReduceDebugMode`. No new infra needed.
- Ticket 1 (fusion pass + `BatchRoundEvaluate` override) — subsumed
  by Phase D. Keep Ticket 1 landed as-is until Phase D lands; then
  its fusion pass + override get deleted in favor of the unified
  grouping logic inside `backend.reduce`.

## Commit discipline

Per-phase commits, one commit per merge-ready slice:

- Phase A: `refactor(reduce): T2-A introduce unified ReduceSpec + reduce trait method (no runtime change)`
- Phase B: `refactor(reduce): T2-B compiler emits Op::Reduce alongside legacy ops`
- Phase C: `refactor(reduce): T2-C runtime flips to Op::Reduce behind JOLT_UNIFIED_REDUCE toggle`
- Phase D: `refactor(reduce): T2-D delete legacy reduce surface — unified path only`
- Phase E (iters): `perf(reduce): T2-E<n> <attack> (-X% prove_ms on sha2-chain @ log_T=16)`

No journal commits in Phases A–D (they're refactors, no perf claim to
revert). Phase E iterations use the standard Perf Loop journal format.
