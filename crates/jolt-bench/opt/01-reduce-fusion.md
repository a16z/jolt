# Ticket 1 — Reduce-side fusion: `fuse_ops` hook + CPU `batch_round_evaluate`

**Status:** in progress (see **Progress** below). Biggest single
architectural win.
**Est. perf win:** ~47s CPU on sha2-chain log_T=16 (factor B dispatch
overhead + part of factor C cache thrash, per
`perf/report_tools/kernel_gap_memo.md` §1).
**Est. effort:** medium (~400-500 LOC + CPU kernel restructure).
**Philosophy check:** handlers unchanged in LOC, compiler unchanged —
rewrite pass lives entirely in `jolt-cpu`.

## Progress

### Done

- **Ticket 0 fully landed** (commit `8befd6613`): `fuse_ops` trait hook on
  `ComputeBackend` (default `None`), `FuseDebugMode` enum + env toggle
  (`JOLT_FUSE_DEBUG=1`), `Executable::shadow_ops` + `has_fuse_debug()`,
  `per_instance_batch_evaluate` reference path extracted from the default
  `batch_round_evaluate` impl, dual-path shadow harness in the
  `BatchRoundEvaluate` handler, and 8 linker-level tests in
  `crates/jolt-equivalence/tests/fuse_equivalence.rs`.
- **Fusion pass `fuse_batch_round_reduces`** — implemented in
  `crates/jolt-cpu/src/fuse.rs` (lives in the CPU crate because the hook
  only matters when paired with a backend that overrides
  `batch_round_evaluate`; promote to `jolt-compute` if a second backend
  ever wants the same pass). Walks each `BatchRoundBegin ..
  BatchRoundFinalize` window, collapses `InstanceReduce` /
  `InstanceSegmentedReduce` into one `Op::BatchRoundEvaluate`, moves
  `BatchAccumulateInstance` + `BatchInactiveContribution` after the fused
  reduce, leaves everything else (materialize / bind / decomp
  `ReadCheckingReduce`+`RafReduce` / scatter / captures) in place. Windows
  with no reducible ops pass through unchanged.
- **`CpuBackend::fuse_ops` opts in** — `crates/jolt-cpu/src/backend.rs`
  returns `Some(fuse_batch_round_reduces(ops))`.
- **6 unit tests** for the pass (same file): empty stream, single window,
  prep ordering, inactive contribution move, no-reduces window skip,
  multi-window independence. All green.

### Not done (what blocks the perf win)

The fused stream emits `Op::BatchRoundEvaluate`, but the handler dispatches
to the default `ComputeBackend::batch_round_evaluate` implementation, which
falls back to `per_instance_batch_evaluate` — a byte-identical per-instance
loop over `reduce` / `segmented_reduce` / `gruen_segmented_reduce`. Zero
perf delta from this alone.

1. **Override `CpuBackend::batch_round_evaluate`** with a cache-friendly
   implementation that actually fuses the work across instances (§4 below).
   This is the ~47s CPU win.
2. **Correctness gate** with `JOLT_FUSE_DEBUG=1` (shadow asserts every
   `BatchRoundEvaluate` against `per_instance_batch_evaluate`) AND
   `JOLT_FUSE_DEBUG=0` (release path — production-representative).
3. **Perf gate** against `perf/baseline-modular-best.json` (current ratchet
   `72638ms` on sha2-chain log_T=16). Accept ≥15%, reject <5%, revert +
   journal on reject per `CLAUDE.md` Perf Loop.

## Remaining work — step-by-step

Read order for someone picking this up cold:

1. `crates/jolt-bench/opt/01-reduce-fusion.md` (this file) — context.
2. `crates/jolt-cpu/src/fuse.rs` — what the pass does today. Pass tests
   show exact shape of the fused stream.
3. `crates/jolt-zkvm/src/runtime/handlers.rs:852` — `BatchRoundEvaluate`
   handler. Already packs specs and calls `backend.batch_round_evaluate(&specs, &challenges)`.
   Do not edit — this stays.
4. `crates/jolt-compute/src/traits.rs:591` — the default
   `batch_round_evaluate` that we need to override for CPU. Read its
   signature and the reference `per_instance_batch_evaluate` immediately
   below it.
5. `crates/jolt-cpu/src/backend.rs:131` — `reduce` (dense dispatch) and
   nearby `segmented_reduce` / `gruen_segmented_reduce`. These are what
   today's default fallback calls N times per batch-round.
6. `crates/jolt-cpu/src/backend.rs:957` — `reduce_dense` / `reduce_dense_fixed` —
   the shape we want to match inside the fused path: single `rayon::fold` /
   `reduce` over `0..half`, with const-generic scratch arrays for Dense
   kernels of known `(NI, NE)`. Study this: the override should reuse this
   pattern, not call back into `reduce_dense`.
7. `crates/jolt-core/src/subprotocols/booleanity.rs:327` — the jolt-core
   inline fold over all RA polynomials is the *target shape*. One Rayon
   region walks all instances per outer chunk.

### Step 1 — implement the CPU override

New function in `crates/jolt-cpu/src/backend.rs`, overriding the default
trait method:

```rust
fn batch_round_evaluate<F: Field>(
    &self,
    specs: &[BatchInstanceSpec<'_, Self, F>],
    challenges: &[F],
) -> Vec<Vec<F>>
```

Design:

1. **Partition `specs` by `BatchReduceKind`** (`Dense` / `Segmented` /
   `Gruen`). Within each partition, every spec uses the same kernel
   *shape* but potentially different kernels and different buffer lengths.
2. **For the `Dense` partition**, group further by `(half_len,
   binding_order, num_evals)`. Instances sharing a group can iterate the
   same `0..half` range; each iteration pulls one pair from each instance's
   input buffer and runs that instance's `evaluate` closure.
3. **Single Rayon region per group**: one `par_iter` over
   `0..half` (with `with_min_len(2048)`, matching `reduce_dense_fixed`),
   fold into `Vec<[F::Accumulator; num_evals]>` — one per instance —
   reduce across chunks, then `FieldAccumulator::reduce` into the output
   `Vec<F>` for each instance. Write results into `results[spec_idx]`.
4. **Segmented / Gruen partitions**: same outer-chunk idea but the
   iteration bound is `outer_size`, and each chunk reads the shared
   `outer_eq` slice once and applies it against every instance's
   inner-size kernel call. Look at
   `crates/jolt-cpu/src/backend.rs` for `segmented_reduce` / the existing
   Gruen path to understand the inner calls we're replacing.
5. **Pre-allocate `results: Vec<Vec<F>>`** sized per spec up front; each
   worker writes to its own slot. Use `UnsafeCell` or slice splitting to
   avoid mutex contention.
6. **Guard on `feature = "parallel"`** — non-parallel builds should call
   `per_instance_batch_evaluate(self, specs, challenges)` directly.

Important constraints (from per-session memories):

- **Outer `par_iter` across instances is the wrong axis.** Instance count
  is ~120 but per-instance work drops off sharply in late rounds; nested
  par_iter inside reduce_dense causes oversubscription. Parallelize across
  *outer chunks*, iterate instances *inside* each chunk.
- **`WideAccumulator` swap caveat** (`feedback_wide_accumulator_cache.md`):
  the per-instance accumulator arrays are small stack allocations — fine
  to keep as `[F::Accumulator; NE]`. Don't materialize a
  `Vec<WideAccumulator>` that scales with instance count × outer_size.
- **Global mutex hazard** (`feedback_global_mutex_guarded_parallel.md` and
  `feedback_handle_mutex_contention.md`): the backend holds no mutexes on
  the hot path, but if you touch `HandleStore`, drop the guard before
  spinning up rayon.
- **Handlers must remain ≤30 LOC** — the override lives in the backend,
  not the handler. Do not touch `crates/jolt-zkvm/src/runtime/handlers.rs`.

### Step 2 — correctness gate

Run with `JOLT_FUSE_DEBUG=1` first — the shadow harness in the
`BatchRoundEvaluate` handler asserts, for every fused dispatch, that the
override's output matches `per_instance_batch_evaluate` byte-for-byte. Any
reordering bug surfaces at the first divergence, not 200 rounds later.

```bash
JOLT_FUSE_DEBUG=1 cargo nextest run -p jolt-equivalence \
  transcript_divergence zkvm_proof_accepted modular_self_verify --cargo-quiet
JOLT_FUSE_DEBUG=1 cargo nextest run -p jolt-equivalence --cargo-quiet
```

Then with debug off (release-representative path):

```bash
cargo nextest run -p jolt-equivalence transcript_divergence \
  zkvm_proof_accepted modular_self_verify --cargo-quiet
cargo nextest run -p jolt-equivalence --cargo-quiet
```

Clippy gate (zero warnings in every mode we ship):

```bash
cargo clippy -p jolt-core --features host --message-format=short -q \
  --all-targets -- -D warnings
cargo clippy -p jolt-core --features host,zk --message-format=short -q \
  --all-targets -- -D warnings
cargo clippy -p jolt-cpu --message-format=short -q --all-targets -- -D warnings
cargo clippy -p jolt-compute --message-format=short -q --all-targets -- -D warnings
```

Any failure → fix or revert. Do not commit a red tree.

### Step 3 — perf gate

Standard Perf Loop measurement:

```bash
cargo run --release -p jolt-bench -- --program sha2-chain \
  --num-iters 16 --log-t 16 --iters 1 --warmup 1 \
  --json perf/last-iter.json
```

Compare modular `prove_ms` against `perf/baseline-modular-best.json`
(current ratchet `72638ms`). Accept thresholds from the Perf Loop:

- **Accept**: ≥5% reduction. Update `perf/baseline-modular-best.json`.
  Append to `perf/history.jsonl`.
- **Target**: ≥15% reduction (~61500ms modular). Ticket projection: ~30%
  (~51000ms).
- **Reject**: <5%. Revert the override + the `fuse_ops` opt-in in
  `CpuBackend` (leave the pass module in place as dormant infra), then
  commit a `journal:` entry.

If rejected, **do not leave the half-done state on the branch** — revert
to the post-ticket-0 baseline and move on.

### Step 4 — commit

Exactly one commit per iteration per `CLAUDE.md`:

- **Improvement**: `perf(fuse): T1 batch-round reduce fusion (-X% prove_ms on sha2-chain @ log_T=16)`
  — include the new baseline number in the body.
- **Reject**: `journal: T1 reverted (<reason>)` — keep the `fuse.rs`
  module + tests + Ticket 0 infra in place; only revert the `fuse_ops`
  opt-in and the override.

## Why this exists

The kernel_gap_memo identifies five architectural causes of the 28.7×
CPU gap in the sumcheck inner loop. This ticket attacks factor B
("per-call dispatch", ~1.5×, ~8.5s) *and* the reduce-half of factor C
("cross-instance cache thrash", ~3.0×, ~20s split between reduce and bind).
Combined projected win: ~47s CPU on the modular stack (from ~66s sumcheck
CPU down to ~19s).

Today the compiler emits one `InstanceReduce` / `InstanceSegmentedReduce`
op per (batch, round, instance) — for sha2-chain log_T=16 that's ~120
instances × 20 rounds × 2 kernels ≈ 4,800 per-instance reduce ops.
Runtime dispatch walks the vec and calls `backend.reduce(...)` /
`backend.segmented_reduce(...)` / `backend.gruen_segmented_reduce(...)`
once per call. Each call:

1. Pays vtable dispatch + handler setup (small but multiplied).
2. Spins up a fresh Rayon parallel region over `half_len` elements
   (~50-100µs fork/join floor per region).
3. Traverses polynomial buffers once, then completes, evicting cache
   before the next instance starts — even though the next instance shares
   the `eq_tensor`/outer-eq and reads adjacent slots.

`jolt-core`'s booleanity sumcheck (`jolt-core/src/subprotocols/booleanity.rs:327`)
uses one fold-reduce over `0..half_len` that walks all RA polynomials
inline — that's the shape we need to match.

The infra is already in place:

- `Op::BatchRoundEvaluate { batch, round, instances: Vec<InstanceEvalKind> }`
  is defined in `crates/jolt-compiler/src/module.rs:1264`.
- `ComputeBackend::batch_round_evaluate` exists as a default method in
  `crates/jolt-compute/src/traits.rs:574` with a byte-identical per-instance
  fallback.
- Handler in `crates/jolt-zkvm/src/runtime/handlers.rs:852` packs specs and
  dispatches — already ≤30 LOC of real logic.

Missing two pieces:

1. **A fusion pass** that rewrites the emitted op stream so `InstanceReduce`
   + `InstanceSegmentedReduce` ops within a batch-round collapse into one
   `BatchRoundEvaluate`.
2. **A CPU override** of `batch_round_evaluate` that does cache-friendly
   inner-chunk-interleaved iteration across instances (single Rayon region).

## Architectural change

### 1. Add `fuse_ops` to `ComputeBackend` (✅ landed via Ticket 0, commit `8befd6613`)

```rust
// crates/jolt-compute/src/traits.rs
pub trait ComputeBackend {
    // ... existing
    fn fuse_ops(&self, _ops: &[Op]) -> Option<Vec<Op>> {
        None
    }
}
```

### 2. Pure pass: collapse reduces within batch-round windows (✅ implemented in `crates/jolt-cpu/src/fuse.rs`)

Sketch below was the original design. Live code + tests live in
`crates/jolt-cpu/src/fuse.rs`; behavior matches this sketch. Moved into
`jolt-cpu` (not `jolt-compute`) because the pass is only meaningful when
paired with a backend override — promote to a shared crate only if a
second backend needs it.

```rust
pub fn fuse_batch_round_reduces(ops: &[Op]) -> Vec<Op> {
    let mut out = Vec::with_capacity(ops.len());
    let mut i = 0;
    while i < ops.len() {
        if let Op::BatchRoundBegin { batch, round, .. } = &ops[i] {
            // Find matching BatchRoundFinalize
            let end = find_finalize(&ops[i..]) + i;
            // Walk the window, hoisting reduces
            let fused = fuse_window(&ops[i..=end], *batch, *round);
            out.extend(fused);
            i = end + 1;
        } else {
            out.push(ops[i].clone());
            i += 1;
        }
    }
    out
}

fn fuse_window(window: &[Op], batch: BatchIdx, round: usize) -> Vec<Op> {
    // Pass 1: collect reduce ops (in order) into InstanceEvalKind list.
    let mut kinds = Vec::new();
    let mut non_reduce = Vec::new();  // everything else, order preserved
    for op in window {
        match op {
            Op::InstanceReduce { instance, kernel, .. } => {
                kinds.push(InstanceEvalKind::Dense {
                    instance: *instance,
                    kernel: *kernel,
                });
            }
            Op::InstanceSegmentedReduce { instance, kernel,
                round_within_phase, segmented, .. } => {
                kinds.push(InstanceEvalKind::Segmented {
                    instance: *instance,
                    kernel: *kernel,
                    round_within_phase: *round_within_phase,
                    segmented: segmented.clone(),
                });
            }
            _ => non_reduce.push(op.clone()),
        }
    }
    // Emit: [non-reduce ops up to but not including the last
    //        BatchAccumulateInstance], then BatchRoundEvaluate,
    //        then all BatchAccumulateInstance + BatchRoundFinalize.
    ...
}
```

**Key ordering constraint:** `BatchAccumulateInstance` reads
`state.last_round_instance_evals[instance.0]` which the reduce op writes.
Fusion must preserve: reduce before accumulate for the same instance.
But the simpler rule — "do ALL reduces first, then ALL accumulates" — is
safe and wins the cache battle because accumulate is cheap scalar work
and doesn't need to overlap with reduce.

### 3. CPU opts into the pass (✅ done)

```rust
// crates/jolt-cpu/src/backend.rs
impl ComputeBackend for CpuBackend {
    fn fuse_ops(&self, ops: &[Op]) -> Option<Vec<Op>> {
        Some(crate::fuse::fuse_batch_round_reduces(ops))
    }
    // ...
}
```

### 4. CPU overrides `batch_round_evaluate` with a cache-friendly impl (⏳ the remaining work)

The shape to match (from `jolt-core/src/subprotocols/booleanity.rs:327`):

```rust
// CPU pseudocode — single Rayon region, interleaved per-instance work
fn batch_round_evaluate<F>(&self,
    specs: &[BatchInstanceSpec<'_, Self, F>],
    challenges: &[F],
) -> Vec<Vec<F>> {
    // Partition specs by BatchReduceKind (Dense | Segmented | Gruen)
    // — each partition has a different kernel signature but within a
    // partition, instances share inner_size and outer_eq layout.
    let partitions = partition_by_kind(specs);

    // Pre-allocate result vecs (one per spec).
    let mut results: Vec<Vec<F>> = specs.iter()
        .map(|s| vec![F::zero(); s.kernel.num_evals()])
        .collect();

    rayon::scope(|scope| {
        for partition in partitions {
            scope.spawn(|_| {
                // For Dense partition:
                //   one par_iter_mut over outer-chunks of the shared
                //   half_len; inside each chunk, iterate all instances
                //   in the partition, reading from their input buffers
                //   and accumulating into a local [F; num_evals] scratch,
                //   then reducing into results[spec_idx].
                //
                // For Segmented/Gruen partitions:
                //   same shape but with outer_eq weighting applied once
                //   per outer slot (shared across instances that share
                //   outer_eq).
                run_partition(partition, challenges, &mut results);
            });
        }
    });

    results
}
```

The win is:

- **One Rayon region per batch-round**, not one per instance. 50-100µs
  fork/join × 4800 → × 20. Savings: ~400ms-500ms pure Rayon overhead.
- **Cache reuse**: each outer-chunk touches the matching slots across all
  instances in sequence. The eq table, kernel weight matrices, and
  outer_eq are fetched once per outer-chunk and reused for every instance.
  Projected cache miss reduction: ~2×, worth several seconds at log_T=16.
- **Dispatch amortized**: handler runs once per batch-round, not per
  instance.

## Code-level sketch — what files change

Already landed (✅):

- `crates/jolt-compute/src/traits.rs` — `fuse_ops` hook, default trait impl
  of `batch_round_evaluate`, `per_instance_batch_evaluate` reference path.
- `crates/jolt-compute/src/linker.rs` — `FuseDebugMode`, `Executable::shadow_ops`,
  `has_fuse_debug()`.
- `crates/jolt-zkvm/src/runtime/handlers.rs:852` — `BatchRoundEvaluate`
  handler with shadow-harness assertion.
- `crates/jolt-cpu/src/fuse.rs` — the pass + 6 unit tests.
- `crates/jolt-cpu/src/lib.rs` — `mod fuse;`
- `crates/jolt-cpu/src/backend.rs` — `fuse_ops` opt-in.
- `crates/jolt-equivalence/tests/fuse_equivalence.rs` — linker-level
  tests for the shadow plumbing.

Still pending (⏳):

- `crates/jolt-cpu/src/backend.rs` — add the `batch_round_evaluate`
  override (see §4 / Step 1). Keep the file organized so the new function
  sits next to `reduce` / `segmented_reduce`.
- **No handler changes.** Confirm the `BatchRoundEvaluate` handler is
  still ≤30 LOC of real logic after the override runs. (It is today.)

## Dependencies

- Ticket 0 (dual-path validation harness) — ✅ landed in commit
  `8befd6613`. Enables `JOLT_FUSE_DEBUG=1` shadow assertions per
  `BatchRoundEvaluate`.

## Correctness gate

Run with `JOLT_FUSE_DEBUG=1` during development — the harness catches
any reordering bugs immediately.

Then full suite:

```
cargo nextest run -p jolt-equivalence transcript_divergence
cargo nextest run -p jolt-equivalence zkvm_proof_accepted
cargo nextest run -p jolt-equivalence modular_self_verify
cargo nextest run -p jolt-equivalence
cargo clippy -p jolt-core --features host,zk --message-format=short -q --all-targets -- -D warnings
cargo clippy -p jolt-cpu --message-format=short -q --all-targets -- -D warnings
cargo clippy -p jolt-compute --message-format=short -q --all-targets -- -D warnings
```

**Critical:** rerun the correctness gate with `JOLT_FUSE_DEBUG=0` too.
The release-mode path (no shadow, no assertions) is the one that will run
in production and must pass independently.

## Perf gate

Standard Perf Loop measurement:

```
cargo run --release -p jolt-bench -- --program sha2-chain \
  --num-iters 16 --log-t 16 --iters 1 --warmup 1 \
  --json perf/last-iter.json
```

**Accept thresholds:**

- Minimum acceptable: ≥15% prove_ms reduction on modular stack. (The
  memo projects ~47s of ~120s CPU = 39%, which even at wall-clock
  parallelism floors should translate to ≥15% wall.)
- Target: ~30% prove_ms reduction. Current ratchet 72638ms → target
  ≤51000ms.
- Reject: <5% improvement (within Perf Loop inconclusive band — revert).

Update `perf/baseline-modular-best.json` on accept.

## Profiling checklist (before committing)

After implementing, re-profile to confirm the attack worked:

```
cargo run --release -p jolt-core profile --name sha2-chain --format chrome
```

Expected post-fusion:

- `BatchRoundEvaluate` span appears with roughly the summed time of the
  old per-instance reduces.
- `InstanceReduce` / `InstanceSegmentedReduce` spans vanish (or appear
  only in the decomp path, which stays per-instance).
- The CPU-work counter track should show ~30-40% less total sumcheck CPU.

If the trace shows the old spans still present, the fusion pass didn't
pick up that window — check the pattern matcher in `fuse_window`.

## Rollback

Pass is opt-in — to revert, change CPU's `fuse_ops` back to the default
`None`. That disables the pass but leaves the plumbing in place. Single
commit revert.

Alternative partial rollback: keep the pass but disable the CPU
`batch_round_evaluate` override. Fused stream still emits
`BatchRoundEvaluate` but the default trait impl falls back to per-instance
— byte-identical to the un-fused path. Useful for isolating a
correctness bug to the kernel override vs. the fusion pass.

## Open questions

- **Q1: Handling of `ReadCheckingReduce` + `RafReduce` within the same
  window.** These are emitted for instance-config (decomp) instances and
  are paired (read-checking runs, then RAF combines). Ticket 1 scope is
  limited to non-decomp instances. Decomp path stays per-instance —
  address separately in a follow-up if it becomes a bottleneck. The pass
  must leave these ops untouched.

- **Q2: `BatchAccumulateInstance` fusion?** Accumulate is cheap
  (~one extend + a few field mults per instance). Not worth fusing in
  this ticket — keep as per-instance. If it later shows up hot in the
  post-fusion profile, add `Op::BatchAccumulateAll` as a tiny follow-up.

- **Q3: Pass cost.** Running once at executable construction over a
  12,000-op vec is cheap (~1ms). Confirmed in profile — allocates
  ~12k ops vec once. If the rewritten stream ends up same-length, we
  could skip the clone. Not worth optimizing in v1.

- **Q4: Fused accumulation within `batch_round_evaluate` — is there a
  win beyond cache?** The booleanity kernel's `[F; 4]` local fold
  accumulator means we can accumulate directly into a shared `[F; 4] *
  num_instances` scratch without intermediate `Vec<F>` allocations per
  instance. This is the deeper win; worth doing in the CPU override,
  but factor out into a helper so the Segmented/Gruen variants share it.

- **Q5: Does `BatchInstanceSpec` need a new field for "shared outer_eq"
  across instances?** Currently each spec carries its own `outer_eq:
  &[F]`. If two instances share the same slice, we can dedupe when
  building the outer-chunk Rayon iterator — but that's a sub-optimization
  inside the CPU override, not a trait change.
