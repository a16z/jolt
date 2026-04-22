# Ticket 1 — Reduce-side fusion: `fuse_ops` hook + CPU `batch_round_evaluate`

**Status:** biggest single architectural win. Attack after ticket 0 lands.
**Est. perf win:** ~47s CPU on sha2-chain log_T=16 (factor B dispatch
overhead + part of factor C cache thrash, per
`perf/report_tools/kernel_gap_memo.md` §1).
**Est. effort:** medium (~400-500 LOC + CPU kernel restructure).
**Philosophy check:** handlers unchanged in LOC, compiler unchanged —
rewrite pass lives entirely in `jolt-compute` / `jolt-cpu`.

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

### 1. Add `fuse_ops` to `ComputeBackend` (lands via Ticket 0)

```rust
// crates/jolt-compute/src/traits.rs
pub trait ComputeBackend {
    // ... existing
    /// Rewrite the emitted op stream with backend-specific fusion rules.
    ///
    /// Default returns `None` (no rewrite — runtime executes the input
    /// stream as-is). Backends opt in by returning `Some(new_stream)`.
    /// Called once at `Executable` construction; result is cached on the
    /// executable.
    fn fuse_ops(&self, ops: &[Op]) -> Option<Vec<Op>> {
        None
    }
}
```

### 2. Pure pass: collapse reduces within batch-round windows

New module `crates/jolt-compute/src/fuse/batch_round.rs`:

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

### 3. CPU opts into the pass

```rust
// crates/jolt-cpu/src/backend.rs
impl ComputeBackend for CpuBackend {
    fn fuse_ops(&self, ops: &[Op]) -> Option<Vec<Op>> {
        Some(jolt_compute::fuse::fuse_batch_round_reduces(ops))
    }
    // ...
}
```

### 4. CPU overrides `batch_round_evaluate` with a cache-friendly impl

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

- **NEW** `crates/jolt-compute/src/fuse/mod.rs` — pass module.
- **NEW** `crates/jolt-compute/src/fuse/batch_round.rs` — the reduce pass.
- **EDIT** `crates/jolt-compute/src/traits.rs` — add `fuse_ops` trait
  method (if not already added by ticket 0).
- **EDIT** `crates/jolt-cpu/src/backend.rs` — implement `fuse_ops`
  (delegate to pass) + override `batch_round_evaluate`.
- **EDIT** `crates/jolt-compiler/src/compiler/mod.rs` (or wherever
  `Executable` is constructed) — call `backend.fuse_ops` and store result.
- **EDIT** `crates/jolt-zkvm/src/runtime/handlers.rs` — no changes
  required; handler already handles `BatchRoundEvaluate`. Verify the
  ≤30 LOC rule still holds.

## Dependencies

- Ticket 0 (dual-path validation harness) — must land first so fusion
  divergence surfaces at a single assertion, not 200 rounds later.

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
