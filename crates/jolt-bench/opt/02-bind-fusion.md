# Ticket 2 — Bind-side fusion: `Op::BatchBind` + CPU fused bind

**Status:** second architectural attack after ticket 1 is in.
**Est. perf win:** ~19s CPU on sha2-chain log_T=16 (remaining half of
factor C cache thrash + factor C' cross-instance bind overhead, per
`perf/report_tools/kernel_gap_memo.md` §1).
**Est. effort:** medium (~300-400 LOC).
**Philosophy check:** adds one new `Op` variant; compiler emission
unchanged; backend fusion pass picks up the new pattern.

## Why this exists

After ticket 1 fuses reduces, the remaining hot spot in the sumcheck
inner loop is `interpolate_inplace` — the per-instance bind that runs
before reduce each round. The per-iter profile shows it at
~19,000ms × 17,416 calls with an average ~1.1ms each. Each call:

1. Allocates a fresh `Vec<F>` of length `half` for the bound output.
2. Spins up a Rayon region for the fold.
3. Writes the result back.

For 4,800 instance-bind ops per sha2-chain log_T=16 prove, we pay
~17k allocations + ~17k Rayon regions + ~17k sequential passes over
adjacent buffers (cache-cold for each instance).

`jolt-core`'s booleanity `ingest_challenge` binds eq + all RA polys in a
single pass (`jolt-core/src/subprotocols/booleanity.rs:383`). Same
architectural shape as ticket 1, applied to bind.

The blocker is structural: today the `Op` enum has `InstanceBind` (single
instance) and `BindCarryBuffers` (per-poly list, per-instance) but no
variable-arity `BatchBind`. Adding it is the bulk of this ticket's work.

## Architectural change

### 1. New op variant

```rust
// crates/jolt-compiler/src/module.rs
/// Fused bind across many active instances in a batch-round.
///
/// Variable-arity generalization of `InstanceBind`. One op per batch-round
/// replaces N per-instance bind ops. The handler packs a
/// `&[BatchBindSpec]` and dispatches once to
/// `ComputeBackend::batch_interpolate_inplace`, which defaults to a
/// per-instance loop (byte-identical fallback) and is overridable for
/// fused bind (shared scratch, single Rayon region).
///
/// Emitted only via the backend's `fuse_ops` pass — the compiler does
/// not emit this variant.
BatchBind {
    batch: BatchIdx,
    round: usize,
    instances: Vec<BindInstanceSpec>,
    challenge: ChallengeIdx,
},

pub struct BindInstanceSpec {
    pub instance: InstanceIdx,
    pub kernel: usize,
    pub carry_polys: Vec<PolynomialId>,  // from BindCarryBuffers, if any
}
```

### 2. Trait method with default

```rust
// crates/jolt-compute/src/traits.rs
pub trait ComputeBackend {
    /// Per-round bind across a batch of sumcheck instances.
    ///
    /// Binds each instance's input buffers (and carry polys) in-place
    /// using the round challenge. Default implementation loops per-spec
    /// via `interpolate_inplace`, matching the current per-instance
    /// dispatch byte-for-byte. Backends override for fused dispatch
    /// (shared scratch, single Rayon region, cache-sequential per-slot
    /// traversal across instances).
    fn batch_interpolate_inplace<F: Field>(
        &self,
        specs: &[BatchBindSpec<'_, Self, F>],
        challenge: F,
        order: BindingOrder,
    ) {
        for spec in specs {
            for poly in &spec.polys {
                self.interpolate_inplace(poly, challenge, order);
            }
        }
    }
}

pub struct BatchBindSpec<'a, B: ComputeBackend + ?Sized, F: Field> {
    pub instance: InstanceIdx,
    pub polys: Vec<&'a mut B::Buffer<F>>,  // de-duplicated
}
```

Note the `&mut` refs in `BatchBindSpec` — binding is in-place. The
handler packing layer in `handlers.rs` needs to carefully de-duplicate
input buffers across instances (some instances share inputs — seen
today by `state.bound_this_round: HashSet<PolynomialId>`).

### 3. Fusion pass extension

Extend `jolt-compute::fuse::batch_round::fuse_window` (ticket 1) to also
collect `InstanceBind` + `BindCarryBuffers` ops within the window and
emit a single `Op::BatchBind`:

```rust
fn fuse_window(window: &[Op], ...) -> Vec<Op> {
    let mut bind_specs = Vec::new();
    let mut reduce_kinds = Vec::new();
    let mut other = Vec::new();
    let mut challenge = None;
    for op in window {
        match op {
            Op::InstanceBind { instance, kernel, challenge: ch, .. } => {
                challenge = Some(*ch);
                bind_specs.push(BindInstanceSpec {
                    instance: *instance,
                    kernel: *kernel,
                    carry_polys: vec![],
                });
            }
            Op::BindCarryBuffers { polys, challenge: ch, .. } => {
                // Attach to the most recent bind spec (same instance).
                challenge = Some(*ch);
                if let Some(last) = bind_specs.last_mut() {
                    last.carry_polys.extend(polys.iter().copied());
                }
            }
            Op::InstanceReduce { .. } | Op::InstanceSegmentedReduce { .. } => {
                // ticket 1 logic
            }
            _ => other.push(op.clone()),
        }
    }
    // Emit: [pre-bind misc ops] → BatchBind → [pre-reduce misc ops]
    //       → BatchRoundEvaluate → BatchAccumulateInstance×N → BatchRoundFinalize
}
```

**Ordering constraint:** bind must run before reduce (reduce reads bound
state). The fused stream preserves this: all binds → all reduces.

**Edge case:** the `Bind` op (used for decomp-instance path) is different
from `InstanceBind` and stays un-fused. Fusion pass matches on
`InstanceBind` only.

### 4. CPU override: fused-bind with shared scratch

```rust
// crates/jolt-cpu/src/backend.rs
impl ComputeBackend for CpuBackend {
    fn batch_interpolate_inplace<F: Field>(...) {
        // 1. Flatten (instance, poly) → flat list of &mut Vec<F>,
        //    with sizes matching (typically half_len).
        // 2. Allocate one persistent scratch vec (ties in with ticket 3).
        // 3. For each unique size group:
        //    rayon::scope |s| {
        //      s.spawn(move |_| {
        //        // Inner-chunk-interleaved: chunk the outer (half_len)
        //        // axis, inside each chunk process all polys.
        //        for chunk in outer_chunks(half_len / THREADS) {
        //          for &mut poly in polys_of_size(size) {
        //            interpolate_chunk_inplace(poly, chunk, challenge);
        //          }
        //        }
        //      });
        //    }
    }
}
```

The win: for ~120 polys of identical size per round, one Rayon scope
instead of 120. At ~50µs fork/join floor × 20 rounds × 120 polys,
eliminating 2400 Rayon regions saves ~120ms just in parallelism
overhead. The bigger win is cache: binding poly_i's slot `2*k..2*k+2`
prefetches the adjacent cache line before poly_{i+1} needs it.

## Code-level sketch — what files change

- **EDIT** `crates/jolt-compiler/src/module.rs` — add `Op::BatchBind`
  variant + `BindInstanceSpec` struct.
- **EDIT** `crates/jolt-compute/src/traits.rs` — add
  `batch_interpolate_inplace` with default.
- **EDIT** `crates/jolt-compute/src/fuse/batch_round.rs` — extend
  `fuse_window` to also collect bind ops.
- **EDIT** `crates/jolt-cpu/src/backend.rs` — override
  `batch_interpolate_inplace`.
- **EDIT** `crates/jolt-zkvm/src/runtime/handlers.rs` — new handler arm
  for `Op::BatchBind`. Must stay ≤30 LOC. Keep the dedupe logic that
  `InstanceBind`/`BindCarryBuffers` handlers have today — instances
  share some input polys (bound_this_round: HashSet<PolynomialId>).

## Dependencies

- Ticket 0 (dual-path validation).
- Ticket 1 (`fuse_ops` trait method + fuse-window framework). Ticket 2
  extends ticket 1's `fuse_window` to also emit `BatchBind`.

Can be developed in parallel with ticket 3 (persistent state) once
ticket 1 lands; ticket 3's scratch-vec infra dovetails with this
ticket's fused-bind inner loop.

## Correctness gate

Same suite as ticket 1. Run with `JOLT_FUSE_DEBUG=1` during dev.

Special attention:

- **De-duplication correctness.** Today's `InstanceBind` handler uses
  `state.bound_this_round` to skip polys already bound this round. The
  fused `BatchBind` must achieve the same: if instance A and instance B
  share input poly P, P must only be bound once per round. Put this in
  a unit test in `jolt-equivalence/tests/fuse_bind_dedupe.rs`.

- **Binding order.** Each kernel declares a `binding_order` (HighToLow /
  LowToHigh). Some instances may differ. The fused impl must partition
  by order and run each partition independently. Today, all instances
  within a batch share the same order — verify this assumption with a
  `debug_assert_eq!` in the pass.

## Perf gate

```
cargo run --release -p jolt-bench -- --program sha2-chain \
  --num-iters 16 --log-t 16 --iters 1 --warmup 1 \
  --json perf/last-iter.json
```

**Accept thresholds:**

- Minimum: ≥10% additional prove_ms reduction on modular stack.
- Target: ~15% additional, bringing cumulative (ticket 1 + 2) to ~45%.
- Reject: <5% (inconclusive band → revert).

## Profiling checklist

After ticket 2, the profile should show:

- `BatchBind` span appears with ~summed time of old `InstanceBind` +
  `BindCarryBuffers`.
- `interpolate_inplace` per-call count drops from ~17k to ~20-40 (one
  per batch-round).
- Rayon fork/join overhead (visible as idle gaps) drops substantially.

## Rollback

Two layers of rollback:

1. **Disable bind fusion** — add a feature-gate or env var to skip
   `BatchBind` emission in the pass. Pass still fuses reduces. Keeps
   ticket 1's wins.
2. **Full revert** — pure revert commit. `Op::BatchBind` variant stays
   in the enum (dead code, not harmful) but the fusion pass stops
   emitting it.

## Open questions

- **Q1: `Bind` (decomp) handling.** `Op::Bind` (for instance-config
  instances) is a different op shape and stays un-fused. Verify this
  doesn't break the pass — pattern match only on `InstanceBind`.

- **Q2: `BindingOrder` partitioning.** Today, do all instances in a
  batch share the same binding_order? Grep
  `crates/jolt-compiler/src/builder.rs` — looks like they do (per
  `kdef.spec.binding_order` — kernel-level). But partition safely in
  the CPU impl anyway.

- **Q3: Is `BatchBind` one op per (batch, round) or one per
  (batch, round, binding_order)?** If the former, the CPU impl
  partitions internally. If the latter, the pass partitions. The former
  is simpler; go with that.

- **Q4: Shared scratch for the fused-bind inner loop.** Each instance's
  bind needs a temporary `[F; 2]` lane accumulator. Stack-local is fine
  for the fold — no heap allocation needed. But if we later move to
  per-slot scratches, ticket 3's persistent state is the place.

- **Q5: Do we benefit from fusing bind and reduce into a single pass?**
  Core's booleanity `ingest_challenge` + `compute_message` happen in
  different phases of the round (bind at end of round i, compute at
  start of round i+1). They don't need to fuse. Ticket 2 keeps them
  separate.
