# Ticket 3 — Persistent per-Executable state (scratch + shared eq handle)

**Status:** unlocks ~4s CPU + headroom for tickets 1 & 2 to scale.
**Est. perf win:** ~4s direct (factor A: redundant temp `Vec<F>`
allocation, per `perf/report_tools/kernel_gap_memo.md` §1) + unknown
headroom for ticket 1/2's CPU kernels.
**Est. effort:** medium — carefully-designed but small surface. Absorbs
the reverted iter-76 eq-handle work.
**Philosophy check:** handlers unchanged; state owned by the
backend's execution context, not a global.

## Why this exists

Profile at iter 93-95 shows `eq_project` at ~9,800ms across 82 calls.
Each call materializes a fresh `Vec<F>` of eq values from scratch. Two
successive rounds share all but the most-recent bound variable — the
previous round's eq table is a valid prefix of this round's, yet we
recompute from scratch.

More generally, several CPU paths allocate temporary `Vec<F>` per call:

- `gruen_segmented_reduce` allocates `[F; degree]` per segment.
- `segmented_reduce` allocates an eval-buffer of size `num_evals`.
- `bind_high_to_low` / `bind_low_to_high` both allocate a fresh
  destination `Vec<F>` of length `half` every call.

Per-call allocation overhead is small individually but accumulates:
memo estimates ~4,000ms CPU in pure allocation/free + page faults.

The prior attack (iter 76) tried to solve this with a global
`Mutex<HandleStore>` — which regressed because the mutex serialized
Rayon workers (`feedback_handle_mutex_contention.md`). The fix is to
scope state to the **Executable** (one per prove call), not global.

This ticket gives the backend a place to stash per-prove state:

1. A pool of reusable scratch vecs, sized by seen sizes.
2. A persistent `EqHandle` that memoizes the shared eq prefix across
   rounds.
3. Per-batch combined-eval buffers that survive across BatchRoundEvaluate
   calls so we don't reallocate per round.

## Architectural change

### 1. `SumcheckState<F>` associated type on `ComputeBackend`

```rust
// crates/jolt-compute/src/traits.rs
pub trait ComputeBackend {
    type SumcheckState<F: Field>: Default;

    /// Called once at the start of each Executable prove run.
    /// Backends may pre-allocate scratch or set up persistent handles.
    fn open_sumcheck_state<F: Field>(&self) -> Self::SumcheckState<F> {
        Self::SumcheckState::<F>::default()
    }

    /// Called once at the end of each prove run. Backends free or
    /// recycle state. Default: drop (state goes out of scope).
    fn close_sumcheck_state<F: Field>(&self, _s: Self::SumcheckState<F>) {}
}
```

The state is passed into `batch_round_evaluate` and
`batch_interpolate_inplace`:

```rust
fn batch_round_evaluate<F: Field>(
    &self,
    specs: &[BatchInstanceSpec<'_, Self, F>],
    challenges: &[F],
    state: &mut Self::SumcheckState<F>,  // NEW
) -> Vec<Vec<F>>;

fn batch_interpolate_inplace<F: Field>(
    &self,
    specs: &[BatchBindSpec<'_, Self, F>],
    challenge: F,
    order: BindingOrder,
    state: &mut Self::SumcheckState<F>,  // NEW
);
```

The runtime threads `state` through the execution loop:

```rust
// crates/jolt-zkvm/src/runtime/ — execute()
let mut sumcheck_state = backend.open_sumcheck_state::<F>();
for op in &executable.ops {
    dispatch_op(op, ..., &mut sumcheck_state);
}
backend.close_sumcheck_state(sumcheck_state);
```

### 2. CPU implementation: `ScratchPool` + `EqHandle`

```rust
// crates/jolt-cpu/src/backend.rs
#[derive(Default)]
pub struct CpuSumcheckState<F: Field> {
    /// Pool of reusable Vec<F> scratch buffers, keyed by capacity
    /// (rounded up to power of two). Reclaimed on drop of the
    /// ScratchGuard returned by `get`.
    scratch: ScratchPool<F>,

    /// Persistent eq-handle memoizing the shared eq prefix. Stores
    /// the evaluation vec from the previous round and lets
    /// `batch_round_evaluate` reuse it when the challenge sequence
    /// matches a known prefix.
    eq_handle: Option<EqHandle<F>>,

    /// Per-batch combined-eval buffer; persists across rounds within
    /// the same batch so we reset (zero) rather than re-allocate.
    combined_buffers: HashMap<BatchIdx, Vec<F>>,
}

pub struct ScratchPool<F: Field> {
    by_size: HashMap<usize, Vec<Vec<F>>>,
}

impl<F: Field> ScratchPool<F> {
    pub fn get(&mut self, size: usize) -> ScratchGuard<'_, F> {
        let bucket = self.by_size.entry(size.next_power_of_two()).or_default();
        let vec = bucket.pop().unwrap_or_else(|| Vec::with_capacity(size));
        ScratchGuard { pool: self, bucket_size: size.next_power_of_two(), vec: Some(vec) }
    }
}

pub struct EqHandle<F: Field> {
    pub r: Vec<F>,              // bound challenges so far
    pub evals: Vec<F>,           // eq table after binding r
}

impl<F: Field> EqHandle<F> {
    /// Extend by one round: bind r_new into the eq table in-place.
    /// Avoids rebuilding the table from scratch.
    pub fn bind(&mut self, r_new: F) { ... }
}
```

The scratch pool's `ScratchGuard` returns the vec to the pool on drop,
clearing it (but preserving capacity).

### 3. Tie-in with tickets 1 & 2

- Ticket 1 (`batch_round_evaluate`) uses `state.scratch.get(num_evals)`
  instead of `vec![F::zero(); num_evals]`.
- Ticket 2 (`batch_interpolate_inplace`) uses the scratch pool for
  lane-local temp vecs if any are needed.
- Both can use `state.eq_handle` as the source-of-truth for the shared
  eq prefix across instances, avoiding the per-instance
  `BuildSegmentedEq` allocations when possible.

## Dependencies

- Ticket 1 must land first (it defines `batch_round_evaluate`'s
  signature; this ticket extends it with `state`).
- Ticket 2 can be developed alongside or after this ticket; if ticket 3
  lands first, ticket 2 picks up the `state` threading.

## Correctness gate

Same suite as tickets 1 & 2. Specific tests to add:

- **Scratch-reuse sanity** — a unit test that runs two prove calls with
  the same Executable (impossible today — Executable is consumed?) or
  two sequential sumchecks and asserts the scratch pool reused buffers.
  Probe via `state.scratch.bucket_size(target).len()`.

- **EqHandle bind correctness** — a unit test that computes the eq
  table fresh at r = [r_0, r_1, r_2] and compares to the handle's
  eq table after binding r_0, r_1, r_2 sequentially. Must be
  bit-exact.

- **Pool isolation** — confirm state from one Executable doesn't leak
  to another (concurrent prove calls don't share the pool). Test by
  running two prove calls on different threads and asserting no
  cross-contamination. Probably ensured by `open_sumcheck_state`
  returning a fresh instance each time.

## Perf gate

```
cargo run --release -p jolt-bench -- --program sha2-chain \
  --num-iters 16 --log-t 16 --iters 1 --warmup 1 \
  --json perf/last-iter.json
```

**Accept thresholds:**

- Minimum: ≥3% additional prove_ms reduction on modular stack.
- Target: ~5-7%, translating to ~2000-3000ms saved.
- Reject: <2% (within noise → revert).

The direct allocation-savings win is small; the real leverage is
ticket 1/2's ability to operate on persistent buffers. If the combined
ticket-1-and-3 delta is larger than ticket-1 alone, credit the
difference to this ticket.

## Profiling checklist

- `eq_project` call count should drop substantially (ideally to 1 per
  batch if the handle catches the shared prefix).
- `allocation` profiling via `--features allocative` should show fewer
  per-call `Vec<F>` allocations inside `reduce`/`segmented_reduce`.
- Page faults (visible in `--features monitor` RSS track) should be
  flat across rounds rather than climbing.

## Rollback

Additive. Reverting removes the associated type + the CPU impl; the
backend falls back to per-call allocation. Tickets 1/2 stay intact
because their signatures include `state: &mut Self::SumcheckState<F>`
with a default.

## Open questions

- **Q1: Associated type vs. struct threading.** Alternative: pass a
  `&mut dyn ScratchPool + EqHandle` instead of an associated type.
  Associated type is zero-cost (`feedback_abstractions_zero_cost.md`)
  and lets each backend define its own shape. Prefer associated type.

- **Q2: Prove-call lifecycle.** Where exactly in `runtime::execute` do
  we call `open_sumcheck_state` / `close_sumcheck_state`? Once per
  `execute` call, or once per `Executable::prove`? Probably once per
  `execute`. Double-check there aren't multiple `execute` calls per
  prove — if there are, state is per-execute and doesn't survive.

- **Q3: Cross-thread prove calls.** If someone runs two provers in
  parallel, each gets its own state. Confirmed by
  `open_sumcheck_state` returning a fresh instance. No shared state,
  no contention.

- **Q4: EqHandle memory.** Keeping the full eq table at each round
  stored in the handle costs `sum_{r=0}^{R} 2^(logT - r)` = 2 * 2^logT
  = 2T field elements. At T = 2^16 and BN254 Fr (32 bytes), that's 4MB
  — acceptable. If tight, store only the final eq table and the
  challenge vec; recompute on demand.

- **Q5: Thread-local scratch pool?** If we want parallel Rayon workers
  to each have their own scratch (no contention), move the pool to
  thread-local storage. Complicates `ScratchGuard` lifetime. Probably
  unnecessary for v1 — the `SumcheckState<F>` is single-threaded
  through `batch_round_evaluate`, and inner workers use stack-local
  scratch.
