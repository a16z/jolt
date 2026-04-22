# Ticket 4 — Shape-preserving Compact / OneHot tags on `DeviceBuffer`

**Status:** highest risk, highest long-term win. Attack only after
tickets 1, 2, 3 are landed and stable.
**Est. perf win:** ~15s CPU on sha2-chain log_T=16 (factor D: field
promotion, per `perf/report_tools/kernel_gap_memo.md` §1). Possibly
more as it unlocks sparse OneHot fast-paths that core enjoys.
**Est. effort:** large — touches most `ComputeBackend` methods and
every compact-scalar site. Plan for 2-3 days of careful work.
**Philosophy check:** a zero-cost abstraction on the IR boundary; the
handler still calls the same method names — dispatch happens inside
the backend via a tag. See `feedback_abstractions_zero_cost.md` and
`feedback_no_smallscalar_trait.md` (we do NOT mirror core's
`SmallScalar` trait — we extend the existing `DeviceBuffer` encoding).

## Why this exists

`jolt-core` stores small-scalar polynomials (`CompactPolynomial<T>`)
with `T ∈ {u8, i8, u16, u32, i64, u128}`, avoiding 32-byte field
element storage and avoiding field-arithmetic cost when the bind step
can short-circuit on zero or small values. For booleanity and
one-hot polynomials, core also has specialized `OneHot` paths that
run at ~1/10th the cost of the equivalent dense-field path.

In the modular stack today, `DeviceBuffer<F>` loses all encoding
information at the `Op` boundary — once a poly becomes a `Buf<B, F>`,
the backend sees only `Vec<F>`. Sparsity is gone, small-scalar is gone.
Every bind/reduce touches full 32-byte field elements even when the
logical data is u8.

The prohibition from user directive (`feedback_no_smallscalar_trait.md`):
**do not add a `SmallScalar` trait + `CompactPolynomial<T, F>` generic
parameter.** The abstraction level in the modular stack is
`ComputeBackend` + `DeviceBuffer` — shape-preservation happens inside
the buffer encoding, not as an extra generic.

## Architectural change

### 1. Extend `DeviceBuffer` with an encoding tag

```rust
// crates/jolt-compute/src/traits.rs (or wherever DeviceBuffer lives)
pub enum DeviceBuffer<F: Field> {
    Dense(Vec<F>),                         // existing
    OneHot { indices: Vec<u32>, len: usize },    // NEW: sparse one-hot
    CompactU8(Vec<u8>),                    // NEW
    CompactU16(Vec<u16>),                  // NEW
    CompactU32(Vec<u32>),                  // NEW
    CompactU64(Vec<u64>),                  // NEW
    CompactI64(Vec<i64>),                  // NEW (for signed advice)
    // Promoted types (existing dense representation after bind):
    PromotedFromU8(Vec<F>),
    PromotedFromU16(Vec<F>),
    // etc — or use a single Dense with a "was_compact" marker
}
```

Alternative flattening: keep a single `Dense(Vec<F>)` variant but add
a `origin: Option<CompactTag>` side-channel. Trade-offs:

- **Union variants**: clean pattern-match dispatch, but every `as_field`
  / `as_field_mut` site must promote-to-dense on demand. More code
  paths but cleaner types.
- **Side-channel tag**: keep existing type shape, pay the tag check in
  hot paths. Less code change; harder to catch regressions in the
  type system.

**Recommendation**: union variants. Promotion is explicit
(`buf.promote_to_dense()`) and one-time per poly.

### 2. Compact-aware ComputeBackend methods

Every method that currently takes `&Buf<B, F>` must decide:

- **Does it need dense semantics?** → call `.promote_to_dense()` at
  entry, memoize on the buffer so the next call sees `Dense`.
- **Can it specialize on compact?** → branch on the tag and dispatch
  to a specialized inner loop. For bind and reduce, this is where the
  wins live.

Example — CPU `batch_interpolate_inplace` with compact support:

```rust
fn batch_interpolate_inplace<F: Field>(
    &self,
    specs: &[BatchBindSpec<'_, Self, F>],
    challenge: F,
    order: BindingOrder,
    state: &mut Self::SumcheckState<F>,
) {
    for spec in specs {
        for poly in &mut spec.polys {
            match poly {
                DeviceBuffer::Dense(v) => bind_dense_inplace(v, challenge, order),
                DeviceBuffer::CompactU8(v) => {
                    let promoted = bind_compact_u8_to_dense(v, challenge, order);
                    **poly = DeviceBuffer::Dense(promoted);
                }
                DeviceBuffer::OneHot { indices, len } => {
                    let promoted = bind_onehot_to_dense(indices, *len, challenge, order);
                    **poly = DeviceBuffer::Dense(promoted);
                }
                // etc.
            }
        }
    }
}
```

The compact→dense bind does half the work of dense→dense (the "zero
half" is free).

### 3. Compact origin surfaced at the `Op` boundary

The compiler needs to know which polynomials are small-scalar so it
can tag the initial materialization. Today, `WitnessProvider` produces
`Vec<F>` blindly. Extend the provider trait:

```rust
// crates/jolt-compute/src/provider.rs or similar
pub trait BufferProvider {
    fn materialize<F: Field>(&self, poly: PolynomialId) -> DeviceBuffer<F>;
    // Default: returns DeviceBuffer::Dense. Witness crates override
    // to return Compact variants for small-scalar polys.
}
```

### 4. Ticket 1/2 kernels become compact-aware

The CPU `batch_round_evaluate` and `batch_interpolate_inplace`
specialize on the tag of the input buffers. For polys that are
promoted early (e.g., advice polys are already dense), this is a no-op
branch. For polys that stay compact through many rounds (one-hot RA
polys — common), the specialized path is dramatically cheaper.

## Code-level sketch — what files change

- **EDIT** `crates/jolt-compute/src/traits.rs` — add variants to
  `DeviceBuffer` + helper methods (`promote_to_dense`, `as_compact_u8`,
  etc).
- **EDIT** every `impl ComputeBackend for X` method that takes
  `&Buf<B, F>` or `&mut Buf<B, F>` — add compact-tag dispatch.
- **EDIT** `crates/jolt-witness/*` — witness providers return
  Compact/OneHot variants when the source poly is small-scalar.
- **EDIT** `crates/jolt-zkvm/src/runtime/handlers.rs` — materialize
  ops flow Compact buffers through; no change to handler LOC count.
- **NEW** `crates/jolt-cpu/src/compact_bind.rs` / `compact_reduce.rs`
  — specialized inner loops per tag. Maybe 6 variants × bind +
  reduce = ~12 tight inner-loop functions.

## Dependencies

- Tickets 1 and 2 must land first — this ticket's inner-loop
  specializations are only profitable inside
  `batch_round_evaluate` / `batch_interpolate_inplace`.
- Ticket 3 (persistent state) — scratch pool should handle promoted
  dense allocations to avoid rebuilding the Dense buffer on every
  round.

## Correctness gate

Same suite plus:

- **Compact→Dense promotion correctness** — for each compact variant,
  a unit test that runs a small prover with the poly as Compact vs.
  Dense-pre-promoted, assert identical transcripts.
- **OneHot bind correctness** — binding a one-hot poly must produce
  the same result as binding the dense expansion. Unit test with
  random challenges × small sizes.
- **Round-trip stability** — a compact poly that gets promoted
  mid-round must not be re-tagged as compact on the next round. The
  variant transition is one-way.

## Perf gate

```
cargo run --release -p jolt-bench -- --program sha2-chain \
  --num-iters 16 --log-t 16 --iters 1 --warmup 1 \
  --json perf/last-iter.json
```

**Accept thresholds:**

- Minimum: ≥8% additional prove_ms reduction on modular stack
  beyond tickets 1+2+3.
- Target: ~12-15%, which should bring cumulative prove_ms to within
  ~3× of core (matching the kernel_gap_memo projection).
- Reject: <5% (inconclusive band → revert).

## Profiling checklist

- `CompactU8` / `CompactU16` bind spans appear in trace.
- Dense-equivalent span time drops proportionally to the compact poly
  population (majority of witness polys are u8/u16 in sha2).
- Field-mult count (approximable via allocative + span timing) drops
  ~2×, matching factor D's 2.5× projection.

## Rollback

Risk is high — many method signatures touched. Rollback via feature
flag:

```rust
#[cfg(feature = "compact_bufs")]
impl DeviceBuffer<F> { ... Compact variants ... }
```

Tickets 1-3 don't rely on compact variants. Defaulting the feature
off gives a safe revert path.

## Open questions

- **Q1: Union variant vs. side-channel tag.** Recommend union.
  Cleaner dispatch; variant explosion is bounded to ~6 types. Revisit
  if the enum becomes unwieldy.

- **Q2: Should `OneHot` promote on first bind, or stay OneHot
  through all rounds?** Core's `RaPolynomial` stays OneHot through
  multiple rounds using a state machine (Round1→Round2→RoundN). This
  ticket could start with "promote on first bind" and defer the full
  state machine to a follow-up. Flagged as a staging question.

- **Q3: Compact variants for intermediate computations.** If a reduce
  produces a `[F; 4]` eval vec, that stays dense. Compact only applies
  to input polynomials, not intermediate results. Confirm this matches
  the current compiler emission.

- **Q4: How does this interact with the fusion pass?** The fusion pass
  doesn't care about buffer encodings — it rewrites ops. The
  `batch_round_evaluate` handler sees a mix of compact and dense
  buffers in `specs[i].inputs`. CPU override handles each per-input.

- **Q5: Witness provider changes.** Today, `jolt-witness` materializes
  polynomials as `Vec<F>`. Extending the provider trait is a breaking
  change across all witness implementors. Plan for a single refactor
  commit that updates all callers.
