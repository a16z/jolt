# jolt-zkvm Cleanup Tracker

Post-E2E code review. Items ordered by priority.

## 1. Wire OpeningStage into pipeline (dead code)

**Status**: TODO
**Files**: `src/prover.rs:115-142`, `src/stages/s8_opening.rs`

`prover.rs` inlines the RLC-reduce + PCS-open logic directly:

```rust
// prover.rs lines 120-141 — inline S8
let (reduced, ()) = <RlcReduction as OpeningReduction<PCS>>::reduce_prover(
    opening_claims, transcript, challenge_fn,
);
reduced.into_iter().map(|claim| {
    let poly: PCS::Polynomial = claim.evaluations.into();
    PCS::open(&poly, &claim.point, claim.eval, &key.pcs_prover_setup, None, transcript)
}).collect()
```

Meanwhile `OpeningStage::prove` in `s8_opening.rs` does the exact same thing, with
404 lines of code and full test coverage, but is never called by the pipeline.

**Root cause**: `OpeningStage` was written as a standalone stage with its own
`prove`/`verify` methods. It doesn't implement `ProverStage` because it has a
different signature (takes `Vec<ProverClaim>` directly). The prover was then
assembled by hand-wiring each step, duplicating the logic.

S8 is fundamentally different from S1-S7 — it does PCS openings, not sumcheck.
It receives all accumulated claims but doesn't produce new ones. Forcing it into
`ProverStage` (which returns `StageBatch` with sumcheck claims/witnesses) would
be a poor fit.

**Decision**: Keep `OpeningStage` as a separate type (not `ProverStage`), but
wire it into the pipeline properly:

```rust
// prover.rs — replace inline S8 with:
let opening_proofs = OpeningStage::<PCS>::prove(
    opening_claims, &key.pcs_prover_setup, transcript, ...
);
```

Wire `OpeningStage::verify()` into the verifier path symmetrically. Delete the
duplicated inline logic.

---

## 2. CommittedTables — verifier needs claim threading, not raw tables

**Status**: TODO
**Files**: `src/host.rs`

`ProveOutput` carries `committed_tables: CommittedTables<F>` — the prover's full
`rd_inc` and `ram_inc` evaluation tables — so `verify_proof` can manually recompute
`claimed_sum` in `build_verifier_descriptors`:

```rust
let claimed_sum: F = (0..n)
    .map(|j| eq_table[j] * (c0 * ram_inc[j] + c1 * rd_inc[j]))
    .sum();
```

This is architecturally wrong: a real verifier never has the prover's tables.
The `claimed_sum` should flow from the claim threading chain — each stage's
output evaluations (already in the proof via `SumcheckStageProof::evaluations`)
feed into the next stage's `claimed_sum` via the IR-defined claim formula.

**Risk**: The manual recomputation can diverge from what the prover actually
claimed, creating a silent correctness bug (either false accepts or rejects).

**Fix**:
1. `StageDescriptor` already carries `claimed_sum`. The verifier should derive
   this from prior stage evaluations + the Spartan witness eval, using the same
   `ClaimDefinition` expressions that define the stage formulas.
2. Delete `CommittedTables` and the `committed_tables` field from `ProveOutput`.
3. `verify_proof` should be self-contained: proof + verifying key = enough.

---

## 3. Wire `jolt-field` Challenge type through pipeline, eliminate challenge_fn

**Status**: TODO
**Files**: jolt-sumcheck, jolt-openings, jolt-verifier, jolt-zkvm (all `challenge_fn` sites)

### Background

`jolt-field` already has a complete `Challenge` infrastructure:

| Type | Description |
|------|-------------|
| `MontU128Challenge<F>` | 125-bit value in two u64 limbs. Uses `Fr::mul_by_hi_2limbs()` for ~1.6x faster `Challenge × Fr` multiply (4×2 limbs vs 4×4). Default. |
| `Mont254BitChallenge<F>` | Full 254-bit challenge. Same speed as `Fr × Fr`. Feature-gated fallback. |
| `Challenge<F>` trait | `From<u128> + Into<F> + Mul<F, Output=F>` — type-safe challenge with field interop |
| `WithChallenge` trait | Associates `Field` with its default `Challenge` type. Bounds enable `F * C → F`. |
| `DefaultChallenge<F>` | Type alias selected by `challenge-254-bit` feature flag |
| `OptimizedMul` | Zero/one short-circuits for the hot path |

This infra is **already used** in jolt-core's old pipeline:
- `SplitEqEvaluator::bind(r: F::Challenge)` — binds with challenge type directly
- `RaPolynomial::bind(r: F::Challenge, ...)` — same

### The problem

The new jolt-zkvm pipeline **doesn't use any of this**. Instead, every function
takes a `challenge_fn: impl Fn(T::Challenge) -> F` closure that converts the
raw `u128` transcript output to a full `F` immediately:

```rust
// Current: every call site
let challenge = challenge_fn(transcript.challenge());  // u128 → F
witness.bind(challenge);  // F × F = full 4×4 Montgomery multiply
```

This throws away the performance benefit of the challenge type. The old pipeline's
bind path used `F::Challenge` for a ~1.6x speedup on the hottest operation in
sumcheck (polynomial binding).

The `challenge_fn` closure is always `|c: u128| F::from(c)` — identical everywhere,
never anything else. It's pure boilerplate threaded through every function in the
stack (prove, prove_stages, verify, sumcheck prover/verifier, RLC reduction, etc.).

### Fix

**Step 1: Change `SumcheckCompute::bind` to take `F::Challenge`**

```rust
// Before
trait SumcheckCompute<F: Field> {
    fn bind(&mut self, challenge: F);
}

// After
trait SumcheckCompute<F: WithChallenge> {
    fn bind(&mut self, challenge: F::Challenge);
}
```

**Step 2: Update the sumcheck prover loop**

```rust
// Before
let challenge = challenge_fn(transcript.challenge());
handler.on_challenge(challenge);
running_claim = round_poly.evaluate(challenge);
witness.bind(challenge);

// After
let raw = F::Challenge::from(transcript.challenge());
handler.on_challenge(raw.into());        // F for handler/transcript
running_claim = round_poly.evaluate(raw.into());  // F for evaluation
witness.bind(raw);                       // Challenge for fast binding
```

**Step 3: Delete `challenge_fn` parameter from all functions**

Replace with `F: WithChallenge` (or `F: From<T::Challenge>` where full
conversion is needed). Conversion happens via `F::from(challenge)` or
`challenge.into()` at the few sites that need full field elements.

**Step 4: Update KernelEvaluator and other new evaluators**

`KernelEvaluator::bind`, `RaVirtualEvaluator::bind`, etc. update signatures
from `fn bind(&mut self, challenge: F)` to `fn bind(&mut self, challenge: F::Challenge)`.

### Performance impact

In batched sumcheck with ~50 polynomials per round, ~20 rounds per stage,
and 7 stages, that's ~7000 bind operations per proof. Each bind does
`challenge × coefficient` for every evaluation point. The 4×2 vs 4×4
Montgomery multiply gives ~1.6x speedup on this operation, which is
~1.3x speedup on the overall bind step (per jolt-field benchmarks).

### Scope

- jolt-sumcheck: `SumcheckCompute::bind`, `SumcheckProver::prove`, `BatchedSumcheckProver::prove`, `SumcheckVerifier::verify`
- jolt-openings: `RlcReduction::reduce_prover/reduce_verifier`
- jolt-verifier: `verify`, `verify_spartan`, `verify_openings`
- jolt-zkvm: `prove`, `prove_stages`, `prove_trace`, `KernelEvaluator::bind`, all stage evaluators

---

## 4. Prover/verifier factory duplication in host.rs

**Status**: TODO
**Files**: `src/host.rs`

`build_prover_stages` and `build_verifier_descriptors` are parallel functions
that both squeeze `c0, c1` from the transcript, compute `num_cycle_vars`,
slice `r_y`, and build stage structures. They must stay in sync but share no code.

**Fix**: Extract common logic into a shared helper:

```rust
struct StageParams<F> {
    c0: F,
    c1: F,
    r_cycle: Vec<F>,
}

fn derive_stage_params<F, T>(tables: &[F], r_y: &[F], transcript: &mut T) -> StageParams<F> { ... }
```

Then both `build_prover_stages` and `build_verifier_descriptors` call it.

Will become more important as more stages are wired (S2, S4-S7 will each add
parallel prover/verifier construction that must match).

---

## 5. stage_span! macro — replace with uniform span

**Status**: TODO
**Files**: `src/pipeline.rs:53-83`

The `stage_span!` macro matches on 9 hardcoded stage name strings to create
per-name `tracing::info_span!` calls. This exists because `tracing` requires
string literals for span names.

If a stage's `name()` return value changes or a new stage is added, the macro
silently falls through to the generic `"stage"` branch — no compile-time check.

**Fix**: Use a single uniform span with the stage name as a field:

```rust
let span = tracing::info_span!("sumcheck_stage", name = stage_name, claims, rounds);
```

Perfetto/Chrome traces will show the name field in the span details. The per-name
literal spans add no real value since Perfetto can filter on field values.

---

## 6. proof.rs — move JoltProvingKey, keep re-exports

**Status**: TODO
**Files**: `src/proof.rs`, `src/preprocessing.rs`

`proof.rs` is 30 lines: `JoltProvingKey` struct + re-exports from jolt-verifier.
`JoltProvingKey` is constructed only in `preprocessing.rs`.

**Fix**: Move `JoltProvingKey` to `preprocessing.rs` where it's built. Keep the
re-exports of `JoltProof`, `SumcheckStageProof`, `JoltError`, etc. since
users will import jolt-zkvm as the primary crate.

---

## 7. extract_claims boilerplate helper

**Status**: TODO
**Files**: All stage files in `src/stages/`

Every stage's `extract_claims` repeats:

```rust
let eval_point: Vec<F> = challenges.iter().rev().copied().collect();
tables.into_iter().map(|evals| {
    let poly = Polynomial::new(evals.clone());
    let eval = poly.evaluate(&eval_point);
    ProverClaim { evaluations: evals, point: eval_point.clone(), eval }
}).collect()
```

**Fix**: Add a helper in `stage.rs`:

```rust
pub fn extract_polynomial_claims<F: Field>(
    challenges: &[F],
    tables: Vec<Vec<F>>,
) -> Vec<ProverClaim<F>> { ... }
```

Saves ~15 lines per stage (8 stages = ~120 lines).

---

## 8. Make host.rs generic over transcript

**Status**: TODO
**Files**: `src/host.rs`

`prove_trace` and `verify_proof` hardcode `Blake2bTranscript::new(b"jolt-v2")`.

**Fix**: Add a transcript type parameter:

```rust
pub fn prove_trace<PCS, T: Transcript>(
    trace: &[Cycle],
    pcs_setup: impl FnOnce(usize) -> (...),
    label: &'static [u8],
) -> Result<ProveOutput<PCS::Field, PCS>, ProveError>
```

Or provide `prove_trace` as the Blake2b convenience and `prove_trace_with`
as the generic version.
