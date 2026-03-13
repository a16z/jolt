# Task 14: Jolt-optimized uniform Spartan in jolt-spartan

**Status:** Done
**Dependencies:** Task 13 (GruenSplitEqPolynomial extraction)
**Blocks:** S1 (Spartan outer sumcheck) in jolt-zkvm

## Objective

Add a **uniform Spartan** variant to `jolt-spartan` that is specifically optimized for the Jolt zkVM's R1CS constraint system. The existing `SpartanKey`/`SpartanProver` in jolt-spartan stores dense A/B/C MLEs — this is the general-purpose path. The Jolt zkVM uses a **uniform** R1CS where constraints are compile-time constants and the key stores no matrices at all. This task adds that specialized path.

## Background: How jolt-core's Spartan works

The outer sumcheck proves `Σ_x eq(τ, x) · [Az(x) · Bz(x) − u · Cz(x)] = 0` where:
- `z` is the R1CS witness vector
- `A`, `B`, `C` are the constraint matrices
- `τ` is a random evaluation point
- `u` is the relaxation factor (1 for standard, variable for relaxed/Nova)

In jolt-core, this is implemented via two phases:

1. **Univariate skip phase** (first `log(num_constraints/num_cycles)` rounds): Exploits the uniform structure — since every cycle has the same constraints, the polynomial has repeated structure that can be evaluated without materializing it. Uses `GruenSplitEqPolynomial::gruen_poly_deg_3` to compute cubic round polynomials.

2. **Streaming/linear phase** (remaining `log(num_cycles)` rounds): After the uniform structure is consumed, the remaining polynomial is evaluated via a streaming sumcheck over the execution trace. Uses `StreamingSumcheck` with `OuterStreamingWindow` + `OuterLinearStage`.

### Key types in jolt-core

| Type | Location | Role |
|------|----------|------|
| `UniformSpartanKey<F>` | `zkvm/r1cs/key.rs` | Stores `num_rows_bits`, `A_mle_indices`, `B_mle_indices`, `C_mle_indices` — sparse matrix indices, NOT dense MLEs |
| `R1CSConstraint` / `LC` / `Term` | `zkvm/r1cs/constraints.rs` | Compile-time constraint definitions |
| `R1CS_CONSTRAINTS` | `zkvm/r1cs/constraints.rs` | Static array of all RISC-V circuit constraints |
| `JoltR1CSInputs` | `zkvm/r1cs/inputs.rs` | Enum of all witness wire types (flags, registers, lookup operands, etc.) |
| `R1CSCycleInputs<F>` | `zkvm/r1cs/inputs.rs` | Per-cycle witness extraction from the execution trace |
| `R1CSEval` | `zkvm/r1cs/evaluation.rs` | Typed Az/Bz product evaluator per cycle |
| `OuterUniSkipProver` | `zkvm/spartan/outer.rs` | Univariate skip phase for the outer sumcheck |
| `OuterRemainingSumcheckParams` | `zkvm/spartan/outer.rs` | Streaming/linear phase parameters |
| `StreamingSumcheck` | `subprotocols/streaming_sumcheck.rs` | Generic streaming sumcheck engine (see below) |

## Design: What to add to jolt-spartan

### 1. Streaming sumcheck engine

Extract `StreamingSumcheck` from `jolt-core/src/subprotocols/streaming_sumcheck.rs` (~208 lines) into `jolt-sumcheck`.

**Traits to add to jolt-sumcheck:**

```rust
/// Computation for one streaming window before materialization.
pub trait StreamingSumcheckWindow<F: Field>: Send + Sync {
    type Shared;
    fn initialize(shared: &mut Self::Shared, window_size: usize) -> Self;
    fn compute_message(&self, shared: &Self::Shared, window_size: usize, previous_claim: F) -> UnivariatePoly<F>;
    fn ingest_challenge(&mut self, shared: &mut Self::Shared, r: F, round: usize);
}

/// Computation for rounds after materialization.
pub trait LinearSumcheckStage<F: Field>: Send + Sync {
    type Shared;
    type Streaming: StreamingSumcheckWindow<F, Shared = Self::Shared>;
    fn initialize(streaming: Option<Self::Streaming>, shared: &mut Self::Shared, window_size: usize) -> Self;
    fn next_window(&mut self, shared: &mut Self::Shared, window_size: usize);
    fn compute_message(&self, shared: &Self::Shared, window_size: usize, previous_claim: F) -> UnivariatePoly<F>;
    fn ingest_challenge(&mut self, shared: &mut Self::Shared, r: F, round: usize);
}
```

Plus the `StreamingSumcheck<F, Schedule, Shared, Streaming, Linear>` struct that implements `SumcheckCompute<F>` by dispatching between phases based on the schedule.

### 2. Uniform Spartan key

Add to jolt-spartan alongside the existing `SpartanKey`:

```rust
/// Sparse representation of a uniform R1CS — same constraints repeated per cycle.
pub struct UniformSpartanKey<F: Field> {
    /// log2(number of constraint rows)
    pub num_rows_bits: usize,
    /// log2(number of cycles)
    pub num_cycles_bits: usize,
    /// Sparse A matrix: for each constraint, list of (column, coefficient) pairs
    pub a_sparse: Vec<Vec<(usize, F)>>,
    /// Sparse B matrix
    pub b_sparse: Vec<Vec<(usize, F)>>,
    /// Sparse C matrix
    pub c_sparse: Vec<Vec<(usize, F)>>,
}
```

This stores the per-constraint sparse structure (not dense MLEs). The `evaluate_inner_sum_product_at_point` method evaluates `Az(r) · Bz(r) − u · Cz(r)` using the sparse representation.

### 3. Uniform outer sumcheck prover

Add a `UniformOuterProver` that implements the two-phase protocol:

```rust
pub struct UniformOuterProver<F, S, W, L>
where
    F: Field,
    S: StreamingSchedule,
    W: StreamingSumcheckWindow<F>,
    L: LinearSumcheckStage<F, Streaming = W>,
{
    key: UniformSpartanKey<F>,
    streaming_sumcheck: StreamingSumcheck<F, S, W::Shared, W, L>,
}
```

The concrete `Window` and `LinearStage` types are parameterized — jolt-zkvm provides them (they depend on the RISC-V witness structure).

### 4. Streaming schedule

```rust
/// Controls when the streaming sumcheck switches from streaming to linear phase.
pub trait StreamingSchedule {
    fn num_rounds(&self) -> usize;
    fn switch_over_point(&self) -> usize;
    fn window_size(&self, round: usize) -> usize;
    fn is_window_start(&self, round: usize) -> bool;
}
```

## What stays in jolt-zkvm (NOT in jolt-spartan)

The following are RISC-V-specific and belong in jolt-zkvm, not jolt-spartan:

- `R1CS_CONSTRAINTS` / `R1CSConstraint` / `LC` / `Term` — the concrete RISC-V circuit definition
- `JoltR1CSInputs` / `R1CSCycleInputs` — RISC-V witness wire types
- `R1CSEval` — per-cycle Az/Bz evaluator using the RISC-V witness
- `OuterStreamingWindow` / `OuterLinearStage` — concrete streaming/linear implementations that read the RISC-V execution trace
- `BytecodePreprocessing` — PC mapping

The boundary is: **jolt-spartan provides the protocol skeleton and key types; jolt-zkvm provides the RISC-V-specific witness evaluation via trait implementations.**

## File plan

| File | Action |
|------|--------|
| `crates/jolt-sumcheck/src/streaming.rs` | **NEW** — `StreamingSumcheckWindow`, `LinearSumcheckStage`, `StreamingSumcheck`, `StreamingSchedule` |
| `crates/jolt-spartan/src/uniform_key.rs` | **NEW** — `UniformSpartanKey<F>` |
| `crates/jolt-spartan/src/uniform_prover.rs` | **NEW** — `UniformOuterProver` using streaming sumcheck |
| `crates/jolt-spartan/src/uniform_verifier.rs` | **NEW** — verifier for uniform Spartan |
| `crates/jolt-spartan/src/lib.rs` | **MODIFY** — add `pub mod uniform_*` |
| `crates/jolt-sumcheck/src/lib.rs` | **MODIFY** — add `pub mod streaming` |

## Verification

```bash
# jolt-sumcheck compiles with streaming types
cargo clippy -p jolt-sumcheck --message-format=short -q --all-targets -- -D warnings

# jolt-spartan compiles with uniform key + prover
cargo clippy -p jolt-spartan --message-format=short -q --all-targets -- -D warnings

# Existing tests pass
cargo nextest run -p jolt-sumcheck --cargo-quiet
cargo nextest run -p jolt-spartan --cargo-quiet
```

## Testing strategy

Unit tests for the uniform key and prover can use small toy R1CS circuits (2-3 constraints, 4-8 cycles) to verify:

1. `UniformSpartanKey::evaluate_inner_sum_product_at_point` matches brute-force Az·Bz − u·Cz
2. The streaming sumcheck round polynomials match the non-streaming (dense) version
3. Full prove/verify round-trip with the uniform prover

Integration with the actual Jolt RISC-V circuit happens later when jolt-zkvm wires S1 to use the uniform prover.

## Reference files

| File | Lines | Notes |
|------|-------|-------|
| `jolt-core/src/subprotocols/streaming_sumcheck.rs` | 208 | → `jolt-sumcheck/src/streaming.rs` |
| `jolt-core/src/zkvm/r1cs/key.rs` | ~200 | → `jolt-spartan/src/uniform_key.rs` |
| `jolt-core/src/zkvm/spartan/outer.rs` | ~800 | Protocol logic → `jolt-spartan`; RISC-V witness → stays in jolt-zkvm |
| `jolt-core/src/subprotocols/univariate_skip.rs` | ~400 | Univariate skip logic used by outer prover; may need partial extraction |
