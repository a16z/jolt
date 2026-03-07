# Task 10: Verifier R1CS Construction

**Status:** DONE
**Phase:** BlindFold Layer 2
**Dependencies:** jolt-ir (R1CS backend), jolt-spartan (R1CS trait)
**Blocks:** Task 12 (BlindFold protocol)

## Objective

Build the verifier R1CS that encodes all deferred sumcheck consistency checks. In committed sumcheck mode, the verifier cannot evaluate round polynomials (only commitments are visible), so the checks $\text{poly}(0) + \text{poly}(1) = \text{running\_sum}$ and $\text{poly}(r_i) = \text{next\_running\_sum}$ are deferred. This task constructs an R1CS that encodes ALL deferred checks across all sumcheck stages, to be proved via Spartan.

## Context

Each sumcheck stage has $n$ rounds. Each round has a round polynomial of degree $d$. The deferred checks per round are:

1. **Running sum check:** $c_0 + c_1 = s_{\text{prev}}$ (coefficients at 0 and 1 sum to previous running sum)
2. **Challenge evaluation:** $\text{poly}(r_i) = s_{\text{next}}$ (evaluating the polynomial at the Fiat-Shamir challenge gives the next running sum)

For a degree-$d$ polynomial with $d+1$ coefficients $c_0, \ldots, c_d$:
- $\text{poly}(0) = c_0$
- $\text{poly}(1) = c_0 + c_1 + \ldots + c_d$
- $\text{poly}(r) = c_0 + c_1 r + c_2 r^2 + \ldots + c_d r^d$

The Fiat-Shamir challenges are **baked into the R1CS matrix coefficients** (not stored as public inputs). This means the R1CS depends on the specific transcript, matching jolt-core's `BakedPublicInputs` pattern.

Reference: `jolt-core/src/subprotocols/blindfold/r1cs.rs`

## Deliverables

### Types

```rust
/// Describes one sumcheck stage's shape for R1CS construction.
pub struct StageConfig<F: Field> {
    /// Number of sumcheck rounds in this stage.
    pub num_rounds: usize,
    /// Degree of the round polynomial.
    pub degree: usize,
    /// Claimed sum (input to the first round).
    pub claimed_sum: F,
}

/// Fiat-Shamir challenges baked into R1CS matrix coefficients.
pub struct BakedPublicInputs<F: Field> {
    /// All challenges across all stages, flattened.
    /// Stage $i$, round $j$ → `challenges[stage_offsets[i] + j]`.
    pub challenges: Vec<F>,
}
```

### R1CS construction

```rust
/// Builds verifier R1CS encoding all deferred sumcheck checks.
///
/// Uses jolt-ir's R1CS backend to emit constraints. Each round
/// produces constraints for the running sum check and challenge
/// evaluation. The challenges are baked into matrix coefficients.
pub fn build_verifier_r1cs<F: Field>(
    stages: &[StageConfig<F>],
    baked_inputs: &BakedPublicInputs<F>,
) -> R1csEmission<F>;
```

### Witness assignment

```rust
/// Materializes the R1CS witness from accumulated prover data.
///
/// The witness contains the round polynomial coefficients from each
/// stage, in order. The prover knows these values (stored in
/// `CommittedRoundData`); the verifier does not.
pub fn assign_witness<F: Field, VC: JoltCommitment>(
    stages: &[StageConfig<F>],
    round_data: &[CommittedRoundData<F, VC>],
) -> Vec<F>;
```

### Key design

Uses `jolt-ir` `ExprBuilder` to author verification constraints:

```rust
let b = ExprBuilder::new();
// For each round: c_0 + c_1 + ... + c_d == running_sum
// This becomes a constraint with baked coefficients (the challenge powers)
```

Each constraint is expressed as an `Expr`, normalized to sum-of-products, and emitted via `R1csEmitter`. This gives us:
- Correctness by construction (jolt-ir handles normalization)
- Clean code (no hand-written sparse matrix entries)
- Automatic compatibility with the Lean4 and circuit backends

## Testing

- **Small case:** Single stage, 2 rounds, degree 1 — manually compute the correct witness, verify $Az \circ Bz = Cz$
- **Multi-stage:** Two stages with different round counts and degrees
- **Negative:** Tamper with one coefficient in the witness → constraint violation
- **Consistency:** Build R1CS, assign witness from a real committed sumcheck output (from jolt-blindfold Layer 1), verify satisfaction via `SimpleR1CS::multiply_witness`

## Files

| File | Change |
|------|--------|
| `jolt-blindfold/src/verifier_r1cs.rs` | New: `StageConfig`, `BakedPublicInputs`, `build_verifier_r1cs`, `assign_witness` |
| `jolt-blindfold/src/error.rs` | New: `BlindFoldError` enum |
| `jolt-blindfold/Cargo.toml` | Add `jolt-ir`, `jolt-spartan` dependencies |

## Reference

- `jolt-core/src/subprotocols/blindfold/r1cs.rs` — `VerifierR1CS`, `VerifierR1CSBuilder`
- `jolt-core/src/subprotocols/blindfold/mod.rs` — `BakedPublicInputs`
- `jolt-ir/src/backends/r1cs.rs` — `R1csEmitter`, `R1csEmission`
