# Task 09: Relaxed R1CS in jolt-spartan

**Status:** Pending
**Phase:** BlindFold Layer 0 (prerequisite)
**Dependencies:** None (self-contained in jolt-spartan)
**Blocks:** Task 11 (Nova folding), Task 12 (BlindFold protocol)

## Objective

Add relaxed R1CS support to jolt-spartan. Relaxed R1CS extends the standard constraint system with a relaxation scalar $u$ and error vector $E$:

$$Az \circ Bz = u \cdot Cz + E$$

Standard R1CS is the special case $u = 1, E = \mathbf{0}$. Nova folding produces relaxed instances, so this is a prerequisite for the BlindFold protocol.

## Context

jolt-spartan currently implements standard R1CS proving: the `R1CS<F>` trait defines `num_constraints()`, `num_variables()`, and `multiply_witness()`. The prover runs an outer sumcheck ($\sum_x eq(x,\tau) \cdot (Az \cdot Bz - Cz)$) and an inner sumcheck for the witness evaluation. Relaxed R1CS modifies the outer summation to include $u$ and $E$.

Reference: `jolt-core/src/subprotocols/blindfold/relaxed_r1cs.rs`

## Deliverables

### `RelaxedR1CS<F>` trait

```rust
/// Extension of `R1CS` for relaxed constraint systems.
///
/// The relaxed system satisfies $Az \circ Bz = u \cdot Cz + E$ where
/// $u$ is a scalar and $E$ is the error vector.
pub trait RelaxedR1CS<F: Field>: R1CS<F> {
    /// Relaxation scalar ($u = 1$ for standard R1CS).
    fn relaxation_scalar(&self) -> F;

    /// Error vector evaluations over the Boolean hypercube.
    /// Length must equal `num_constraints()` (padded to power of 2).
    fn error_vector(&self) -> &[F];
}
```

### `SimpleRelaxedR1CS<F>` concrete type

A test-friendly implementation wrapping `SimpleR1CS` with explicit `u` and `E` fields.

### Relaxed prover/verifier

- `SpartanProver::prove_relaxed` — outer sumcheck sums $\sum_x eq(x,\tau) \cdot (Az(x) \cdot Bz(x) - u \cdot Cz(x) - E(x))$
- `SpartanVerifier::verify_relaxed` — checks the relaxed relation
- Inner sumcheck: the witness polynomial evaluation must account for the error vector contribution

### Modified summation

The outer sumcheck now proves:
$$\sum_x \widetilde{eq}(x, \tau) \cdot \left(\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - u \cdot \widetilde{Cz}(x) - \widetilde{E}(x)\right) = 0$$

This adds two more polynomial evaluations per round (the $u \cdot Cz$ and $E$ terms).

### Error handling

Add `SpartanError::RelaxedConstraintViolation` variant.

## Testing

- **Degenerate case:** `u = 1, E = 0` must produce identical proofs to standard `SpartanProver::prove`
- **Non-trivial relaxation:** Manually construct a relaxed instance with $u \neq 1$ and non-zero $E$ such that the relation holds, prove and verify
- **Rejection:** Tamper with $u$ or $E$ → verification fails
- **Property:** For any valid standard R1CS proof, wrapping it with `u=1, E=0` as relaxed and re-proving must succeed

## Files

| File | Change |
|------|--------|
| `jolt-spartan/src/r1cs.rs` | Add `RelaxedR1CS` trait, `SimpleRelaxedR1CS` impl |
| `jolt-spartan/src/prover.rs` | Add `prove_relaxed` with modified outer summation |
| `jolt-spartan/src/verifier.rs` | Add `verify_relaxed` with modified checks |
| `jolt-spartan/src/error.rs` | Add `RelaxedConstraintViolation` variant |
| `jolt-spartan/src/lib.rs` | Re-exports, tests |

## Reference

- `jolt-core/src/subprotocols/blindfold/relaxed_r1cs.rs` — existing relaxed R1CS types
- `jolt-core/src/subprotocols/blindfold/spartan.rs` — relaxed Spartan prover/verifier
- Nova paper §3 (relaxed R1CS definition)
