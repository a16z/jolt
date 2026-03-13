# Task 11: Nova Folding

**Status:** DONE
**Phase:** BlindFold Layer 3
**Dependencies:** Task 09 (relaxed R1CS in jolt-spartan), Task 10 (verifier R1CS)
**Blocks:** Task 12 (BlindFold protocol)

## Objective

Implement Nova folding for relaxed R1CS instances. Nova folding combines a real instance $(u_1, W_1, E_1)$ with a random satisfying instance $(u_2, W_2, E_2)$ via a cross-term $T$. The folded instance hides the real witness (one-time pad), enabling zero-knowledge.

## Context

The BlindFold protocol constructs a verifier R1CS (Task 10) and materializes a witness. The prover has a real instance/witness pair that satisfies this R1CS. To hide the witness, Nova folds it with a random satisfying instance:

1. Prover generates a random satisfying instance $(u_2, W_2, E_2)$
2. Prover computes the cross-term $T$ and commits to it
3. Verifier sends a challenge $r$
4. Both sides compute the folded instance: $u' = u_1 + r u_2$, $W' = W_1 + r W_2$, $E' = E_1 + r T + r^2 E_2$
5. Prover computes the folded witness (can evaluate the folded R1CS)

The folded instance satisfies the relaxed R1CS (Task 09's `RelaxedR1CS`).

Reference: `jolt-core/src/subprotocols/blindfold/folding.rs`

## Deliverables

### Types

```rust
/// Public instance for a relaxed R1CS.
///
/// Generic over `VC: JoltCommitment` — commitments to W and E
/// use the same vector commitment scheme as the round polynomials.
pub struct RelaxedInstance<F: Field, VC: JoltCommitment> {
    /// Relaxation scalar.
    pub u: F,
    /// Commitment to the witness vector.
    pub w_commitment: VC::Commitment,
    /// Commitment to the error vector.
    pub e_commitment: VC::Commitment,
    /// Public inputs (if any).
    pub public_inputs: Vec<F>,
}

/// Private witness for a relaxed R1CS.
pub struct RelaxedWitness<F: Field> {
    /// Witness vector.
    pub w: Vec<F>,
    /// Error vector.
    pub e: Vec<F>,
}
```

### Cross-term computation

```rust
/// Computes the Nova cross-term $T$ from two instance/witness pairs.
///
/// $$T = Az_1 \circ Bz_2 + Az_2 \circ Bz_1 - u_1 Cz_2 - u_2 Cz_1$$
///
/// where $z_i$ is the full assignment (public inputs ∥ witness).
pub fn compute_cross_term<F: Field>(
    r1cs: &impl R1CS<F>,
    w1: &RelaxedWitness<F>,
    w2: &RelaxedWitness<F>,
    u1: F,
    u2: F,
    public_inputs_1: &[F],
    public_inputs_2: &[F],
) -> Vec<F>;
```

### Folding

```rust
/// Folds two relaxed instances into one.
///
/// $$u' = u_1 + r \cdot u_2$$
/// $$\text{com}_{W'} = \text{com}_{W_1} + r \cdot \text{com}_{W_2}$$
/// $$\text{com}_{E'} = \text{com}_{E_1} + r \cdot \text{com}_T + r^2 \cdot \text{com}_{E_2}$$
///
/// Requires `VC::Commitment: JoltGroup` for additive homomorphism.
pub fn fold_instances<F: Field, VC: JoltCommitment>(
    inst1: &RelaxedInstance<F, VC>,
    inst2: &RelaxedInstance<F, VC>,
    t_commitment: VC::Commitment,
    challenge: F,
) -> RelaxedInstance<F, VC>
where
    VC::Commitment: JoltGroup;

/// Folds two relaxed witnesses into one.
///
/// $$W' = W_1 + r \cdot W_2$$
/// $$E' = E_1 + r \cdot T + r^2 \cdot E_2$$
pub fn fold_witnesses<F: Field>(
    wit1: &RelaxedWitness<F>,
    wit2: &RelaxedWitness<F>,
    cross_term: &[F],
    challenge: F,
) -> RelaxedWitness<F>;
```

### Random instance generation

```rust
/// Generates a random satisfying instance for the one-time pad.
///
/// 1. Sample random witness $W_2$
/// 2. Compute $Az_2, Bz_2, Cz_2$
/// 3. Set $u_2 = 1$ (or random non-zero)
/// 4. Compute $E_2 = Az_2 \circ Bz_2 - u_2 \cdot Cz_2$ (error that makes it satisfy)
/// 5. Commit to $W_2$ and $E_2$
pub fn sample_random_instance<F: Field, VC: JoltCommitment>(
    r1cs: &impl R1CS<F>,
    num_public_inputs: usize,
    vc_setup: &VC::Setup,
    rng: &mut impl CryptoRngCore,
) -> (RelaxedInstance<F, VC>, RelaxedWitness<F>)
where
    VC::Commitment: JoltGroup;
```

### Key constraint

`VC::Commitment: JoltGroup` — the additive group structure on commitments is needed for `fold_instances` to compute linear combinations of commitments. This is the key generic constraint that enables commitment-agnostic folding.

## Testing

- **Correctness:** Fold two known satisfying instances, verify the result satisfies the relaxed R1CS
- **Identity:** Folding with `r = 0` returns the first instance unchanged
- **Cross-term:** Verify $T$ satisfies the algebraic identity from the Nova paper
- **Random instance:** Verify `sample_random_instance` produces a satisfying pair
- **End-to-end:** Build verifier R1CS (Task 10), assign witness, create real relaxed instance, generate random instance, fold, verify folded instance via relaxed Spartan (Task 09)

## Files

| File | Change |
|------|--------|
| `jolt-blindfold/src/folding.rs` | New: all folding types and functions |
| `jolt-blindfold/Cargo.toml` | Ensure `jolt-crypto` dependency (for `JoltGroup`) |

## Reference

- `jolt-core/src/subprotocols/blindfold/folding.rs` — existing Nova folding
- Nova paper §4 (folding scheme)
- `jolt-crypto/src/group.rs` — `JoltGroup` trait (additive notation)
