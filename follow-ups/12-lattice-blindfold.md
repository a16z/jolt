# Lattice BlindFold

## What

A variant of BlindFold that uses lattice-based commitments (additively homomorphic over Z_q) instead of Pedersen commitments over elliptic curves. This is the ZK layer for Hachi (lattice Jolt).

## Why

BlindFold's core mechanism is:
1. Prover commits to sumcheck round polynomial coefficients (using an additively homomorphic scheme)
2. The verifier's sumcheck checks are encoded as R1CS constraints
3. Nova folds the real instance with a random instance (using the homomorphism)
4. Spartan proves the folded R1CS

Steps 1 and 3 require additive homomorphism. Pedersen over elliptic curves provides this, but is not post-quantum. Lattice commitments (e.g., based on RLWE or MLWE) are additively homomorphic AND post-quantum.

The protocol structure is identical — only the commitment operations change.

## Scope

**What stays the same:**
- R1CS derivation from Module (02) — same constraints
- Nova folding logic — same cross-term computation, same relaxed R1CS structure
- Spartan outer/inner sumcheck — same protocol
- Witness assignment — same variables

**What changes:**
- `PedersenCommitment<G>` → `LatticeCommitment` — different commitment type, different operations
- Commitment key: EC generator points → lattice public matrix
- Homomorphic addition: EC point addition → vector addition mod q
- Scalar multiplication: EC scalar mul → matrix-vector product mod q
- Commitment opening: discrete log relation → short vector (SIS) relation
- Verification equation: pairing check → lattice norm check

**Abstraction:** BlindFold should be generic over a `HomomorphicCommitment` trait:

```rust
trait HomomorphicCommitment {
    type Commitment;
    type Key;
    type Opening;
    
    fn commit(key: &Self::Key, values: &[F], randomness: &Self::Opening) -> Self::Commitment;
    fn add(a: &Self::Commitment, b: &Self::Commitment) -> Self::Commitment;
    fn scale(c: &Self::Commitment, scalar: F) -> Self::Commitment;
    fn verify(key: &Self::Key, commitment: &Self::Commitment, values: &[F], opening: &Self::Opening) -> bool;
}
```

Pedersen and Lattice both implement this. BlindFold is generic over it.

## How It Fits

`jolt-blindfold` is generic over the commitment scheme. The Hachi instantiation provides the lattice commitment. The standard instantiation provides Pedersen.

```rust
// Standard Jolt
BlindFoldProver::<PedersenCommitment<Bn254>>::prove(...)

// Hachi
BlindFoldProver::<LatticeCommitment>::prove(...)
```

## Dependencies

- BlindFold rewrite (04) — must be generic over commitment scheme
- Hachi field + PCS (11) — the lattice commitment shares parameters with the lattice PCS

## Unblocks

- Post-quantum ZK Jolt (complete Hachi pipeline)

## Open Questions

- **Parameter sharing:** The lattice PCS and lattice BlindFold both use lattice parameters (dimension, modulus). Should they share a common parameter set, or are their requirements different?
- **Commitment size:** Lattice commitments are vectors, not single group elements. This increases the size of BlindFold proofs (each sumcheck round has a commitment). What's the overhead?
- **Soundness:** The SIS assumption underlying lattice commitment opening has different tightness than discrete log. How does this affect BlindFold's overall soundness guarantee?
- **Folding correctness:** Nova folding relies on the homomorphism being "exact" (no error growth). Lattice commitments have error terms that grow with operations. Does error accumulation across folding steps affect correctness? Is there a bound on the number of folds?
