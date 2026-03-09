# jolt-blindfold

ZK layer for Jolt sumcheck: committed round handlers and BlindFold accumulator.

## Purpose

This crate makes sumcheck proofs zero-knowledge by replacing cleartext polynomial
absorption with commitment-based absorption. Instead of appending round polynomial
coefficients to the Fiat-Shamir transcript (which would leak the witness), the
prover commits to them via a vector commitment scheme and appends only the
commitment. The actual consistency checks (`poly(0) + poly(1) == running_sum`)
are deferred to the BlindFold protocol, which encodes them into a verifier R1CS
proved via Nova folding and Spartan.

## Public API

### Committed Sumcheck

- **`CommittedRoundHandler`** -- Prover-side `RoundHandler`: commits coefficients, stores blindings.
- **`CommittedRoundVerifier`** -- Verifier-side `RoundVerifier`: absorbs commitments, defers checks.
- **`CommittedSumcheckProof`** -- Public proof (commitments only, no coefficients).
- **`CommittedRoundData`** -- Private prover data (coefficients + blindings) for BlindFold.
- **`CommittedSumcheckOutput`** -- Combined proof + round data returned from `finalize()`.

### BlindFold Protocol

- **`BlindFoldAccumulator`** -- Collects `CommittedRoundData` across all sumcheck stages.
- **`BlindFoldProver`** / **`BlindFoldVerifier`** -- Full BlindFold protocol (Nova folding + Spartan over verifier R1CS).
- **`BlindFoldProof`** -- Serializable proof artifact.
- **`StageConfig`** -- Per-stage configuration for the verifier R1CS.
- **`BakedPublicInputs`** -- Fiat-Shamir-derived values baked into R1CS matrix coefficients.

### Nova Folding

- **`RelaxedInstance`** / **`RelaxedWitness`** -- Relaxed R1CS instance and witness.
- **`fold_instances`** / **`fold_witnesses`** / **`fold_scalar`** -- Instance/witness folding operations.
- **`compute_cross_term`** -- Cross-term computation for folding.
- **`sample_random_witness`** -- Random satisfying instance generation.

All types are generic over `JoltCommitment` -- Pedersen, hash-based, or
lattice-based commitment schemes can be substituted.

## Feature Flags

No feature flags. The crate is unconditionally compiled.

## Dependency Position

```
jolt-blindfold
  +-- jolt-field        (scalar field)
  +-- jolt-poly         (UnivariatePoly)
  +-- jolt-transcript   (Fiat-Shamir)
  +-- jolt-sumcheck     (RoundHandler / RoundVerifier traits)
  +-- jolt-crypto       (JoltCommitment, Pedersen)
  +-- jolt-spartan      (Spartan prover/verifier for BlindFold R1CS)
  +-- jolt-openings     (CommitmentScheme traits)
```

## License

MIT
