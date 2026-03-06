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

| Type | Role |
|------|------|
| `CommittedRoundHandler` | Prover-side `RoundHandler` — commits coefficients, stores blindings |
| `CommittedRoundVerifier` | Verifier-side `RoundVerifier` — absorbs commitments, defers checks |
| `CommittedSumcheckProof` | Public proof (commitments only, no coefficients) |
| `CommittedRoundData` | Private prover data (coefficients + blindings) for BlindFold |
| `CommittedSumcheckOutput` | Combined proof + round data returned from `finalize()` |
| `BlindFoldAccumulator` | Collects `CommittedRoundData` across all sumcheck stages |
| `ZkStageData` | Per-stage wrapper stored in the accumulator |

All types are generic over `JoltCommitment` — Pedersen, hash-based, or
lattice-based commitment schemes can be substituted.

## Feature Flags

No feature flags. The crate is unconditionally compiled.

## Dependency Position

```
jolt-blindfold
  ├── jolt-field        (scalar field)
  ├── jolt-poly         (UnivariatePoly)
  ├── jolt-transcript   (Fiat-Shamir)
  ├── jolt-sumcheck     (RoundHandler / RoundVerifier traits)
  └── jolt-crypto       (JoltCommitment, Pedersen)
```

## Usage

```rust
use jolt_blindfold::{CommittedRoundHandler, CommittedRoundVerifier, BlindFoldAccumulator};
use jolt_sumcheck::{SumcheckProver, SumcheckVerifier};

// Prover: create a committed handler and run sumcheck
let handler = CommittedRoundHandler::<F, VC, _>::new(&setup, &mut rng);
let output = SumcheckProver::prove_with_handler(&claim, &mut witness, &mut transcript, cast, handler);

// Send output.proof to verifier; keep output.round_data private
accumulator.push_stage(output.round_data);

// Verifier: replay with committed verifier
let verifier = CommittedRoundVerifier::<VC>::new();
SumcheckVerifier::verify_with_handler(&claim, &proof.round_commitments, &mut transcript, cast, &verifier);
```
