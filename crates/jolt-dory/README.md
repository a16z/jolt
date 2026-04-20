# jolt-dory

Dory commitment scheme implementation for the Jolt zkVM.

Part of the [Jolt](https://github.com/a16z/jolt) zkVM.

## Overview

Wraps the [Dory](https://eprint.iacr.org/2020/1274) polynomial commitment scheme for BN254 with transparent setup, logarithmic proof size, and logarithmic verification. Supports streaming commitment and additive homomorphism for batch opening reduction.

Implements `CommitmentScheme`, `AdditivelyHomomorphic`, `StreamingCommitment`, and `ZkOpeningScheme` from `jolt-openings`.

## Public API

- **`DoryScheme`** — Implements all four PCS traits. Static methods: `setup_prover`, `setup_verifier`, `extract_pedersen_setup`.
- **`DoryCommitment`** — BN254 pairing target element (GT).
- **`DoryProof`** — Single opening proof.
- **`DoryProverSetup`** / **`DoryVerifierSetup`** — Prover and verifier SRS.
- **`DoryPartialCommitment`** — Intermediate state for streaming commitment.
- **`DoryHint`** — Row commitments reusable as opening proof hint.

## Dependency Position

```
jolt-field ─┐
jolt-poly  ─┤
jolt-transcript ─┼─► jolt-dory
jolt-crypto ─┤
jolt-openings ─┘
```

## License

MIT
