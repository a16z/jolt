# jolt-sumcheck

`jolt-sumcheck` provides generic clear sumcheck proving and verification without
selecting a curve, polynomial commitment scheme, stock transcript, or thread
pool. Its default feature set is empty.

## Features

- `parallel`: parallel polynomial operations.
- `transcript-blake2b`, `transcript-keccak`, `transcript-poseidon`: stock
  transcript implementations. Poseidon selects BN254 because that sponge is
  field-specific.
- `committed`: commitment-backed round proving. It exposes the generic
  `VectorCommitment` boundary without selecting a curve backend.
- `r1cs`: lowering of sumcheck verifier equations into the dependency-light
  R1CS builder.

For a custom transcript, implement `jolt_transcript::Transcript`. Scalar types
used with the stock proof and prover types implement `jolt_field::JoltField`;
the verifier's custom-round surface uses the smaller `SumcheckScalar` bundle.
After verification, the caller must check the returned `EvaluationClaim`
against its polynomial oracle or commitment. This includes zero-variable claims.

Run the minimal example with:

```bash
cargo run -p jolt-sumcheck --example clear_custom_transcript --no-default-features
```

The fixed transcript in that example makes the algebraic API easy to inspect;
it is not a production Fiat–Shamir transcript. Applications must use a
cryptographic transcript with adequate challenge entropy.

The isolated fixture is checked outside Cargo's Jolt workspace feature
unification with:

```bash
scripts/check-external-sumcheck.sh
```

An external project should pin all Jolt packages to the same full Git revision:

```toml
[dependencies]
jolt-sumcheck = { git = "https://github.com/a16z/jolt", rev = "FULL_COMMIT_SHA", default-features = false }
jolt-field = { git = "https://github.com/a16z/jolt", rev = "FULL_COMMIT_SHA", default-features = false, features = ["solinas"] }
jolt-poly = { git = "https://github.com/a16z/jolt", rev = "FULL_COMMIT_SHA", default-features = false }
jolt-transcript = { git = "https://github.com/a16z/jolt", rev = "FULL_COMMIT_SHA", default-features = false }
```

This branch's external-consumer validation used implementation revision
`5a01f83ce8da90afc942be8af7256fcd5e627fd0`. Use a reachable revision from the
upstream repository after the branch is pushed.

Cargo only honors `[patch]` tables in the consuming workspace root. The minimal
configuration above does not require Jolt's Arkworks patches. BN254 and Poseidon
consumers must copy the exact Arkworks patch/source requirements documented by
the pinned Jolt revision into their root manifest.
