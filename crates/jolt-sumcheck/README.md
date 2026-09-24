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

Polynomial storage, compression, round generation, and the shared prover and
verifier engines require `jolt_field::Field`. The stock clear recorder and
proof-verification conveniences additionally require
`jolt_transcript::AppendToTranscript`, because they absorb field elements.
Implement `jolt_transcript::Transcript` to choose challenge generation; stock
hash transcripts retain their `CanonicalEncoding` requirement. Jolt's own
optimized kernels and commitment-backed modes may require the broader
`JoltField` bundle at their integration boundary.

After verification, the caller must check the returned `EvaluationClaim`
against its polynomial oracle or commitment. This includes zero-variable claims.

Run the minimal example with:

```bash
cargo run -p jolt-sumcheck --example clear_custom_transcript --no-default-features
```

The example transcript tracks every absorb but returns fixed challenges, which
makes the Fiat–Shamir schedule and algebraic API easy to inspect. It is not a
production transcript; applications must use a cryptographic transcript with
adequate challenge entropy.

The isolated fixture runs outside Cargo's Jolt workspace feature unification:

```bash
cargo nextest run \
  --manifest-path crates/jolt-sumcheck/tests/external-consumer/Cargo.toml \
  --locked
```

Check its minimal dependency boundary separately with:

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

Cargo only honors `[patch]` tables in the consuming workspace root. The minimal
configuration above does not require Jolt's Arkworks patches. BN254 and Poseidon
consumers must copy the exact Arkworks patch/source requirements documented by
the pinned Jolt revision into their root manifest.
