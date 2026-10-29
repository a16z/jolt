# jolt-sumcheck

`jolt-sumcheck` provides generic clear sumcheck proving and verification without
selecting a curve, polynomial commitment scheme, stock transcript, or thread
pool. Its default feature set is empty.

## Features

- `parallel`: parallel polynomial operations.
- `prover`: the prove side — the batched round loop and the recorders'
  rounds. Verifier builds leave it off.
- `transcript-blake2b`, `transcript-keccak`: the sponges a transcript can draw
  field challenges from. Keccak is jolt-transcript's default.
- `committed`: commitment-backed round proving. It exposes the generic
  `VectorCommitment` boundary without selecting a curve backend.
- `r1cs`: lowering of sumcheck verifier equations into the dependency-light
  R1CS builder.

Polynomial storage, compression, round generation, and the shared prover and
verifier engines require `jolt_field::Field`. The stock clear recorder and
proof-verification conveniences additionally require spongefish's
`Encoding`, because they absorb field elements. Challenges come from
`jolt_transcript::FieldTranscript`, which spongefish's prover and verifier
states implement through the field's `ChallengeCodec`. Jolt's own
optimized kernels and commitment-backed modes may require the broader
`JoltField` bundle at their integration boundary.

After verification, the caller must check the returned `EvaluationClaim`
against its polynomial oracle or commitment. This includes zero-variable claims.

Run the minimal example with:

```bash
cargo run -p jolt-sumcheck --example clear_sumcheck --features prover
```

The example proves and verifies a one-round clear sumcheck over a Keccak
transcript.

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
jolt-sumcheck = { git = "https://github.com/a16z/jolt", rev = "FULL_COMMIT_SHA", default-features = false, features = ["prover"] }
jolt-field = { git = "https://github.com/a16z/jolt", rev = "FULL_COMMIT_SHA", default-features = false, features = ["solinas"] }
jolt-poly = { git = "https://github.com/a16z/jolt", rev = "FULL_COMMIT_SHA", default-features = false }
jolt-transcript = { git = "https://github.com/a16z/jolt", rev = "FULL_COMMIT_SHA", default-features = false, features = ["solinas", "transcript-keccak"] }
```

Cargo only honors `[patch]` tables in the consuming workspace root. The minimal
configuration above does not require Jolt's Arkworks patches. BN254
consumers must copy the exact Arkworks patch/source requirements documented by
the pinned Jolt revision into their root manifest.
