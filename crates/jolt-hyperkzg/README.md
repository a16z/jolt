# Clear binary HyperKZG

This crate implements ordinary BN254 multilinear commitment/opening through
`jolt_openings::CommitmentScheme`. The standalone public API is its first
external contract; the planned generic Spartan prover/verifier will consume
it. There is no hiding, committed-round protocol, or batch/homomorphic API.

## Statement and setup contract

An honest commitment treats the high-to-low multilinear evaluation table as
univariate coefficients against imported G1 powers. Tables have `2^ell`
entries with `ell >= 1`; constant polynomials must be represented by a
two-entry constant table. Queries have exactly `ell` coordinates.

`CommitmentScheme::setup(HyperKZGSetupParams)` imports typed BN254 G1 powers,
G2, beta-G2, an authenticated setup-policy ID, and `max_public_degree`.
It checks the imported capacity, nonidentity powers, and that the declared
public degree covers the imported powers. It **does not** validate progression
of the powers, ceremony provenance, or secrecy of the trapdoor. Those are
trusted importer preconditions. Production applications must authenticate
both key material and policy; setup must not be taken from the proof.
No production function generates or accepts a secret scalar.

BN254 point deserialization delegates to `jolt-crypto`'s checked canonical
point decoding, including subgroup validation. Applications own bounded
container decoding and rejection of trailing serialized container bytes.
Identity proof points are allowed because zero polynomials are legitimate;
identity setup powers are rejected.

Imported capacity is an honest-prover/API bound. `max_public_degree` is the
authenticated bound on **all** public powers under the same trapdoor, not
only the imported slice. Local truncation does not lower this adversarial
bound. It is transcript-bound policy, not a claim verified by a pairing.

The conditional extraction contract needed by a consumer is: a polynomial
of degree at most the full public bound is fixed when its commitment is
made; later valid openings refer to that same polynomial. Its first `2^ell`
coefficients determine the extracted multilinear vector. Do not infer the
stronger equality between the original commitment and a commitment to the
truncated vector. For example, with powers through degree three, the
polynomial `X^3` can pass an arity-one opening at zero with value zero while
its projected vector is `[0,0]`.

For Spartan, the extracted vector must be fixed before its witness-dependent
challenges. Mapping this conditional statement to a complete cryptographic
extraction and Fiat–Shamir composition theorem remains a review gate. This
implementation does not certify that theorem or a production ceremony.

## Source-to-code map

The clear binary construction is adapted from Jolt donor
`fd2a7a3996ed34635dc0c8a3d0337d1024b1db6a`,
`crates/jolt-hyperkzg/src/{scheme,kzg}.rs`, following
[Gemini §2.4.2](https://eprint.iacr.org/2022/420).

| Relation / invariant | Implementation | Evidence |
|---|---|---|
| Coefficient commitment | `HyperKZGProverSetup::commit_coefficients` | Known-answer commitment for coefficients `[1,2,3,4]` |
| Suffix binary fold; last fold equals claimed value | `HyperKZGScheme::open_table`, `verify_opening` | Independent multilinear evaluation oracle at arities 1–6; false claims rejected |
| Fold relation `2r P_next(r²) = r(1-x)(P(r)+P(-r)) + x(P(r)-P(-r))` | `verify_opening` | All transmitted evaluation/fold components individually tampered |
| Univariate synthetic division and KZG batching | `kzg.rs` | Explicit quotient coefficients; wrong statement, witness, and key rejection |
| Exact shape and nonzero fold challenge | `check_arity`, `verify_opening` | Malformed lengths and out-of-range arities; challenge rejection implemented |
| Commitment/query/claim fixed before opening challenges | `append_statement` | Statement/key-policy tampering and final transcript agreement |
| Global-degree extraction and complete FS reduction | Consumer security contract above | **Not established by these tests** |

The scheme owns a versioned prefix binding setup policy, full public degree,
imported capacity, G1/G2/beta-G2, commitment, query length/coordinates, and
claimed value. The subsequent order preserves the donor: fold commitments,
challenge `r`, evaluations at `[r,-r,r²]`, polynomial batching challenge,
three witnesses, pairing batching challenge. Zero `r` rejects without retries.
Distinct point checks are unnecessary for separate single-point KZG openings.
This prefix intentionally changes the historical proof format.

Proving is not constant-time. This clear protocol reveals its evaluation
messages; callers needing ZK must use a separately specified construction.

## Validation

```sh
cargo nextest run -p jolt-hyperkzg --cargo-quiet
cargo clippy -p jolt-hyperkzg --all-targets -- -D warnings
cargo fmt -p jolt-hyperkzg --check
```

No speed, EVM gas, or zero-knowledge claim is made. The implementation uses
the existing group MSM and pairing backend and recomputes the original
commitment during opening to own its complete transcript statement.

## Opt-in BN254 transcript policy

The standalone caller policy is
`jolt_transcript::Bn254WideBlake2bTranscript::new(application_label)`.
Pass that same transcript through the complete proof, including the PCS opening;
Spartan's PCS-generic API remains unchanged. Both parties must select this policy
and authenticate the same relation/setup policy identifier. The fixed session
`bn254-blake2b-wide384-v1` binds the sampler and Blake2b-512 spongefish backend
before any Spartan tau or PCS challenge; the application label is a big-endian u64 byte length followed by a padded
32-byte word (maximum label length 32). Old sampler proofs are incompatible.

`challenge`, `challenge_scalar`, and inherited vector draws all reduce three
128-bit scalar draws as `(a * 2^128 + b) * 2^128 + c` modulo BN254 Fr.
Conditional on independent uniform underlying byte blocks, maximum output mass
is at most `1/p + 2^-384`. Poseidon and arbitrary user transcript implementations
are outside this policy. This opt-in changes no shared Jolt transcript default.
The standalone acceptance, malformed-proof, and replay tests use this policy
with HyperKZG; Spartan also exercises Dory. Independent integer reduction vectors
and scalar-byte decoding tests live in `jolt-transcript/src/wide.rs`.
Online extraction, complete-SRS assumptions, and a backend-specific Fiat–Shamir
composition theorem remain separate unproved deployment obligations.
