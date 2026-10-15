# Clear binary HyperKZG

This crate implements ordinary BN254 multilinear commitment/opening through
`jolt_openings::CommitmentScheme`. The `jolt-spartan-prover` tests exercise it
as one of two PCS backends (Dory is the other); no workspace crate links it in
production yet. There is no hiding, committed-round protocol, or
batch/homomorphic API.

## Statement and setup contract

An honest commitment treats the high-to-low multilinear evaluation table as
univariate coefficients against imported G1 powers. Tables have `2^ell`
entries with `ell >= 1`; constant polynomials must be represented by a
two-entry constant table. Queries have exactly `ell` coordinates.

`CommitmentScheme::setup(HyperKZGSetupParams)` imports typed BN254 G1 powers,
G2, beta-G2, and an authenticated setup-policy ID. It checks the imported
capacity and nonidentity powers. It **does not** validate progression
of the powers, ceremony provenance, or secrecy of the trapdoor. Those are
trusted importer preconditions. Production applications must authenticate
both key material and policy; setup must not be taken from the proof.
No production function generates or accepts a secret scalar.

BN254 point deserialization delegates to `jolt-crypto`'s checked canonical
point decoding, including subgroup validation. Applications own bounded
container decoding and rejection of trailing serialized container bytes.
Identity proof points are allowed because zero polynomials are legitimate;
identity setup powers are rejected.

The opening hint is the commitment returned by `commit`; `open` binds it into
the statement and recomputes the commitment only when no hint is supplied. A
hint that is not the polynomial's commitment yields a proof the verifier
rejects.

Imported capacity is an honest-prover/API bound, not an adversarial degree
bound: **all** public powers under the same trapdoor count, not only the
imported slice, and local truncation does not lower that bound. The
authenticated `setup_id` identifies the ceremony and its full published degree.

The conditional extraction contract needed by a consumer is: a polynomial
of degree at most the ceremony's full published degree is fixed when its
commitment is made; later valid openings refer to that same polynomial. Its
first `2^ell` coefficients determine the extracted multilinear vector. Do not
infer the stronger equality between the original commitment and a commitment
to the truncated vector. For example, with powers through degree three, the
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
| Univariate synthetic division and KZG batching | `kzg.rs` | Round trips at arities 1–6; wrong statement, witness, and key rejection |
| Exact shape and nonzero fold challenge | `check_arity`, `verify_opening` | Malformed lengths and out-of-range arities; zero challenge rejected by prover and verifier |
| Commitment/query/claim fixed before opening challenges | `append_statement` | Statement/key-policy tampering and final transcript agreement |
| Global-degree extraction and complete FS reduction | Consumer security contract above | **Not established by these tests** |

`HyperKZGVerifierSetup::append_statement` absorbs a versioned prefix binding
setup policy, imported capacity, G1/G2/beta-G2, commitment, query
length/coordinates, and claimed value. Then come fold commitments, challenge
`r`, evaluations at `[r,-r,r²]`, polynomial batching challenge, three
witnesses, pairing batching challenge. Zero `r` rejects without retries.
Distinct point checks are unnecessary for separate single-point KZG openings.

Proving is not constant-time. This clear protocol reveals its evaluation
messages; callers needing ZK must use a separately specified construction.
No speed, EVM gas, or zero-knowledge claim is made.

## Validation

```sh
cargo nextest run -p jolt-hyperkzg --cargo-quiet
cargo clippy -p jolt-hyperkzg --all-targets -- -D warnings
cargo fmt -p jolt-hyperkzg --check
```
