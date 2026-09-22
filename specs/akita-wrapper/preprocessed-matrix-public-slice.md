# Conditional v2 preprocessing and public-column PCS slice

This implements trusted preprocessing and the public-column opening stage of the repaired preprocessed-matrix contract. It is **not a complete SPARK verifier**, a succinct R1CS argument, a zero-knowledge argument, or a resolution of the joint online extraction/ROM security gate. The unchanged direct Spartan verifier remains the full-relation reference. Do not expose `verify_public` as application proof acceptance.

Base: `2e7c833f8`. Source contract: original-workspace `specs/akita-wrapper/preprocessed-matrix-protocol.md`, independently reviewed by `preprocessed-matrix-review.md`. The serialization-only fixture comes from the independent Python key-vector artifact; its digest is `89ceda1f5b77b7f4f08b2fea7ae920b9d19d0f1ae1bbd66d599aa2a638052c1b`.

## Protocol-to-code map

- `jolt-crypto/src/ec/bn254/mod.rs`: compressed point encoding has one owner; existing transcript append delegates to it without changing bytes.
- `jolt-hyperkzg/src/types.rs`: `HyperKZGVerifierSetup::binding` validates the actual opaque setup and exports its canonical setup-binding preimage. No external mutable metadata is accepted as a setup.
- `jolt-spartan-verifier/src/preprocessed.rs`: exact 476-byte version-2 key, parameterized BLAKE2b-256 digest, checked dimensions/capacity, authenticated expected key ID and setup, concrete wide384 initialization retaining the post-tau `spartan-outer` zero marker, claim-before-selector schedule, and actual HyperKZG verification of public columns.
- `jolt-spartan-prover/src/preprocessed.rs`: existing linear-combination normalization, original-matrix digest, private-column reindexing, padded operations and cumulative audit tables, three actual commitments, and an actual merged public-column opening.
- The opt-in `preprocessed` feature adds no matrix-oracle trait and does not alter v1 protocol behavior. This version deliberately has a different key/transcript contract.

For `R` padded rows and `p` public columns including ONE, the public table is `T R` entries with `T=next_power_of_two(3p)`. Slab `Mp+j` is the row vector of public column `j` of matrix `M`. Prover sends all `3p` evaluations before the selector. The PCS query concatenates selector coordinates then row coordinates, with zero padding for unused slabs. The result is the weighted public contribution only.

The operations table has 16 slabs of N entries: row addresses A/B/C, row read timestamps A/B/C, private column addresses A/B/C, column read timestamps A/B/C, values A/B/C, then zero. Entries are normalized row-major; each matrix pads with `(0,0,0)`. Timestamp counters persist across A/B/C and include these dummy accesses. The memory table contains final row counters followed by final column counters. They are committed but their memory argument is **not** implemented in this slice.

## Discriminating fixture and boundaries

The live fixture has four rows, eight total columns, p=3, five private variables, padded private size eight, and N=4. It contains duplicate/cancelling sparse entries and zero coefficients. Its independently calculated audit checkpoints distinguish both reset-at-matrix errors and skipped dummy accesses. Test-only powers use known beta seven and are unsuitable for deployment.

Tests run the complete existing direct Spartan proof/verification with actual PCS. Separately, they execute v2 initialization, actual existing outer sumcheck proving/verification, outer-claim validation, matrix challenge draws, actual public PCS proof/verification, and the inner input marker. The public contribution is checked against the live direct matrix evaluator at the resulting non-Boolean point. This is a prefix composition test, **not an implemented full v2 verifier**.

The exact final prefix state is `8d2b88e6eaed657ed8b8f71ca0a683810d4cc2853c982b93085660e268d9781a`; the next challenge's canonical little-endian bytes are `58fe74b79a1064ba7b14053d87b90bb8ac4afd9c9efb749c813182d243bc0401`. These are pinned transition-regression vectors from the live owner implementations, not an independent proof of Fiat–Shamir security. The separately generated 476-byte key fixture is independently checked byte for byte.

Tamper cases cover changed public-column claims, shortened claims, changed opening witness, changed row point, wrong expected key ID, query dimensions, and invalid public/private shapes. The known relation and direct evaluator provide correctness checks independent of the new table encoder.

## Remaining module and ambiguity register

1. No private dereference commitment, sparse matrix product sumcheck, GKR product trees, or timestamp multiset audit exists yet. No mocked opening substitutes for them. Implement those next against the pinned primary SPARK construction before claiming the requested complete 4x8 v2 relation.
2. The key constructor assumes authenticated trusted preprocessing. The application must authenticate the expected key ID; an attacker choosing their own key and expected ID is outside this contract. A canonical byte decoder with point validity checks and bounded proof-container decoding remains absent.
3. The public helper accepts an already initialized wide transcript and explicit row point/weights. Production acceptance must bind these to the outer sumcheck, public input and witness commitment. Only the test currently composes that prefix; the helper alone does not bind arbitrary supplied weights to a relation proof.
4. The joint-extraction/adaptive projected-vector argument and Fiat–Shamir/ROM theorem remain the reviewed S1 security gate. These tests do not close it. No ZK masking, EVM verifier, gas measurement, whole-wrapper performance, or proof-size improvement is claimed.
5. SRS policy/authentication and maximum degree are bound to the actual imported setup. Commitment vector preprocessing is linear; succinctness must come from the remaining verified evaluation argument, not from hashing a key.

## Validation

Commands and final outcomes are appended after scoped validation. Raw logs are retained under `/private/tmp/preprocessed-matrix-*.log`; they are local evidence, not portable CI artifacts.

Initial source: `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=../crypto-r1cs/target cargo clippy -p jolt-spartan-prover --features preprocessed --lib -q -- -D warnings` passed (`/private/tmp/preprocessed-matrix-clippy.log`). Initial three tests passed with `cargo nextest run -p jolt-spartan-prover --features preprocessed --lib --cargo-quiet`, run `e98ce34f-b123-4741-ae2e-866f13318b43` (`/private/tmp/preprocessed-matrix-tests.log`). Vector capture rerun passed, run `34ccb2b2-d4f0-4277-bef9-b39ca25dcb72` (`/private/tmp/preprocessed-matrix-vector.log`). The initial tests emitted an unused `Option` warning, now corrected.

Final source fixes the warning, replaces temporary vector prints with exact assertions, adds a malformed PCS witness assertion, and checks the independent serialization fixture. Final `cargo nextest run -p jolt-spartan-prover -p jolt-spartan-verifier --features preprocessed --lib --cargo-quiet` passed all four tests, run `fea97588-9031-4998-ae5f-7069d91f5ccf` (`/private/tmp/preprocessed-matrix-final-tests.log`). All commands use `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=../crypto-r1cs/target`.

`cargo clippy -p jolt-spartan-prover -p jolt-spartan-verifier --features preprocessed --all-targets -q -- -D warnings` passed. The same command with `--no-default-features` passed. Logs: `/private/tmp/preprocessed-matrix-final-clippy.log` and `/private/tmp/preprocessed-matrix-minimal-clippy.log`. Validation paused when disk fell below 12 GiB and resumed only after conductor-managed recovery.
`cargo fmt -q`, `git diff --check`, and Taplo on both changed manifests passed. Taplo required unsandboxed execution after the known macOS system-configuration panic; it then returned zero. The independent original-workspace `python3 specs/akita-wrapper/preprocessed-matrix-key-vector.py` returned PASS for 476 bytes, the expected key ID and two post-tau marker appends.

Final minimal check: the same four-test nextest command with `--no-default-features` passed (`/private/tmp/preprocessed-matrix-minimal-tests.log`). Both crates have no default features, so this reused the exact compiled feature graph. The real full direct-Spartan control is already part of `real_outer_sumcheck_and_public_pcs_v2_prefix`; a redundant broader direct test suite was not run. No workspace-wide or ZK tests were run for this opt-in clear prototype.
