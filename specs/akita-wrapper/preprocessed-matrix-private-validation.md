# Conditional private SPARK prototype: validation packet

2026-09-22. Source checkpoint `38f6c75a5b177272ef2a793f9cf2e7da65f18437`, based on accepted public slice `1ef064749090ee7395d65e489c42f63b3fb5a8c6`, now has the requested discriminating controls and successful scoped execution. The exact final source pin is the commit carrying this packet. This supersedes the checkpoint's **unvalidated** status for this tested source; it does not certify a production SNARK.

The full 4-row, 8-total-column, p=3 relation now executes real proving and verification through public PCS, private product/memory reductions, all three private PCS openings, and the original witness opening. The verifier owns a fresh wide384 v2 transcript and authenticates the expected key and actual setup before query derivation. It never substitutes direct private matrix evaluation. Direct matrices remain only in honest proving/reference tests.

## What changed after source review

1. Refactored private witness preparation from downstream proof construction. Honest proving still checks roots and half sums before proceeding. The malicious test prover alone skips that local assertion, appends the exact prescribed claims, and generates **real downstream product proofs and PCS openings**. There is no production challenge override or mock opening.
2. Coherent timestamp-reset control changes the read/audit tables, recomputes their actual commitments, and uses the resulting synthetic key. Verification rejects specifically at `Relation("memory roots")`.
3. Coherent unshifted-address control restores original column numbers, recomputes cumulative timestamps/audits and commitments, and regenerates dereferences and downstream proofs. It retains the intended correctly shifted matrix values. Verification rejects specifically at `Relation("matrix half sums")`.
4. These are deliberately different keys with malformed trusted preprocessing, not attacks replacing the application's authenticated key. They distinguish the algebraic checks from downstream Fiat–Shamir divergence caused by mutating a completed proof.
5. A positive four-tree product network with zero leaves in different positions proves and verifies, with terminal values compared against direct MLE evaluation. It confirms zero factors do not trigger division or rejection. Empty A/B/C with N=2 also passes full v2 and exercises the zero-round bottom layer with six dot triples.
6. Retained the actual full prover transcript internally for regression testing, then pinned the completed transcript state/challenge and payload counts. The public proving API still owns its session.

The only initial compilation defect was projecting `Output` through `CommitmentScheme` instead of its supertrait owner. The implementation now imports concrete `Bn254G1` with its actual optional `jolt-crypto` dependency under `preprocessed`. Clippy repairs were an implicit clone, `is_multiple_of`, a test loop, and replacing temporary vector-capture prints with assertions. No algebraic test failed during these runs.

## Full transcript and actual small-fixture size

The final full-v2 transcript state is:

`27e411988a6e871c399b6f3396714e8ee10b4001df89ff27413d9575ea35f0f9`

The next wide challenge in canonical little-endian Fr bytes is:

`c42db9a9a4497d32c4b85c7972907aff7fb8276f50e562a5dc9f4b6522c8c909`

These are regression vectors from the live protocol owners, not independent evidence of a Fiat–Shamir theorem. The separately generated real test-only setup vector is independent of Rust serialization and also passes byte-for-byte comparison; changed setup ID, maximum degree and beta reject at the runtime entry.

The actual proof object contains **248 Fr and 36 G1 elements**, including the witness commitment, dereference commitment and **five** PCS opening proofs (four matrix plus one witness). This is **9,088 bytes** at 32 bytes per scalar/compressed point. The pinned bincode encoding is **9,176 bytes**, including its containers. These counts concern only this 4x8 fixture; they are not norm-circuit or full-wrapper costs, an EVM ABI measurement, a benchmark, or a claim of improved tradeoffs. Test-only known-trapdoor powers are unsuitable for deployment.

## Commands and outcomes

All compilation used `CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=../crypto-r1cs/target`; it began only after the conductor released the recursion trial's resource gate. Exact commands, directly observed exit codes, local log paths and log SHA-256 hashes are in [private-spark-command-results.json](private-spark-command-results.json). Initial failures are retained.

- Final scoped nextest: **10/10 passed**, exit0, run `3324bbfe-7a1c-4c7e-b692-c4c63f47b06e`. Covers public and full private acceptance; all committed slab/claim/opening mutations; fixed dimensions; coherent timestamp/address controls; non-Boolean points; empty N=2; setup and transcript vectors.
- Existing direct-Spartan regression: **9/9 passed**, exit0, run `cf25dac8-c28c-4d9f-94d7-04c351741bd9`. Required because v1 and v2 now share the unchanged outer terminal identity; includes Dory and HyperKZG paths.
- Scoped all-target clippy for both Spartan crates with `preprocessed`: exit0.
- Rustfmt, Taplo, independent setup-vector generator and diff checks: exit0.

Both Spartan crates define no default features, so repeating these commands with `--no-default-features` would resolve the same selected feature graph. It was intentionally not represented as distinct coverage. No workspace-wide, ZK, EVM or large circuit benchmarks were run.

## Remaining gates

Independent review must inspect the final source and executed controls before integration. Application authentication and correctly normalized trusted preprocessing remain hypotheses. Canonical runtime key decoding, resource-bounded proof decoding, trailing-byte rejection and on-chain ABI are still deployment work: derived Deserialize vectors alone do not enforce resource limits.

Joint adaptive projected-vector extraction and its operational/ROM interface remain explicit security gates; no theorem is established by these tests. ZK and complete Akita-verifier/wrapper circuit binding are not implemented by this slice. No remote push or full recursive proof run was performed by this worker.
