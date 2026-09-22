# Private SPARK source checkpoint — NOT VALIDATED

2026-09-22. Builds are paused by the conductor while the full recursion trial runs. This source checkpoint extends accepted public-slice `1ef064749` toward the complete reviewed v2 relation. **No new Rust source in this checkpoint has compiled, passed tests, or received independent review.** Formatting/syntax parsing and source inspection are not correctness evidence. The earlier public-slice packet remains historical evidence for its exact commit only.

## Implemented source boundary

The existing verifier crate now contains concrete fixed-shape product reduction state and the full application-facing `ComputationKey::verify`: fresh wide384 v2 initialization; existing outer sumcheck; actual public PCS; existing inner sumcheck; private SPARK; and the actual original witness opening. Full verification does not retain or scan private matrices.

The private verifier checks sixteen roots and six half-dot claims; twelve length-N operation trees and four length-L memory trees; exactly j cubic rounds at layer j; claim-before-coefficient and terminal-before-eta ordering; six triple-product half-dots only at the operation bottom; and three actual HyperKZG openings for dereference, operations and audit tables. It checks the dot leaves and all read/write/init/audit hash equations. No mocked opening, direct-private fallback, division by fingerprint factors, or second identity-polynomial formula is used. `IdentityPolynomial` and `EqPolynomial` retain their existing owners.

The existing prover crate retains authenticated immutable direct matrices for honest proving, host integer address lists from normalization, and actual committed tables. It builds product trees by repeated matching-half multiplication, adds bottom dot products, uses the existing sumcheck prover, and invokes actual PCS opening paths. Direct matrix evaluation exists only in honest proving and the reference tests. The original outer terminal check is factored once for both v1 and v2.

All changes follow original `preprocessed-matrix-protocol.md` and accepted `preprocessed-matrix-public-review.md`. The pinned reference remains Microsoft/Spartan `d62b961f9497e3c07a921b6da1457cad467598d9`, source `product_tree.rs` and `sparse_mlpoly.rs` archived under `/private/tmp/spartan-matrix-sources`. This uses the reviewed adapted transcript, not Merlin compatibility.

## Tests written, not executed

- Full 4x8,p3 relation, duplicate/cancelling sparse entries and empty C: actual four matrix openings plus witness opening.
- Two non-Boolean sparse queries with unequal row/private dimensions and leading-zero extension.
- Completely empty A/B/C relation with N=2: top layer has zero sumcheck rounds but still authenticates six dot triples.
- Changes to every root, half-dot, dereference slab, operation slab (including zero), audit slab, private matrix claim, each private opening witness and witness opening; tree terminals, dot terminals, layer counts, vector lengths, public inputs/key ID and query dimensions.
- Independently encoded real test-only beta=1 setup preimage (224 bytes), plus changed setup ID, maximum degree and beta at full verifier entry. Fixture script cites pinned Arkworks BN254 generator coordinates and explicit compressed-positive flag. Digest `020ddd7807a613cdd969d49ec8324ea6a13b263b90cf199be42ed23b62bdb73b`. It is not a deployment ceremony.

## Exact next validation command, after conductor release

In `preprocessed-matrix-r1cs`:

```
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=../crypto-r1cs/target cargo nextest run -p jolt-spartan-prover -p jolt-spartan-verifier --features preprocessed --lib --cargo-quiet
```

First resolve compilation/algebraic failures against the specification. Then run scoped all-target clippy, minimal feature selection, existing direct-Spartan regression for the shared outer-check extraction, fmt and diff checks. Preserve logs and actual exit codes. Do not promote this checkpoint based on the old four passing public tests.

## Remaining acceptance and security obligations

The written implementation must pass live honest and adversarial checks before any correctness claim. Additional discriminating timestamp-reset/unshifted-address controls and zero-fingerprint product-network cases should be assessed before review; the current corruption tests are not a replacement for those algebraic controls. Full transcript and payload counts must be measured after it runs. Canonical runtime key decoding, bounded proof decoding, trailing-byte policy and on-chain ABI remain deployment gates. Application authentication of the expected key ID and setup provenance remain external requirements.

Joint adaptive projected-vector extraction and the operational/ROM interface remain reviewed S1 gates; this source does not close them. No ZK, full-wrapper acceptance, EVM gas, proof-size improvement, or performance claim is made. No remote push and no new proof/trace run occurred during this source-only checkpoint.
