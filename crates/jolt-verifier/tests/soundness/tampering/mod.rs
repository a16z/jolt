// The `not(akita)` modules tamper the base (dory) proof shape, which does not
// exist under the akita feature — one compiled verifier runs exactly one
// protocol. The packed pipeline gets its own typed tamper suite (`akita`:
// clear-claim wire sweep, commitment-byte sweeps, proof-shape and presence
// tampers); only the shape-agnostic `manifest` checks run under both.
// Same per-family split for field-inline: legacy-fixture-driven suites are
// FR-off only (the FR-on verifier rejects legacy proofs at the
// protocol-config gate); the FR wire cells get their own typed suite over
// modular-prover fixtures (`field_inline`). The shape-agnostic `manifest`
// checks run under every family.
// The akita sweep is legacy-fixture-driven too, so it is FR-off like the
// other legacy suites (its exhaustive claim destructuring would otherwise
// need the FR wire cells no legacy fixture can populate).
#[cfg(all(
    feature = "prover-fixtures",
    feature = "akita",
    not(feature = "field-inline")
))]
pub mod akita;
#[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
pub mod commitments;
#[cfg(not(feature = "akita"))]
pub mod configs;
#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    not(feature = "zk")
))]
pub mod field_inline;
pub mod manifest;
#[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
pub mod openings;
#[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
pub mod preamble;
#[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
pub mod proof_shape;
#[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
pub mod sumcheck;
#[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
pub mod zk;
