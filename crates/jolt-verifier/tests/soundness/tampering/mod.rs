// The `not(akita)` modules tamper the base (dory) proof shape, which does not exist under the
// akita feature — one compiled verifier runs exactly one protocol. The packed pipeline gets
// its own typed tamper suite (`akita`: clear-claim wire sweep, commitment-byte sweeps,
// proof-shape and presence tampers); only the shape-agnostic `manifest` checks run under both.
// Same per-family split for field-inline: legacy-fixture-driven suites run only when
// field-inline is disabled (the verifier with field-inline enabled rejects legacy proofs at
// the protocol-config gate); the field-inline wire cells get their own typed suite over
// modular-prover fixtures (`field_inline`). The ordinary `sumcheck`, `openings`, and
// `commitments` sweeps run under both field-inline configurations over
// `ordinary_tamper_bases()` (legacy muldiv with field-inline disabled, the modular eq-MLE
// fixture with field-inline enabled), so the ordinary stage payloads are also rejected by the
// composed verifier; their legacy-advice-only tests keep field-inline disabled. The
// shape-agnostic `manifest` checks run under every family. The akita sweep is
// legacy-fixture-driven too, so it also runs with field-inline disabled (its exhaustive claim
// destructuring would otherwise need the field-inline wire cells no legacy fixture can
// populate).
#[cfg(all(
    feature = "prover-fixtures",
    feature = "akita",
    not(feature = "field-inline")
))]
pub mod akita;
#[cfg(not(feature = "akita"))]
pub mod commitments;
#[cfg(not(feature = "akita"))]
pub mod configs;
#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    not(feature = "akita"),
    not(feature = "zk")
))]
pub mod field_inline;
pub mod manifest;
#[cfg(not(feature = "akita"))]
pub mod openings;
#[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
pub mod preamble;
#[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
pub mod proof_shape;
#[cfg(not(feature = "akita"))]
pub mod sumcheck;
#[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
pub mod zk;
