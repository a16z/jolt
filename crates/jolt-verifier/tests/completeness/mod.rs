// The legacy-generated fixture suites cannot run with field-inline enabled: legacy proofs pin
// the field-inline axis disabled, which the verifier with field-inline enabled rejects at the
// protocol-config gate. The field-inline path gets its own modular-prover-backed module
// (`field_inline`).
#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "field-inline")
))]
pub mod advice;
#[cfg(all(feature = "prover-fixtures", feature = "akita"))]
pub mod akita;
#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    not(feature = "akita")
))]
pub mod field_inline;
#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "field-inline")
))]
pub mod standard;
#[cfg(not(feature = "field-inline"))]
pub mod zk;
