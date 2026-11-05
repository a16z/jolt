// Legacy-generated ZK fixtures; only when field-inline is disabled (see
// soundness/tampering/mod.rs), and BlindFold's only (Whiteout's are
// `soundness/whiteout.rs`).
#[cfg(not(any(feature = "field-inline", feature = "akita")))]
pub mod zk;
