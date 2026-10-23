//! The Akita prove path: native trace-group commitments and heterogeneous batched openings.

pub mod preprocessing;
mod prover;
mod setup;
pub use setup::one_hot_trace_setup_shape;
// A Whiteout build proves only through the prover-fixtures seam until W6d.
#[cfg(any(not(feature = "zk"), feature = "prover-fixtures"))]
mod stage0;
// Whiteout's stage 8 is the masked Akita opening (W6d).
#[cfg(not(feature = "zk"))]
mod stage8;
pub mod witness;
pub use prover::prove;
#[cfg(any(not(feature = "zk"), feature = "prover-fixtures"))]
mod spine;
#[cfg(all(feature = "zk", feature = "prover-fixtures"))]
pub use spine::{prove_masked_stages, MaskedStagesProof};
