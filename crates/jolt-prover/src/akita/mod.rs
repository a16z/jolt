//! The Akita prove path: native trace-group commitments and heterogeneous batched openings.

pub mod preprocessing;
#[cfg(not(feature = "zk"))]
mod prover;
mod setup;
pub use setup::one_hot_trace_setup_shape;
mod spine;
mod stage0;
// Whiteout's stage 8 is the masked Akita opening (`whiteout`).
#[cfg(not(feature = "zk"))]
mod stage8;
mod trace_group;
#[cfg(feature = "zk")]
mod whiteout;
pub mod witness;
#[cfg(not(feature = "zk"))]
pub use prover::prove;
#[cfg(feature = "zk")]
pub use whiteout::prove;
#[cfg(all(feature = "zk", feature = "prover-fixtures"))]
pub use whiteout::{faults as whiteout_faults, secrets as whiteout_secrets};
