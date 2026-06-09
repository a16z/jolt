//! Typed wrapper verifier protocol phases.

pub mod hyperkzg;
pub mod r1cs_relation;
pub mod spartan;
#[cfg(feature = "zk")]
pub mod zk;
