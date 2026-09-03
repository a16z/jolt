//! HyperKZG wrapper of the Jolt/Dory verifier.
//!
//! The verifier's work is encoded as (R) a row table for the stage algebra,
//! (T1) a Blake3 transcript table, and (T2) a non-native limb table for the
//! Dory final check. One batched sumcheck stream and one HyperKZG opening
//! verify all three tables and their copy links.

pub mod hash_table;
pub mod limb_table;
pub mod profile;
pub mod relation;
pub mod relation_table;
pub mod stream;
pub mod wrap;
