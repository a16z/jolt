//! Spartan + HyperKZG wrapper of the Jolt/Dory verifier.
//!
//! Layer 1 proves the Jolt verifier's work as (R) a small Spartan R1CS for the
//! sumcheck-round algebra, (T1) a Blake3 transcript table and (T2) a non-native
//! limb-arithmetic table for the Dory final check, all verified by one batched
//! sumcheck stream and one HyperKZG opening. Layer 2 (Groth16) is out of this
//! crate.

pub mod hash_table;
pub mod limb_table;
pub mod profile;
pub mod relation;
pub mod spartan;
pub mod stream;
