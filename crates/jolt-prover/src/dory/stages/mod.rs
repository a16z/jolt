//! The Dory path's protocol-specific pipeline ends: stage 0 (streaming
//! per-polynomial witness commitments) and stage 8 (the RLC-batched joint
//! opening). The shared stage 1–7 recipes live in [`crate::stages`].

pub mod stage0;
pub mod stage8;
