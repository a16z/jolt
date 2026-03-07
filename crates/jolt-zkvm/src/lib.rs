//! Top-level zkVM prover and verifier orchestration.
//!
//! Composes all Jolt sub-crates into a complete proving system for
//! RISC-V (RV64IMAC) execution traces. See `crates/zkvm_spec.md`
//! for the full design specification.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`claims`] | IR-based claim definitions (single source of truth) |
//! | [`tags`] | Opaque polynomial and sumcheck identity tags |

pub mod claims;
pub mod tags;
