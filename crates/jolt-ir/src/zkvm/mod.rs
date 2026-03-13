//! Jolt zkVM protocol definitions.
//!
//! This module contains the concrete claim definitions and polynomial/sumcheck
//! identity tags for the Jolt zkVM proving system. Both the prover (jolt-zkvm)
//! and verifier (jolt-verifier) consume these definitions as the single source
//! of truth for all claim formulas.
//!
//! # Submodules
//!
//! - [`tags`] — Opaque `u64` identifiers for polynomials and sumcheck instances.
//! - [`claims`] — [`ClaimDefinition`](crate::ClaimDefinition) constructors for
//!   every sumcheck instance in the Jolt pipeline.

pub mod claims;
pub mod tags;
