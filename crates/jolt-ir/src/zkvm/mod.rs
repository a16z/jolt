//! Jolt zkVM protocol definitions.
//!
//! This module contains the concrete claim definitions and [`PolynomialId`](crate::PolynomialId)
//! identifiers for the Jolt zkVM proving system. Both the prover (jolt-zkvm)
//! and verifier (jolt-verifier) consume these definitions as the single source
//! of truth for all claim formulas.
//!
//! # Submodules
//!
//! - [`claims`] — [`ClaimDefinition`](crate::ClaimDefinition) constructors for
//!   every sumcheck instance in the Jolt pipeline.

pub mod claims;
