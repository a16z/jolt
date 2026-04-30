//! R1CS constraint data structures for the Jolt proving system.
//!
//! This crate provides the core types for uniform Rank-1 Constraint Systems:
//!
//! - [`ConstraintMatrices`] — per-cycle sparse A, B, C matrices
//! - [`R1csKey`] — preprocessed key combining matrices with runtime dimensions
//! - [`R1csSource`] — materializes R1CS-derived polynomials (Az, Bz, Cz, etc.)
//! - [`constraints::rv64`] — Jolt RV64IMAC variable layout and dimension constants

pub mod constraint;
pub mod constraints;
pub mod key;
pub mod provider;
pub mod row_dots;

pub use constraint::ConstraintMatrices;
pub use key::R1csKey;
pub use provider::{R1csSource, SpartanChallenges};
pub use row_dots::{R1csRowDotSlice, R1csRowDotTable};
