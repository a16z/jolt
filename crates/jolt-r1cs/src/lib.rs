//! R1CS constraint data structures for the Jolt proving system.
//!
//! This crate provides the core types for uniform Rank-1 Constraint Systems:
//!
//! - [`ConstraintMatrices`] — per-cycle sparse A, B, C matrices
//! - [`R1csKey`] — preprocessed key combining matrices with runtime dimensions
//! - [`R1csSource`] — materializes R1CS-derived polynomials (Az, Bz, Cz, etc.)
//! - [`R1csColumn`] — names the derived polynomial columns (Az/Bz/Cz/…)
//! - [`constraints::rv64`] — Jolt RV64IMAC variable layout and dimension constants
//! - [`constraints::field_constraints`] — native field-inline constraint layout
//! - [`constraints::jolt`] — compile-time feature-gated composition of Jolt R1CS constraints

pub mod builder;
pub mod column;
pub mod constraint;
pub mod constraints;
pub mod key;
pub mod lowering;
pub mod nonnative;
pub mod provider;

pub use builder::{AssignedScalar, LinearCombination, R1csBuilder, R1csBuilderError, Variable};
pub use column::R1csColumn;
pub use constraint::{
    ConstraintMatrices, ConstraintMatrixEvalError, MatrixColumnContributions, SparseRow,
    WeightedMatrixColumns,
};
pub use key::R1csKey;
pub use lowering::{
    assert_claim_expr_eq, lower_claim_expr, ClaimLoweringError, ClaimSourceTable, ClaimSources,
    SourceValue,
};
pub use nonnative::FqVar;
pub use provider::{R1csSource, SpartanChallenges};
