//! Symbolic expression IR for sumcheck claim formulas.
//!
//! `jolt-ir` is the single source of truth for all claim-level expressions in
//! the Jolt zkVM. Each sumcheck claim formula is written once as an [`Expr`] and
//! all backends (evaluation, BlindFold R1CS, Lean4, circuit transpilation)
//! derive from it.
//!
//! # Quick start
//!
//! ```
//! use jolt_ir::{ExprBuilder, ClaimDefinition, OpeningBinding, PolynomialId};
//!
//! // Booleanity check: γ · (H² − H)
//! let b = ExprBuilder::new();
//! let h = b.opening(0);
//! let gamma = b.challenge(0);
//! let expr = b.build(gamma * (h * h - h));
//!
//! // Normalize to sum-of-products for R1CS emission
//! let sop = expr.to_sum_of_products();
//! assert_eq!(sop.len(), 2); // γ·H·H and -γ·H
//! ```
mod backends;
mod builder;
mod claim;
mod expr;
mod kernel;
mod normalize;
mod polynomial_id;
mod visitor;

pub mod protocol;
pub mod toom_cook;
pub mod zkvm;

pub use builder::{ExprBuilder, ExprHandle};
pub use claim::{ClaimDefinition, OpeningBinding};
pub use expr::{Expr, ExprArena, ExprId, ExprNode, Var};
pub use kernel::{KernelDescriptor, KernelShape, TensorSplit};
pub use normalize::{SopTerm, SopValue, SumOfProducts};
pub use polynomial_id::PolynomialId;
pub use visitor::ExprVisitor;

pub use backends::circuit::CircuitEmitter;
pub use backends::r1cs::{LcTerm, LinearCombination, R1csConstraint, R1csEmission, R1csVar};

#[cfg(feature = "z3")]
pub use backends::z3::Z3Emitter;
