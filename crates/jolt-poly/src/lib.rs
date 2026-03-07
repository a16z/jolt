//! Polynomial types and operations for multilinear, univariate, and
//! specialized polynomials. Backend-agnostic and reusable outside Jolt.
//!
//! This crate provides the core polynomial abstractions used throughout
//! the Jolt zkVM proving system:
//!
//! - [`Polynomial`]: Evaluation table over the Boolean hypercube, generic over
//!   coefficient type (`Field` for dense, primitives like `u8`/`bool` for compact)
//! - [`EqPolynomial`]: Equality polynomial for sumcheck evaluation
//! - [`UnivariatePoly`]: Coefficient-form univariate polynomial with Lagrange interpolation
//! - [`CompressedPoly`]: Compressed univariate with the linear term omitted (for proof size)
//! - [`IdentityPolynomial`]: Maps hypercube points to their integer index

mod compressed_univariate;
mod cpu_polynomial;
mod eq;
mod identity;
pub mod lagrange;
mod multilinear;
mod source;
mod univariate;

pub use compressed_univariate::CompressedPoly;
pub use cpu_polynomial::Polynomial;
pub use eq::EqPolynomial;
pub use identity::IdentityPolynomial;
pub use multilinear::{MultilinearBinding, MultilinearEvaluation};
pub use source::{EvaluationSource, RlcSource};
pub use univariate::{UnivariatePoly, UnivariatePolynomial};
