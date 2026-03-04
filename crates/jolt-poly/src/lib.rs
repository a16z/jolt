//! Polynomial types and operations for multilinear, univariate, and
//! specialized polynomials. Backend-agnostic and reusable outside Jolt.
//!
//! This crate provides the core polynomial abstractions used throughout
//! the Jolt zkVM proving system:
//!
//! - [`DensePolynomial`]: Full evaluation table over the Boolean hypercube
//! - [`CompactPolynomial`]: Memory-efficient storage using small scalar types
//! - [`EqPolynomial`]: Equality polynomial for sumcheck evaluation
//! - [`UnivariatePoly`]: Coefficient-form univariate polynomial
//! - [`IdentityPolynomial`]: Maps hypercube points to their integer index
//! - [`LagrangePolynomial`]: Basis evaluation and interpolation over integer domains

mod compact;
mod dense;
mod eq;
mod identity;
mod lagrange;
pub mod serde_canonical;
mod traits;
mod univariate;

pub use compact::{CompactPolynomial, SmallScalar};
pub use dense::DensePolynomial;
pub use eq::EqPolynomial;
pub use identity::IdentityPolynomial;
pub use lagrange::LagrangePolynomial;
pub use traits::MultilinearPolynomial;
pub use univariate::UnivariatePoly;
