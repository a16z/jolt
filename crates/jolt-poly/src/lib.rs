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

mod binding;
mod compressed_univariate;
mod cpu_polynomial;
mod eq;
mod eq_plus_one;
mod identity;
mod lt;
pub mod lagrange;
pub mod math;
mod multilinear;
mod one_hot;
mod source;
pub mod thread;
mod univariate;

pub use binding::BindingOrder;
pub use compressed_univariate::CompressedPoly;
pub use cpu_polynomial::Polynomial;
pub use eq::EqPolynomial;
pub use eq_plus_one::{EqPlusOnePolynomial, EqPlusOnePrefixSuffix};
pub use identity::IdentityPolynomial;
pub use lt::LtPolynomial;
pub use multilinear::{MultilinearBinding, MultilinearEvaluation};
pub use one_hot::OneHotPolynomial;
pub use source::{MultilinearPoly, RlcSource};
pub use univariate::{UnivariatePoly, UnivariatePolynomial};
