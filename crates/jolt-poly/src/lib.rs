//! Polynomial types and operations for multilinear, univariate, and
//! specialized polynomials. Backend-agnostic and reusable outside Jolt.
//!
//! This crate provides the core polynomial abstractions used throughout
//! the Jolt zkVM proving system. Polynomials are represented by their
//! evaluations over the Boolean hypercube, with specialized compact
//! representations for small-scalar coefficients.
//!
//! # Core Traits
//!
//! - [`MultilinearEvaluation`]: Point evaluation interface — `num_vars()`, `len()`, `evaluate(point)`
//! - [`MultilinearBinding`]: In-place variable binding for sumcheck — `bind(scalar)`
//! - [`UnivariatePolynomial`]: Shared interface for univariate types — `degree()`
//!
//! # Polynomial Types
//!
//! - [`Polynomial<F: Field>`]: Full evaluation table as `Vec<F>`, with in-place binding and arithmetic
//! - [`Polynomial<T>`] (compact): When `T` is a small primitive (`bool`, `u8`, etc.), stores
//!   evaluations natively and promotes to field on demand (up to 32x memory reduction)
//! - [`EqPolynomial`]: Equality polynomial `eq(x, r)`, materialized via bottom-up doubling
//! - [`EqPlusOnePolynomial`]: Successor polynomial `eq+1(x, y)` evaluating to 1 when `y = x + 1`
//! - [`EqPlusOnePrefixSuffix`]: Prefix-suffix decomposition of `eq+1` for sqrt-sized sumcheck buffers
//! - [`LtPolynomial`]: Less-than polynomial `LT(x, r)` with split optimization for sqrt-sized buffers
//! - [`IdentityPolynomial`]: Maps hypercube points to their integer index
//! - [`UnivariatePoly`]: Coefficient-form univariate with Lagrange interpolation and compression
//! - [`CompressedPoly`]: Compressed univariate with the linear term omitted (one field element saved per round)
//!
//! # Streaming and Sparse Access
//!
//! - [`MultilinearPoly`]: Core trait for multilinear access — evaluation, row iteration, fold, sparsity hints
//! - [`RlcSource`]: Lazy random linear combination of multiple [`MultilinearPoly`] sources
//! - [`OneHotPolynomial`]: Sparse polynomial where each row has at most one nonzero entry (value 1),
//!   enabling O(T) PCS commit via generator lookup
//!
//! # Binding
//!
//! - [`BindingOrder`]: Controls variable binding order during sumcheck (MSB-first vs LSB-first)
//!
//! # Utility Modules
//!
//! - [`lagrange`]: Lagrange interpolation, symmetric power sums, polynomial multiplication,
//!   Newton-form interpolation over integer domains
//! - [`math`]: Bit-manipulation utilities on `usize` via the `Math` trait (`pow2`, `log_2`)
//! - [`thread`]: `drop_in_background_thread` (rayon) and `unsafe_allocate_zero_vec` (zero-init allocation)

mod binding;
mod compressed_univariate;
mod dense;
mod eq;
mod eq_plus_one;
mod identity;
pub mod lagrange;
mod lt;
pub mod math;
mod multilinear;
mod one_hot;
pub mod thread;
mod univariate;

pub use binding::BindingOrder;
pub use compressed_univariate::CompressedPoly;
pub use dense::Polynomial;
pub use eq::EqPolynomial;
pub use eq_plus_one::{EqPlusOnePolynomial, EqPlusOnePrefixSuffix};
pub use identity::IdentityPolynomial;
pub use lt::LtPolynomial;
pub use multilinear::{MultilinearBinding, MultilinearEvaluation, MultilinearPoly, RlcSource};
pub use one_hot::OneHotPolynomial;
pub use univariate::{UnivariatePoly, UnivariatePolynomial};
