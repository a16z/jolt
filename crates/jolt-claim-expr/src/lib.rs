//! Protocol-free symbolic claim expressions and sumcheck descriptions.

#![forbid(unsafe_code)]
#![deny(
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes
)]

mod claims;
pub mod formula_error;
mod ops;
mod sumcheck;
mod symbolic;

pub use claims::{challenge, constant, derived, opening, Expr, Source, Term};
pub use sumcheck::SumcheckDomain;
pub use symbolic::{referenced_openings, SymbolicSumcheck};
