#![deny(warnings, unused, nonstandard_style)]
#![allow(clippy::type_complexity, clippy::upper_case_acronyms)]
use std::ops::MulAssign;

pub mod commitments;
pub mod gipa;
pub mod inner_products;
pub mod tipa;

pub mod poly_commit;

pub type Error = anyhow::Error;

//TODO: helper function for mul because relying on MulAssign
pub(crate) fn mul_helper<T: MulAssign<F> + Clone, F: Clone>(t: &T, f: &F) -> T {
    let mut clone = t.clone();
    clone.mul_assign(f.clone());
    clone
}

#[derive(Debug, thiserror::Error)]
pub enum InnerProductArgumentError {
    #[error("left length, right length: {0}, {1}")]
    MessageLengthInvalid(usize, usize),
    #[error("inner product not sound")]
    InnerProductInvalid,
}
