#![deny(warnings, unused, nonstandard_style)]
#![allow(clippy::type_complexity, clippy::upper_case_acronyms)]

pub mod afgho;
pub mod gipa;
pub mod inner_products;
pub mod tipa;

pub mod poly_commit;

pub type Error = anyhow::Error;
