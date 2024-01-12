#![allow(non_snake_case)]
#![allow(clippy::assertions_on_result_states)]
#![allow(clippy::needless_range_loop)]
#![feature(extend_one)]
#![feature(associated_type_defaults)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub mod benches;
pub mod jolt;
pub mod lasso;
mod msm;
mod poly;
pub mod r1cs;
mod subprotocols;
mod utils;

// Benchmarks
pub use crate::poly::dense_mlpoly::bench::dense_ml_poly_bench;
pub use crate::subprotocols::sumcheck::bench::sumcheck_bench;

#[cfg(test)]
mod e2e_test;