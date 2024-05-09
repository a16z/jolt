#![allow(non_snake_case)]
#![allow(clippy::assertions_on_result_states)]
#![allow(clippy::needless_range_loop)]
#![feature(extend_one)]
#![feature(associated_type_defaults)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(iter_next_chunk)]
#![allow(long_running_const_eval)]

// Note: Used exclusively by const fn BiniusConstructable::compute_powers. 
// Can be removed with a manual const fn for BinaryField multiplication.
#![feature(const_trait_impl)]
#![feature(effects)]
#![feature(const_refs_to_cell)]

pub mod benches;
pub mod field;
pub mod host;
pub mod jolt;
pub mod lasso;
pub mod msm;
pub mod poly;
pub mod r1cs;
mod subprotocols;
pub mod utils;

// Benchmarks
pub use crate::subprotocols::sumcheck::bench::sumcheck_bench;
