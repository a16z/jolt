#![allow(non_snake_case)]
#![allow(clippy::assertions_on_result_states)]
#![feature(extend_one)]
#![feature(generic_const_exprs)]
#![feature(associated_type_defaults)]

extern crate core;
extern crate digest;
extern crate merlin;
extern crate rand;
extern crate sha3;

#[cfg(feature = "multicore")]
extern crate rayon;

mod commitments;
mod dense_mlpoly;
mod errors;
mod math;
mod nizk;
mod product_tree;
pub mod random;
pub mod sparse_mlpoly;
mod sumcheck;
mod transcript;
mod unipoly;
mod utils;
mod gaussian_elimination;
pub mod bench;

#[cfg(test)]
mod e2e_test;