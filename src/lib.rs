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

pub mod bench;
mod commitments;
mod dense_mlpoly;
mod errors;
mod gaussian_elimination;
mod grand_product;
mod math;
mod msm;
mod nizk;
pub mod random;
pub mod sparse_mlpoly;
mod sumcheck;
mod transcript;
mod unipoly;
mod utils;

#[cfg(test)]
mod e2e_test;
