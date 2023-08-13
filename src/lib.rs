#![allow(non_snake_case)]
#![allow(clippy::assertions_on_result_states)]
#![feature(extend_one)]
#![feature(associated_type_defaults)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate core;
extern crate digest;
extern crate merlin;
extern crate rand;
extern crate sha3;

#[cfg(feature = "multicore")]
extern crate rayon;

pub mod benches;
pub mod lasso;
mod msm;
mod poly;
mod subprotocols;
pub mod subtables;
pub mod utils;

#[cfg(test)]
mod e2e_test;
