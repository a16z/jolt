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

pub mod lasso;
pub mod subtables;
pub mod benches;
mod msm;
mod poly;
mod subprotocols;
mod utils;

#[cfg(test)]
mod e2e_test;
