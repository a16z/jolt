//! Arkworks-backed field implementations.
//!
//! Provides the BN254 scalar field (`Fr`) and its low-level arithmetic
//! (Montgomery/Barrett reduction, precomputed lookup tables, sparse multiplication).

pub mod bn254;
#[allow(dead_code)]
pub(crate) mod bn254_ops;
pub mod montgomery_impl;
pub mod wide_accumulator;
