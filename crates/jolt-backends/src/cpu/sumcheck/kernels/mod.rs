//! CPU sumcheck kernel implementations.
//!
//! Coarse protocol kernels should live here and use `sparse_product` only as a
//! fallback/reference layer. For the first CPU backend, performance beats
//! internal modularity: relation-specific kernels may duplicate code and keep
//! protocol-shaped state when that is what preserves the optimized `jolt-core`
//! algorithm. The stable abstraction is the request/result boundary.

pub(super) mod spartan_outer;
pub(super) mod spartan_product;
pub(super) mod stage3_shift;
pub(super) mod stage7_advice;
pub(super) mod stage7_hamming;

pub(super) mod regular_batch;
mod sparse_product;

pub(super) use sparse_product::{evaluate_linear_product_queries, evaluate_row_product_queries};
