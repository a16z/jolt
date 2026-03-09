//! [`SumcheckCompute`](jolt_sumcheck::SumcheckCompute) implementations.
//!
//! Each module provides a round-polynomial evaluator for one type of
//! sumcheck instance in the Jolt pipeline. Evaluators hold polynomial
//! evaluation tables and perform the per-round computation.

pub mod eq_product;
pub mod formula;
pub mod hamming;
pub mod kernel;
pub mod mles_product_sum;
pub mod ra_poly;
pub mod ra_virtual;
