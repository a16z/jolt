//! [`SumcheckCompute`](jolt_sumcheck::SumcheckCompute) implementations.
//!
//! Each module implements the sumcheck witness computation for one type
//! of sumcheck instance in the Jolt pipeline. Witnesses hold polynomial
//! data and perform the per-round computation.

pub mod eq_product;
pub mod formula;
pub mod hamming;
