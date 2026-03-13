//! [`SumcheckCompute`](jolt_sumcheck::SumcheckCompute) implementations.
//!
//! Evaluators hold polynomial evaluation tables and perform per-round
//! sumcheck computation. The generic [`KernelEvaluator`](kernel::KernelEvaluator)
//! handles all compositions via compiled kernels from [`catalog`] descriptors.

pub mod catalog;
pub mod kernel;
pub mod mles_product_sum;
pub mod ra_poly;
pub mod ra_virtual;
pub mod segmented;
