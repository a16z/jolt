//! Optimized kernels: legacy-prover algorithms ported behind the same
//! [`PrepareKernel`](crate::PrepareKernel) seam the reference tier fills.
//! Byte parity with the reference kernels (identical round polynomials and
//! output claims) is the correctness bar; only data structures and
//! algorithms change.

pub mod booleanity;
pub mod ram_hamming_booleanity;
