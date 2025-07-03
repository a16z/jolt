//! This module provides the specialized sum-check precompile operators for Jolt's ONNX execution.
//! These precompile's are designed for when it is more efficient to prove an ONNX operator with a direct sum-check protocol,
//! rather than via lookups.
//!
//! On the provers end precompiles are typically defined by their witness values in the execution trace.

pub mod conv;
pub mod matmult;
pub mod sum;
pub mod sumcheck_engine;

use crate::jolt_onnx::precompiles::conv::ConvPrecompile;
use matmult::MatMultPrecompile;
use serde::{Deserialize, Serialize};

/// Specifies the ONNX precompile operators used in the Jolt ONNX VM.
/// Used to specifiy the precompile type and its input's in the [`JoltONNXTraceStep`]
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum PrecompileOperators {
    /// Matrix multiplication precompile.
    MatMult(MatMultPrecompile),
    /// Conv operator precompile.
    Conv(ConvPrecompile),
}
