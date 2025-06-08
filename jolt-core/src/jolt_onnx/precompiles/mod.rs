//! This module provides the specialized sum-check precompile operators for Jolt's ONNX execution.

pub mod conv;
pub mod conv1d;
pub mod matmult;
pub mod sumcheck_engine;

use matmult::MatMultPrecompile;
use serde::{Deserialize, Serialize};

/// Specifies the ONNX precompile operators used in the Jolt ONNX VM.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum PrecompileOperators {
    /// Matrix multiplication precompile.
    MatMult(MatMultPrecompile),
}
