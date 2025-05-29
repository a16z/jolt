//! This module provides the specialized sum-check precompile operators for Jolt's ONNX execution.

pub mod matmult;
pub mod sumcheck_engine;
use matmult::MatMultPrecompile;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum PrecompileOperators {
    MatMult(MatMultPrecompile),
}
