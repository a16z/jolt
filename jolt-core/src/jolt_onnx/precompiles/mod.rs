//! This module implements the sum-check precompiles for ONNX operators.
pub mod matmult;
pub mod sumcheck_engine;
use matmult::MatMultPrecompile;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum PrecompileOperators {
    MatMult(MatMultPrecompile),
}
