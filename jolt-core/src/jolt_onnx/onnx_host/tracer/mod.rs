//! This module provides an interface for tracing ONNX models.

// TODO: Still need to decide on panic strategy — this module is unwrap/expect-heavy.
// Plan is to keep unwraps/expects where panics help catch dev bugs, and switch to proper error handling for actual runtime errors.

use model::QuantizedONNXModel;
use std::path::PathBuf;
use tensor::LiteTensor;

pub mod model;
pub mod tensor;
pub mod trace;

#[cfg(test)]
mod tests;

/// Generate's an execution trace for an ONNX model
pub fn trace(model_path: &PathBuf, input: &[f32]) -> LiteTensor {
    let model = QuantizedONNXModel::parse(model_path);
    model.execute(input)
}
