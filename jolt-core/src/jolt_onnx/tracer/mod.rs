//! This module provides an interface for tracing ONNX models.

// TODO: Still need to decide on panic strategy — this module is unwrap/expect-heavy.
// Plan is to keep unwraps/expects where panics help catch dev bugs, and switch to proper error handling for actual runtime errors.

use model::QuantizedONNXModel;
use std::path::PathBuf;

use crate::jolt_onnx::common::onnx_trace::{JoltONNXDevice, ONNXTraceRow};

pub mod model;
pub mod tensor;
pub mod trace;

#[cfg(test)]
mod tests;

/// Generate's an execution trace for an ONNX model
pub fn trace(model_path: &PathBuf, input: &[f32]) -> (Vec<ONNXTraceRow>, JoltONNXDevice) {
    let mut model = QuantizedONNXModel::parse(model_path);
    let output = model.execute(input);
    let device = JoltONNXDevice::new(input.to_vec(), output.data.clone());
    let execution_trace = model.tracer.rows.clone();
    (execution_trace, device)
}
