//! This module provides an interface for tracing ONNX models.

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
    let output = model.execute_quantized(input);
    let device = JoltONNXDevice::new(input.to_vec(), output.dequantize().data);
    let execution_trace = model.tracer.rows.clone();
    (execution_trace, device)
}
