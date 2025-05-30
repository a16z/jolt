//! This module provides a runtime and a corresponding tracer for ONNX models.
//! We use the tracer to generate an execution trace for a given ONNX model which will serve as input to the Jolt VM.

use crate::jolt_onnx::common::onnx_trace::{JoltONNXDevice, ONNXTraceRow};
use model::QuantizedONNXModel;
use std::path::PathBuf;

pub mod model;
pub mod tensor;
pub mod trace;

#[cfg(test)]
mod tests;

/// Given a models path and its corresponding input generate an execution trace for an ONNX model
pub fn trace(model_path: &PathBuf, input: &[f32]) -> (Vec<ONNXTraceRow>, JoltONNXDevice) {
    let mut model = QuantizedONNXModel::parse(model_path);
    let output = model.execute_quantized(input);
    let device = JoltONNXDevice::new(input.to_vec(), output.dequantize().data);
    (model.tracer.rows, device)
}
