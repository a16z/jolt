//! Provides an API for constructing and tracing ONNX programs for integration with the proving system.
//!
//! This module enables loading ONNX models, managing their inputs, and preparing them for use in
//! the zkVM.

use onnx_tracer::{
    tensor::Tensor,
    trace_types::{ONNXCycle, ONNXInstr},
};
use std::path::PathBuf;

/// Represents an ONNX program with tracing capabilities.
/// The model binary is specified by a `PathBuf`, and model inputs are stored for inference.
pub struct ONNXProgram {
    /// The path to the ONNX model file.
    pub model_path: PathBuf,
    /// The inputs to the ONNX model, represented as a tensor.
    /// We quantize inputs to i128 bits for compatibility with the zkVM.
    /// We limit batch size to 1 for simplicity.
    pub inputs: Tensor<i128>,
}

impl ONNXProgram {
    /// Get the [`ONNXProgram`] bytecode in a format accessible to the constraint system.
    /// Called during pre-processsing time. We use this preprocessed bytecode to prove correctness of the bytecode trace.
    pub fn decode(&self) -> Vec<ONNXInstr> {
        onnx_tracer::decode(&self.model_path)
    }

    /// Get the execution trace for the [`ONNXProgram`].
    /// Called during proving time.
    pub fn trace(&self) -> Vec<ONNXCycle> {
        onnx_tracer::trace(&self.model_path, &self.inputs)
    }
}
