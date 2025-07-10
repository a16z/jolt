//! Provides an API for constructing and tracing ONNX programs for integration with the proving system.
//!
//! This module enables loading ONNX models, managing their inputs, and preparing them for use in
//! the zkVM.

use onnx_tracer::tensor::Tensor;
use std::path::PathBuf;

/// Represents an ONNX program with tracing capabilities.
/// The model binary is specified by a `PathBuf`, and model inputs are stored for inference.
pub struct ONNXProgram {
    path: PathBuf,
    inputs: Tensor<i128>, // We quantize inputs to i28 bits
}

impl ONNXProgram {
    /// Get the [`ONNXProgram`] bytecode in a format accessible to the constraint system.
    /// Called during pre-processsing time.
    pub fn decode() {}

    /// Get the execution trace for the [`ONNXProgram`].
    /// Called during proving time.
    pub fn trace() {}
}
