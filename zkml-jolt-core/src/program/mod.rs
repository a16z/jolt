//! Provides an API for constructing and extracting execution-trace's from ONNX programs for
//! integration with the Jolt proving system.
//!
//! This module enables loading ONNX models, managing their inputs, and preparing them for use in
//! the zkVM, such as decoding them and tracing their execution.

use onnx_tracer::{
    tensor::Tensor,
    trace_types::{ONNXCycle, ONNXInstr},
};
use std::path::PathBuf;

/// Represents an ONNX program with tracing capabilities.
/// The model binary is specified by a `PathBuf`, and model inputs are stored for inference.
pub struct ONNXProgram {
    /// The path to the ONNX model file on disk.
    ///
    /// This field specifies the location of the binary ONNX model to be loaded, decoded, & extract an execution trace from.
    pub model_path: PathBuf,
    /// The inputs to the ONNX model, represented as a tensor.
    ///
    /// # Note:
    ///    - We quantize inputs to i128 bits for compatibility with the zkVM.
    ///    - We limit batch size to 1 for simplicity.
    pub inputs: Tensor<i128>,
}

impl ONNXProgram {
    /// Used to preprocess the ONNX program bytecode for the Jolt proving system.
    /// We preprocess the [`ONNXProgram`] bytecode into a format accessible to the constraint system.
    /// Called during pre-processsing time.
    ///
    /// # Returns
    ///  - `Vec<ONNXInstr>`: The preprocessed bytecode. This preprocessed bytecode serves as the read-only memory storing the program bytecode
    ///    we perform lookups into.
    pub fn decode(&self) -> Vec<ONNXInstr> {
        onnx_tracer::decode(&self.model_path)
    }

    /// Get the execution trace for the [`ONNXProgram`].
    /// Called during proving time.
    ///
    /// # Returns
    ///  - `Vec<ONNXCycle>`: A step by step record of what the ONNX runtime did over the course of its execution.
    pub fn trace(&self) -> Vec<ONNXCycle> {
        onnx_tracer::trace(&self.model_path, &self.inputs)
    }
}

impl ONNXProgram {
    /// Create a new ONNX program with the specified model path and inputs.
    ///
    /// # Arguments
    /// - `model_path`: The path to the ONNX model file.
    /// - `inputs`: The inputs to the ONNX model as a tensor.
    pub fn new(model_path: PathBuf, inputs: Tensor<i128>) -> Self {
        Self { model_path, inputs }
    }
}
