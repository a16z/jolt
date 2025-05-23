//! This module provides a tracer for the ONNX runtime.
//!
//! # Overview
//!
//! - [`tracer`] - This module provides a tracer for the ONNX runtime.
//! - [`ONNXProgram`] - Represents an ONNX model

use super::tracer;
use std::path::PathBuf;

/// This type represents an ONNX model
pub struct ONNXProgram {
    pub model: PathBuf,
    pub input: Vec<f32>,
}

impl ONNXProgram {
    /// Create a new instance of [`ONNXProgram`]
    pub fn new(model: &str, input: &[f32]) -> Self {
        Self {
            model: PathBuf::from(model),
            input: input.to_vec(),
        }
    }
}

impl ONNXProgram {
    /// Parse the ONNX model, quantize it & get the execution trace
    pub fn trace(&self) {
        tracer::trace(&self.model, &self.input);
    }
}
