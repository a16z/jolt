//! This module provides a tracer for the ONNX runtime.
//!
//! # Overview
//!
//! - [`tracer`] - This module provides a tracer for the ONNX runtime.
//! - [`ONNXProgram`] - Represents an ONNX model

pub mod tracer;

#[cfg(test)]
mod tests;

use std::path::PathBuf;

/// This type represents an ONNX model
pub struct ONNXProgram {
    pub model: PathBuf,
}

impl ONNXProgram {
    /// Create a new instance of [`ONNXProgram`]
    pub fn new(model: &str) -> Self {
        Self {
            model: PathBuf::from(model),
        }
    }
}

impl ONNXProgram {
    /// Parse the ONNX model, quantize it & get the execution trace
    pub fn trace(&self) {
        tracer::trace(&self.model);
    }
}
