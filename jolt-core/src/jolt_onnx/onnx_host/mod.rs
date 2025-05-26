//! This module provides a tracer for the ONNX runtime.
//!
//! # Overview
//!
//! - [`tracer`] - This module provides a tracer for the ONNX runtime.
//! - [`ONNXProgram`] - Represents an ONNX model

use crate::jolt::vm::JoltTraceStep;

use super::{
    common::onnx_trace::JoltONNXDevice, trace::onnx::onnxrow_to_lookup, tracer,
    vm::onnx_vm::ONNXInstructionSet,
};
use std::path::PathBuf;

/// This type represents an ONNX model
pub struct ONNXProgram {
    /// The path to the ONNX model file
    pub model: PathBuf,
    /// The input to the ONNX model
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
    #[tracing::instrument(skip_all, name = "ONNXProgram::trace")]
    /// Parse the ONNX model, quantize it & get the execution trace
    pub fn trace(&self) -> (JoltONNXDevice, Vec<JoltTraceStep<ONNXInstructionSet>>) {
        let (raw_trace, io_device) = tracer::trace(&self.model, &self.input);
        let trace = raw_trace
            .iter()
            .flat_map(|row| {
                let lookups = onnxrow_to_lookup(row);
                if let Some(lookups) = lookups {
                    lookups
                        .into_iter()
                        .map(|lookup| {
                            let mut step = JoltTraceStep::no_op();
                            step.instruction_lookup = Some(lookup);
                            step
                        })
                        .collect::<Vec<_>>()
                } else {
                    vec![JoltTraceStep::no_op()]
                }
            })
            .collect::<Vec<_>>();
        (io_device, trace)
    }
}
