//! This module provides an ONNX program type that can be traced and executed using Jolt's VM.
//! It includes functionality to parse ONNX models, quantize them, and generate execution traces.
//! The ONNX program can be used to run inference on ONNX models and verify the results using Jolt's proof system.

use super::{
    common::onnx_trace::JoltONNXDevice,
    trace::onnx::onnxrow_to_lookup,
    tracer::{self, model::QuantizedONNXModel},
    vm::{onnx_vm::ONNXInstructionSet, JoltONNXTraceStep},
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
    #[tracing::instrument(skip_all, name = "ONNXProgram::decode")]
    /// Parse the ONNX program
    pub fn decode(&self) -> QuantizedONNXModel {
        QuantizedONNXModel::parse(&self.model)
    }

    #[tracing::instrument(skip_all, name = "ONNXProgram::trace")]
    /// Parse the ONNX model, quantize it & get the execution trace
    pub fn trace(&self) -> (JoltONNXDevice, Vec<JoltONNXTraceStep<ONNXInstructionSet>>) {
        let (raw_trace, io_device) = tracer::trace(&self.model, &self.input);
        let trace = raw_trace
            .iter()
            .flat_map(|row| {
                let lookups = onnxrow_to_lookup(row);
                if let Some(lookups) = lookups {
                    lookups
                        .into_iter()
                        .map(|lookup| {
                            let mut step = JoltONNXTraceStep::no_op();
                            step.instruction_lookup = Some(lookup);
                            step
                        })
                        .collect::<Vec<_>>()
                } else {
                    vec![JoltONNXTraceStep::no_op()]
                }
            })
            .collect::<Vec<_>>();
        (io_device, trace)
    }
}
