//! This module provides a [`Tracer`] for the ONNX runtime, that captures the pre-execution and
//! post-execution state of each layer in the model.
//!
//! The ONNX runtime will use this to construct an execution trace.

use super::tensor::QuantizedTensor;
use crate::jolt_onnx::common::onnx_trace::{LayerState, ONNXInstruction, ONNXTraceRow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A tracer for ONNX models that captures the execution trace
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Tracer {
    /// Execution trace of the ONNX model.
    pub rows: Vec<ONNXTraceRow>,
}

impl Tracer {
    /// Create a new instance of [`ONNXTraceRow`]
    pub fn start_instruction(&mut self, inst: ONNXInstruction) {
        self.rows.push(ONNXTraceRow {
            instruction: inst,
            layer_state: LayerState::default(),
        });
    }

    /// Capture the input values of a layer.
    /// Each node/layer in the ONNX model takes in a string of input names, which map to tensors in the `io_map`.
    /// We capture these input tensors before the layer is executed.
    pub fn capture_pre_state(&mut self, io_map: &HashMap<String, QuantizedTensor>) {
        let row = self.rows.last_mut().unwrap();
        let mut input_vals = Vec::new();
        for input_name in row.instruction.inputs.iter() {
            let input_tensor = io_map.get(input_name).unwrap();
            input_vals.push(input_tensor.clone());
        }
        row.layer_state.input_vals = Some(input_vals);
    }

    /// Capture the output values of a layer.
    /// Each node/layer in the ONNX model produces a string of output names, which map to tensors in the `io_map`.
    /// After the layer is executed, we capture the output tensors values.
    pub fn capture_post_state(&mut self, io_map: &HashMap<String, QuantizedTensor>) {
        let row = self.rows.last_mut().unwrap();
        let mut output_vals = Vec::new();
        for output_name in row.instruction.outputs.iter() {
            let output_tensor = io_map.get(output_name).unwrap();
            output_vals.push(output_tensor.clone());
        }
        row.layer_state.output_vals = Some(output_vals);
    }
}
