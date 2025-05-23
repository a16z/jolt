//! This module provides a [`Tracer`] for the ONNX runtime.

use super::tensor::{LiteTensor, QuantizedLiteTensor};
use crate::jolt_onnx::common::onnx_trace::{LayerState, ONNXInstruction, ONNXTraceRow};
use std::collections::HashMap;

#[derive(Default, Debug)]
pub struct Tracer {
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

    /// Capture the input values of a layer
    pub fn capture_pre_state(&mut self, io_map: &HashMap<String, QuantizedLiteTensor>) {
        let row = self.rows.last_mut().unwrap();
        let mut input_vals = Vec::new();
        for input_name in row.instruction.inputs.iter() {
            let input_tensor = io_map.get(input_name).unwrap();
            input_vals.push(input_tensor.clone());
        }
        row.layer_state.input_vals = Some(input_vals);
    }

    /// Capture the output values of a layer
    pub fn capture_post_state(&mut self, io_map: &HashMap<String, QuantizedLiteTensor>) {
        let row = self.rows.last_mut().unwrap();
        let mut output_vals = Vec::new();
        for output_name in row.instruction.outputs.iter() {
            let output_tensor = io_map.get(output_name).unwrap();
            output_vals.push(output_tensor.clone());
        }
        row.layer_state.output_vals = Some(output_vals);
    }
}
