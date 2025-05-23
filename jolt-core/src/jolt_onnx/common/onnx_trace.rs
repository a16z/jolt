//! This module provides the types that are used to construct the execution trace from an ONNX runtime context.

use crate::jolt_onnx::onnx_host::tracer::tensor::LiteTensor;
use serde::{Deserialize, Serialize};
use tract_onnx::pb::NodeProto;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ONNXTraceRow {
    pub instruction: ONNXInstruction,
    pub layer_state: LayerState,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerState {
    pub input_vals: Option<Vec<LiteTensor>>,
    pub output_vals: Option<Vec<LiteTensor>>,
}

/// Represents a single layer (node) in the ONNX model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ONNXInstruction {
    pub opcode: Operator,
    pub attributes: Option<Vec<f32>>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl ONNXInstruction {
    /// Create a new instance of [`ONNXInstruction`]
    pub fn new(opcode: Operator) -> Self {
        Self {
            opcode,
            attributes: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Decorate opcode with their attributes - uses the ONNX model's initializers
    /// to get the serialized data for the attributes.
    pub fn decorate(&mut self, node_proto: &NodeProto) {
        self.inputs = node_proto.input.clone();
        self.outputs = node_proto.output.clone();
        match self.opcode {
            Operator::MatMul => {
                // Get the alpha and beta values from the node attributes
                self.decorate_matmul(node_proto);
            }
            Operator::Relu => {}
        }
    }

    /// Add the alpha and beta values to the instruction's attributes
    fn decorate_matmul(&mut self, node_proto: &NodeProto) {
        let (alpha, beta) = alpha_beta(node_proto);
        self.attributes = Some(vec![alpha, beta]);
    }
}

/// Represents an operator in the ONNX model
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operator {
    MatMul,
    Relu,
}

/// Parse the alpha and beta values from the node attributes
fn alpha_beta(node_proto: &NodeProto) -> (f32, f32) {
    let attribute = |name: &str| {
        node_proto
            .attribute
            .iter()
            .find(|&a| a.name == name)
            .map(|a| a.f)
            .unwrap_or(1.0)
    };
    let alpha = attribute("alpha");
    let beta = attribute("beta");
    (alpha, beta)
}

/// I/O for ONNX runtime context.
/// The inputs and outputs are part of the public inputs to the proof.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct JoltONNXDevice {
    pub inputs: Vec<f32>,
    pub outputs: Vec<f32>,
    pub panic: bool,
}

impl JoltONNXDevice {
    /// Create a new instance of [`JoltONNXDevice`]
    pub fn new(inputs: Vec<f32>, outputs: Vec<f32>) -> Self {
        Self {
            inputs,
            outputs,
            panic: false,
        }
    }
}
