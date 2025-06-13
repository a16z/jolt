//! This module provides the types that are used to construct the execution trace from an ONNX runtime context.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tract_onnx::pb::NodeProto;

use crate::jolt_onnx::tracer::tensor::QuantizedTensor;

/// Represents a row in the execution trace
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ONNXTraceRow {
    /// The instruction that this row represents
    pub instruction: ONNXInstruction,
    /// The state of the layer during execution, including input and output values
    pub layer_state: LayerState,
}

/// Stores the input and output values of a layer
/// during the execution of the ONNX model
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerState {
    /// The input values to the layer, which are the quantized tensors
    pub input_vals: Option<Vec<QuantizedTensor>>,
    /// The output values from the layer, which are the quantized tensors
    pub output_vals: Option<Vec<QuantizedTensor>>,
}

/// Represents a single layer (node) in the ONNX model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ONNXInstruction {
    /// The operator that this instruction represents
    pub opcode: Operator,
    /// Optional attributes for the operator, such as alpha and beta for MatMul
    pub attributes: Option<HashMap<String, Vec<u64>>>,
    /// The inputs to the operator, which are the names of the tensors
    pub inputs: Vec<String>,
    /// The outputs of the operator, which are the names of the tensors
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

    /// Decorate the instruction with inputs and outputs from the ONNX node
    /// Additionally, it handles specific operators like MatMul to extract attributes such as alpha and beta.
    pub fn decorate(&mut self, node_proto: &NodeProto) {
        self.inputs = node_proto.input.clone();
        self.outputs = node_proto.output.clone();
        match self.opcode {
            Operator::MatMul => {
                // Get the alpha and beta values from the node attributes
                self.decorate_matmul(node_proto);
            }
            Operator::Relu => {}
            Operator::Conv => {
                self.decorate_conv(node_proto);
            }
        }
    }

    /// Add the alpha and beta values to the instruction's attributes
    fn decorate_matmul(&mut self, node_proto: &NodeProto) {
        let (alpha, beta) = alpha_beta(node_proto);
        self.attributes = Some(HashMap::from([
            ("alpha".to_string(), vec![alpha.to_bits() as u64]),
            ("beta".to_string(), vec![beta.to_bits() as u64]),
        ]));
    }

    ///
    fn decorate_conv(&mut self, node_proto: &NodeProto) {
        let get_attr = |name: &str| -> Vec<i64> {
            node_proto
                .attribute
                .iter()
                .find(|x| x.name.contains(name))
                .map_or_else(Vec::new, |x| x.ints.clone())
        };

        let (strides, pads, _kernel_shape, dilations) = (
            get_attr("strides"),
            get_attr("pads"),
            get_attr("kernel_shape"),
            get_attr("dilations"),
        );
    }
}

/// Represents an operator in the ONNX model
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operator {
    /// Matrix multiplication operator
    MatMul, // TODO: If operator has bias, handle it separately with an add operator
    /// Rectified Linear Unit (ReLU) activation function
    /// This is a non-linear activation function that outputs the input directly if it is positive;
    /// otherwise, it outputs zero.
    Relu,
    /// Convolution operator
    Conv,
}

/// Used to decorate the matmul operator with its attributes.
/// Give [`NodeProto`] parse the alpha and beta values from the node attributes
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
    /// Inputs to the ONNX model
    pub inputs: Vec<f32>,
    /// Outputs from the ONNX model
    pub outputs: Vec<f32>,
    /// Panic flag to indicate if the ONNX model execution panicked
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
