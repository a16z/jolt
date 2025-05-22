//! This module provides an interface for tracing ONNX models.

// TODO: Still need to decide on panic strategy — this module is unwrap/expect-heavy.
// Plan is to keep unwraps/expects where panics help catch dev bugs, and switch to proper error handling for actual runtime errors.

use std::{path::PathBuf, str::FromStr};
use tract_onnx::pb::tensor_shape_proto::dimension::Value::{DimParam, DimValue};
use tract_onnx::{
    pb::{type_proto::Value, GraphProto},
    prelude::*,
};

/// Parse the ONNX model & quantize it.
pub fn parse(model_path: &PathBuf) -> QuantizedONNXModel {
    let graph = computational_graph(model_path);

    // Get input shape. This is used to track the i/o shapes of the rest of the model layers.
    println!("Input shape: {:?}", input_shape(&graph));
    let mut instrs = Vec::new();
    for (i, node) in graph.node.iter().enumerate() {
        instrs.push(ONNXInstruction::new(
            Operator::from_str(node.op_type.as_str()).unwrap(),
        ));
    }
    QuantizedONNXModel::new(instrs)
}

/// Generate's an execution trace for an ONNX model
pub fn trace(model_path: &PathBuf) {
    let model = parse(model_path);
    println!("Model: {model:#?}");
}

/// Represents a topologically-sorted, quantized  ONNX model
#[derive(Debug)]
pub struct QuantizedONNXModel {
    instrs: Vec<ONNXInstruction>,
}

impl QuantizedONNXModel {
    /// Create a new instance of [`QuantizedONNXModel`]
    pub fn new(instrs: Vec<ONNXInstruction>) -> Self {
        Self { instrs }
    }
}

/// Represents a single layer (node) in the ONNX model
#[derive(Debug)]
pub struct ONNXInstruction {
    opcode: Operator,
}

impl ONNXInstruction {
    /// Create a new instance of [`ONNXInstruction`]
    pub fn new(opcode: Operator) -> Self {
        Self { opcode }
    }
}

/// Represents an operator in the ONNX model
#[derive(Debug)]
pub enum Operator {
    MatMul,
    Relu,
}

impl FromStr for Operator {
    type Err = String;
    /// Match the [`Operator`] type to its string representation
    fn from_str(op: &str) -> Result<Self, Self::Err> {
        match op {
            "Gemm" => Ok(Operator::MatMul),
            "Relu" => Ok(Operator::Relu),
            _ => Err(format!(
                "Could not match instruction {op} to ONNX operator set."
            )),
        }
    }
}

/// Get a computational graph from a path.
fn computational_graph(model_path: &PathBuf) -> GraphProto {
    // Load the ONNX model using tract — we use proto to get the raw model,
    // i.e., the model without any optimizations — this makes the parsing more predictable. (TODO: we can change this in the future)
    let model = tract_onnx::onnx().proto_model_for_path(model_path).unwrap();
    model.graph.unwrap()
}

/// Get the input shape from a graph.
/// We restrict batch size to 1 for now. (TODO: we can change this in the future)
fn input_shape(graph: &GraphProto) -> (usize, usize) {
    let input = graph
        .input
        .first()
        .expect("Graph should have at least one input"); // TODO: Allow for multiple inputs
    let tensor_type = input
        .r#type
        .as_ref()
        .and_then(|t| t.value.as_ref())
        .map(|v| match v {
            Value::TensorType(tensor) => tensor,
        })
        .expect("Input should have tensor type");
    let shape = tensor_type
        .shape
        .as_ref()
        .expect("Tensor type should have a shape");
    let dim = shape
        .dim
        .get(1)
        .expect("Shape should have at least two dimensions");
    let size = match dim.value.as_ref().expect("Dimension should have a value") {
        DimValue(size) => *size,
        DimParam(_) => panic!("Dynamic input shape is not supported"),
    };
    (1, size as usize)
}
