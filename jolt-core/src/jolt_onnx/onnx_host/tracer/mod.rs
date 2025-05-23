//! This module provides an interface for tracing ONNX models.

// TODO: Still need to decide on panic strategy — this module is unwrap/expect-heavy.
// Plan is to keep unwraps/expects where panics help catch dev bugs, and switch to proper error handling for actual runtime errors.

use crate::jolt_onnx::utils::create_tensor;
use std::collections::HashMap;
use std::ops::Deref;
use std::{path::PathBuf, str::FromStr};
use tract_onnx::pb::tensor_shape_proto::dimension::Value::{DimParam, DimValue};
use tract_onnx::pb::{NodeProto, TensorProto};
use tract_onnx::{
    pb::{type_proto::Value, GraphProto},
    prelude::*,
};

#[cfg(test)]
mod tests;

/// Parse the ONNX model & quantize it.
pub fn parse(model_path: &PathBuf) -> QuantizedONNXModel {
    let graph = computational_graph(model_path);

    // Get input shape. This is used to track the i/o shapes of the rest of the model layers.
    let input_shape = input_shape(&graph);
    let mut instrs = Vec::new();
    let initializer_map = ONNXInitializerMap::new(&graph.initializer);
    for node in graph.node.iter() {
        let mut instruction =
            ONNXInstruction::new(Operator::from_str(node.op_type.as_str()).unwrap());

        // Decorate the instruction with its attributes
        instruction.decorate(node);
        instrs.push(instruction);
    }
    QuantizedONNXModel::new(initializer_map, input_shape, instrs)
}

/// Generate's an execution trace for an ONNX model
pub fn trace(model_path: &PathBuf) {
    let model = parse(model_path);
    println!("Model: {model:#?}");
}

/// Represents a topologically-sorted, quantized  ONNX model
#[derive(Debug)]
pub struct QuantizedONNXModel {
    input_shape: (usize, usize),
    instrs: Vec<ONNXInstruction>,
    initializer_map: ONNXInitializerMap,
}

impl QuantizedONNXModel {
    /// Create a new instance of [`QuantizedONNXModel`]
    pub fn new(
        initializer_map: ONNXInitializerMap,
        input_shape: (usize, usize),
        instrs: Vec<ONNXInstruction>,
    ) -> Self {
        Self {
            initializer_map,
            input_shape,
            instrs,
        }
    }
}

/// Represents a single layer (node) in the ONNX model
#[derive(Debug)]
pub struct ONNXInstruction {
    opcode: Operator,
    attributes: Option<Vec<f32>>,
    inputs: Vec<String>,
    outputs: Vec<String>,
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
    fn decorate(&mut self, node_proto: &NodeProto) {
        match self.opcode {
            Operator::MatMul => {
                self.decorate_matmul(node_proto);
            }
            Operator::Relu => {}
        }
    }

    fn decorate_matmul(&mut self, node_proto: &NodeProto) {
        let (alpha, beta) = alpha_beta(node_proto);
        self.attributes = Some(vec![alpha, beta]);
        self.inputs = node_proto.input.clone();
        self.outputs = node_proto.output.clone();
    }
}

/// Represents an operator in the ONNX model
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone)]
/// Represents a [`tract_onnx`] tensor for this codebase
pub struct LiteTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl LiteTensor {
    fn transposed(&self, alpha: f32) -> LiteTensor {
        let mut tensor_shape = self.shape.clone();
        // Reverse the shape to get the correct dimensions for transposing
        tensor_shape.reverse();
        let tensor_data = &self.data;
        let (m, n) = (tensor_shape[0], tensor_shape[1]);
        let mut transposed_data = vec![0.0; tensor_data.len()];

        // Transpose the data matrix
        for i in 0..m {
            for j in 0..n {
                transposed_data[i * n + j] = tensor_data[j * m + i] * alpha;
            }
        }
        LiteTensor {
            shape: tensor_shape,
            data: transposed_data,
        }
    }

    /// Multiply tensor with a scalar
    fn multiply(&self, beta: f32) -> LiteTensor {
        let tensor_data = &self.data;
        let tensor_shape = self.shape.clone();
        let multiplied_data = tensor_data.iter().map(|&x| x * beta).collect::<Vec<f32>>();
        LiteTensor {
            shape: tensor_shape,
            data: multiplied_data,
        }
    }
}

impl From<Tensor> for LiteTensor {
    fn from(tensor: Tensor) -> Self {
        let shape = tensor.shape().to_vec();
        let data = tensor.as_slice::<f32>().unwrap().to_vec();
        Self { shape, data }
    }
}

#[derive(Debug, Clone)]
/// Represents the initializers in the ONNX model
pub struct ONNXInitializerMap(HashMap<String, LiteTensor>);

impl Deref for ONNXInitializerMap {
    type Target = HashMap<String, LiteTensor>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ONNXInitializerMap {
    /// Create a new instance of [`ONNXInitializer`]
    pub fn new(initializers: &[TensorProto]) -> Self {
        let mut initializers_map = HashMap::new();
        for initializer in initializers.iter() {
            let dt = tract_onnx::pb::tensor_proto::DataType::from_i32(initializer.data_type)
                .unwrap()
                .try_into()
                .unwrap();
            let shape: Vec<usize> = initializer.dims.iter().map(|&i| i as usize).collect();
            let value = create_tensor(shape, dt, &initializer.raw_data).unwrap();
            let key = initializer.name.to_string();
            initializers_map.insert(key, value.into());
        }
        Self(initializers_map)
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
    // TODO: Allow for multiple inputs
    assert!(
        graph.input.len() == 1,
        "Graph should have exactly one input"
    );
    let input = graph.input.first().unwrap();
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

///
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
