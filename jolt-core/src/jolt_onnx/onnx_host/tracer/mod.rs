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

/// Parse the ONNX model & quantize it from `model_path`
pub fn parse(model_path: &PathBuf) -> QuantizedONNXModel {
    let graph = computational_graph(model_path);
    let input_shape = input_shape(&graph);

    // Get constant-inputs of the graph.
    let initializer_map = ONNXInitializerMap::new(&graph.initializer);
    let mut instrs = Vec::new();
    for node in graph.node.iter() {
        let mut instruction =
            ONNXInstruction::new(Operator::from_str(node.op_type.as_str()).unwrap());

        // Decorate the instruction with its attributes. For example, the MatMul operator has
        // attributes for alpha and beta, which are used to scale the input tensors.
        // The attributes are stored in the ONNX model's initializers, so we need to get the
        // serialized data for the attributes.
        instruction.decorate(node);
        instrs.push(instruction);
    }
    QuantizedONNXModel::new(initializer_map, input_shape, instrs)
}

/// Generate's an execution trace for an ONNX model
pub fn trace(model_path: &PathBuf, input: &[f32]) -> LiteTensor {
    let model = parse(model_path);
    model.execute(input)
}

/// Represents a topologically-sorted, quantized  ONNX model
#[derive(Debug)]
pub struct QuantizedONNXModel {
    input_shape: (usize, usize),
    instrs: Vec<ONNXInstruction>,
    initializer_map: ONNXInitializerMap,
}

impl QuantizedONNXModel {
    /// Execute the model on a given input
    pub fn execute(&self, input: &[f32]) -> LiteTensor {
        // let mut node_outputs: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();
        let mut input_values = HashMap::<String, LiteTensor>::new();
        let input = LiteTensor::from(Tensor::from_shape(&[1, input.len()], input).unwrap());
        input_values.insert(
            "input".to_string(), // TODO: Make this more robust
            input.clone(),
        );
        for (key, value) in self.initializer_map.iter() {
            input_values.insert(key.clone(), value.clone());
        }

        for instr in self.instrs.iter() {
            match instr.opcode {
                Operator::MatMul => {
                    let a = input_values.get(&instr.inputs[0]).unwrap(); // shape: [M, K]
                    let b = input_values.get(&instr.inputs[1]).unwrap(); // shape: [N, K]
                    let c = input_values.get(&instr.inputs[2]).unwrap(); // shape: [N]
                    let (alpha, beta) = {
                        let attributes = instr.attributes.as_ref().unwrap();
                        (attributes[0], attributes[1])
                    };

                    let m = a.shape[0]; // rows in A
                    let k = a.shape[1]; // cols in A == cols in B^T
                    let n = b.shape[0]; // rows in B == output cols

                    // Output shape is [M, N]
                    let mut result = vec![0.0; m * n];

                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0;
                            for t in 0..k {
                                let a_val = a.data[i * k + t]; // A[i][t]
                                let b_val = b.data[j * k + t]; // B[j][t] → B^T[t][j]
                                sum += a_val * b_val;
                            }
                            let bias = if beta != 0.0 { beta * c.data[j] } else { 0.0 };
                            result[i * n + j] = alpha * sum + bias;
                        }
                    }

                    let output_tensor = LiteTensor {
                        shape: vec![m, n],
                        data: result,
                    };

                    input_values.insert(instr.outputs[0].clone(), output_tensor);
                }

                Operator::Relu => {
                    let a = input_values.get(&instr.inputs[0]).unwrap();
                    let relu_data = a
                        .data
                        .iter()
                        .map(|&x| if x < 0.0 { 0.0 } else { x })
                        .collect::<Vec<f32>>();
                    let output_tensor = LiteTensor {
                        shape: a.shape.clone(),
                        data: relu_data,
                    };
                    input_values.insert(instr.outputs[0].clone(), output_tensor);
                }
            }
        }
        // Get the output tensor
        let output_tensor = input_values
            .get(&self.instrs.last().unwrap().outputs[0])
            .unwrap();
        output_tensor.clone()
    }

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

    fn decorate_matmul(&mut self, node_proto: &NodeProto) {
        let (alpha, beta) = alpha_beta(node_proto);
        self.attributes = Some(vec![alpha, beta]);
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
/// Stores the constant-inputs of the computational graph; represents the initializers in the ONNX model.
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
