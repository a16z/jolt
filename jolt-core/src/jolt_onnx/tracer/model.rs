//! This module provides a way to parse and execute ONNX models.

// TODO:
//       * Still need to decide on panic strategy — this module is unwrap/expect-heavy.
//       Plan is to keep unwraps/expects where panics help catch dev bugs, and switch to proper error handling for actual runtime errors.
//
//       * Refactor duplicated code in `execute` and `execute_quantized`

use super::tensor::{LiteTensor, QuantizedLiteTensor};
use super::trace::Tracer;
use crate::jolt_onnx::common::onnx_trace::{ONNXInstruction, Operator};
use crate::jolt_onnx::utils::create_tensor;
use itertools::Itertools;
use std::collections::HashMap;
use std::ops::Deref;
use std::{path::PathBuf, str::FromStr};
use tract_onnx::pb::tensor_shape_proto::dimension::Value::{DimParam, DimValue};
use tract_onnx::pb::TensorProto;
use tract_onnx::{
    pb::{type_proto::Value, GraphProto},
    prelude::*,
};

/// Represents a topologically-sorted ONNX model
#[derive(Debug)]
pub struct QuantizedONNXModel {
    pub instrs: Vec<ONNXInstruction>,
    initializer_map: ONNXInitializerMap,
    input_shape: Vec<usize>,
    /// A tracer that captures the execution trace of the model
    pub tracer: Tracer,
}

impl QuantizedONNXModel {
    /// Parse the ONNX model & quantize it from `model_path`.
    /// Given a path output a [`QuantizedONNXModel`] type.
    pub fn parse(model_path: &PathBuf) -> Self {
        let graph = computational_graph(model_path);

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
        Self::new(initializer_map, instrs, input_shape(&graph))
    }

    /// Track the i/o shapes of the model layers. Useful for preprocessing, i.e. get the public parameters of the model.
    pub fn track_io_shapes(&self) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut io_shapes = Vec::new();
        let mut input_shape = self.input_shape.clone();
        for instr in self.instrs.iter() {
            let layer_input_shape = input_shape.clone();
            let output_shape = match instr.opcode {
                Operator::MatMul => {
                    // MatMul: Y = alpha * A * B^T + beta * C
                    // A: [M, K], B: [N, K], C: [N]
                    let a_shape = input_shape.clone();
                    let b_shape = self
                        .initializer_map
                        .get(&instr.inputs[1])
                        .unwrap()
                        .shape
                        .clone();
                    vec![a_shape[0], b_shape[0]] // Output shape is [M, N]
                }
                Operator::Relu => {
                    // Relu: Y = max(0, X)
                    input_shape.clone() // Output shape is same as input shape
                }
            };
            io_shapes.push((layer_input_shape, output_shape.clone()));
            // Update input shape for the next layer
            input_shape = output_shape.clone();
        }
        io_shapes
    }

    /// Execute the model on a given input
    pub fn execute(&mut self, input: &[f32]) -> LiteTensor {
        //
        let mut io_map = HashMap::<String, LiteTensor>::new();
        let input = LiteTensor::from(Tensor::from_shape(&[1, input.len()], input).unwrap());
        io_map.insert(
            "input".to_string(), // TODO: Make this more robust
            input.clone(),
        );
        for (key, value) in self.initializer_map.iter() {
            io_map.insert(key.clone(), value.clone());
        }

        for instr in self.instrs.iter() {
            match instr.opcode {
                Operator::MatMul => {
                    // Y = alpha * A * B^T + beta * C
                    // A: [M, K], B: [N, K], C: [N]

                    // TODO: I do not think it is guaranteed that instr.inputs[0] will be a, and instr.inputs[1] will be b, etc...

                    let a = io_map.get(&instr.inputs[0]).unwrap(); // shape: [M, K]
                    let b = io_map.get(&instr.inputs[1]).unwrap(); // shape: [N, K]
                    let c = io_map.get(&instr.inputs[2]).unwrap(); // shape: [N]
                    let (alpha, beta) = {
                        let attributes = instr.attributes.as_ref().unwrap();
                        (attributes[0], attributes[1])
                    };

                    // rows in A
                    let m = a.shape[0];
                    // cols in A == cols in B^T
                    let k = a.shape[1];
                    // rows in B == output cols
                    let n = b.shape[0];

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
                    io_map.insert(instr.outputs[0].clone(), output_tensor);
                }

                Operator::Relu => {
                    let a = io_map.get(&instr.inputs[0]).unwrap();
                    let relu_data = a
                        .data
                        .iter()
                        .map(|&x| if x < 0.0 { 0.0 } else { x })
                        .collect::<Vec<f32>>();
                    let output_tensor = LiteTensor {
                        shape: a.shape.clone(),
                        data: relu_data,
                    };
                    io_map.insert(instr.outputs[0].clone(), output_tensor);
                }
            }
        }

        // Get the output tensor // TODO: Make this more robust
        let output_tensor = io_map.get(&self.instrs.last().unwrap().outputs[0]).unwrap();
        output_tensor.clone()
    }

    /// Execute the model with quantization on a given input
    pub fn execute_quantized(&mut self, input: &[f32]) -> QuantizedLiteTensor {
        //
        let mut io_map = HashMap::<String, QuantizedLiteTensor>::new();
        let input =
            LiteTensor::from(Tensor::from_shape(&[1, input.len()], input).unwrap()).quantize();
        io_map.insert(
            "input".to_string(), // TODO: Make this more robust
            input.clone(),
        );
        for (key, value) in self.initializer_map.iter() {
            io_map.insert(key.clone(), value.quantize());
        }

        for instr in self.instrs.iter() {
            self.tracer.start_instruction(instr.clone());
            self.tracer.capture_pre_state(&io_map);
            match instr.opcode {
                Operator::MatMul => {
                    // Y = alpha * A * B^T + beta * C
                    // A: [M, K], B: [N, K], C: [N]

                    // TODO: I do not think it is guaranteed that instr.inputs[0] will be a, and instr.inputs[1] will be b, etc...

                    let a = io_map.get(&instr.inputs[0]).unwrap(); // shape: [M, K]
                    let b = io_map.get(&instr.inputs[1]).unwrap(); // shape: [N, K]
                    let c = io_map.get(&instr.inputs[2]).unwrap(); // shape: [N]
                    let (alpha, beta) = {
                        let attributes = instr.attributes.as_ref().unwrap();
                        (attributes[0] as i32, attributes[1] as i32)
                    };
                    // rows in A
                    let m = a.shape[0];
                    // cols in A == cols in B^T
                    let k = a.shape[1];
                    // rows in B == output cols
                    let n = b.shape[0];

                    let a_zp = a.zero_point as i32;
                    let b_zp = b.zero_point as i32;

                    // Output shape is [M, N]
                    let mut result = vec![0i32; m * n];

                    for i in 0..m {
                        for j in 0..n {
                            let mut acc = 0i32;
                            for t in 0..k {
                                let a_val = a.data[i * k + t] as i32 - a_zp;
                                let b_val = b.data[j * k + t] as i32 - b_zp;
                                acc += a_val * b_val;
                            }

                            let bias = if beta != 0 {
                                beta * c.data[j] as i32
                            } else {
                                0
                            };
                            result[i * n + j] = alpha * acc + bias;
                        }
                    }

                    // Requantize back to i8 (simulate TFLite or tract post-processing)
                    let output_scale = a.scale * b.scale + c.scale;
                    let output_zero_point = 0; // can be set based on min/max of result if needed

                    let quantized_result: Vec<i8> = result
                        .iter()
                        .map(|&x| {
                            ((x as f32 / output_scale) + output_zero_point as f32)
                                .round()
                                .clamp(-128.0, 127.0) as i8
                        })
                        .collect();

                    let output_tensor = QuantizedLiteTensor {
                        shape: vec![m, n],
                        data: quantized_result,
                        scale: output_scale,
                        zero_point: output_zero_point,
                    };

                    io_map.insert(instr.outputs[0].clone(), output_tensor);
                }

                Operator::Relu => {
                    let a = io_map.get(&instr.inputs[0]).unwrap();
                    let relu_data = a
                        .data
                        .iter()
                        .map(|&x| if x < 0 { 0 } else { x })
                        .collect_vec();
                    let output_tensor = QuantizedLiteTensor {
                        shape: a.shape.clone(),
                        data: relu_data,
                        scale: a.scale,
                        zero_point: a.zero_point,
                    };
                    io_map.insert(instr.outputs[0].clone(), output_tensor);
                }
            }
            self.tracer.capture_post_state(&io_map);
        }

        // Get the output tensor // TODO: Make this more robust
        let output_tensor = io_map.get(&self.instrs.last().unwrap().outputs[0]).unwrap();
        output_tensor.clone()
    }

    /// Create a new instance of [`QuantizedONNXModel`]
    pub fn new(
        initializer_map: ONNXInitializerMap,
        instrs: Vec<ONNXInstruction>,
        input_shape: Vec<usize>,
    ) -> Self {
        Self {
            initializer_map,
            instrs,
            input_shape,
            tracer: Tracer::default(),
        }
    }
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
/// Stores the constant-inputs of the computational graph; represents the initializers in the ONNX model.
pub struct QuantONNXInitializerMap(HashMap<String, QuantizedLiteTensor>);

impl Deref for QuantONNXInitializerMap {
    type Target = HashMap<String, QuantizedLiteTensor>;
    fn deref(&self) -> &Self::Target {
        &self.0
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
    /// Create a new instance of [`ONNXInitializerMap`]
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

    /// Quantize the initializers
    pub fn quantize(&self) -> QuantONNXInitializerMap {
        let mut initializers_map: HashMap<String, QuantizedLiteTensor> = HashMap::new();
        for (key, value) in self.0.iter() {
            let quantized = value.quantize();
            initializers_map.insert(key.clone(), quantized);
        }
        QuantONNXInitializerMap(initializers_map)
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
fn input_shape(graph: &GraphProto) -> Vec<usize> {
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
    vec![1, size as usize]
}
