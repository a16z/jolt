//! This module provides a way to parse and execute ONNX models.

use super::tensor::QuantizedTensor;
use super::trace::Tracer;
use crate::jolt_onnx::common::onnx_trace::{ONNXInstruction, Operator};
use crate::jolt_onnx::utils::create_tensor;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Deref;
use std::{path::PathBuf, str::FromStr};
use tract_onnx::pb::tensor_shape_proto::dimension::Value::{DimParam, DimValue};
use tract_onnx::pb::TensorProto;
use tract_onnx::{
    pb::{type_proto::Value, GraphProto},
    prelude::*,
};

/// Represents a topologically-sorted ONNX model, i.e. we store the operators in the order they are executed.
/// We also store the initializers of the model, which are the constant values used in the model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantizedONNXModel {
    /// Stores the instructions of the model, i.e. the operators and their attributes.
    pub instrs: Vec<ONNXInstruction>,
    /// Stores the constant-inputs of the computational graph; represents the initializers in the ONNX model.
    initializer_map: ONNXInitializerMap,
    /// Helps track the i/o shapes of the model layers.
    input_shape: Vec<usize>,
    /// Captures the execution trace of the model
    pub tracer: Tracer, // TODO: probably should not be part of the model.
}

impl QuantizedONNXModel {
    /// Parse the ONNX model & quantize it from `model_path`.
    /// Given a path output a [`QuantizedONNXModel`] type.
    pub fn parse(model_path: &PathBuf) -> Self {
        // Use tract to parse the ONNX model into a [`GraphProto`] type.
        // This  [`GraphProto`] is the raw ONNX model type, i.e. it is not optimized.
        // We will use this raw tract ONNX model type to help parse the model into our own model type.
        let graph = computational_graph(model_path);

        // Get constant-values of the graph.
        let initializer_map = ONNXInitializerMap::new(&graph.initializer);

        // --- Build the CodeMap ---
        // Map the `node.op_type: string` to the ONNXOperator enum.
        let instrs = graph
            .node
            .iter()
            .map(|node| {
                let mut instruction =
                    ONNXInstruction::new(Operator::from_str(node.op_type.as_str()).unwrap());
                // Update the instructions expected input and output names and
                // decorate the instruction with attributes if applicable.
                instruction.decorate(node);
                instruction
            })
            .collect_vec();
        Self::new(
            initializer_map,
            instrs,
            input_shape(&graph).expect("ONNX model should have a valid input shape"),
        )
    }

    /// Given the parsed ONNX model, store the input and output shapes of the model at each layer.
    /// Track the i/o shapes of the model layers. Useful for preprocessing, i.e. get the public parameters of the model.
    pub fn layers_io_shapes(&self) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut io_shapes = Vec::new();

        // Use the initial input shape to determine the i/o shapes of the model layers down the graph.
        let mut input_shape = self.input_shape.clone();

        // Iterate through the layers of the model and compute the input and output shapes
        // based on the operator type.
        for instr in self.instrs.iter() {
            // Store input shape for the current layer
            let layer_input_shape = input_shape.clone();

            // Determine the output shape based on the operator type
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
                Operator::Conv => {
                    todo!()
                }
            };

            // Store the input and output shapes for the current layer
            io_shapes.push((layer_input_shape, output_shape.clone()));

            // Update input shape for the next layer
            input_shape = output_shape.clone();
        }
        io_shapes
    }

    /// Execute the model with quantization on a given input
    pub fn execute_quantized(&mut self, input: &[f32]) -> QuantizedTensor {
        //
        let mut io_map = HashMap::<String, QuantizedTensor>::new();
        let input = QuantizedTensor::from(Tensor::from_shape(&[1, input.len()], input).unwrap());
        io_map.insert(
            "input".to_string(), // TODO: Make this more robust
            input.clone(),
        );
        for (key, value) in self.initializer_map.iter() {
            io_map.insert(key.clone(), value.clone());
        }

        for instr in self.instrs.iter() {
            self.tracer.start_instruction(instr.clone());
            self.tracer.capture_pre_state(&io_map);
            match instr.opcode {
                Operator::MatMul => {
                    // --- Gemm ---
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

                    // Output shape is [M, N]
                    let mut result = vec![0i32; m * n];
                    for i in 0..m {
                        for j in 0..n {
                            let mut acc = 0i32;
                            for t in 0..k {
                                let a_val = a.data[i * k + t] as i32;
                                let b_val = b.data[j * k + t] as i32;
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

                    // Requantize back to i8
                    let output_scale = a.scale * b.scale;
                    // TODO: Figure out a way around this otherwise we will have to prove this computation is correct
                    let quantized_result: Vec<i8> = result
                        .iter()
                        .map(|&x| (x as f32 / output_scale).round().clamp(-128.0, 127.0) as i8)
                        .collect();
                    let output_tensor = QuantizedTensor {
                        shape: vec![m, n],
                        data: quantized_result,
                        scale: output_scale,
                    };
                    io_map.insert(instr.outputs[0].to_string(), output_tensor);
                }

                Operator::Relu => {
                    let a = io_map.get(&instr.inputs[0]).unwrap();
                    let relu_data = a.data.iter().map(|&x| x.max(0)).collect_vec();
                    let output_tensor = QuantizedTensor {
                        shape: a.shape.clone(),
                        data: relu_data,
                        scale: a.scale,
                    };
                    io_map.insert(instr.outputs[0].to_string(), output_tensor);
                }
                Operator::Conv => {
                    todo!()
                }
            }
            self.tracer.capture_post_state(&io_map);
        }

        // Get the output tensor
        // TODO: Make this more robust, i.e. it is not guaranteed that the last instruction will have a single output
        // and that the output will be the first output.
        // For now, we assume that the last instruction has a single output and that it is the first output.
        // This is a hacky way to get the output tensor, we should improve this in the future.
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
            "Conv" => Ok(Operator::Conv),
            _ => Err(format!(
                "Could not match instruction {op} to ONNX operator set."
            )),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
/// Stores the constant-inputs of the computational graph; represents the initializers in the ONNX model.
pub struct ONNXInitializerMap(HashMap<String, QuantizedTensor>);

impl Deref for ONNXInitializerMap {
    type Target = HashMap<String, QuantizedTensor>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ONNXInitializerMap {
    /// Create a new instance of [`ONNXInitializerMap`].
    /// We convert the raw data from the initializers into `QuantizedTensor` types.
    pub fn new(initializers: &[TensorProto]) -> Self {
        let mut initializers_map = HashMap::new();
        for initializer in initializers.iter() {
            let dt = tract_onnx::pb::tensor_proto::DataType::from_i32(initializer.data_type)
                .and_then(|dt| dt.try_into().ok())
                .expect("Unsupported data type in ONNX initializer");
            let shape: Vec<usize> = initializer.dims.iter().map(|&i| i as usize).collect();
            let tensor = create_tensor(shape, dt, &initializer.raw_data).unwrap();
            let key = initializer.name.to_string();
            initializers_map.insert(key, tensor.into());
        }
        Self(initializers_map)
    }
}

/// Get the computational graph from a path.
fn computational_graph(model_path: &PathBuf) -> GraphProto {
    // Load the ONNX model using tract — we use proto to get the raw model,
    // i.e., the model without any optimizations — this makes the parsing more predictable. (TODO: we can change this in the future)
    let model = tract_onnx::onnx().proto_model_for_path(model_path).unwrap();
    model.graph.unwrap()
}

// TODO: Allow for dynamic batch sizes

/// Get the input shape from a graph, or None if anything is missing/unsupported.
/// We still only support batch-size=1 and a static 2nd dim.
fn input_shape(graph: &GraphProto) -> Option<Vec<usize>> {
    // 1) get the sole input
    let input = graph.input.first()?;

    // 2) drill into its TypeProto → Value → TensorType
    let ty = input.r#type.as_ref()?.value.as_ref()?;
    let Value::TensorType(tensor) = ty;

    // 3) get the shape and its 2nd dimension
    let dim = tensor.shape.as_ref()?.dim.get(1)?;

    // 4) extract a concrete usize from DimValue, bail on DimParam
    let size = match dim.value.as_ref()? {
        DimValue(n) => *n as usize,
        DimParam(_) => return None,
    };
    let batch_size = 1; // We only support batch size of 1 for now
    Some(vec![batch_size, size])
}
