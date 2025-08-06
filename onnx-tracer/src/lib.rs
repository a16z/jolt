//! # zkml-jolt ONNX Tracer Library
//!
//! This library provides utilities for converting computational graphs, such as neural networks in ONNX format, into ZK-circuits suitable for zero-knowledge proof systems. It is designed to facilitate the extraction, transformation, and tracing of ONNX models for use in the Jolt zkVM and proof system.
//!
//! ## Overview
//!
//! The main components of the library include:
//!
//! - **circuit**: Methods for configuring tensor operations and assigning values to them in a Halo2 circuit.
//! - **fieldutils**: Utilities for converting between Halo2 Field types and integers.
//! - **graph**: Methods for loading ONNX format models and automatically laying them out in a Halo2 circuit.
//! - **constants**: Constants used throughout the library, including bytecode and configuration values.
//! - **logger**: Logging utilities.
//! - **tensor**: Multi-dimensional tensor implementation and utilities.
//! - **trace_types**: Types representing execution traces and instructions for ONNX models.
//!
//! ## Key Functions
//!
//! - `decode`: Given a file path, decodes the ONNX model binary into a vector of `ONNXInstr`, representing the program code for the zkVM.
//! - `trace`: Provides an API to obtain the execution trace for an ONNX model and its inference input, producing a step-by-step record of VM state transitions.
//! - `execution_trace`: Internal function that runs the model and extracts its execution trace.
//! - `model`: Loads an ONNX model from a file path and returns a `Model` instance for further processing.
//! - `decode_node`: Converts a `NodeType` and its program counter into an `ONNXInstr`, used during decoding.
//!
//! ## Configuration
//!
//! - `RunArgs`: Struct containing parameters specific to a proving run, such as batch size, scale multipliers, and quantization denominators. Implements `Default` for easy initialization.
//! - `parse_key_val`: Utility function for parsing key-value pairs from strings, used for command-line argument parsing.
//!
//! ## Usage
//!
//! This library is intended for use in systems that require verifiable execution of neural network inference, such as zkML applications. It provides the necessary abstractions to load ONNX models, trace their execution, and prepare them for zero-knowledge proof generation.
//!
//! ## Extensibility
//!
//! The library is modular, allowing for extension and customization of tensor operations, model loading, and execution tracing. Advanced configuration options are available via `RunArgs` for fine-tuning proving runs.

// we allow this for our dynamic range based indexing scheme
#![allow(clippy::single_range_in_vec_init)]
#![allow(clippy::empty_docs)]

use crate::{
    circuit::ops::poly::PolyOp,
    constants::BYTECODE_PREPEND_NOOP,
    graph::{
        model::{Model, NodeType},
        utilities::{create_const_node, create_input_node, create_polyop_node},
    },
    tensor::Tensor,
    trace_types::{ONNXCycle, ONNXInstr},
};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::{fs::File, path::PathBuf};
/// Methods for configuring tensor operations and assigning values to them in a Halo2
/// circuit.
pub mod circuit;
/// Utilities for converting from Halo2 Field types to integers (and vice-versa).
pub mod fieldutils;
/// Methods for loading onnx format models and automatically laying them out in
/// a Halo2 circuit.
pub mod graph;
// /// An implementation of multi-dimensional tensors.
pub mod constants;
pub mod logger;
pub mod tensor;
pub mod trace_types;

/// The denominator in the fixed point representation used when quantizing inputs
pub type Scale = i32;

/// Given a file path decode the binary into [ONNXInstr] format
/// (i.e., the program code we will be reading into, in the zkVM)
///
/// # Returns
/// A vector of [`ONNXInstr`] representing the program code.
pub fn decode(model_path: &PathBuf) -> Vec<ONNXInstr> {
    decode_model(model(model_path))
}

/// Converts a [`Model`] into a vector of [`ONNXInstr`].
/// This function extracts the nodes from the model and decodes them into [`ONNXInstr`]'s.
pub fn decode_model(model: Model) -> Vec<ONNXInstr> {
    model.graph.nodes.iter().map(decode_node).collect()
}

/// Provides a simple API to obtain the execution trace for an ONNX model.
/// Use this to extract the execution trace from an ONNX model and its inference input, so it can be fed into the Jolt system.
///
/// An execution trace is, a step-by-step record of what the VM did over the course of its execution.
/// Roughly speaking, the trace describes just the changes to virtual machine state at each step of its execution (this includes read operations).
/// These state transitions are later checked & verified in the Jolt proof system, ensuring the prover possesses a valid execution trace for the given model and input.
pub fn trace(model_path: &PathBuf, input: &Tensor<i128>) -> Vec<ONNXCycle> {
    execution_trace(model(model_path), input)
}

/// Given a model and input extract the execution trace
pub fn execution_trace(model: Model, input: &Tensor<i128>) -> Vec<ONNXCycle> {
    // Run the model with the provided inputs.
    // The internal model tracer will automatically capture the execution trace during the forward pass
    let _ = model
        .forward(&[input.clone()])
        .expect("Failed to run model");
    let execution_trace = model.tracer.execution_trace.borrow().clone();
    execution_trace
}

/// Given a file path, load the ONNX model and return a [`Model`].
/// This function is used to initialize the model for further processing.
pub fn model(model_path: &PathBuf) -> Model {
    let mut file = File::open(model_path).expect("Failed to open ONNX model");
    // Default RunArgs (batch_size=1 by default)
    let run_args = RunArgs::default();
    // Load the model
    Model::new(&mut file, &run_args)
}

/// Converts a [`NodeType`] and its program counter into an [`ONNXInstr`].
/// This helper keeps the [decode] function concise and focused.
///
/// # NOTE:
/// Adds 1 to pc to account for prepended no-op
pub fn decode_node((pc, node): (&usize, &NodeType)) -> ONNXInstr {
    match node {
        NodeType::Node(node) => node.decode(*pc + BYTECODE_PREPEND_NOOP),
        NodeType::SubGraph { .. } => {
            todo!()
        }
    }
}

/// Parameters specific to a proving run
#[derive(Debug, Args, Deserialize, Serialize, Clone, PartialEq, PartialOrd)]
pub struct RunArgs {
    /// Hand-written parser for graph variables, eg. batch_size=1
    #[arg(short = 'V', long, value_parser = parse_key_val::<String, usize>, default_value =
  "batch_size=1", value_delimiter = ',')]
    pub variables: Vec<(String, usize)>,
    /// if the scale is ever > scale_rebase_multiplier * input_scale then the scale is
    /// rebased
    // to input_scale (this a more advanced parameter, use with caution)
    #[arg(long, default_value = "1")]
    pub scale_rebase_multiplier: u32,
    /// The denominator in the fixed point representation used when quantizing inputs
    #[arg(short = 'S', long, default_value = "7", allow_hyphen_values = true)]
    pub input_scale: Scale,
    /// The denominator in the fixed point representation used when quantizing
    /// parameters
    #[arg(long, default_value = "7", allow_hyphen_values = true)]
    pub param_scale: Scale,
    //     // /// The tolerance for error on model outputs
    //     // #[arg(short = 'T', long, default_value = "0")]
    //     // pub tolerance: Tolerance,
    //     /// The min and max elements in the lookup table input column
    //     #[arg(short = 'B', long, value_parser = parse_tuple::<i128>, default_value =
    // "(-32768,32768)")]     pub lookup_range: (i128, i128),
    //     /// The log_2 number of rows
    //     #[arg(short = 'K', long, default_value = "17")]
    //     pub logrows: u32,
    //     /// The log_2 number of rows
    //     #[arg(short = 'N', long, default_value = "2")]
    //     pub num_inner_cols: usize,
}

impl Default for RunArgs {
    fn default() -> Self {
        Self {
            input_scale: 7,
            param_scale: 7,
            scale_rebase_multiplier: 1,
            variables: vec![("batch_size".to_string(), 1)],
            // tolerance: Tolerance::default(),
            // lookup_range: (-32768, 32768),
            // logrows: 17,
            // num_inner_cols: 2,
        }
    }
}

/// Parse a single key-value pair
fn parse_key_val<T, U>(
    s: &str,
) -> Result<(T, U), Box<dyn std::error::Error + Send + Sync + 'static>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
    U: std::str::FromStr,
    U::Err: std::error::Error + Send + Sync + 'static,
{
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{s}`"))?;
    Ok((s[..pos].parse()?, s[pos + 1..].parse()?))
}

// impl RunArgs {
//     ///
//     pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
//         if self.scale_rebase_multiplier < 1 {
//             return Err("scale_rebase_multiplier must be >= 1".into());
//         }
//         if self.lookup_range.0 > self.lookup_range.1 {
//             return Err("lookup_range min is greater than max".into());
//         }
//         if self.logrows < 1 {
//             return Err("logrows must be >= 1".into());
//         }
//         if self.num_inner_cols < 1 {
//             return Err("num_inner_cols must be >= 1".into());
//         }
//         Ok(())
//     }

//     /// Export the ezkl configuration as json
//     pub fn as_json(&self) -> Result<String, Box<dyn std::error::Error>> {
//         let serialized = match serde_json::to_string(&self) {
//             Ok(s) => s,
//             Err(e) => {
//                 return Err(Box::new(e));
//             }
//         };
//         Ok(serialized)
//     }
//     /// Parse an ezkl configuration from a json
//     pub fn from_json(arg_json: &str) -> Result<Self, serde_json::Error> {
//         serde_json::from_str(arg_json)
//     }
// }

// /// Parse a tuple
// fn parse_tuple<T>(s: &str) -> Result<(T, T), Box<dyn std::error::Error + Send +
// Sync + 'static>> where
//     T: std::str::FromStr + Clone,
//     T::Err: std::error::Error + Send + Sync + 'static,
// {
//     let res = s.trim_matches(|p| p == '(' || p == ')').split(',');

//     let res = res
//         .map(|x| {
//             // remove blank space
//             let x = x.trim();
//             x.parse::<T>()
//         })
//         .collect::<Result<Vec<_>, _>>()?;
//     if res.len() != 2 {
//         return Err("invalid tuple".into());
//     }
//     Ok((res[0].clone(), res[1].clone()))
// }

/* ********************** Testing Model's ********************** */

/// # Program in (opcode, inputs) tuple format:
/// [(0, input, []), (1, add, [0, 0]), (2, sub, [1, 0]), (3, mul, [1, 2]), (4, add, [2, 3]), (5, output, [4])]
pub fn custom_addsubmul_model() -> Model {
    const SCALE: i32 = 7;
    const NODE_OUTPUT_IDX: usize = 0;
    let out_dims = vec![1, 4];
    let mut model = Model::default();

    // Create and insert input node
    let input_node = create_input_node(SCALE, out_dims.clone(), 0, 2);
    model.insert_node(input_node);

    // Create and insert add node
    let add_node = create_polyop_node(
        PolyOp::Add,
        SCALE,
        vec![(0, NODE_OUTPUT_IDX), (0, NODE_OUTPUT_IDX)],
        out_dims.clone(),
        1,
        2,
    );
    model.insert_node(add_node);

    // Create and insert sub node
    let sub_node = create_polyop_node(
        PolyOp::Sub,
        SCALE,
        vec![(1, NODE_OUTPUT_IDX), (0, NODE_OUTPUT_IDX)],
        out_dims.clone(),
        2,
        1,
    );
    model.insert_node(sub_node);

    // Create and insert mul node
    let mul_node = create_polyop_node(
        PolyOp::Mult,
        SCALE,
        vec![(1, NODE_OUTPUT_IDX), (2, NODE_OUTPUT_IDX)],
        out_dims.clone(),
        3,
        1,
    );
    model.insert_node(mul_node);

    // Create and insert second add node
    let add_node2 = create_polyop_node(
        PolyOp::Add,
        SCALE,
        vec![(2, NODE_OUTPUT_IDX), (3, NODE_OUTPUT_IDX)],
        out_dims.clone(),
        4,
        1,
    );
    model.insert_node(add_node2);

    // Set inputs and outputs
    model.set_inputs(vec![0]);
    model.set_outputs(vec![(4, NODE_OUTPUT_IDX)]);
    model
}

/// # Program in (idx, opcode, inputs) tuple format:
/// [(0, input, []), (1, const, []), (2, add, [0, 1]), (3, sub, [0, 1]), (4, mul, [2, 3]), (5, output, [4])]
pub fn custom_addsubmulconst_model() -> Model {
    const SCALE: i32 = 7;
    const NODE_OUTPUT_IDX: usize = 0;
    let mut model = Model::default();

    // Create and insert input node
    let input_node = create_input_node(SCALE, vec![1], 0, 2);
    model.insert_node(input_node);

    // Create constant tensor and node
    let mut const_tensor = Tensor::new(Some(&[50i128]), &[1, 1]).unwrap();
    const_tensor.set_scale(7);
    let const_node = create_const_node(
        const_tensor,
        Tensor::new(Some(&[]), &[0]).unwrap(),
        SCALE,
        vec![1],
        1,
        2,
    );
    model.insert_node(const_node);

    // Create and insert add node
    let add_node = create_polyop_node(
        PolyOp::Add,
        SCALE,
        vec![(0, NODE_OUTPUT_IDX), (1, NODE_OUTPUT_IDX)],
        vec![1],
        2,
        1,
    );
    model.insert_node(add_node);

    // Create and insert sub node
    let sub_node = create_polyop_node(
        PolyOp::Sub,
        SCALE,
        vec![(0, NODE_OUTPUT_IDX), (1, NODE_OUTPUT_IDX)],
        vec![1],
        3,
        1,
    );
    model.insert_node(sub_node);

    // Create and insert mul node
    let mul_node = create_polyop_node(
        PolyOp::Mult,
        SCALE,
        vec![(2, NODE_OUTPUT_IDX), (3, NODE_OUTPUT_IDX)],
        vec![1],
        4,
        1,
    );
    model.insert_node(mul_node);

    // Set inputs and outputs
    model.set_inputs(vec![0]);
    model.set_outputs(vec![(4, NODE_OUTPUT_IDX)]);
    model
}

/// # Program in (idx, opcode, inputs) tuple format:
/// [(0, input, []), (1, add, [0, 0]), (2, sub, [1, 0]), (3, mul, [1, 2]), (4, add, [2, 3]), (5, output, [4])]
pub fn scalar_addsubmul_model() -> Model {
    const SCALE: i32 = 7;
    const NODE_OUTPUT_IDX: usize = 0;
    let out_dims = vec![1];
    let mut model = Model::default();

    // Create and insert input node
    let input_node = create_input_node(SCALE, out_dims.clone(), 0, 2);
    model.insert_node(input_node);

    // Create and insert add node
    let add_node = create_polyop_node(
        PolyOp::Add,
        SCALE,
        vec![(0, NODE_OUTPUT_IDX), (0, NODE_OUTPUT_IDX)],
        out_dims.clone(),
        1,
        2,
    );
    model.insert_node(add_node);

    // Create and insert sub node
    let sub_node = create_polyop_node(
        PolyOp::Sub,
        SCALE,
        vec![(1, NODE_OUTPUT_IDX), (0, NODE_OUTPUT_IDX)],
        out_dims.clone(),
        2,
        1,
    );
    model.insert_node(sub_node);

    // Create and insert mul node
    let mul_node = create_polyop_node(
        PolyOp::Mult,
        SCALE,
        vec![(1, NODE_OUTPUT_IDX), (2, NODE_OUTPUT_IDX)],
        out_dims.clone(),
        3,
        1,
    );
    model.insert_node(mul_node);

    // Create and insert second add node
    let add_node2 = create_polyop_node(
        PolyOp::Add,
        SCALE,
        vec![(2, NODE_OUTPUT_IDX), (3, NODE_OUTPUT_IDX)],
        out_dims.clone(),
        4,
        1,
    );
    model.insert_node(add_node2);

    // Set inputs and outputs
    model.set_inputs(vec![0]);
    model.set_outputs(vec![(4, NODE_OUTPUT_IDX)]);
    model
}
