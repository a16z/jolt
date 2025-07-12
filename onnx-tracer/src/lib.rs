//! A library for turning computational graphs, such as neural networks, into
//! ZK-circuits.

// we allow this for our dynamic range based indexing scheme
#![allow(clippy::single_range_in_vec_init)]
#![allow(clippy::empty_docs)]

use crate::{
    graph::model::{Model, NodeType},
    trace_types::ONNXInstr,
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
    model(model_path)
        .graph
        .nodes
        .iter()
        .map(decode_node)
        .collect()
}

/// Provides a simple API to obtain the execution trace for an ONNX model.
/// Use this to extract the execution trace from an ONNX model and input, so it can be supplied to the Jolt system.
///
/// The execution trace (or transcript) records the changes to the CPU state at each cycle of execution,
/// effectively capturing a step-by-step log of the VM's actions during model inference.
/// These state transitions are later verified in the Jolt proof system, ensuring the prover possesses a valid execution trace for the given model and input.
pub fn trace() {
    todo!()
}

/// Given a file path, load the ONNX model and return a [`Model`].
/// This function is used to initialize the model for further processing.
fn model(model_path: &PathBuf) -> Model {
    let mut file = File::open(model_path).expect("Failed to open ONNX model");
    // Default RunArgs (batch_size=1 by default)
    let run_args = RunArgs::default();
    // Load the model
    Model::new(&mut file, &run_args)
}

/// Converts a [`NodeType`] and its program counter into an [`ONNXInstr`].
/// This helper keeps the [decode] function concise and focused.
fn decode_node((pc, node): (&usize, &NodeType)) -> ONNXInstr {
    match node {
        NodeType::Node(node) => node.decode(*pc),
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
