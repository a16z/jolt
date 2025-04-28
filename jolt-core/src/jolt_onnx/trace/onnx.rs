extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tract_onnx::prelude::*;

use crate::utils::errors::ONNXError;

// Define an enum for operation types with explicit numeric values
// Including all binary and element-wise operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(usize)]
pub enum OperationType {
    // Basic operations
    Input = 0,
    Const = 1,
    Cast = 2,
    Conv = 3,
    MatMul = 4,
    MaxPool = 5,
    EinSum = 6,
    Relu = 7,
    Sigmoid = 8,
    TypedBinOp = 9,
    ElementWiseOp = 10,
    AddAxis = 11,
    Reshape = 12,
    RmAxis = 13,
    Gather = 14,
    Reduce = 15,
    Softmax = 16,

    // Binary operations
    Add = 17,
    Sub = 18,
    Mul = 19,
    Div = 20,
    Pow = 21,
    Max = 22,
    Min = 23,
    Rem = 24,
    ShiftLeft = 25,
    ShiftRight = 26,

    // Element-wise operations
    Abs = 27,
    Exp = 28,
    Ln = 29,
    Square = 30,
    Sqrt = 31,
    Recip = 32,
    Rsqrt = 33,
    Ceil = 34,
    Floor = 35,
    Round = 36,
    Cos = 37,
    Sin = 38,
    Tan = 39,
    Acos = 40,
    Asin = 41,
    Atan = 42,
    Cosh = 43,
    Sinh = 44,
    Tanh = 45,
    Erf = 46,
    Atanh = 47,
    Acosh = 48,
    Asinh = 49,
    Neg = 50,
    Sign = 51,

    // Fallback for unknown operations
    Unknown = 52,
}

#[cfg(feature = "host")]
impl OperationType {
    /// Convert from string to operation type
    ///
    /// This method maps operation names from ONNX model nodes to their corresponding
    /// OperationType enum values. It handles various naming conventions and aliases
    /// used in ONNX models.
    pub fn from_str(s: &str) -> Self {
        match s {
            "Source" => OperationType::Input,
            "Const" => OperationType::Const,
            "Cast" => OperationType::Cast,
            "Conv" => OperationType::Conv,
            "MatMul" | "Gemm" => OperationType::MatMul,
            "MaxPool" => OperationType::MaxPool,
            "EinSum" => OperationType::EinSum,
            "Relu" => OperationType::Relu,
            "Sigmoid" => OperationType::Sigmoid,
            "AddAxis" => OperationType::AddAxis,
            "Reshape" => OperationType::Reshape,
            "Softmax" => OperationType::Softmax,

            // Operations with prefixes
            name if name.starts_with("RmAxis") => OperationType::RmAxis,
            name if name.starts_with("Gather") => OperationType::Gather,
            name if name.starts_with("Reduce") => OperationType::Reduce,

            // Binary operations
            "Add" => OperationType::Add,
            "Sub" => OperationType::Sub,
            "Mul" => OperationType::Mul,
            "Div" => OperationType::Div,
            "Pow" => OperationType::Pow,
            "Max" => OperationType::Max,
            "Min" => OperationType::Min,
            "Rem" => OperationType::Rem,
            "ShiftLeft" => OperationType::ShiftLeft,
            "ShiftRight" => OperationType::ShiftRight,

            // Element-wise operations
            "Abs" => OperationType::Abs,
            "Exp" => OperationType::Exp,
            "Ln" => OperationType::Ln,
            "Square" => OperationType::Square,
            "Sqrt" => OperationType::Sqrt,
            "Recip" => OperationType::Recip,
            "Rsqrt" => OperationType::Rsqrt,
            "Ceil" => OperationType::Ceil,
            "Floor" => OperationType::Floor,
            "Round" => OperationType::Round,
            "Cos" => OperationType::Cos,
            "Sin" => OperationType::Sin,
            "Tan" => OperationType::Tan,
            "Acos" => OperationType::Acos,
            "Asin" => OperationType::Asin,
            "Atan" => OperationType::Atan,
            "Cosh" => OperationType::Cosh,
            "Sinh" => OperationType::Sinh,
            "Tanh" => OperationType::Tanh,
            "Erf" => OperationType::Erf,
            "Atanh" => OperationType::Atanh,
            "Acosh" => OperationType::Acosh,
            "Asinh" => OperationType::Asinh,
            "Neg" => OperationType::Neg,
            "Sign" => OperationType::Sign,

            // Unknown operation
            _ => {
                eprintln!("Unknown operation type: {}", s);
                OperationType::Unknown
            }
        }
    }

    /// Convert to usize value
    pub fn as_usize(&self) -> usize {
        *self as usize
    }
}

/// Represents a node in the computational graph
#[derive(Serialize, Deserialize)]
pub struct GraphNode {
    pub id: usize,
    pub op_type: OperationType,
    pub name: String,
}

/// Represents the computational graph
#[derive(Serialize, Deserialize)]
pub struct ComputationalGraph {
    pub nodes: Vec<GraphNode>,
    pub input_count: usize,
    pub output_count: usize,
}

impl ComputationalGraph {
    /// Creates a new empty computational graph
    pub fn new() -> Self {
        ComputationalGraph {
            nodes: Vec::new(),
            input_count: 0,
            output_count: 0,
        }
    }

    /// Print the computational graph
    pub fn print(&self) {
        println!("Computational Graph:");
        println!("Number of inputs: {}", self.input_count);
        println!("Number of outputs: {}", self.output_count);
        println!("Nodes ({}): ", self.nodes.len());

        // Count operation types
        let mut op_counts: HashMap<OperationType, usize> = HashMap::new();
        for node in &self.nodes {
            *op_counts.entry(node.op_type).or_insert(0) += 1;
        }

        // Print node count by type
        println!("\nOperation types:");
        for (op_type, count) in op_counts {
            println!("  {:?}: {} nodes", op_type, count);
        }

        // Print node details
        println!("\nNode details:");
        for node in &self.nodes {
            println!("  Node {}: {}", node.id, node.name);
            println!("    Type: {:?}", node.op_type);
        }
    }
}

/// ONNX Parser
pub struct ONNXParser;

impl ONNXParser {
    /// Load an ONNX model from a file
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<ComputationalGraph, ONNXError> {
        // Create an empty computational graph
        let mut graph = ComputationalGraph::new();

        // Load the ONNX model using tract
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| ONNXError::InvalidModel(format!("Failed to load model: {}", e)))?;

        // Count inputs and outputs
        graph.input_count = model.inputs.len();
        graph.output_count = model.outputs.len();

        // Process each node in the model
        for (id, node) in model.nodes.iter().enumerate() {
            // Get the operation name
            let op_name = node.op.name();

            // Map the operation name to OperationType using the from_str method
            let op_type = OperationType::from_str(op_name.as_ref());

            // Create the graph node
            let graph_node = GraphNode {
                id,
                op_type,
                name: node.name.clone(),
            };

            // Add the node to the graph
            graph.nodes.push(graph_node);
        }

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::ONNXParser;

    #[test]
    fn print_model() {
        let model_path = "onnx/linear.onnx";
        let graph = ONNXParser::load_model(model_path).unwrap();
        graph.print();
    }
}
