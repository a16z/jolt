extern crate alloc;
use crate::jolt::instruction::or::ORInstruction;
use crate::jolt::instruction::xor::XORInstruction;
use crate::jolt::vm::JoltTraceStep;
use crate::jolt_onnx::vm::onnx_vm::ONNX;
use crate::utils::errors::ONNXError;
use crate::{
    jolt::instruction::and::ANDInstruction, jolt_onnx::instruction::relu::ReLUInstruction,
};
use alloc::string::String;
use alloc::vec::Vec;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::rv_trace::MemoryLayout;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracer::ELFInstruction;
use tract_onnx::prelude::*;

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

    // Bitwise operations
    And = 52,
    Or = 53,
    Xor = 54,

    // Other operation
    Clip = 55,
}

impl OperationType {
    /// Convert from string to operation type
    ///
    /// This method maps operation names from ONNX model nodes to their corresponding
    /// OperationType enum values. It handles various naming conventions and aliases
    /// used in ONNX models.
    pub fn parse(s: &str) -> Self {
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
            // Bitwise operations
            "And" => OperationType::And,
            "Or" => OperationType::Or,
            "Xor" => OperationType::Xor,

            // Other operations
            "Clip" => OperationType::Clip,

            // Unknown operation
            _ => {
                panic!("Unknown operation type: {s}");
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
    /// Creates a new instance of [`ComputationalGraph`]
    pub fn new(nodes: Vec<GraphNode>, input_count: usize, output_count: usize) -> Self {
        ComputationalGraph {
            nodes,
            input_count,
            output_count,
        }
    }

    /// Creates a new empty computational graph
    pub fn empty() -> Self {
        ComputationalGraph {
            nodes: Vec::new(),
            input_count: 0,
            output_count: 0,
        }
    }

    #[cfg(test)]
    /// Print the computational graph
    pub fn print(&self) {
        use std::collections::HashMap;

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

    // TODO: Implement a proper execution trace

    pub fn trace(&self) -> Vec<JoltTraceStep<ONNX>> {
        let mut trace = Vec::new();
        for node in &self.nodes {
            let opcode = match node.op_type {
                OperationType::And => Some(ANDInstruction::default().into()),
                OperationType::Or => Some(ORInstruction::default().into()),
                OperationType::Xor => Some(XORInstruction::default().into()),
                OperationType::Relu => Some(ReLUInstruction::default().into()),
                OperationType::Clip => None,
                OperationType::Input => None,
                _ => {
                    panic!("Unsupported operation type for tracing: {:?}", node.op_type);
                }
            };
            let mut step = JoltTraceStep::<ONNX>::no_op();
            step.instruction_lookup = opcode;
            trace.push(step);
        }
        trace
    }
}

/// ONNX Parser
pub struct ONNXParser;

impl ONNXParser {
    /// Load an ONNX model from a file
    pub fn load_model<P: AsRef<Path>>(path: P) -> Result<ComputationalGraph, ONNXError> {
        // Load the ONNX model using tract
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| ONNXError::InvalidModel(format!("Failed to load model: {e}")))?;

        let mut nodes = Vec::new();
        // Process each node in the model
        for (id, node) in model.nodes.iter().enumerate() {
            // Get the operation name
            let op_name = node.op.name();

            // Map the operation name to OperationType using the from_str method
            let op_type = OperationType::parse(op_name.as_ref());

            // Create the graph node
            let graph_node = GraphNode {
                id,
                op_type,
                name: node.name.clone(),
            };

            // Add the node to the graph
            nodes.push(graph_node);
        }

        Ok(ComputationalGraph::new(
            nodes,
            model.inputs.len(),
            model.outputs.len(),
        ))
    }
}

#[allow(clippy::too_long_first_doc_paragraph)]
/// Represented as a "peripheral device" in the RISC-V emulator, this captures
/// all reads from the reserved memory address space for program inputs and all writes
/// to the reserved memory address space for program outputs.
/// The inputs and outputs are part of the public inputs to the proof.
#[derive(
    Debug, Clone, PartialEq, Serialize, Deserialize, CanonicalSerialize, CanonicalDeserialize,
)]
pub struct JoltONNXDevice {
    pub inputs: Vec<u8>,
    pub outputs: Vec<u8>,
    pub panic: bool,
    pub memory_layout: MemoryLayout,
}

impl JoltONNXDevice {
    pub fn new(max_input_size: u64, max_output_size: u64) -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            panic: false,
            memory_layout: MemoryLayout::new(max_input_size, max_output_size),
        }
    }
}

/// Trivial [`TryFrom`] trait implementation for [`ONNX`] to make [`JoltInstructionSet`] trait happy
impl TryFrom<&ELFInstruction> for ONNX {
    type Error = &'static str;

    #[rustfmt::skip] // keep matches pretty
    fn try_from(_: &ELFInstruction) -> Result<Self, Self::Error> {
        Err("No corresponding ONNX instruction")
    }
}

#[cfg(test)]
mod tests {
    use super::ONNXParser;

    #[test]
    fn print_model() {
        let model_path = "../onnx/bitwise_test.onnx";
        let graph = ONNXParser::load_model(model_path).unwrap();
        graph.print();
    }
}
