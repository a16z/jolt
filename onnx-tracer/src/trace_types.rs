//! Type library used to build the execution trace.
//! Used to format the bytecode and define each instr flags and memory access patterns.
//! Used by the runtime to generate an execution trace for ONNX runtime execution.

use crate::graph::node::Outlet;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Represents a single ONNX instruction parsed from the model.
/// The ONNX model is converted into a sequence of [`ONNXInstr`]s, forming the program code.
/// During runtime, the program counter (PC) is used to fetch the next instruction from this sequence.
/// Represents a single ONNX instruction in the bytecode sequence.
///
/// Each `ONNXInstr` contains the program counter address, the operation code,
/// and up to two input tensor operands (as `Outlet`s) that specify the sources
/// of tensor data in the computation graph. The operands are optional and may
/// be `None` if the instruction requires fewer than two inputs.
///
/// # Fields
/// - `address`: The program counter (PC) address of this instruction in the bytecode.
/// - `opcode`: The operation code (opcode) that defines the instruction's function.
/// - `ts1`: The first input tensor operand, specified as an `Option<Outlet>`. Analogous to the `rs1` register in RISC-V.
/// - `ts2`: The second input tensor operand, specified as an `Option<Outlet>`. Analogous to the `rs2` register in RISC-V.
pub struct ONNXInstr {
    /// The program counter (PC) address of this instruction in the bytecode.
    pub address: usize,
    /// The operation code (opcode) that defines the instruction's function.
    pub opcode: ONNXOpcode,
    /// The first input tensor operand, specified as an `Outlet`.
    /// An `Outlet` uniquely identifies the output of a node in the computation graph,
    /// effectively serving as a pointer to a specific tensor value produced during model execution.
    /// The `Outlet` is represented as a tuple `(node_idx, output_idx)`, where `node_idx` refers to the index of a node in the computation graph,
    /// and `output_idx` specifies which output of that node is being referenced. During execution, the runtime locates the node at `node_idx`
    /// in the graph, then selects its `output_idx`-th output tensor as the input for this instruction. This allows precise identification and
    /// retrieval of the required tensor value: `graph.nodes[node_idx].outputs[output_idx]`.
    /// At runtime, the executor uses this outlet to fetch the actual tensor data needed as input.
    /// If the instruction requires fewer than two inputs, this will be `None`.
    /// Conceptually, `ts1` is analogous to the `rs1` register specifier in RISC-V,
    /// as both indicate the source location (address or index) of an operand.
    pub ts1: Option<Outlet>,
    /// The second input tensor operand, also specified as an `Outlet`.
    /// Like `ts1`, this outlet identifies another tensor value in the computation graph,
    /// allowing the runtime to fetch the correct input during execution.
    /// If the instruction requires only one or zero inputs, this will be `None`.
    /// This field is analogous to the `rs2` register specifier in RISC-V,
    /// serving to specify the address or index of the second operand.
    pub ts2: Option<Outlet>,
}

// TODO: Expand the instruction set architecture (ISA):
//       For phase 1, we focus on supporting text-classification models.
//       This reduced ISA currently includes only the opcodes commonly used in such models.
//       Future phases should extend this set to support a broader range of ONNX operations.

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Operation code uniquely identifying each ONNX instruction's function
pub enum ONNXOpcode {
    Constant,
    Input,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Relu,
    MatMult,
    Gather,
    Transpose,
    Sqrt,
    ReduceMean,
    Sigmoid,
    Softmax,
}
