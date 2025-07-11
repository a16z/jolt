//! Type library used to build the execution trace.
//! Used to format the bytecode and define each instr flags and memory access patterns.
//! Used by the runtime to generate an execution trace for ONNX runtime execution.

use crate::graph::node::Outlet;

/// Represents a single ONNX instruction parsed from the model.
/// The ONNX model is converted into a sequence of [`ONNXInstr`]s, forming the program code.
/// During runtime, the program counter (PC) is used to fetch the next instruction from this sequence.
pub struct ONNXInstr {
    /// The program counter (PC) address of this instruction in the bytecode.
    pub address: usize,
    /// The operation code (opcode) that defines the instruction's function.
    pub opcode: ONNXOpcode,
    /// The first input tensor operand, specified as an `Outlet`.
    /// An `Outlet` uniquely identifies the output of a node in the computation graph,
    /// effectively serving as a pointer to a specific tensor value produced during model execution.
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

/// Operation code uniquely identifying each ONNX instruction's function
pub enum ONNXOpcode {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Relu,
    MatMult,
    Gather,
    Transpose,
    Sigmoid,
    ReduceMean,
    Softmax,
    Sqrt,
    Constant,
}
