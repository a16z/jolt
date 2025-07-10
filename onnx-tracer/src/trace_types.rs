//! Type library used to build the execution trace.
//! Used to format the bytecode and define each instr flags and memory access patterns.
//! Used by the runtime to generate an execution trace for ONNX runtime execution.

/// Represents a single ONNX instruction parsed from the model.
/// The ONNX model is converted into a sequence of [`ONNXInstr`]s, forming the program code.
/// During runtime, the program counter (PC) is used to fetch the next instruction from this sequence.
pub struct ONNXInstr {
    /// The program counter (PC) address of this instruction in the bytecode.
    pub address: usize,
    /// The operation code (opcode) that defines the instruction's function.
    pub opcode: ONNXOpcode,
    /// The first tensor source operand, analogous to rs1 in RISC-V.
    pub ts1: usize,
    /// The second tensor source operand, analogous to rs2 in RISC-V.
    pub ts2: usize,
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
}
