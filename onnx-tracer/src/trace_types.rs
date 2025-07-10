//! Type library used to build the execution trace.
//! Used to format the bytecode and define each instr flags and memory access patterns.
//! Used by the runtime to generate an execution trace for ONNX runtime execution.

/// Represents a single ONNX instruction parsed from the model.
/// The ONNX model is converted into a sequence of [`ONNXInstr`]s, forming the program code.
/// During runtime, the program counter (PC) is used to fetch the next instruction from this sequence.
pub struct ONNXInstr {
    address: usize,
    opcode: ONNXOpcode,
    /// Analogous to rs1 & rs2 except it maps addresses to tensors instead of registers
    ts1: usize,
    ts2: usize,
}

/// Operation code uniquely identifying each ONNX instruction's function
pub enum ONNXOpcode {
    Add,
    Sub,
    Mul,
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
