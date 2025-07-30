//! Type library used to build the execution trace.
//! Used to format the bytecode and define each instr flags and memory access patterns.
//! Used by the runtime to generate an execution trace for ONNX runtime execution.

use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use strum::EnumCount;
use strum_macros::EnumCount as EnumCountMacro;

/// Represents a step in the execution trace, where an execution trace is a `Vec<ONNXCycle>`.
/// Records what the VM did at a cycle of execution.
/// Constructed at each step in the VM execution cycle, documenting instr, reads & state changes (writes).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ONNXCycle {
    pub instr: ONNXInstr,
    pub memory_state: MemoryState,
    pub advice_value: Option<i128>,
}

impl ONNXCycle {
    pub fn no_op() -> Self {
        ONNXCycle {
            instr: ONNXInstr::no_op(),
            memory_state: MemoryState::default(),
            advice_value: None,
        }
    }

    pub fn td(&self) -> usize {
        self.instr.td.unwrap_or(0)
    }

    // HACKS(Forpee)
    pub fn td_write(&self) -> u64 {
        self.memory_state
            .td_post_val
            .as_ref()
            .map(|t| t.inner[0] as i64 as u64)
            .unwrap_or(0)
    }

    pub fn ts1_val(&self) -> u64 {
        self.memory_state
            .ts1_val
            .as_ref()
            .map(|t| t.inner[0] as i64 as u64)
            .unwrap_or(0)
    }

    pub fn ts2_val(&self) -> u64 {
        self.memory_state
            .ts2_val
            .as_ref()
            .map(|t| t.inner[0] as i64 as u64)
            .unwrap_or(0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Default, Serialize, Deserialize, PartialOrd, Ord)]
pub struct MemoryState {
    pub ts1_val: Option<Tensor<i128>>,
    pub ts2_val: Option<Tensor<i128>>,
    pub td_post_val: Option<Tensor<i128>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Represents a single ONNX instruction parsed from the model.
/// Represents a single ONNX instruction in the program code.
///
/// Each `ONNXInstr` contains the program counter address, the operation code,
/// and up to two input tensor operands that specify the sources
/// of tensor data in the computation graph. The operands are optional and may
/// be `None` if the instruction requires fewer than two inputs.
///
/// # Fields
/// - `address`: The program counter (PC) address of this instruction in the bytecode.
/// - `opcode`: The operation code (opcode) that defines the instruction's function.
/// - `ts1`: The first input tensor operand, specified as an `Option<usize>`, representing the index of a node in the computation graph. Analogous to the `rs1` register in RISC-V.
/// - `ts2`: The second input tensor operand, specified as an `Option<usize>`, representing the index of a node in the computation graph. Analogous to the `rs2` register in RISC-V.
///
/// The ONNX model is converted into a sequence of [`ONNXInstr`]s, forming the program code.
/// During runtime, the program counter (PC) is used to fetch the next instruction from this read-only memory storing the program bytecode.
pub struct ONNXInstr {
    /// The program counter (PC) address of this instruction in the bytecode.
    pub address: usize,
    /// The operation code (opcode) that defines the instruction's function.
    pub opcode: ONNXOpcode,
    /// The first input tensor operand, specified as the index of a node in the computation graph.
    /// This index (`node_idx`) identifies which node's output tensor will be used as input for this instruction.
    /// Since each node produces only one output tensor in this simplified ISA, the index is sufficient.
    /// If the instruction requires fewer than two inputs, this will be `None`.
    /// Conceptually, `ts1` is analogous to the `rs1` register specifier in RISC-V,
    /// as both indicate the source location (address or index) of an operand.
    pub ts1: Option<usize>,
    /// The second input tensor operand, also specified as the index of a node in the computation graph.
    /// Like `ts1`, this index identifies another node whose output tensor will be used as input.
    /// If the instruction requires only one or zero inputs, this will be `None`.
    /// This field is analogous to the `rs2` register specifier in RISC-V,
    /// serving to specify the address or index of the second operand.
    pub ts2: Option<usize>,
    /// The destination tensor index, which is the index of the node in the computation graph
    /// where the result of this instruction will be stored.
    /// This is analogous to the `rd` register specifier in RISC-V, indicating
    /// where the result of the operation should be written.
    pub td: Option<usize>,
    pub imm: Option<i128>,
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

/// Boolean flags used in Jolt's R1CS constraints (`opflags` in the Jolt paper).
/// Note that the flags below deviate somewhat from those described in Appendix A.1
/// of the Jolt paper.
#[derive(Clone, Copy, Debug, PartialEq, EnumCountMacro)]
pub enum CircuitFlags {
    // /// 1 if the second instruction operand is `imm`; 0 otherwise.
    // RightOperandIsImm,
    /// 1 if the first instruction operand is RS1 value; 0 otherwise.
    LeftOperandIsRs1Value,
    /// 1 if the first instruction operand is RS2 value; 0 otherwise.
    RightOperandIsRs2Value,
    /// 1 if the first lookup operand is the sum of the two instruction operands.
    AddOperands,
    /// 1 if the first lookup operand is the difference between the two instruction operands.
    SubtractOperands,
    /// 1 if the first lookup operand is the product of the two instruction operands.
    MultiplyOperands,
    // /// 1 if the instruction is a load (i.e. `LW`)
    // Load,
    // /// 1 if the instruction is a store (i.e. `SW`)
    // Store,
    // /// 1 if the instruction is a jump (i.e. `JAL`, `JALR`)
    // Jump,
    // /// 1 if the instruction is a branch (i.e. `BEQ`, `BNE`, etc.)
    // Branch,
    /// 1 if the lookup output is to be stored in `rd` at the end of the step.
    WriteLookupOutputToRD,
    // /// 1 if the instruction is "inline", as defined in Section 6.1 of the Jolt paper.
    // InlineSequenceInstruction,
    // /// 1 if the instruction is an assert, as defined in Section 6.1.1 of the Jolt paper.
    // Assert,
    // /// Used in virtual sequences; the program counter should be the same for the full sequence.
    // DoNotUpdateUnexpandedPC,
    // /// Is (virtual) advice instruction
    // Advice,
}

pub const NUM_CIRCUIT_FLAGS: usize = CircuitFlags::COUNT;

impl ONNXInstr {
    #[rustfmt::skip]
    pub fn to_circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];

        flags[CircuitFlags::LeftOperandIsRs1Value as usize] = matches!(
            self.opcode,
            ONNXOpcode::Add
            | ONNXOpcode::Sub
            | ONNXOpcode::Mul
        );

        flags[CircuitFlags::RightOperandIsRs2Value as usize] = matches!(
            self.opcode,
            ONNXOpcode::Add
            | ONNXOpcode::Sub
            | ONNXOpcode::Mul
        );

        flags[CircuitFlags::AddOperands as usize] = matches!(
            self.opcode,
            ONNXOpcode::Add,
        );

        flags[CircuitFlags::SubtractOperands as usize] = matches!(
            self.opcode,
            ONNXOpcode::Sub,
        );

        flags[CircuitFlags::MultiplyOperands as usize] = matches!(
            self.opcode,
            ONNXOpcode::Mul,
        );

        flags[CircuitFlags::WriteLookupOutputToRD as usize] = matches!(
            self.opcode,
            ONNXOpcode::Add
            | ONNXOpcode::Sub
            | ONNXOpcode::Mul
        );

        flags
    }
}

impl ONNXInstr {
    pub fn no_op() -> Self {
        ONNXInstr {
            address: 0,
            opcode: ONNXOpcode::Noop,
            ts1: None,
            ts2: None,
            td: None,
            imm: None,
            virtual_sequence_remaining: None,
        }
    }
}

// TODO: Expand the instruction set architecture (ISA):
//       For phase 1, we focus on supporting text-classification models.
//       This reduced ISA currently includes only the opcodes commonly used in such models.
//       Future phases should extend this set to support a broader range of ONNX operations.

#[derive(Clone, Hash, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Operation code uniquely identifying each ONNX instruction's function
pub enum ONNXOpcode {
    Noop,
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
    /// Used for the ReduceMean operator, which is internally converted to a
    /// combination of Sum and Div operations.
    Sum,
    MeanOfSquares,
    Sigmoid,
    Softmax,
    RebaseScale(Box<ONNXOpcode>),

    // Virtual instructions
    VirtualAdvice,
    VirtualAssertValidSignedRemainder,
    VirtualAssertValidDiv0,
    VirtualMove,
    VirtualAssertEq,
}

impl ONNXOpcode {
    pub fn into_bitflag(self) -> u64 {
        match self {
            ONNXOpcode::Noop => 1u64 << 0,
            ONNXOpcode::Constant => 1u64 << 1,
            ONNXOpcode::Input => 1u64 << 2,
            ONNXOpcode::Add => 1u64 << 3,
            ONNXOpcode::Sub => 1u64 << 4,
            ONNXOpcode::Mul => 1u64 << 5,
            ONNXOpcode::Div => 1u64 << 6,
            ONNXOpcode::Pow => 1u64 << 7,
            ONNXOpcode::Relu => 1u64 << 8,
            ONNXOpcode::MatMult => 1u64 << 9,
            ONNXOpcode::Gather => 1u64 << 10,
            ONNXOpcode::Transpose => 1u64 << 11,
            ONNXOpcode::Sqrt => 1u64 << 12,
            ONNXOpcode::Sum => 1u64 << 13,
            ONNXOpcode::MeanOfSquares => 1u64 << 14,
            ONNXOpcode::Sigmoid => 1u64 << 15,
            ONNXOpcode::Softmax => 1u64 << 16,
            _ => panic!("ONNXOpcode not implemented in into_bitflag"),
        }
    }
}
