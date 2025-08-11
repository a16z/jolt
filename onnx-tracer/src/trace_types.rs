//! Type library used to build the execution trace.
//! Used to format the bytecode and define each instr flags and memory access patterns.
//! Used by the runtime to generate an execution trace for ONNX runtime execution.

use crate::{
    constants::{MAX_TENSOR_SIZE, ZERO_ADDR_PREPEND},
    tensor::Tensor,
};
use core::panic;
use rand::{rngs::StdRng, RngCore};
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};
use strum::EnumCount;
use strum_macros::EnumCount as EnumCountMacro;

/// Represents a step in the execution trace, where an execution trace is a `Vec<ONNXCycle>`.
/// Records what the VM did at a cycle of execution.
/// Constructed at each step in the VM execution cycle, documenting instr, reads & state changes (writes).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ONNXCycle {
    pub instr: ONNXInstr,
    pub memory_state: MemoryState,
    pub advice_value: Option<Tensor<i128>>,
}

impl ONNXCycle {
    pub fn no_op() -> Self {
        ONNXCycle {
            instr: ONNXInstr::no_op(),
            memory_state: MemoryState::default(),
            advice_value: None,
        }
    }

    pub fn random(opcode: ONNXOpcode, rng: &mut StdRng) -> Self {
        ONNXCycle {
            instr: ONNXInstr::dummy(opcode),
            memory_state: MemoryState::random(rng),
            advice_value: Some(Tensor::from(
                (0..MAX_TENSOR_SIZE).map(|_| rng.next_u64() as u32 as i32 as i128),
            )),
        }
    }

    // # NOTE: Adds [ZERO_ADDR_PREPEND] to the orignal traced value
    pub fn td(&self) -> usize {
        self.instr.td.map_or(0, |td| td + ZERO_ADDR_PREPEND)
    }

    // # NOTE: Adds [ZERO_ADDR_PREPEND] to the orignal traced value
    pub fn ts1(&self) -> usize {
        self.instr.ts1.map_or(0, |ts1| ts1 + ZERO_ADDR_PREPEND)
    }

    // # NOTE: Adds [ZERO_ADDR_PREPEND] to the orignal traced value
    pub fn ts2(&self) -> usize {
        self.instr.ts2.map_or(0, |ts2| ts2 + ZERO_ADDR_PREPEND)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Default, Serialize, Deserialize, PartialOrd, Ord)]
pub struct MemoryState {
    pub ts1_val: Option<Tensor<i128>>,
    pub ts2_val: Option<Tensor<i128>>,
    pub td_pre_val: Option<Tensor<i128>>,
    pub td_post_val: Option<Tensor<i128>>,
}

impl MemoryState {
    pub fn random(rng: &mut StdRng) -> Self {
        MemoryState {
            ts1_val: Some(
                Tensor::new(Some(&[rng.next_u64() as u32 as i32 as i128]), &[1]).unwrap(),
            ),
            ts2_val: Some(
                Tensor::new(Some(&[rng.next_u64() as u32 as i32 as i128]), &[1]).unwrap(),
            ),
            td_pre_val: Some(
                Tensor::new(Some(&[rng.next_u64() as u32 as i32 as i128]), &[1]).unwrap(),
            ),
            td_post_val: Some(
                Tensor::new(Some(&[rng.next_u64() as u32 as i32 as i128]), &[1]).unwrap(),
            ),
        }
    }
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
    pub imm: Option<Tensor<i128>>, // Immediate value, if applicable
    /// If this instruction is part of a "virtual sequence" (see Section 6.2 of the
    /// Jolt paper), then this contains the number of virtual instructions after this
    /// one in the sequence. I.e. if this is the last instruction in the sequence,
    /// `virtual_sequence_remaining` will be Some(0); if this is the penultimate instruction
    /// in the sequence, `virtual_sequence_remaining` will be Some(1); etc.
    pub virtual_sequence_remaining: Option<usize>,
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryOp {
    Read(u64, u64),       // (address, value)
    Write(u64, u64, u64), // (address, old_value, new_value)
}

impl MemoryOp {
    pub fn noop_read() -> Self {
        Self::Read(0, 0)
    }

    pub fn noop_write() -> Self {
        Self::Write(0, 0, 0)
    }

    pub fn address(&self) -> u64 {
        match self {
            MemoryOp::Read(a, _) => *a,
            MemoryOp::Write(a, _, _) => *a,
        }
    }
}

impl ONNXCycle {
    #[allow(clippy::type_complexity)]
    /// Converts the cycle's tensor state into memory operation tuples for ts1, ts2, and td.
    ///
    /// Each returned tuple contains:
    /// - A vector of memory addresses, obtained via `get_tensor_addresses`.
    /// - A vector of normalized values (u64), padded with zeros up to `MAX_TENSOR_SIZE`.
    ///
    /// Panics if any underlying tensor's length exceeds `MAX_TENSOR_SIZE`.
    pub fn to_memory_ops(
        &self,
    ) -> (
        (Vec<usize>, Vec<u64>),
        (Vec<usize>, Vec<u64>),
        (Vec<usize>, Vec<u64>, Vec<u64>),
    ) {
        let ts1 = (get_tensor_addresses(self.ts1()), self.ts1_vals());
        let ts2 = (get_tensor_addresses(self.ts2()), self.ts2_vals());
        let td = (
            get_tensor_addresses(self.td()),
            self.td_pre_vals(),
            self.td_post_vals(),
        );
        (ts1, ts2, td)
    }

    /// Returns normalized and padded values for ts1.
    ///
    /// - Normalizes each element of `ts1_val` via `normalize`.
    /// - Pads the resulting Vec<u64> with zeros up to `MAX_TENSOR_SIZE`.
    ///
    /// If no `ts1_val` is present, returns a zero-filled Vec<u64> of length `MAX_TENSOR_SIZE`.
    ///
    /// # Panics
    /// Panics if the tensor's length exceeds `MAX_TENSOR_SIZE`.
    pub fn ts1_vals(&self) -> Vec<u64> {
        self.build_vals(self.memory_state.ts1_val.as_ref(), "ts1_val")
    }

    /// Returns normalized and padded values for ts2.
    ///
    /// Behaves like `ts1_vals`, but for `ts2_val`.
    pub fn ts2_vals(&self) -> Vec<u64> {
        self.build_vals(self.memory_state.ts2_val.as_ref(), "ts2_val")
    }

    /// Returns normalized and padded post-execution values for td.
    ///
    /// - Normalizes each element of `td_post_val` via `normalize`.
    /// - Pads the Vec<u64> with zeros up to `MAX_TENSOR_SIZE`.
    ///
    /// If no `td_post_val` is present, returns a zero-filled Vec<u64> of length `MAX_TENSOR_SIZE`.
    ///
    /// # Panics
    /// Panics if `td_post_val`'s length exceeds `MAX_TENSOR_SIZE`.
    pub fn td_post_vals(&self) -> Vec<u64> {
        self.build_vals(self.memory_state.td_post_val.as_ref(), "td_post_val")
    }

    /// Returns a zero-filled Vec<u64> for pre-execution values of td.
    ///
    /// Currently always zeros; may change for const opcodes.
    pub fn td_pre_vals(&self) -> Vec<u64> {
        self.build_vals(self.memory_state.td_pre_val.as_ref(), "td_pre_val")
    }

    /// Helper to build normalized and padded u64 values from an optional TensorValue.
    ///
    /// - `tensor_opt`: Optional reference to the raw tensor values.
    /// - `name`: Used in panic message if length exceeds limit.
    ///
    /// # Panics
    /// - Panics if the tensor's length exceeds `MAX_TENSOR_SIZE`.
    /// ---
    /// Returns a Vec<u64> of normalized values, padded with zeros to `MAX_TENSOR_SIZE`.
    fn build_vals(&self, tensor_opt: Option<&Tensor<i128>>, name: &str) -> Vec<u64> {
        match tensor_opt {
            Some(t) => {
                assert!(
                    t.inner.len() <= MAX_TENSOR_SIZE,
                    "{} length exceeds MAX_TENSOR_SIZE; actual length = {}, MAX_TENSOR_SIZE = {}",
                    name,
                    t.inner.len(),
                    MAX_TENSOR_SIZE
                );
                let mut vals: Vec<u64> = t.inner.iter().map(normalize).collect();
                vals.resize(MAX_TENSOR_SIZE, 0);
                vals
            }
            None => vec![0u64; MAX_TENSOR_SIZE],
        }
    }

    /// Returns the optional tensor for ts1 (unmodified).
    pub fn ts1_val_raw(&self) -> Option<&Tensor<i128>> {
        self.memory_state.ts1_val.as_ref()
    }

    /// Returns the optional tensor for ts2 (unmodified).
    pub fn ts2_val_raw(&self) -> Option<&Tensor<i128>> {
        self.memory_state.ts2_val.as_ref()
    }

    /// Returns the optional tensor for td_post (unmodified).
    pub fn td_post_val_raw(&self) -> Option<&Tensor<i128>> {
        self.memory_state.td_post_val.as_ref()
    }

    /// Returns the optional tensor for advice.
    /// # Note normalizes the advice value to u64 and pads it to `MAX_TENSOR_SIZE`.
    /// # Panics if the advice value's length exceeds `MAX_TENSOR_SIZE`.
    pub fn advice_value(&self) -> Option<Vec<u64>> {
        self.advice_value.as_ref().map(|adv| {
            assert!(
                adv.inner.len() <= MAX_TENSOR_SIZE,
                "advice_value length exceeds MAX_TENSOR_SIZE"
            );
            let mut vals: Vec<u64> = adv.inner.iter().map(normalize).collect();
            vals.resize(MAX_TENSOR_SIZE, 0);
            vals
        })
    }

    pub fn imm(&self) -> Vec<u64> {
        self.instr.imm()
    }
}

/// Converts a tensor index to a vector of addresses.
/// Used in the zkVM to track all the onnx runtime machine tensor read and write addresses.
pub fn get_tensor_addresses(t: usize) -> Vec<usize> {
    let mut addresses = Vec::new();
    for i in 0..MAX_TENSOR_SIZE {
        addresses.push(t * MAX_TENSOR_SIZE + i);
    }
    addresses
}

// HACK(Forpee): This is a temporary function to normalize i128 values to u64 for the jolt execution trace.
/// Normalizes an i128 value to u64 by casting it through i32 and u32.
/// # Panics
/// Panics if the value's absolute value exceeds `i128::from(u32::MAX)`.
/// This is to ensure that the immediate value can be safely normalized to u32 and then store in 64 bits.
fn normalize(value: &i128) -> u64 {
    // TODO: Temp assert. We will remove this when we migrate runtime to 32-bit quant strat.
    assert!(
        value.abs() <= i128::from(u32::MAX),
        "Value out of bounds for normalization"
    );
    *value as i32 as u32 as u64
}

/// Boolean flags used in Jolt's R1CS constraints (`opflags` in the Jolt paper).
/// Note that the flags below deviate somewhat from those described in Appendix A.1
/// of the Jolt paper.
#[derive(Clone, Copy, Debug, PartialEq, EnumCountMacro)]
pub enum CircuitFlags {
    /// 1 if the first instruction operand is TS1 value; 0 otherwise.
    LeftOperandIsTs1Value,
    /// 1 if the first instruction operand is TS2 value; 0 otherwise.
    RightOperandIsTs2Value,
    /// 1 if the second instruction operand is `imm`; 0 otherwise.
    RightOperandIsImm,
    /// 1 if the first lookup operand is the sum of the two instruction operands.
    AddOperands,
    /// 1 if the first lookup operand is the difference between the two instruction operands.
    SubtractOperands,
    /// 1 if the first lookup operand is the product of the two instruction operands.
    MultiplyOperands,
    /// 1 if the lookup output is to be stored in `td` at the end of the step.
    WriteLookupOutputToTD,
    /// 1 if the instruction is "inline", as defined in Section 6.1 of the Jolt paper.
    InlineSequenceInstruction,
    /// 1 if the instruction is an assert, as defined in Section 6.1.1 of the Jolt paper.
    Assert,
    /// Used in virtual sequences; the program counter should be the same for the full sequence.
    DoNotUpdateUnexpandedPC,
    /// Is (virtual) advice instruction
    Advice,
    /// Is constant instruction
    Const,
}

pub const NUM_CIRCUIT_FLAGS: usize = CircuitFlags::COUNT;

impl ONNXInstr {
    #[rustfmt::skip]
    pub fn to_circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let mut flags = [false; NUM_CIRCUIT_FLAGS];

        flags[CircuitFlags::LeftOperandIsTs1Value as usize] = matches!(
            self.opcode,
            ONNXOpcode::Add
            | ONNXOpcode::Sub
            | ONNXOpcode::Mul
            | ONNXOpcode::VirtualMove
            | ONNXOpcode::VirtualAssertValidSignedRemainder
            | ONNXOpcode::VirtualAssertValidDiv0
            | ONNXOpcode::VirtualAssertEq
            | ONNXOpcode::Gte
        );

        flags[CircuitFlags::RightOperandIsTs2Value as usize] = matches!(
            self.opcode,
            ONNXOpcode::Add
            | ONNXOpcode::Sub
            | ONNXOpcode::Mul
            | ONNXOpcode::VirtualAssertValidSignedRemainder
            | ONNXOpcode::VirtualAssertValidDiv0
            | ONNXOpcode::VirtualAssertEq
            | ONNXOpcode::Gte
        );

        flags[CircuitFlags::RightOperandIsImm as usize] = matches!(
            self.opcode,
            | ONNXOpcode::VirtualMove
        );

        flags[CircuitFlags::AddOperands as usize] = matches!(
            self.opcode,
            ONNXOpcode::Add
            | ONNXOpcode::VirtualMove
        );

        flags[CircuitFlags::SubtractOperands as usize] = matches!(
            self.opcode,
            ONNXOpcode::Sub,
        );

        flags[CircuitFlags::MultiplyOperands as usize] = matches!(
            self.opcode,
            ONNXOpcode::Mul,
        );

        flags[CircuitFlags::WriteLookupOutputToTD as usize] = matches!(
            self.opcode,
            ONNXOpcode::Add
            | ONNXOpcode::Sub
            | ONNXOpcode::Mul
            | ONNXOpcode::VirtualAdvice
            | ONNXOpcode::VirtualMove
            | ONNXOpcode::VirtualConst
        );

        flags[CircuitFlags::Advice as usize] = matches!(
            self.opcode,
            ONNXOpcode::VirtualAdvice
        );

        flags[CircuitFlags::Const as usize] = matches!(
            self.opcode,
            ONNXOpcode::VirtualConst
        );

        flags[CircuitFlags::Assert as usize] = matches!(
            self.opcode,
            ONNXOpcode::VirtualAssertValidSignedRemainder
            | ONNXOpcode::VirtualAssertValidDiv0
            | ONNXOpcode::VirtualAssertEq
        );

        flags[CircuitFlags::InlineSequenceInstruction as usize] =
            self.virtual_sequence_remaining.is_some();
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] =
            self.virtual_sequence_remaining.unwrap_or(0) != 0;

        flags
    }
}

pub trait InterleavedBitsMarker {
    fn is_interleaved_operands(&self) -> bool;
}

impl InterleavedBitsMarker for [bool; NUM_CIRCUIT_FLAGS] {
    fn is_interleaved_operands(&self) -> bool {
        !self[CircuitFlags::AddOperands]
            && !self[CircuitFlags::SubtractOperands]
            && !self[CircuitFlags::MultiplyOperands]
            && !self[CircuitFlags::Advice]
            && !self[CircuitFlags::Const]
    }
}

impl Index<CircuitFlags> for [bool; NUM_CIRCUIT_FLAGS] {
    type Output = bool;
    fn index(&self, index: CircuitFlags) -> &bool {
        &self[index as usize]
    }
}

impl IndexMut<CircuitFlags> for [bool; NUM_CIRCUIT_FLAGS] {
    fn index_mut(&mut self, index: CircuitFlags) -> &mut bool {
        &mut self[index as usize]
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

    pub fn dummy(opcode: ONNXOpcode) -> Self {
        ONNXInstr {
            address: 0,
            opcode,
            ts1: None,
            ts2: None,
            td: None,
            imm: None,
            virtual_sequence_remaining: None,
        }
    }

    pub fn imm(&self) -> Vec<u64> {
        match self.imm.clone() {
            Some(imm) => {
                assert!(
                    imm.inner.len() <= MAX_TENSOR_SIZE,
                    "imm length exceeds MAX_TENSOR_SIZE"
                );
                let mut vals: Vec<u64> = imm.inner.iter().map(normalize).collect();
                vals.resize(MAX_TENSOR_SIZE, 0);
                vals
            }
            None => vec![0u64; MAX_TENSOR_SIZE],
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
    Gte,
    Reshape,

    // Virtual instructions
    VirtualAdvice,
    VirtualAssertValidSignedRemainder,
    VirtualAssertValidDiv0,
    VirtualMove,
    VirtualAssertEq,
    VirtualConst,
}

impl ONNXOpcode {
    // TODO: Refactor bitflag generation to be more extensible.
    // Currently uses manual bit shifting due to RebaseScale variant containing
    // a Box<ONNXOpcode>, which prevents simple discriminant-based conversion.
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

            // Virtual instructions
            ONNXOpcode::VirtualAdvice => 1u64 << 17,
            ONNXOpcode::VirtualAssertValidSignedRemainder => 1u64 << 18,
            ONNXOpcode::VirtualAssertValidDiv0 => 1u64 << 19,
            ONNXOpcode::VirtualMove => 1u64 << 20,
            ONNXOpcode::VirtualAssertEq => 1u64 << 21,
            ONNXOpcode::VirtualConst => 1u64 << 22,

            ONNXOpcode::Gte => 1u64 << 23,
            ONNXOpcode::Reshape => 1u64 << 24,
            _ => panic!("ONNXOpcode {self:#?} not implemented in into_bitflag"),
        }
    }
}
