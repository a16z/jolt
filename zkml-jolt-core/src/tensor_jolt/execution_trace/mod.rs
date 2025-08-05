use crate::jolt::JoltProverPreprocessing;
use itertools::Itertools;
use jolt_core::jolt::instruction::LookupQuery;
use jolt_core::poly::one_hot_polynomial::OneHotPolynomial;
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
    },
    utils::{interleave_bits, transcript::Transcript},
};
use onnx_tracer::constants::MAX_TENSOR_SIZE;
use onnx_tracer::trace_types::ONNXOpcode;
use onnx_tracer::trace_types::{CircuitFlags, ONNXCycle};
use onnx_tracer::trace_types::{NUM_CIRCUIT_FLAGS, ONNXInstr};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::jolt::instruction::{add::ADD, mul::MUL, sub::SUB};
use jolt_core::jolt::{instruction::InstructionLookup, lookup_table::LookupTables};

pub const WORD_SIZE: usize = 32;

pub type ExecutionTrace = Vec<JoltONNXCycle>;
pub type ONNXLookup = Vec<ElementWiseLookup>;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct MemoryOps {
    ts1_read: (usize, Vec<u64>),
    ts2_read: (usize, Vec<u64>),
    td_write: (usize, Vec<u64>, Vec<u64>),
}

impl MemoryOps {
    pub fn no_op() -> Self {
        MemoryOps {
            ts1_read: (0, vec![0; MAX_TENSOR_SIZE]),
            ts2_read: (0, vec![0; MAX_TENSOR_SIZE]),
            td_write: (0, vec![0; MAX_TENSOR_SIZE], vec![0; MAX_TENSOR_SIZE]),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct JoltONNXCycle {
    pub instruction_lookups: Option<ONNXLookup>,
    pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],
    pub memory_ops: MemoryOps,
    pub instr: ONNXInstr,
}

// TODO(Forpee): Refactor these clones in JoltONNXCycle::ts1_read, ts2_read, td_write
impl JoltONNXCycle {
    /// # Returns: (address, read_value)
    pub fn ts1_read(&self) -> (usize, Vec<u64>) {
        self.memory_ops.ts1_read.clone()
    }

    /// # Returns: (address, read_value)
    pub fn ts2_read(&self) -> (usize, Vec<u64>) {
        self.memory_ops.ts2_read.clone()
    }

    /// # Returns: (address, pre_value, post_value)
    pub fn td_write(&self) -> (usize, Vec<u64>, Vec<u64>) {
        self.memory_ops.td_write.clone()
    }

    pub fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        self.circuit_flags
    }

    pub fn instr(&self) -> ONNXInstr {
        self.instr.clone()
    }

    pub fn bytecode_line(&self) -> ONNXInstr {
        self.instr.clone()
    }

    pub fn no_op() -> Self {
        JoltONNXCycle {
            instruction_lookups: None,
            circuit_flags: [false; NUM_CIRCUIT_FLAGS],
            memory_ops: MemoryOps::no_op(),
            instr: ONNXInstr::no_op(),
        }
    }
}

impl From<ONNXCycle> for JoltONNXCycle {
    fn from(raw_cycle: ONNXCycle) -> Self {
        let mut cycle = JoltONNXCycle::no_op();
        let (ts1_read, ts2_read, td_write) = raw_cycle.to_tensor_memory_ops();
        cycle.memory_ops = MemoryOps {
            ts1_read,
            ts2_read,
            td_write,
        };
        cycle.circuit_flags = raw_cycle.instr.to_circuit_flags();
        cycle.instr = raw_cycle.instr;
        // TODO(Forpee): Refactor this footgun
        cycle.populate_instruction_lookups();
        cycle
    }
}

pub fn jolt_trace(raw_trace: Vec<ONNXCycle>) -> ExecutionTrace {
    raw_trace.into_iter().map(JoltONNXCycle::from).collect()
}

pub trait WitnessGenerator {
    fn generate_witness<F, PCS, ProofTranscript>(
        &self,
        trace: &[JoltONNXCycle],
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript;

    fn len() -> usize;

    fn from_index(index: usize) -> Self;

    fn to_index(&self) -> usize;
}

impl InstructionLookup<WORD_SIZE> for JoltONNXCycle {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        self.instruction_lookups
            .as_ref()
            .and_then(|lookups| lookups.first().and_then(|lookup| lookup.lookup_table()))
    }
}

impl<const WORD_SIZE: usize> ONNXLookupQuery<WORD_SIZE> for JoltONNXCycle {
    /// Returns a tuple of the instruction's inputs. If the instruction has only one input,
    /// one of the tuple values will be 0.
    fn to_instruction_inputs(&self) -> (Vec<u64>, Vec<i64>) {
        self.instruction_lookups.as_ref().map_or(
            (vec![0; MAX_TENSOR_SIZE], vec![0; MAX_TENSOR_SIZE]),
            |lookups| {
                lookups
                    .iter()
                    .map(|lookup| lookup.to_instruction_inputs())
                    .unzip()
            },
        )
    }

    /// Returns a tuple of the instruction's lookup operands. By default, these are the
    /// same as the instruction inputs returned by `to_instruction_inputs`, but in some cases
    /// (e.g. ADD, MUL) the instruction inputs are combined to form a single lookup operand.
    fn to_lookup_operands(&self) -> (Vec<u64>, Vec<u64>) {
        self.instruction_lookups.as_ref().map_or(
            (vec![0; MAX_TENSOR_SIZE], vec![0; MAX_TENSOR_SIZE]),
            |lookups| {
                lookups
                    .iter()
                    .map(|lookup| lookup.to_lookup_operands())
                    .unzip()
            },
        )
    }

    /// Converts this instruction's operands into a lookup index (as used in sparse-dense Shout).
    /// By default, interleaves the two bits of the two operands together.
    fn to_lookup_index(&self) -> Vec<u64> {
        self.instruction_lookups
            .as_ref()
            .map_or(vec![0; MAX_TENSOR_SIZE], |lookups| {
                lookups
                    .iter()
                    .map(|lookup| lookup.to_lookup_index())
                    .collect()
            })
    }

    /// Computes the output lookup entry for this instruction as a u64.
    fn to_lookup_output(&self) -> Vec<u64> {
        self.instruction_lookups
            .as_ref()
            .map_or(vec![0; MAX_TENSOR_SIZE], |lookups| {
                lookups
                    .iter()
                    .map(|lookup| lookup.to_lookup_output())
                    .collect()
            })
    }
}

pub trait ONNXLookupQuery<const WORD_SIZE: usize> {
    /// Returns a tuple of the instruction's inputs. If the instruction has only one input,
    /// one of the tuple values will be 0.
    fn to_instruction_inputs(&self) -> (Vec<u64>, Vec<i64>);

    /// Returns a tuple of the instruction's lookup operands. By default, these are the
    /// same as the instruction inputs returned by `to_instruction_inputs`, but in some cases
    /// (e.g. ADD, MUL) the instruction inputs are combined to form a single lookup operand.
    fn to_lookup_operands(&self) -> (Vec<u64>, Vec<u64>);

    /// Converts this instruction's operands into a lookup index (as used in sparse-dense Shout).
    /// By default, interleaves the two bits of the two operands together.
    fn to_lookup_index(&self) -> Vec<u64>;

    /// Computes the output lookup entry for this instruction as a u64.
    fn to_lookup_output(&self) -> Vec<u64>;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CommittedPolynomials {
    /* R1CS aux variables */
    /// The "left" input to the current instruction. Typically either the
    /// rs1 value or the current program counter.
    LeftInstructionInput,
    /// The "right" input to the current instruction. Typically either the
    /// rs2 value or the immediate value.
    RightInstructionInput,
    /// Product of `LeftInstructionInput` and `RightInstructionInput`
    Product,
    /// Whether the current instruction should write the lookup output to
    /// the destination register
    WriteLookupOutputToRD,
    /// Inc polynomial for the registers instance of Twist
    RdInc,
    /// One-hot ra polynomial for the instruction lookups instance of Shout.
    /// There are four (d=4) of these polynomials, `InstructionRa(0) .. InstructionRa(3)`
    InstructionRa(usize),
}

pub const ALL_COMMITTED_POLYNOMIALS: [CommittedPolynomials; 8] = [
    CommittedPolynomials::LeftInstructionInput,
    CommittedPolynomials::RightInstructionInput,
    CommittedPolynomials::Product,
    CommittedPolynomials::WriteLookupOutputToRD,
    CommittedPolynomials::InstructionRa(0),
    CommittedPolynomials::InstructionRa(1),
    CommittedPolynomials::InstructionRa(2),
    CommittedPolynomials::InstructionRa(3),
];

impl WitnessGenerator for CommittedPolynomials {
    fn generate_witness<F, PCS, ProofTranscript>(
        &self,
        trace: &[JoltONNXCycle],
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        match self {
            // CommittedPolynomials::LeftInstructionInput => {
            //     let coeffs: Vec<u64> = trace
            //         .par_iter()
            //         .map(|cycle| {
            //             cycle
            //                 .to_lookup()
            //                 .map(|lookup| LookupQuery::<32>::to_instruction_inputs(&lookup).0)
            //                 .unwrap_or_default()
            //         })
            //         .collect();
            //     coeffs.into()
            // }
            // CommittedPolynomials::RightInstructionInput => {
            //     let coeffs: Vec<i64> = trace
            //         .par_iter()
            //         .map(|cycle| {
            //             cycle
            //                 .to_lookup()
            //                 .map(|lookup| LookupQuery::<32>::to_instruction_inputs(&lookup).1)
            //                 .unwrap_or_default()
            //         })
            //         .collect();
            //     coeffs.into()
            // }
            // CommittedPolynomials::Product => {
            //     let coeffs: Vec<u64> = trace
            //         .par_iter()
            //         .map(|cycle| {
            //             cycle
            //                 .to_lookup()
            //                 .map(|lookup| {
            //                     let (left_input, right_input) =
            //                         LookupQuery::<32>::to_instruction_inputs(&lookup);
            //                     if left_input.checked_mul(right_input as u64).is_none() {
            //                         panic!(
            //                             "At cycle {cycle:?} Overflow in multiplication: {left_input} * {right_input}"
            //                         );
            //                     }
            //                     left_input * right_input as u64
            //                 })
            //                 .unwrap_or_default()
            //         })
            //         .collect();
            //     coeffs.into()
            // }
            // CommittedPolynomials::WriteLookupOutputToRD => {
            //     let coeffs: Vec<u8> = trace
            //         .par_iter()
            //         .map(|cycle| {
            //             let flag = cycle.instr.to_circuit_flags()
            //                 [CircuitFlags::WriteLookupOutputToRD as usize];
            //             (cycle.td() as u8) * (flag as u8)
            //         })
            //         .collect();
            //     coeffs.into()
            // }
            // CommittedPolynomials::RdInc => {
            //     let coeffs: Vec<i64> = trace
            //         .par_iter()
            //         .map(|cycle| {
            //             let pre_val = cycle.td_pre_val();
            //             let post_val = cycle.td_post_val();
            //             post_val as i64 - pre_val as i64
            //         })
            //         .collect();
            //     coeffs.into()
            // }
            // CommittedPolynomials::InstructionRa(i) => {
            //     if *i > 3 {
            //         panic!("Unexpected i: {i}");
            //     }
            //     let addresses: Vec<usize> = trace
            //         .par_iter()
            //         .map(|cycle| {
            //             let lookup_index = cycle.to_lookup().map_or(0, |x| x.to_lookup_index());
            //             let k = (lookup_index >> (16 * (3 - i))) % (1 << 16);
            //             k as usize
            //         })
            //         .collect();
            //     MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(addresses, 1 << 16))
            // }
            _ => todo!("Witness generation for {self:?} is not implemented"),
        }
    }

    fn len() -> usize {
        ALL_COMMITTED_POLYNOMIALS.len()
    }

    fn from_index(index: usize) -> Self {
        ALL_COMMITTED_POLYNOMIALS[index]
    }

    fn to_index(&self) -> usize {
        ALL_COMMITTED_POLYNOMIALS
            .iter()
            .find_position(|poly| *poly == self)
            .unwrap()
            .0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JoltONNXR1CSInputs {
    Rd, // Virtual (bytecode rv)
    RdWriteValue,
    LeftInstructionInput,  // to_lookup_query -> to_instruction_operands
    RightInstructionInput, // to_lookup_query -> to_instruction_operands
    LeftLookupOperand,     // Virtual (instruction raf)
    RightLookupOperand,    // Virtual (instruction raf)
    Product,               // LeftInstructionOperand * RightInstructionOperand
    WriteLookupOutputToRD,
    LookupOutput, // Virtual (instruction rv)
    OpFlags(CircuitFlags),
}

/// This const serves to define a canonical ordering over inputs (and thus indices
/// for each input). This is needed for sumcheck.
pub const ALL_R1CS_INPUTS: [JoltONNXR1CSInputs; 13] = [
    JoltONNXR1CSInputs::LeftInstructionInput,
    JoltONNXR1CSInputs::RightInstructionInput,
    JoltONNXR1CSInputs::Product,
    JoltONNXR1CSInputs::WriteLookupOutputToRD,
    JoltONNXR1CSInputs::Rd,
    JoltONNXR1CSInputs::RdWriteValue,
    JoltONNXR1CSInputs::LeftLookupOperand,
    JoltONNXR1CSInputs::RightLookupOperand,
    JoltONNXR1CSInputs::LookupOutput,
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::AddOperands),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD),
];

/// The subset of `ALL_R1CS_INPUTS` that are committed. The rest of
/// the inputs are virtual polynomials.
pub const COMMITTED_R1CS_INPUTS: [JoltONNXR1CSInputs; 4] = [
    JoltONNXR1CSInputs::LeftInstructionInput,
    JoltONNXR1CSInputs::RightInstructionInput,
    JoltONNXR1CSInputs::Product,
    JoltONNXR1CSInputs::WriteLookupOutputToRD,
];

impl WitnessGenerator for JoltONNXR1CSInputs {
    /// The total number of unique constraint inputs
    fn len() -> usize {
        ALL_R1CS_INPUTS.len()
    }

    /// Converts an index to the corresponding constraint input.
    fn from_index(index: usize) -> Self {
        ALL_R1CS_INPUTS[index]
    }

    /// Converts a constraint input to its index in the canonical
    /// ordering over inputs given by `ALL_R1CS_INPUTS`.
    fn to_index(&self) -> usize {
        match ALL_R1CS_INPUTS.iter().position(|x| x == self) {
            Some(index) => index,
            None => panic!("Invalid variant {self:?}"),
        }
    }

    fn generate_witness<F, PCS, ProofTranscript>(
        &self,
        trace: &[JoltONNXCycle],
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        match self {
            // JoltONNXR1CSInputs::Rd => {
            //     let coeffs: Vec<u8> = trace.par_iter().map(|cycle| cycle.td() as u8).collect();
            //     coeffs.into()
            // }
            // JoltONNXR1CSInputs::RdWriteValue => {
            //     let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.td_post_val()).collect();
            //     coeffs.into()
            // }
            // JoltONNXR1CSInputs::LeftInstructionInput => {
            //     CommittedPolynomials::LeftInstructionInput.generate_witness(preprocessing, trace)
            // }
            // JoltONNXR1CSInputs::RightInstructionInput => {
            //     CommittedPolynomials::RightInstructionInput.generate_witness(preprocessing, trace)
            // }
            // JoltONNXR1CSInputs::LeftLookupOperand => {
            //     let coeffs: Vec<u64> = trace
            //         .par_iter()
            //         .map(|cycle| {
            //             cycle
            //                 .to_lookup()
            //                 .map(|lookup| LookupQuery::<32>::to_lookup_operands(&lookup).0)
            //                 .unwrap_or(0)
            //         })
            //         .collect();
            //     coeffs.into()
            // }
            // JoltONNXR1CSInputs::RightLookupOperand => {
            //     let coeffs: Vec<u64> = trace
            //         .par_iter()
            //         .map(|cycle| {
            //             cycle
            //                 .to_lookup()
            //                 .map(|lookup| LookupQuery::<32>::to_lookup_operands(&lookup).1)
            //                 .unwrap_or(0)
            //         })
            //         .collect();
            //     coeffs.into()
            // }
            // JoltONNXR1CSInputs::Product => {
            //     CommittedPolynomials::Product.generate_witness(preprocessing, trace)
            // }
            // JoltONNXR1CSInputs::WriteLookupOutputToRD => {
            //     CommittedPolynomials::WriteLookupOutputToRD.generate_witness(preprocessing, trace)
            // }
            // JoltONNXR1CSInputs::LookupOutput => {
            //     let coeffs: Vec<u64> = trace
            //         .par_iter()
            //         .map(|cycle| {
            //             cycle
            //                 .to_lookup()
            //                 .map(|lookup| LookupQuery::<32>::to_lookup_output(&lookup))
            //                 .unwrap_or_default()
            //         })
            //         .collect();
            //     coeffs.into()
            // }
            // JoltONNXR1CSInputs::OpFlags(flag) => {
            //     let coeffs: Vec<u8> = trace
            //         .par_iter()
            //         .map(|cycle| cycle.instr.to_circuit_flags()[*flag as usize] as u8)
            //         .collect();
            //     coeffs.into()
            // }
            _ => todo!("Implement witness generation for this input"),
        }
    }
}

macro_rules! define_lookup_enum {
    (
        enum $enum_name:ident,
        const $word_size:ident,
        trait $trait_name:ident,
        $($variant:ident : $inner:ty),+ $(,)?
    ) => {
        #[derive(Clone, Debug, Serialize, Deserialize)]
        pub enum $enum_name {
            $(
                $variant($inner),
            )+
        }

        impl $trait_name<$word_size> for $enum_name {
            fn to_instruction_inputs(&self) -> (u64, i64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_instruction_inputs(),
                    )+
                }
            }

            fn to_lookup_index(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_index(),
                    )+
                }
            }

            fn to_lookup_operands(&self) -> (u64, u64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_operands(),
                    )+
                }
            }

            fn to_lookup_output(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_output(),
                    )+
                }
            }
        }
    };
}

define_lookup_enum!(
    enum ElementWiseLookup,
    const WORD_SIZE,
    trait LookupQuery,
    Add: ADD<WORD_SIZE>,
    Sub: SUB<WORD_SIZE>,
    Mul: MUL<WORD_SIZE>,
);

impl InstructionLookup<WORD_SIZE> for ElementWiseLookup {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        match self {
            ElementWiseLookup::Add(add) => add.lookup_table(),
            ElementWiseLookup::Sub(sub) => sub.lookup_table(),
            ElementWiseLookup::Mul(mul) => mul.lookup_table(),
        }
    }
}

impl JoltONNXCycle {
    pub fn populate_instruction_lookups(&mut self) {
        self.instruction_lookups = self.to_instruction_lookups();
    }
    pub fn to_instruction_lookups(&self) -> Option<ONNXLookup> {
        let (_, ts1) = self.ts1_read();
        let (_, ts2) = self.ts2_read();
        match self.instr().opcode {
            ONNXOpcode::Add => Some(
                (0..MAX_TENSOR_SIZE)
                    .map(|i| ElementWiseLookup::Add(ADD(ts1[i], ts2[i])))
                    .collect(),
            ),
            ONNXOpcode::Mul => Some(
                (0..MAX_TENSOR_SIZE)
                    .map(|i| ElementWiseLookup::Mul(MUL(ts1[i], ts2[i])))
                    .collect(),
            ),
            ONNXOpcode::Sub => Some(
                (0..MAX_TENSOR_SIZE)
                    .map(|i| ElementWiseLookup::Sub(SUB(ts1[i], ts2[i])))
                    .collect(),
            ),
            _ => None, // TODO(Forpee): Figure out how to ensure sparse-dense-shout handles none to still be no-op lookup queries of len MAX_TENSOR_SIZE
        }
    }
}
