use crate::jolt::JoltProverPreprocessing;
use crate::jolt::lookup_trace::LookupTrace;
use crate::tensor_jolt::instruction::ONNXLookupQuery;
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
use onnx_tracer::trace_types::{CircuitFlags, ONNXCycle};
use onnx_tracer::trace_types::{NUM_CIRCUIT_FLAGS, ONNXInstr};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub type ExecutionTrace = Vec<JoltONNXCycle>;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct JoltONNXCycle {
    pub instruction_lookup: Option<ONNXLookup>,
    pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],
}

impl JoltONNXCycle {
    /// # Returns: (address, read_value)
    pub fn ts1_read(&self) -> (usize, Vec<u64>) {
        todo!()
    }

    /// # Returns: (address, read_value)
    pub fn ts2_read(&self) -> (usize, Vec<u64>) {
        todo!()
    }

    /// # Returns: (address, pre_value, post_value)
    pub fn td_write(&self) -> (usize, Vec<u64>, Vec<u64>) {
        todo!()
    }

    pub fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        self.circuit_flags
    }

    pub fn instr(&self) -> ONNXInstr {
        todo!()
    }

    pub fn bytecode_line(&self) -> ONNXInstr {
        todo!()
    }
}

pub trait WitnessGenerator {
    fn generate_witness<F, PCS, ProofTranscript>(
        &self,
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[JoltONNXCycle],
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript;

    fn len() -> usize;

    fn from_index(index: usize) -> Self;

    fn to_index(&self) -> usize;
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum ONNXLookup {}

impl<const WORD_SIZE: usize> ONNXLookupQuery<WORD_SIZE> for JoltONNXCycle {
    /// Returns a tuple of the instruction's inputs. If the instruction has only one input,
    /// one of the tuple values will be 0.
    fn to_instruction_inputs(&self) -> (u64, i64) {
        todo!()
    }

    /// Returns a tuple of the instruction's lookup operands. By default, these are the
    /// same as the instruction inputs returned by `to_instruction_inputs`, but in some cases
    /// (e.g. ADD, MUL) the instruction inputs are combined to form a single lookup operand.
    fn to_lookup_operands(&self) -> (u64, u64) {
        todo!()
    }

    /// Converts this instruction's operands into a lookup index (as used in sparse-dense Shout).
    /// By default, interleaves the two bits of the two operands together.
    fn to_lookup_index(&self) -> u64 {
        todo!()
    }

    /// Computes the output lookup entry for this instruction as a u64.
    fn to_lookup_output(&self) -> u64 {
        todo!()
    }
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
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[JoltONNXCycle],
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
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[JoltONNXCycle],
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
