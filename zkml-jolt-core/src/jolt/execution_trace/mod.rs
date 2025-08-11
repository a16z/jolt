use crate::jolt::JoltProverPreprocessing;
use crate::jolt::instruction::VirtualInstructionSequence;
use crate::jolt::instruction::div::DIVInstruction;
use crate::jolt::instruction::precompile::reduce_sum::ReduceSumInstruction;
use crate::jolt::instruction::virtual_advice::ADVICEInstruction;
use crate::jolt::instruction::virtual_const::ConstInstruction;
use crate::utils::u64_vec_to_i128_iter;
use itertools::Itertools;
use jolt_core::jolt::instruction::LookupQuery;
use jolt_core::poly::one_hot_polynomial::OneHotPolynomial;
use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial,
    },
    utils::transcript::Transcript,
};
use onnx_tracer::constants::{
    MAX_TENSOR_SIZE, TEST_TENSOR_REGISTER_COUNT, VIRTUAL_TENSOR_REGISTER_COUNT,
};
use onnx_tracer::tensor::Tensor;
use onnx_tracer::trace_types::ONNXOpcode;
use onnx_tracer::trace_types::{CircuitFlags, ONNXCycle};
use onnx_tracer::trace_types::{NUM_CIRCUIT_FLAGS, ONNXInstr};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::jolt::instruction::{
    add::ADD, beq::BEQInstruction, ge::GEInstruction, mul::MUL, sub::SUB,
    virtual_assert_valid_div0::AssertValidDiv0Instruction,
    virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
    virtual_move::MOVEInstruction,
};
use jolt_core::jolt::{instruction::InstructionLookup, lookup_table::LookupTables};

pub const WORD_SIZE: usize = 32;

pub type ExecutionTrace = Vec<JoltONNXCycle>;
pub type ONNXLookup = Vec<ElementWiseLookup>;

#[derive(Clone, Serialize, Deserialize)]
pub struct JoltONNXCycle {
    pub instruction_lookups: Option<ONNXLookup>,
    pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],
    pub memory_ops: MemoryOps,
    pub instr: ONNXInstr,
    pub advice_value: Option<Vec<u64>>,
    pub precompile: Option<Precompile>,
}

// TODO(Forpee): Refactor these clones in JoltONNXCycle::ts1_read, ts2_read, td_write
impl JoltONNXCycle {
    /// # Returns: (address, read_value)
    pub fn ts1_read(&self) -> (Vec<usize>, Vec<u64>) {
        self.memory_ops.ts1_read.clone()
    }

    /// # Returns: (address, read_value)
    pub fn ts2_read(&self) -> (Vec<usize>, Vec<u64>) {
        self.memory_ops.ts2_read.clone()
    }

    /// # Returns: (address, pre_value, post_value)
    pub fn td_write(&self) -> (Vec<usize>, Vec<u64>, Vec<u64>) {
        self.memory_ops.td_write.clone()
    }

    pub fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        self.circuit_flags
    }

    pub fn instr(&self) -> &ONNXInstr {
        &self.instr
    }

    pub fn bytecode_line(&self) -> &ONNXInstr {
        &self.instr
    }

    pub fn no_op() -> Self {
        JoltONNXCycle {
            instruction_lookups: None,
            circuit_flags: [false; NUM_CIRCUIT_FLAGS],
            memory_ops: MemoryOps::no_op(),
            instr: ONNXInstr::no_op(),
            advice_value: None,
            precompile: None,
        }
    }
}

impl JoltONNXCycle {
    // One public entry: builds a fully-initialized, valid cycle.
    pub fn from_raw(raw: &ONNXCycle) -> Self {
        // populate memory ops & advice first (needed by lookups)
        let (ts1_read, ts2_read, td_write) = raw.to_memory_ops();
        let mut cycle = JoltONNXCycle {
            memory_ops: MemoryOps {
                ts1_read,
                ts2_read,
                td_write,
            },
            circuit_flags: raw.instr.to_circuit_flags(),
            instr: raw.instr.clone(),
            advice_value: raw.advice_value(),
            ..JoltONNXCycle::no_op()
        };

        // now safely populate lookups
        cycle.populate_instruction_lookups_internal();
        cycle.populate_precompile();
        cycle
    }

    fn populate_instruction_lookups_internal(&mut self) {
        self.instruction_lookups = self.to_instruction_lookups();
    }

    fn populate_precompile(&mut self) {
        let (_, ts1) = self.ts1_read();
        match self.instr().opcode {
            ONNXOpcode::Sum => {
                let precompile = ReduceSumInstruction::<WORD_SIZE>(ts1);
                self.precompile = Some(Precompile::ReduceSum(precompile));
            }
            _ => {
                // No precompile for other opcodes
                self.precompile = None;
            }
        }
    }
}

impl From<&ONNXCycle> for JoltONNXCycle {
    fn from(raw: &ONNXCycle) -> Self {
        JoltONNXCycle::from_raw(raw)
    }
}

// ---- trace build: expand + prestate + convert ----

// Helper: resolve a virtual tensor register index (readability, fewer magic offsets).
#[inline]
fn vtr_index(td: usize) -> usize {
    td - TEST_TENSOR_REGISTER_COUNT as usize
}

// Expand Virtual instructions, maintain virtual prestate as we go, then convert to Jolt cycles.
pub fn jolt_execution_trace(raw_trace: Vec<ONNXCycle>) -> ExecutionTrace {
    // State for virtual tensor registers
    let mut vtr = vec![vec![0u64; MAX_TENSOR_SIZE]; VIRTUAL_TENSOR_REGISTER_COUNT as usize];

    let mut out = Vec::with_capacity(raw_trace.len());

    for raw in raw_trace {
        // Expand (virtualize) if needed
        let expanded: Vec<ONNXCycle> = match raw.instr.opcode {
            ONNXOpcode::Div => DIVInstruction::<32>::virtual_trace(raw),
            _ => vec![raw],
        };

        for mut cycle in expanded {
            if let (true, Some(rem)) = (
                cycle.instr.virtual_sequence_remaining.is_some(),
                cycle.instr.virtual_sequence_remaining,
            ) {
                if rem != 0 {
                    if let Some(td) = cycle.instr.td {
                        let idx = vtr_index(td);
                        // store pre-state
                        cycle.memory_state.td_pre_val =
                            Some(Tensor::from(u64_vec_to_i128_iter(&vtr[idx])));
                        // sanity check
                        assert_eq!(cycle.td_pre_vals(), vtr[idx], "cycle: {cycle:#?}");
                        // update post-state
                        vtr[idx] = cycle.td_post_vals();
                    }
                }
            }

            // Convert now that the cycle is fully prepared
            out.push(JoltONNXCycle::from(&cycle));
        }
    }

    out
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct MemoryOps {
    ts1_read: (Vec<usize>, Vec<u64>),
    ts2_read: (Vec<usize>, Vec<u64>),
    td_write: (Vec<usize>, Vec<u64>, Vec<u64>),
}

impl MemoryOps {
    pub fn no_op() -> Self {
        MemoryOps {
            ts1_read: (vec![0usize; MAX_TENSOR_SIZE], vec![0; MAX_TENSOR_SIZE]),
            ts2_read: (vec![0usize; MAX_TENSOR_SIZE], vec![0; MAX_TENSOR_SIZE]),
            td_write: (
                vec![0usize; MAX_TENSOR_SIZE],
                vec![0; MAX_TENSOR_SIZE],
                vec![0; MAX_TENSOR_SIZE],
            ),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub enum Precompile {
    ReduceSum(ReduceSumInstruction<WORD_SIZE>),
}

impl ONNXLookupQuery<WORD_SIZE> for Precompile {
    fn to_instruction_inputs(&self) -> (Vec<u64>, Vec<i64>) {
        match self {
            Precompile::ReduceSum(instr) => instr.to_instruction_inputs(),
        }
    }

    fn to_lookup_operands(&self) -> (Vec<u64>, Vec<u64>) {
        match self {
            Precompile::ReduceSum(instr) => instr.to_lookup_operands(),
        }
    }

    fn to_lookup_index(&self) -> Vec<u64> {
        match self {
            Precompile::ReduceSum(instr) => instr.to_lookup_index(),
        }
    }

    fn to_lookup_output(&self) -> Vec<u64> {
        match self {
            Precompile::ReduceSum(instr) => instr.to_lookup_output(),
        }
    }
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
        if let Some(precompile) = &self.precompile {
            return precompile.to_instruction_inputs();
        }

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
        if let Some(precompile) = &self.precompile {
            return precompile.to_lookup_operands();
        }
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
        if let Some(precompile) = &self.precompile {
            return precompile.to_lookup_index();
        }
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
        if let Some(precompile) = &self.precompile {
            return precompile.to_lookup_output();
        }
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
    LeftInstructionInput(usize),
    /// The "right" input to the current instruction. Typically either the
    /// rs2 value or the immediate value.
    RightInstructionInput(usize),
    /// Product of `LeftInstructionInput` and `RightInstructionInput`
    Product(usize),
    /// Td * CircuitFlag::WriteLookupOutputToTD
    TdProdFlag(usize),
    // /// Whether the current instruction should write the lookup output to
    // /// the destination register
    WriteLookupOutputToTD(usize),
    // /// Inc polynomial for the registers instance of Twist
    // RdInc,
    /// One-hot ra polynomial for the instruction lookups instance of Shout.
    /// There are four (d=4) of these polynomials, `InstructionRa(0) .. InstructionRa(3)`
    InstructionRa(usize),
}

macro_rules! fill_array_committed {
    ($arr:ident, $idx:ident, $variant:ident) => {{
        let mut i = 0;
        while i < MAX_TENSOR_SIZE {
            $arr[$idx] = CommittedPolynomials::$variant(i);
            $idx += 1;
            i += 1;
        }
    }};
}

pub const ALL_COMMITTED_POLYNOMIALS: [CommittedPolynomials; 5 * MAX_TENSOR_SIZE + 4] = {
    let mut arr = [CommittedPolynomials::LeftInstructionInput(0); 5 * MAX_TENSOR_SIZE + 4];
    let mut idx = 0;
    fill_array_committed!(arr, idx, LeftInstructionInput);
    fill_array_committed!(arr, idx, RightInstructionInput);
    fill_array_committed!(arr, idx, Product);
    fill_array_committed!(arr, idx, TdProdFlag);
    fill_array_committed!(arr, idx, WriteLookupOutputToTD);
    arr[idx] = CommittedPolynomials::InstructionRa(0);
    arr[idx + 1] = CommittedPolynomials::InstructionRa(1);
    arr[idx + 2] = CommittedPolynomials::InstructionRa(2);
    arr[idx + 3] = CommittedPolynomials::InstructionRa(3);
    arr
};

impl WitnessGenerator for CommittedPolynomials {
    fn generate_witness<F, PCS, ProofTranscript>(
        &self,
        trace: &[JoltONNXCycle],
        _preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        match self {
            CommittedPolynomials::LeftInstructionInput(i) => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        ONNXLookupQuery::<WORD_SIZE>::to_instruction_inputs(cycle)
                            .0
                            .get(*i)
                            .cloned()
                            .unwrap()
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::RightInstructionInput(i) => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| {
                        ONNXLookupQuery::<WORD_SIZE>::to_instruction_inputs(cycle)
                            .1
                            .get(*i)
                            .cloned()
                            .unwrap()
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::Product(i) => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (left_input, right_input) =
                            ONNXLookupQuery::<WORD_SIZE>::to_instruction_inputs(cycle);
                        left_input[*i] * right_input[*i] as u64
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::TdProdFlag(i) => {
                let coeffs: Vec<u32> = trace
                    .par_iter()
                    .map(|cycle| {
                        let flag = cycle.instr.to_circuit_flags()
                            [CircuitFlags::WriteLookupOutputToTD as usize];
                        (cycle.td_write().0[*i] as u32) * (flag as u8 as u32)
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::WriteLookupOutputToTD(i) => {
                let coeffs: Vec<u32> = trace
                    .par_iter()
                    .map(|cycle| {
                        let flag = cycle.instr.to_circuit_flags()
                            [CircuitFlags::WriteLookupOutputToTD as usize];
                        (cycle.td_write().0[*i] as u32)
                            * (flag as u8 as u32)
                            * ((*i < cycle.instr.active_output_elements) as u8 as u32)
                    })
                    .collect();
                coeffs.into()
            }

            // TODO: Openings: https://github.com/ICME-Lab/zkml-jolt/issues/66
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

            // FIXME: I think all polynomials in Spartan have to be the same length (either T or T * MAX_TENSOR_SIZE)
            CommittedPolynomials::InstructionRa(i) => {
                if *i > 3 {
                    panic!("Unexpected i: {i}");
                }
                let addresses: Vec<usize> = trace
                    .par_iter()
                    .flat_map(|cycle| {
                        ONNXLookupQuery::<WORD_SIZE>::to_lookup_index(cycle)
                            .iter()
                            .map(|lookup_index| {
                                let k = (lookup_index >> (16 * (3 - i))) % (1 << 16);
                                k as usize
                            })
                            .collect_vec()
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(addresses, 1 << 16))
            }
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
    Td(usize), // Virtual (bytecode rv)
    TdWriteValue(usize),
    LeftInstructionInput(usize), // to_lookup_query -> to_instruction_operands
    RightInstructionInput(usize), // to_lookup_query -> to_instruction_operands
    LeftLookupOperand(usize),    // Virtual (instruction raf)
    RightLookupOperand(usize),   // Virtual (instruction raf)
    Product(usize),              // LeftInstructionOperand * RightInstructionOperand
    LookupOutput(usize),         // Virtual (instruction rv)
    WriteLookupOutputToTD(usize),
    OpFlags(CircuitFlags),
    PC,               // Virtual (bytecode raf)
    UnexpandedPC,     // Virtual (bytecode rv)
    NextUnexpandedPC, // Virtual (spartan shift sumcheck)
    NextPC,           // Virtual (spartan shift sumcheck)
    ActiveOutput(usize),
    TdProdFlag(usize), // Td * CircuitFlag::WriteLookupOutputToTD
}

macro_rules! fill_array_r1cs_inputs {
    ($arr:ident, $idx:ident, $variant:ident) => {{
        let mut i = 0;
        while i < MAX_TENSOR_SIZE {
            $arr[$idx] = JoltONNXR1CSInputs::$variant(i);
            $idx += 1;
            i += 1;
        }
    }};
}

/// This const serves to define a canonical ordering over inputs (and thus indices
/// for each input). This is needed for sumcheck.
pub const ALL_R1CS_INPUTS: [JoltONNXR1CSInputs; 11 * MAX_TENSOR_SIZE + 12] = {
    let mut arr = [JoltONNXR1CSInputs::Td(0); 11 * MAX_TENSOR_SIZE + 12];
    let mut idx = 0;
    fill_array_r1cs_inputs!(arr, idx, Td);
    fill_array_r1cs_inputs!(arr, idx, TdWriteValue);
    fill_array_r1cs_inputs!(arr, idx, LeftInstructionInput);
    fill_array_r1cs_inputs!(arr, idx, RightInstructionInput);
    fill_array_r1cs_inputs!(arr, idx, Product);
    fill_array_r1cs_inputs!(arr, idx, LeftLookupOperand);
    fill_array_r1cs_inputs!(arr, idx, RightLookupOperand);
    fill_array_r1cs_inputs!(arr, idx, LookupOutput);
    fill_array_r1cs_inputs!(arr, idx, WriteLookupOutputToTD);
    fill_array_r1cs_inputs!(arr, idx, ActiveOutput);
    fill_array_r1cs_inputs!(arr, idx, TdProdFlag);
    arr[idx] = JoltONNXR1CSInputs::OpFlags(CircuitFlags::AddOperands);
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::OpFlags(CircuitFlags::SubtractOperands);
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands);
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToTD);
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::OpFlags(CircuitFlags::Assert);
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC);
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction);
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::OpFlags(CircuitFlags::SumOperands);
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::PC;
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::UnexpandedPC;
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::NextUnexpandedPC;
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::NextPC;

    arr
};

// /// The subset of `ALL_R1CS_INPUTS` that are committed. The rest of
// /// the inputs are virtual polynomials.
// pub const COMMITTED_R1CS_INPUTS: [JoltONNXR1CSInputs; 3] = [
//     JoltONNXR1CSInputs::LeftInstructionInput,
//     JoltONNXR1CSInputs::RightInstructionInput,
//     JoltONNXR1CSInputs::Product,
//     // JoltONNXR1CSInputs::WriteLookupOutputToRD,
// ];

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
            JoltONNXR1CSInputs::PC => {
                let coeffs: Vec<u64> = preprocessing
                    .shared
                    .bytecode
                    .map_trace_to_pc(trace)
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::NextPC => {
                let coeffs: Vec<u64> = preprocessing
                    .shared
                    .bytecode
                    .map_trace_to_pc(&trace[1..])
                    .chain(rayon::iter::once(0))
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::UnexpandedPC => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| cycle.instr().address as u64)
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::NextUnexpandedPC => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        let do_not_update_pc =
                            cycle.circuit_flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize];
                        if do_not_update_pc {
                            cycle.instr().address as u64
                        } else {
                            cycle.instr().address as u64 + 1
                        }
                    })
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::Td(i) => {
                let coeffs: Vec<u32> = trace
                    .par_iter()
                    .map(|cycle| cycle.td_write().0.get(*i).cloned().unwrap() as u32)
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::TdWriteValue(i) => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| cycle.td_write().2.get(*i).cloned().unwrap())
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::LeftInstructionInput(i) => {
                CommittedPolynomials::LeftInstructionInput(*i)
                    .generate_witness(trace, preprocessing)
            }
            JoltONNXR1CSInputs::RightInstructionInput(i) => {
                CommittedPolynomials::RightInstructionInput(*i)
                    .generate_witness(trace, preprocessing)
            }
            JoltONNXR1CSInputs::LeftLookupOperand(i) => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        ONNXLookupQuery::<WORD_SIZE>::to_lookup_operands(cycle)
                            .0
                            .get(*i)
                            .cloned()
                            .unwrap()
                    })
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::RightLookupOperand(i) => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        ONNXLookupQuery::<WORD_SIZE>::to_lookup_operands(cycle)
                            .1
                            .get(*i)
                            .cloned()
                            .unwrap()
                    })
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::Product(i) => {
                CommittedPolynomials::Product(*i).generate_witness(trace, preprocessing)
            }
            JoltONNXR1CSInputs::WriteLookupOutputToTD(i) => {
                CommittedPolynomials::WriteLookupOutputToTD(*i)
                    .generate_witness(trace, preprocessing)
            }
            JoltONNXR1CSInputs::LookupOutput(i) => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        ONNXLookupQuery::<WORD_SIZE>::to_lookup_output(cycle)
                            .get(*i)
                            .cloned()
                            .unwrap()
                    })
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::OpFlags(flag) => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| cycle.instr.to_circuit_flags()[*flag as usize] as u8)
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::TdProdFlag(i) => {
                CommittedPolynomials::TdProdFlag(*i).generate_witness(trace, preprocessing)
            }
            JoltONNXR1CSInputs::ActiveOutput(i) => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| (*i < cycle.instr.active_output_elements) as u8)
                    .collect();
                coeffs.into()
            }
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

        impl InstructionLookup<$word_size> for $enum_name {
            fn lookup_table(&self) -> Option<LookupTables<$word_size>> {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.lookup_table(),
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
    Ge: GEInstruction<WORD_SIZE>,
    Advice: ADVICEInstruction<WORD_SIZE>,
    VirtualAssertValidSignedRemainder: AssertValidSignedRemainderInstruction<WORD_SIZE>,
    VirtualAssertValidDiv0: AssertValidDiv0Instruction<WORD_SIZE>,
    VirtualAssertEq: BEQInstruction<WORD_SIZE>,
    VirtualMove: MOVEInstruction<WORD_SIZE>,
    VirtualConst: ConstInstruction<WORD_SIZE>,
);

impl JoltONNXCycle {
    fn to_instruction_lookups(&self) -> Option<ONNXLookup> {
        let (_, ts1) = self.ts1_read();
        let (_, ts2) = self.ts2_read();
        let imm = self.instr.imm();
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
            ONNXOpcode::VirtualAdvice => {
                let advice_value = self.advice_value.as_ref().unwrap();
                (0..MAX_TENSOR_SIZE)
                    .map(|i| ElementWiseLookup::Advice(ADVICEInstruction(advice_value[i])))
                    .collect::<Vec<_>>()
                    .into()
            }
            ONNXOpcode::VirtualAssertValidSignedRemainder => Some(
                (0..MAX_TENSOR_SIZE)
                    .map(|i| {
                        ElementWiseLookup::VirtualAssertValidSignedRemainder(
                            AssertValidSignedRemainderInstruction(ts1[i], ts2[i]),
                        )
                    })
                    .collect(),
            ),
            ONNXOpcode::VirtualAssertValidDiv0 => Some(
                (0..MAX_TENSOR_SIZE)
                    .map(|i| {
                        ElementWiseLookup::VirtualAssertValidDiv0(AssertValidDiv0Instruction(
                            ts1[i], ts2[i],
                        ))
                    })
                    .collect(),
            ),
            ONNXOpcode::VirtualAssertEq => Some(
                (0..MAX_TENSOR_SIZE)
                    .map(|i| ElementWiseLookup::VirtualAssertEq(BEQInstruction(ts1[i], ts2[i])))
                    .collect(),
            ),
            ONNXOpcode::VirtualMove => Some(
                (0..MAX_TENSOR_SIZE)
                    .map(|i| ElementWiseLookup::VirtualMove(MOVEInstruction(ts1[i])))
                    .collect(),
            ),
            ONNXOpcode::VirtualConst => Some(
                (0..MAX_TENSOR_SIZE)
                    .map(|i| ElementWiseLookup::VirtualConst(ConstInstruction(imm[i])))
                    .collect(),
            ),
            ONNXOpcode::Gte => Some(
                (0..MAX_TENSOR_SIZE)
                    .map(|i| ElementWiseLookup::Ge(GEInstruction(ts1[i], ts2[i])))
                    .collect(),
            ),
            _ => None,
        }
    }
}

impl std::fmt::Debug for JoltONNXCycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lookup_summary = match self.lookup_kind() {
            Some(name) => format!("Some({name})"),
            None => "None".to_string(),
        };

        f.debug_struct("JoltONNXCycle")
            .field("address", &self.instr.address)
            .field("opcode", &self.instr.opcode)
            .field("instruction", &self.instr)
            .field("active_flags", &self.active_circuit_flags())
            .field("lookup", &lookup_summary)
            .field("ts1_read", &format_memory_op_ranges(&self.ts1_read()))
            .field("ts2_read", &format_memory_op_ranges(&self.ts2_read()))
            .field("td_write", &format_write_op_ranges(&self.td_write()))
            .field("advice", &self.advice_value)
            .finish()
    }
}

impl JoltONNXCycle {
    // Helper method to get only the active circuit flags
    fn active_circuit_flags(&self) -> Vec<String> {
        use onnx_tracer::trace_types::CircuitFlags;
        let mut active = Vec::new();

        if self.circuit_flags[CircuitFlags::LeftOperandIsTs1Value as usize] {
            active.push("LeftOperandIsTs1Value".to_string());
        }
        if self.circuit_flags[CircuitFlags::RightOperandIsTs2Value as usize] {
            active.push("RightOperandIsTs2Value".to_string());
        }
        if self.circuit_flags[CircuitFlags::RightOperandIsImm as usize] {
            active.push("RightOperandIsImm".to_string());
        }
        if self.circuit_flags[CircuitFlags::AddOperands as usize] {
            active.push("AddOperands".to_string());
        }
        if self.circuit_flags[CircuitFlags::SubtractOperands as usize] {
            active.push("SubtractOperands".to_string());
        }
        if self.circuit_flags[CircuitFlags::MultiplyOperands as usize] {
            active.push("MultiplyOperands".to_string());
        }
        if self.circuit_flags[CircuitFlags::WriteLookupOutputToTD as usize] {
            active.push("WriteLookupOutputToTD".to_string());
        }
        if self.circuit_flags[CircuitFlags::InlineSequenceInstruction as usize] {
            active.push("InlineSequenceInstruction".to_string());
        }
        if self.circuit_flags[CircuitFlags::Assert as usize] {
            active.push("Assert".to_string());
        }
        if self.circuit_flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] {
            active.push("DoNotUpdateUnexpandedPC".to_string());
        }
        if self.circuit_flags[CircuitFlags::Advice as usize] {
            active.push("Advice".to_string());
        }
        if self.circuit_flags[CircuitFlags::Const as usize] {
            active.push("Const".to_string());
        }
        if self.circuit_flags[CircuitFlags::SumOperands as usize] {
            active.push("SumOperands".to_string());
        }

        // Compile-time check that we've handled all flags.
        // Will error if you add a new flag and forget to update this.
        const _: () = {
            let _ = [(); (NUM_CIRCUIT_FLAGS == 13) as usize - 1];
        };

        if active.is_empty() {
            active.push("None".to_string());
        }

        active
    }

    // Show None or Some(ElementWiseLookupVariant)
    fn lookup_kind(&self) -> Option<&'static str> {
        self.instruction_lookups
            .as_ref()
            .and_then(|v| v.first())
            .map(|e| match e {
                ElementWiseLookup::Add(_) => "Add",
                ElementWiseLookup::Sub(_) => "Sub",
                ElementWiseLookup::Mul(_) => "Mul",
                ElementWiseLookup::Ge(_) => "Ge",
                ElementWiseLookup::Advice(_) => "Advice",
                ElementWiseLookup::VirtualAssertValidSignedRemainder(_) => {
                    "VirtualAssertValidSignedRemainder"
                }
                ElementWiseLookup::VirtualAssertValidDiv0(_) => "VirtualAssertValidDiv0",
                ElementWiseLookup::VirtualAssertEq(_) => "VirtualAssertEq",
                ElementWiseLookup::VirtualMove(_) => "VirtualMove",
                ElementWiseLookup::VirtualConst(_) => "VirtualConst",
            })
    }
}

// Helper: format tensor memory operations, knowing addresses are always contiguous ranges
fn format_memory_op_ranges((addresses, values): &(Vec<usize>, Vec<u64>)) -> String {
    if addresses.is_empty() {
        return "[]".to_string();
    }
    let start_addr = addresses[0];
    let end_addr = start_addr + values.len();

    format!(
        "[{}..{}: [{}]]",
        start_addr,
        end_addr,
        values
            .iter()
            .map(|&v| (v as u32 as i32).to_string())
            .join(", ")
    )
}

fn format_write_op_ranges(
    (addresses, pre_vals, post_vals): &(Vec<usize>, Vec<u64>, Vec<u64>),
) -> String {
    if addresses.is_empty() {
        return "[]".to_string();
    }
    let start_addr = addresses[0];
    let end_addr = start_addr + pre_vals.len();

    format!(
        "[{}..{}: pre=[{}], post=[{}]]",
        start_addr,
        end_addr,
        pre_vals
            .iter()
            .map(|&v| (v as u32 as i32).to_string())
            .join(", "),
        post_vals
            .iter()
            .map(|&v| (v as u32 as i32).to_string())
            .join(", ")
    )
}

#[cfg(test)]
pub fn check_mcc(execution_trace: &ExecutionTrace) {
    let tensor_heap_addresses: Vec<usize> = execution_trace
        .iter()
        .map(|cycle| cycle.td_write().0.last().unwrap() + 1)
        .collect();
    let tensor_heap_K = tensor_heap_addresses
        .iter()
        .max()
        .unwrap()
        .next_power_of_two();
    let mut tensor_heap = vec![0u64; tensor_heap_K];
    for (i, cycle) in execution_trace.iter().enumerate() {
        // check reads

        // ts1 read
        let (ts1_read_addresses, ts1_read_values) = cycle.ts1_read();
        for (addr, value) in itertools::izip!(ts1_read_addresses.iter(), ts1_read_values.iter()) {
            assert_eq!(
                tensor_heap[*addr], *value,
                "TS1 READ error at cycle_{i}: {cycle:#?}; Expected: {}, got: {} at address {addr} ",
                tensor_heap[*addr], *value
            );
        }

        // ts2 read
        let (ts2_read_addresses, ts2_read_values) = cycle.ts2_read();
        for (addr, value) in itertools::izip!(ts2_read_addresses.iter(), ts2_read_values.iter()) {
            assert_eq!(
                tensor_heap[*addr], *value,
                "TS2 READ error at cycle_{i}: {cycle:#?}; Expected: {}, got: {} at address {addr} ",
                tensor_heap[*addr], *value
            );
        }

        // check writes
        let (td_write_addresses, td_pre_values, td_write_values) = cycle.td_write();
        for (addr, pre_val, post_val) in itertools::izip!(
            td_write_addresses.iter(),
            td_pre_values.iter(),
            td_write_values.iter()
        ) {
            assert_eq!(
                tensor_heap[*addr], *pre_val,
                "TD WRITE error at cycle_{i}: {cycle:#?}; Expected pre-state: {pre_val}, got: {} at address {addr} ",
                tensor_heap[*addr]
            );
            tensor_heap[*addr] = *post_val;
        }
    }
}
