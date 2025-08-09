use crate::jolt::JoltProverPreprocessing;
use crate::jolt::instruction::VirtualInstructionSequence;
use crate::jolt::instruction::div::DIVInstruction;
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
    MAX_TENSOR_SIZE, TEST_TENSOR_REGISTER_COUNT, VIRTUAL_TENSOR_REGISTER_COUNT, ZERO_ADDR_PREPEND,
};
use onnx_tracer::tensor::Tensor;
use onnx_tracer::trace_types::ONNXOpcode;
use onnx_tracer::trace_types::{CircuitFlags, ONNXCycle};
use onnx_tracer::trace_types::{NUM_CIRCUIT_FLAGS, ONNXInstr};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::jolt::instruction::{
    add::ADD, beq::BEQInstruction, mul::MUL, sub::SUB,
    virtual_assert_valid_div0::AssertValidDiv0Instruction,
    virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
    virtual_move::MOVEInstruction,
};
use jolt_core::jolt::{instruction::InstructionLookup, lookup_table::LookupTables};

pub const WORD_SIZE: usize = 32;

pub type ExecutionTrace = Vec<JoltONNXCycle>;
pub type ONNXLookup = Vec<ElementWiseLookup>;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct JoltONNXCycle {
    pub instruction_lookups: Option<ONNXLookup>,
    pub circuit_flags: [bool; NUM_CIRCUIT_FLAGS],
    pub memory_ops: MemoryOps,
    pub instr: ONNXInstr,
    pub advice_value: Option<Vec<u64>>,
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
        }
    }
}

impl From<ONNXCycle> for JoltONNXCycle {
    fn from(raw_cycle: ONNXCycle) -> Self {
        let mut cycle = JoltONNXCycle::no_op();
        // 1. Set memory ops
        let (ts1_read, ts2_read, td_write) = raw_cycle.to_memory_ops();
        cycle.memory_ops = MemoryOps {
            ts1_read,
            ts2_read,
            td_write,
        };

        // 2. Set circuit flags
        cycle.circuit_flags = raw_cycle.instr.to_circuit_flags();

        // 3. Set bytecode line
        cycle.instr = raw_cycle.instr.clone();

        // 4. Set advice value
        cycle.advice_value = raw_cycle.advice_value();

        // 5. Populate instruction lookups
        // TODO(Forpee): Refactor this footgun (we should prevent a user from calling this method before memory_ops and advice is set).
        //               Builder pattern might be a good idea.
        cycle.populate_instruction_lookups();
        cycle
    }
}

pub fn jolt_execution_trace(raw_trace: Vec<ONNXCycle>) -> ExecutionTrace {
    let mut expanded_trace: Vec<ONNXCycle> = raw_trace
        .into_iter()
        .flat_map(|cycle| match cycle.instr.opcode {
            ONNXOpcode::Div => DIVInstruction::<32>::virtual_trace(cycle),
            _ => vec![cycle],
        })
        .collect();
    // Populate the pre-state of virtual tensor registers
    populate_virtual_prestate(&mut expanded_trace);
    // println!("Expanded trace: {expanded_trace:#?}");
    expanded_trace
        .into_iter()
        .map(JoltONNXCycle::from)
        .collect()
}

pub fn populate_virtual_prestate(raw_trace: &mut [ONNXCycle]) {
    let mut virtual_tensor_registers =
        vec![vec![0u64; MAX_TENSOR_SIZE]; VIRTUAL_TENSOR_REGISTER_COUNT as usize];
    for cycle in raw_trace.iter_mut() {
        let is_virtual = cycle.instr.virtual_sequence_remaining.is_some();
        let not_last_virtual_instr = cycle.instr.virtual_sequence_remaining.unwrap_or(0) != 0;
        if is_virtual && not_last_virtual_instr {
            if let Some(td) = cycle.instr.td {
                // store pre-state
                cycle.memory_state.td_pre_val = Some(Tensor::from(u64_vec_to_i128_iter(
                    &virtual_tensor_registers[td - TEST_TENSOR_REGISTER_COUNT as usize],
                )));

                assert_eq!(
                    cycle.td_pre_vals(),
                    virtual_tensor_registers[td - TEST_TENSOR_REGISTER_COUNT as usize],
                    "cycle: {cycle:#?}"
                );
                // write to virtual tensor register
                virtual_tensor_registers[td - TEST_TENSOR_REGISTER_COUNT as usize] =
                    cycle.td_post_vals();
            }
        }
    }
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
    LeftInstructionInput(usize),
    /// The "right" input to the current instruction. Typically either the
    /// rs2 value or the immediate value.
    RightInstructionInput(usize),
    /// Product of `LeftInstructionInput` and `RightInstructionInput`
    Product(usize),
    // /// Whether the current instruction should write the lookup output to
    // /// the destination register
    WriteLookupOutputToTD(usize),
    // /// Inc polynomial for the registers instance of Twist
    // RdInc,
    /// One-hot ra polynomial for the instruction lookups instance of Shout.
    /// There are four (d=4) of these polynomials, `InstructionRa(0) .. InstructionRa(3)`
    InstructionRa(usize),
}

pub const ALL_COMMITTED_POLYNOMIALS: [CommittedPolynomials; 4 * MAX_TENSOR_SIZE + 4] = {
    let mut arr = [CommittedPolynomials::LeftInstructionInput(0); 4 * MAX_TENSOR_SIZE + 4];
    let mut idx = 0;
    while idx < MAX_TENSOR_SIZE {
        arr[idx] = CommittedPolynomials::LeftInstructionInput(idx);
        idx += 1;
    }
    let mut j = 0;
    while j < MAX_TENSOR_SIZE {
        arr[idx] = CommittedPolynomials::RightInstructionInput(j);
        idx += 1;
        j += 1;
    }
    let mut k = 0;
    while k < MAX_TENSOR_SIZE {
        arr[idx] = CommittedPolynomials::Product(k);
        idx += 1;
        k += 1;
    }
    let mut n = 0;
    while n < MAX_TENSOR_SIZE {
        arr[idx] = CommittedPolynomials::WriteLookupOutputToTD(n);
        idx += 1;
        n += 1;
    }
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
            CommittedPolynomials::WriteLookupOutputToTD(i) => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| {
                        let flag = cycle.instr.to_circuit_flags()
                            [CircuitFlags::WriteLookupOutputToTD as usize];
                        (cycle.td_write().0[*i] as u8) * (flag as u8)
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
    Rd(usize), // Virtual (bytecode rv)
    RdWriteValue(usize),
    LeftInstructionInput(usize), // to_lookup_query -> to_instruction_operands
    RightInstructionInput(usize), // to_lookup_query -> to_instruction_operands
    LeftLookupOperand(usize),    // Virtual (instruction raf)
    RightLookupOperand(usize),   // Virtual (instruction raf)
    Product(usize),              // LeftInstructionOperand * RightInstructionOperand
    LookupOutput(usize),         // Virtual (instruction rv)
    WriteLookupOutputToTD(usize),
    OpFlags(CircuitFlags),
}

/// This const serves to define a canonical ordering over inputs (and thus indices
/// for each input). This is needed for sumcheck.
pub const ALL_R1CS_INPUTS: [JoltONNXR1CSInputs; 9 * MAX_TENSOR_SIZE + 4] = {
    let mut arr = [JoltONNXR1CSInputs::Rd(0); 9 * MAX_TENSOR_SIZE + 4];
    let mut idx = 0;
    while idx < MAX_TENSOR_SIZE {
        arr[idx] = JoltONNXR1CSInputs::Rd(idx);
        idx += 1;
    }
    let mut h = 0;
    while h < MAX_TENSOR_SIZE {
        arr[idx] = JoltONNXR1CSInputs::RdWriteValue(h);
        idx += 1;
        h += 1;
    }
    let mut i = 0;
    while i < MAX_TENSOR_SIZE {
        arr[idx] = JoltONNXR1CSInputs::LeftInstructionInput(i);
        idx += 1;
        i += 1;
    }
    let mut j = 0;
    while j < MAX_TENSOR_SIZE {
        arr[idx] = JoltONNXR1CSInputs::RightInstructionInput(j);
        idx += 1;
        j += 1;
    }
    let mut k = 0;
    while k < MAX_TENSOR_SIZE {
        arr[idx] = JoltONNXR1CSInputs::Product(k);
        idx += 1;
        k += 1;
    }
    let mut l = 0;
    while l < MAX_TENSOR_SIZE {
        arr[idx] = JoltONNXR1CSInputs::LeftLookupOperand(l);
        idx += 1;
        l += 1;
    }
    let mut m = 0;
    while m < MAX_TENSOR_SIZE {
        arr[idx] = JoltONNXR1CSInputs::RightLookupOperand(m);
        idx += 1;
        m += 1;
    }
    let mut n = 0;
    while n < MAX_TENSOR_SIZE {
        arr[idx] = JoltONNXR1CSInputs::LookupOutput(n);
        idx += 1;
        n += 1;
    }
    let mut o = 0;
    while o < MAX_TENSOR_SIZE {
        arr[idx] = JoltONNXR1CSInputs::WriteLookupOutputToTD(o);
        idx += 1;
        o += 1;
    }
    arr[idx] = JoltONNXR1CSInputs::OpFlags(CircuitFlags::AddOperands);
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::OpFlags(CircuitFlags::SubtractOperands);
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands);
    idx += 1;
    arr[idx] = JoltONNXR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToTD);
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
            JoltONNXR1CSInputs::Rd(i) => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| cycle.td_write().0.get(*i).cloned().unwrap() as u8)
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::RdWriteValue(i) => {
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
    Advice: ADVICEInstruction<WORD_SIZE>,
    VirtualAssertValidSignedRemainder: AssertValidSignedRemainderInstruction<WORD_SIZE>,
    VirtualAssertValidDiv0: AssertValidDiv0Instruction<WORD_SIZE>,
    VirtualAssertEq: BEQInstruction<WORD_SIZE>,
    VirtualMove: MOVEInstruction<WORD_SIZE>,
    VirtualConst: ConstInstruction<WORD_SIZE>,
);

impl InstructionLookup<WORD_SIZE> for ElementWiseLookup {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        match self {
            ElementWiseLookup::Add(add) => add.lookup_table(),
            ElementWiseLookup::Sub(sub) => sub.lookup_table(),
            ElementWiseLookup::Mul(mul) => mul.lookup_table(),
            ElementWiseLookup::Advice(advice) => advice.lookup_table(),
            ElementWiseLookup::VirtualAssertValidSignedRemainder(assert) => assert.lookup_table(),
            ElementWiseLookup::VirtualAssertValidDiv0(assert) => assert.lookup_table(),
            ElementWiseLookup::VirtualAssertEq(beq) => beq.lookup_table(),
            ElementWiseLookup::VirtualMove(move_instr) => move_instr.lookup_table(),
            ElementWiseLookup::VirtualConst(const_instr) => const_instr.lookup_table(),
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
            _ => None,
        }
    }
}
