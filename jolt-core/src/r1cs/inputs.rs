#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use crate::impl_r1cs_input_lc_conversions;
use crate::jolt::instruction::{CircuitFlags, InstructionFlags, LookupQuery};
use crate::jolt::vm::JoltProverPreprocessing;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::utils::transcript::Transcript;

use super::key::UniformSpartanKey;
use super::spartan::UniformSpartanProof;

use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use tracer::instruction::RV32IMCycle;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSProof<F: JoltField, ProofTranscript: Transcript> {
    pub key: UniformSpartanKey<F>,
    pub proof: UniformSpartanProof<F, ProofTranscript>,
    pub _marker: PhantomData<ProofTranscript>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JoltR1CSInputs {
    PC,           // Virtual (bytecode raf)
    UnexpandedPC, // Virtual (bytecode rv)
    Rd,           // Virtual (bytecode rv)
    Imm,          // Virtual (bytecode rv)
    RamAddress,   // Virtual (RAM raf)
    Rs1Value,     // Virtual (registers rv)
    Rs2Value,     // Virtual (registers rv)
    RdWriteValue,
    RamReadValue, // Virtual (RAM rv)
    RamWriteValue,
    LeftInstructionInput,  // to_lookup_query -> to_instruction_operands
    RightInstructionInput, // to_lookup_query -> to_instruction_operands
    LeftLookupOperand,     // Virtual (instruction raf)
    RightLookupOperand,    // Virtual (instruction raf)
    Product,               // LeftInstructionOperand * RightInstructionOperand
    WriteLookupOutputToRD,
    WritePCtoRD,
    ShouldBranch,
    NextUnexpandedPC, // Virtual (spartan shift sumcheck)
    NextPC,           // Virtual (spartan shift sumcheck)
    LookupOutput,     // Virtual (instruction rv)
    OpFlags(CircuitFlags),
}

/// This const serves to define a canonical ordering over inputs (and thus indices
/// for each input). This is needed for sumcheck.
pub const ALL_R1CS_INPUTS: [JoltR1CSInputs; 37] = [
    JoltR1CSInputs::PC,
    JoltR1CSInputs::UnexpandedPC,
    JoltR1CSInputs::Rd,
    JoltR1CSInputs::Imm,
    JoltR1CSInputs::RamAddress,
    JoltR1CSInputs::Rs1Value,
    JoltR1CSInputs::Rs2Value,
    JoltR1CSInputs::RdWriteValue,
    JoltR1CSInputs::RamReadValue,
    JoltR1CSInputs::RamWriteValue,
    JoltR1CSInputs::LeftInstructionInput,
    JoltR1CSInputs::RightInstructionInput,
    JoltR1CSInputs::LeftLookupOperand,
    JoltR1CSInputs::RightLookupOperand,
    JoltR1CSInputs::Product,
    JoltR1CSInputs::WriteLookupOutputToRD,
    JoltR1CSInputs::WritePCtoRD,
    JoltR1CSInputs::ShouldBranch,
    JoltR1CSInputs::NextUnexpandedPC,
    JoltR1CSInputs::NextPC,
    JoltR1CSInputs::LookupOutput,
    JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value),
    JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value),
    JoltR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC),
    JoltR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
    JoltR1CSInputs::OpFlags(CircuitFlags::AddOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
    JoltR1CSInputs::OpFlags(CircuitFlags::Load),
    JoltR1CSInputs::OpFlags(CircuitFlags::Store),
    JoltR1CSInputs::OpFlags(CircuitFlags::Jump),
    JoltR1CSInputs::OpFlags(CircuitFlags::Branch),
    JoltR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD),
    JoltR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction),
    JoltR1CSInputs::OpFlags(CircuitFlags::Assert),
    JoltR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC),
    JoltR1CSInputs::OpFlags(CircuitFlags::Advice),
];

impl JoltR1CSInputs {
    /// The total number of unique constraint inputs
    pub fn num_inputs() -> usize {
        ALL_R1CS_INPUTS.len()
    }

    /// Converts an index to the corresponding constraint input.
    pub fn from_index(index: usize) -> Self {
        ALL_R1CS_INPUTS[index]
    }

    /// Converts a constraint input to its index in the canonical
    /// ordering over inputs given by `ALL_R1CS_INPUTS`.
    pub fn to_index(&self) -> usize {
        match ALL_R1CS_INPUTS.iter().position(|x| x == self) {
            Some(index) => index,
            None => panic!("Invalid variant {self:?}"),
        }
    }

    pub fn generate_witness<F, PCS, ProofTranscript>(
        &self,
        trace: &[RV32IMCycle],
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        match self {
            JoltR1CSInputs::PC => {
                let coeffs: Vec<u64> = preprocessing
                    .shared
                    .bytecode
                    .map_trace_to_pc(trace)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::NextPC => {
                let coeffs: Vec<u64> = preprocessing
                    .shared
                    .bytecode
                    .map_trace_to_pc(&trace[1..])
                    .chain(rayon::iter::once(0))
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::UnexpandedPC => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| cycle.instruction().normalize().address as u64)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::Rd => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| cycle.rd_write().0 as u8)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::Imm => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| cycle.instruction().normalize().operands.imm)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::RamAddress => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| cycle.ram_access().address() as u64)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::Rs1Value => {
                let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.rs1_read().1).collect();
                coeffs.into()
            }
            JoltR1CSInputs::Rs2Value => {
                let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.rs2_read().1).collect();
                coeffs.into()
            }
            JoltR1CSInputs::RdWriteValue => {
                let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.rd_write().2).collect();
                coeffs.into()
            }
            JoltR1CSInputs::RamReadValue => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| match cycle.ram_access() {
                        tracer::instruction::RAMAccess::Read(read) => read.value,
                        tracer::instruction::RAMAccess::Write(write) => write.pre_value,
                        tracer::instruction::RAMAccess::NoOp => 0,
                    })
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::RamWriteValue => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| match cycle.ram_access() {
                        tracer::instruction::RAMAccess::Read(read) => read.value,
                        tracer::instruction::RAMAccess::Write(write) => write.post_value,
                        tracer::instruction::RAMAccess::NoOp => 0,
                    })
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::LeftInstructionInput => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_instruction_inputs(cycle).0)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::RightInstructionInput => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_instruction_inputs(cycle).1)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::LeftLookupOperand => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_lookup_operands(cycle).0)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::RightLookupOperand => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| LookupQuery::<32>::to_lookup_operands(cycle).1)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::Product => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        let (left_input, right_input) =
                            LookupQuery::<32>::to_instruction_inputs(cycle);
                        left_input * right_input as u64
                    })
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::WriteLookupOutputToRD => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| {
                        let flag = cycle.instruction().circuit_flags()
                            [CircuitFlags::WriteLookupOutputToRD as usize];
                        (cycle.rd_write().0 as u8) * (flag as u8)
                    })
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::WritePCtoRD => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| {
                        let flag = cycle.instruction().circuit_flags()[CircuitFlags::Jump as usize];
                        (cycle.rd_write().0 as u8) * (flag as u8)
                    })
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::LookupOutput => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(LookupQuery::<32>::to_lookup_output)
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::NextUnexpandedPC => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        let is_branch =
                            cycle.instruction().circuit_flags()[CircuitFlags::Branch as usize];
                        let should_branch =
                            is_branch && LookupQuery::<32>::to_lookup_output(cycle) != 0;
                        let instr = cycle.instruction().normalize();
                        if should_branch {
                            (instr.address as i64 + instr.operands.imm) as u64
                        } else {
                            // JoltR1CSInputs::NextPCJump
                            let is_jump =
                                cycle.instruction().circuit_flags()[CircuitFlags::Jump as usize];
                            let do_not_update_pc = cycle.instruction().circuit_flags()
                                [CircuitFlags::DoNotUpdateUnexpandedPC as usize];
                            if is_jump {
                                LookupQuery::<32>::to_lookup_output(cycle)
                            } else if do_not_update_pc {
                                instr.address as u64
                            } else {
                                instr.address as u64 + 4
                            }
                        }
                    })
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::ShouldBranch => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| {
                        let is_branch =
                            cycle.instruction().circuit_flags()[CircuitFlags::Branch as usize];
                        (LookupQuery::<32>::to_lookup_output(cycle) as u8) * is_branch as u8
                    })
                    .collect();
                coeffs.into()
            }
            JoltR1CSInputs::OpFlags(flag) => {
                // TODO(moodlezoup): Boolean polynomial
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| cycle.instruction().circuit_flags()[*flag as usize] as u8)
                    .collect();
                coeffs.into()
            }
        }
    }
}

impl_r1cs_input_lc_conversions!(JoltR1CSInputs);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_index_to_index() {
        for i in 0..JoltR1CSInputs::num_inputs() {
            assert_eq!(i, JoltR1CSInputs::from_index(i).to_index());
        }
        for var in ALL_R1CS_INPUTS {
            assert_eq!(
                var,
                JoltR1CSInputs::from_index(JoltR1CSInputs::to_index(&var))
            );
        }
    }
}
