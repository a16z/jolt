#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use super::spartan::UniformSpartanProof;
use crate::jolt::vm::JoltProverPreprocessing;
use jolt_core::field::JoltField;
use jolt_core::jolt::instruction::{InstructionFlags, LookupQuery};
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::r1cs::key::UniformSpartanKey;
use jolt_core::r1cs::ops::{LC, Term, Variable};
use jolt_core::utils::transcript::Transcript;
use onnx_tracer::trace_types::{CircuitFlags, ONNXCycle};
use rayon::prelude::*;
use std::fmt::Debug;
use std::marker::PhantomData;

pub struct R1CSProof<F: JoltField, ProofTranscript: Transcript> {
    pub key: UniformSpartanKey<F>,
    pub proof: UniformSpartanProof<F, ProofTranscript>,
    pub _marker: PhantomData<ProofTranscript>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JoltONNXR1CSInputs {
    // PC, // Virtual (bytecode raf)
    // UnexpandedPC, // Virtual (bytecode rv)
    Rd, // Virtual (bytecode rv)
    // Imm,          // Virtual (bytecode rv)
    // RamAddress,   // Virtual (RAM raf)
    // Rs1Value, // Virtual (registers rv)
    // Rs2Value, // Virtual (registers rv)
    RdWriteValue,
    // RamReadValue, // Virtual (RAM rv)
    // RamWriteValue,
    LeftInstructionInput,  // to_lookup_query -> to_instruction_operands
    RightInstructionInput, // to_lookup_query -> to_instruction_operands
    LeftLookupOperand,     // Virtual (instruction raf)
    RightLookupOperand,    // Virtual (instruction raf)
    Product,               // LeftInstructionOperand * RightInstructionOperand
    WriteLookupOutputToRD,
    // WritePCtoRD,
    // ShouldBranch,
    // NextUnexpandedPC, // Virtual (spartan shift sumcheck)
    // NextPC,           // Virtual (spartan shift sumcheck)
    LookupOutput, // Virtual (instruction rv)
    OpFlags(CircuitFlags),
}

/// This const serves to define a canonical ordering over inputs (and thus indices
/// for each input). This is needed for sumcheck.
pub const ALL_R1CS_INPUTS: [JoltONNXR1CSInputs; 12] = [
    JoltONNXR1CSInputs::LeftInstructionInput,
    JoltONNXR1CSInputs::RightInstructionInput,
    JoltONNXR1CSInputs::Product,
    JoltONNXR1CSInputs::WriteLookupOutputToRD,
    // JoltONNXR1CSInputs::WritePCtoRD,
    // JoltONNXR1CSInputs::ShouldBranch,
    // JoltONNXR1CSInputs::PC,
    // JoltONNXR1CSInputs::UnexpandedPC,
    JoltONNXR1CSInputs::Rd,
    // JoltONNXR1CSInputs::Imm,
    // JoltONNXR1CSInputs::RamAddress,
    // JoltONNXR1CSInputs::Rs1Value,
    // JoltONNXR1CSInputs::Rs2Value,
    JoltONNXR1CSInputs::RdWriteValue,
    // JoltONNXR1CSInputs::RamReadValue,
    // JoltONNXR1CSInputs::RamWriteValue,
    JoltONNXR1CSInputs::LeftLookupOperand,
    JoltONNXR1CSInputs::RightLookupOperand,
    // JoltONNXR1CSInputs::NextUnexpandedPC,
    // JoltONNXR1CSInputs::NextPC,
    JoltONNXR1CSInputs::LookupOutput,
    // JoltONNXR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsRs1Value),
    // JoltONNXR1CSInputs::OpFlags(CircuitFlags::RightOperandIsRs2Value),
    // JoltONNXR1CSInputs::OpFlags(CircuitFlags::LeftOperandIsPC),
    // JoltONNXR1CSInputs::OpFlags(CircuitFlags::RightOperandIsImm),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::AddOperands),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::SubtractOperands),
    JoltONNXR1CSInputs::OpFlags(CircuitFlags::MultiplyOperands),
    // JoltONNXR1CSInputs::OpFlags(CircuitFlags::Load),
    // JoltONNXR1CSInputs::OpFlags(CircuitFlags::Store),
    // JoltONNXR1CSInputs::OpFlags(CircuitFlags::Jump),
    // JoltONNXR1CSInputs::OpFlags(CircuitFlags::Branch),
    // JoltONNXR1CSInputs::OpFlags(CircuitFlags::WriteLookupOutputToRD),
    // JoltONNXR1CSInputs::OpFlags(CircuitFlags::InlineSequenceInstruction),
    // JoltONNXR1CSInputs::OpFlags(CircuitFlags::Assert),
    // JoltONNXR1CSInputs::OpFlags(CircuitFlags::DoNotUpdateUnexpandedPC),
    // JoltONNXR1CSInputs::OpFlags(CircuitFlags::Advice),
];

/// The subset of `ALL_R1CS_INPUTS` that are committed. The rest of
/// the inputs are virtual polynomials.
pub const COMMITTED_R1CS_INPUTS: [JoltONNXR1CSInputs; 4] = [
    JoltONNXR1CSInputs::LeftInstructionInput,
    JoltONNXR1CSInputs::RightInstructionInput,
    JoltONNXR1CSInputs::Product,
    JoltONNXR1CSInputs::WriteLookupOutputToRD,
    // JoltONNXR1CSInputs::WritePCtoRD,
    // JoltONNXR1CSInputs::ShouldBranch,
];

impl JoltONNXR1CSInputs {
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
        trace: &[ONNXCycle],
        preprocessing: &JoltProverPreprocessing<F, ProofTranscript>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        match self {
            // JoltONNXR1CSInputs::PC => {
            //     let coeffs: Vec<u64> = preprocessing
            //         .shared
            //         .bytecode
            //         .map_trace_to_pc(trace)
            //         .collect();
            //     coeffs.into()
            // }
            //     JoltONNXR1CSInputs::NextPC => {
            //         let coeffs: Vec<u64> = preprocessing
            //             .shared
            //             .bytecode
            //             .map_trace_to_pc(&trace[1..])
            //             .chain(rayon::iter::once(0))
            //             .collect();
            //         coeffs.into()
            //     }
            //     JoltONNXR1CSInputs::UnexpandedPC => {
            //         let coeffs: Vec<u64> = trace
            //             .par_iter()
            //             .map(|cycle| cycle.instruction().normalize().address as u64)
            //             .collect();
            //         coeffs.into()
            //     }
            //     JoltONNXR1CSInputs::Rd => {
            //         let coeffs: Vec<u8> = trace
            //             .par_iter()
            //             .map(|cycle| cycle.rd_write().0 as u8)
            //             .collect();
            //         coeffs.into()
            //     }
            //     JoltONNXR1CSInputs::Imm => {
            //         let coeffs: Vec<i64> = trace
            //             .par_iter()
            //             .map(|cycle| cycle.instruction().normalize().operands.imm)
            //             .collect();
            //         coeffs.into()
            //     }
            //     JoltONNXR1CSInputs::RamAddress => {
            //         let coeffs: Vec<u64> = trace
            //             .par_iter()
            //             .map(|cycle| cycle.ram_access().address() as u64)
            //             .collect();
            //         coeffs.into()
            //     }
            //     JoltONNXR1CSInputs::Rs1Value => {
            //         let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.rs1_read().1).collect();
            //         coeffs.into()
            //     }
            //     JoltONNXR1CSInputs::Rs2Value => {
            //         let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.rs2_read().1).collect();
            //         coeffs.into()
            //     }
            JoltONNXR1CSInputs::RdWriteValue => {
                let coeffs: Vec<F> = trace
                    .par_iter()
                    .map(|cycle| F::from_i128(cycle.td_write()))
                    .collect();
                coeffs.into()
            }
            //     JoltONNXR1CSInputs::RamReadValue => {
            //         let coeffs: Vec<u64> = trace
            //             .par_iter()
            //             .map(|cycle| match cycle.ram_access() {
            //                 tracer::instruction::RAMAccess::Read(read) => read.value,
            //                 tracer::instruction::RAMAccess::Write(write) => write.pre_value,
            //                 tracer::instruction::RAMAccess::NoOp => 0,
            //                 tracer::instruction::RAMAccess::Atomic(_) => {
            //                     unimplemented!("Atomic instructions are mapped to virtual sequences")
            //                 }
            //             })
            //             .collect();
            //         coeffs.into()
            //     }
            //     JoltONNXR1CSInputs::RamWriteValue => {
            //         let coeffs: Vec<u64> = trace
            //             .par_iter()
            //             .map(|cycle| match cycle.ram_access() {
            //                 tracer::instruction::RAMAccess::Read(read) => read.value,
            //                 tracer::instruction::RAMAccess::Write(write) => write.post_value,
            //                 tracer::instruction::RAMAccess::NoOp => 0,
            //                 tracer::instruction::RAMAccess::Atomic(_) => {
            //                     unimplemented!("Atomic instructions are mapped to virtual sequences")
            //                 }
            //             })
            //             .collect();
            //         coeffs.into()
            //     }
            // JoltONNXR1CSInputs::LeftInstructionInput => {
            //     CommittedPolynomials::LeftInstructionInput.generate_witness(preprocessing, trace)
            // }
            // JoltONNXR1CSInputs::RightInstructionInput => {
            //     CommittedPolynomials::RightInstructionInput.generate_witness(preprocessing, trace)
            // }
            //     JoltONNXR1CSInputs::LeftLookupOperand => {
            //         let coeffs: Vec<u64> = trace
            //             .par_iter()
            //             .map(|cycle| LookupQuery::<32>::to_lookup_operands(cycle).0)
            //             .collect();
            //         coeffs.into()
            //     }
            //     JoltONNXR1CSInputs::RightLookupOperand => {
            //         let coeffs: Vec<u64> = trace
            //             .par_iter()
            //             .map(|cycle| LookupQuery::<32>::to_lookup_operands(cycle).1)
            //             .collect();
            //         coeffs.into()
            //     }
            //     JoltONNXR1CSInputs::Product => {
            //         CommittedPolynomials::Product.generate_witness(preprocessing, trace)
            //     }
            //     JoltONNXR1CSInputs::WriteLookupOutputToRD => {
            //         CommittedPolynomials::WriteLookupOutputToRD.generate_witness(preprocessing, trace)
            //     }
            //     JoltONNXR1CSInputs::WritePCtoRD => {
            //         CommittedPolynomials::WritePCtoRD.generate_witness(preprocessing, trace)
            //     }
            //     JoltONNXR1CSInputs::LookupOutput => {
            //         let coeffs: Vec<u64> = trace
            //             .par_iter()
            //             .map(LookupQuery::<32>::to_lookup_output)
            //             .collect();
            //         coeffs.into()
            //     }
            //     JoltONNXR1CSInputs::NextUnexpandedPC => {
            //         let coeffs: Vec<u64> = trace
            //             .par_iter()
            //             .map(|cycle| {
            //                 let is_branch =
            //                     cycle.instruction().circuit_flags()[CircuitFlags::Branch as usize];
            //                 let should_branch =
            //                     is_branch && LookupQuery::<32>::to_lookup_output(cycle) != 0;
            //                 let instr = cycle.instruction().normalize();
            //                 if should_branch {
            //                     (instr.address as i64 + instr.operands.imm) as u64
            //                 } else {
            //                     // JoltONNXR1CSInputs::NextPCJump
            //                     let is_jump =
            //                         cycle.instruction().circuit_flags()[CircuitFlags::Jump as usize];
            //                     let do_not_update_pc = cycle.instruction().circuit_flags()
            //                         [CircuitFlags::DoNotUpdateUnexpandedPC as usize];
            //                     if is_jump {
            //                         LookupQuery::<32>::to_lookup_output(cycle)
            //                     } else if do_not_update_pc {
            //                         instr.address as u64
            //                     } else {
            //                         instr.address as u64 + 4
            //                     }
            //                 }
            //             })
            //             .collect();
            //         coeffs.into()
            //     }
            //     JoltONNXR1CSInputs::ShouldBranch => {
            //         CommittedPolynomials::ShouldBranch.generate_witness(preprocessing, trace)
            //     }
            JoltONNXR1CSInputs::OpFlags(flag) => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| cycle.instr.to_circuit_flags()[*flag as usize] as u8)
                    .collect();
                coeffs.into()
            }
            _ => unimplemented!(),
        }
    }
}

impl Into<Variable> for JoltONNXR1CSInputs {
    fn into(self) -> Variable {
        Variable::Input(self.to_index())
    }
}

impl Into<Term> for JoltONNXR1CSInputs {
    fn into(self) -> Term {
        Term(Variable::Input(self.to_index()), 1)
    }
}

impl Into<LC> for JoltONNXR1CSInputs {
    fn into(self) -> LC {
        Term(Variable::Input(self.to_index()), 1).into()
    }
}

/// Newtype wrapper to allow conversion from a vector of inputs to LC.
pub struct InputVec(pub Vec<JoltONNXR1CSInputs>);

impl Into<LC> for InputVec {
    fn into(self) -> LC {
        let terms: Vec<Term> = self.0.into_iter().map(Into::into).collect();
        LC::new(terms)
    }
}

impl<T: Into<LC>> std::ops::Add<T> for JoltONNXR1CSInputs {
    type Output = LC;
    fn add(self, rhs: T) -> Self::Output {
        let lhs_lc: LC = self.into();
        let rhs_lc: LC = rhs.into();
        lhs_lc + rhs_lc
    }
}
impl<T: Into<LC>> std::ops::Sub<T> for JoltONNXR1CSInputs {
    type Output = LC;
    fn sub(self, rhs: T) -> Self::Output {
        let lhs_lc: LC = self.into();
        let rhs_lc: LC = rhs.into();
        lhs_lc - rhs_lc
    }
}
impl std::ops::Mul<i64> for JoltONNXR1CSInputs {
    type Output = Term;
    fn mul(self, rhs: i64) -> Self::Output {
        Term(Variable::Input(self.to_index()), rhs)
    }
}
impl std::ops::Mul<JoltONNXR1CSInputs> for i64 {
    type Output = Term;
    fn mul(self, rhs: JoltONNXR1CSInputs) -> Self::Output {
        Term(Variable::Input(rhs.to_index()), self)
    }
}
impl std::ops::Add<JoltONNXR1CSInputs> for i64 {
    type Output = LC;
    fn add(self, rhs: JoltONNXR1CSInputs) -> Self::Output {
        let term1 = Term(Variable::Input(rhs.to_index()), 1);
        let term2 = Term(Variable::Constant, self);
        LC::new(vec![term1, term2])
    }
}
impl std::ops::Sub<JoltONNXR1CSInputs> for i64 {
    type Output = LC;
    fn sub(self, rhs: JoltONNXR1CSInputs) -> Self::Output {
        let term1 = Term(Variable::Input(rhs.to_index()), -1);
        let term2 = Term(Variable::Constant, self);
        LC::new(vec![term1, term2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_index_to_index() {
        for i in 0..JoltONNXR1CSInputs::num_inputs() {
            assert_eq!(i, JoltONNXR1CSInputs::from_index(i).to_index());
        }
        for var in ALL_R1CS_INPUTS {
            assert_eq!(
                var,
                JoltONNXR1CSInputs::from_index(JoltONNXR1CSInputs::to_index(&var))
            );
        }
    }
}
