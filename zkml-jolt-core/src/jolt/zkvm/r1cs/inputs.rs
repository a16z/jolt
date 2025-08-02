#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use super::spartan::UniformSpartanProof;
use crate::jolt::lookup_trace::LookupTrace;
use crate::jolt::witness::CommittedPolynomials;
use crate::jolt::zkvm::JoltProverPreprocessing;
use jolt_core::field::JoltField;
use jolt_core::jolt::instruction::LookupQuery;
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
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        match self {
            JoltONNXR1CSInputs::Rd => {
                let coeffs: Vec<u8> = trace.par_iter().map(|cycle| cycle.td() as u8).collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::RdWriteValue => {
                let coeffs: Vec<u64> = trace.par_iter().map(|cycle| cycle.td_post_val()).collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::LeftInstructionInput => {
                CommittedPolynomials::LeftInstructionInput.generate_witness(preprocessing, trace)
            }
            JoltONNXR1CSInputs::RightInstructionInput => {
                CommittedPolynomials::RightInstructionInput.generate_witness(preprocessing, trace)
            }
            JoltONNXR1CSInputs::LeftLookupOperand => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        cycle
                            .to_lookup()
                            .map(|lookup| LookupQuery::<32>::to_lookup_operands(&lookup).0)
                            .unwrap_or(0)
                    })
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::RightLookupOperand => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        cycle
                            .to_lookup()
                            .map(|lookup| LookupQuery::<32>::to_lookup_operands(&lookup).1)
                            .unwrap_or(0)
                    })
                    .collect();
                coeffs.into()
            }
            JoltONNXR1CSInputs::Product => {
                CommittedPolynomials::Product.generate_witness(preprocessing, trace)
            }
            JoltONNXR1CSInputs::WriteLookupOutputToRD => {
                CommittedPolynomials::WriteLookupOutputToRD.generate_witness(preprocessing, trace)
            }
            JoltONNXR1CSInputs::LookupOutput => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        cycle
                            .to_lookup()
                            .map(|lookup| LookupQuery::<32>::to_lookup_output(&lookup))
                            .unwrap_or_default()
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

impl From<JoltONNXR1CSInputs> for Variable {
    fn from(input: JoltONNXR1CSInputs) -> Variable {
        Variable::Input(input.to_index())
    }
}

impl From<JoltONNXR1CSInputs> for Term {
    fn from(input: JoltONNXR1CSInputs) -> Term {
        Term(Variable::Input(input.to_index()), 1)
    }
}

impl From<JoltONNXR1CSInputs> for LC {
    fn from(input: JoltONNXR1CSInputs) -> LC {
        Term(Variable::Input(input.to_index()), 1).into()
    }
}

/// Newtype wrapper to allow conversion from a vector of inputs to LC.
pub struct InputVec(pub Vec<JoltONNXR1CSInputs>);

impl From<InputVec> for LC {
    fn from(input_vec: InputVec) -> LC {
        let terms: Vec<Term> = input_vec.0.into_iter().map(Into::into).collect();
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
