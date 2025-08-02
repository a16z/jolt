use itertools::Itertools;
use onnx_tracer::trace_types::{CircuitFlags, ONNXCycle};
use rayon::prelude::*;

use jolt_core::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::MultilinearPolynomial, one_hot_polynomial::OneHotPolynomial,
    },
    utils::transcript::Transcript,
};

use jolt_core::jolt::instruction::LookupQuery;

use crate::jolt::{JoltProverPreprocessing, lookup_trace::LookupTrace};

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

impl CommittedPolynomials {
    pub fn len() -> usize {
        ALL_COMMITTED_POLYNOMIALS.len()
    }

    pub fn from_index(index: usize) -> Self {
        ALL_COMMITTED_POLYNOMIALS[index]
    }

    pub fn to_index(&self) -> usize {
        ALL_COMMITTED_POLYNOMIALS
            .iter()
            .find_position(|poly| *poly == self)
            .unwrap()
            .0
    }

    pub fn generate_witness<F, PCS, ProofTranscript>(
        &self,
        _preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[ONNXCycle],
    ) -> MultilinearPolynomial<F>
    where
        F: JoltField,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
        ProofTranscript: Transcript,
    {
        match self {
            CommittedPolynomials::LeftInstructionInput => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        cycle
                            .to_lookup()
                            .map(|lookup| LookupQuery::<32>::to_instruction_inputs(&lookup).0)
                            .unwrap_or_default()
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::RightInstructionInput => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| {
                        cycle
                            .to_lookup()
                            .map(|lookup| LookupQuery::<32>::to_instruction_inputs(&lookup).1)
                            .unwrap_or_default()
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::Product => {
                let coeffs: Vec<u64> = trace
                    .par_iter()
                    .map(|cycle| {
                        cycle
                            .to_lookup()
                            .map(|lookup| {
                                let (left_input, right_input) =
                                    LookupQuery::<32>::to_instruction_inputs(&lookup);
                                if left_input.checked_mul(right_input as u64).is_none() {
                                    panic!(
                                        "At cycle {cycle:?} Overflow in multiplication: {left_input} * {right_input}"
                                    );
                                }
                                left_input * right_input as u64
                            })
                            .unwrap_or_default()
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::WriteLookupOutputToRD => {
                let coeffs: Vec<u8> = trace
                    .par_iter()
                    .map(|cycle| {
                        let flag = cycle.instr.to_circuit_flags()
                            [CircuitFlags::WriteLookupOutputToRD as usize];
                        (cycle.td() as u8) * (flag as u8)
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::RdInc => {
                let coeffs: Vec<i64> = trace
                    .par_iter()
                    .map(|cycle| {
                        let pre_val = cycle.td_pre_val();
                        let post_val = cycle.td_post_val();
                        post_val as i64 - pre_val as i64
                    })
                    .collect();
                coeffs.into()
            }
            CommittedPolynomials::InstructionRa(i) => {
                if *i > 3 {
                    panic!("Unexpected i: {i}");
                }
                let addresses: Vec<usize> = trace
                    .par_iter()
                    .map(|cycle| {
                        let lookup_index = cycle.to_lookup().map_or(0, |x| x.to_lookup_index());
                        let k = (lookup_index >> (16 * (3 - i))) % (1 << 16);
                        k as usize
                    })
                    .collect();
                MultilinearPolynomial::OneHot(OneHotPolynomial::from_indices(addresses, 1 << 16))
            }
        }
    }
}
