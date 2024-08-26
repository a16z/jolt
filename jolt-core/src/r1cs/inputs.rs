#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use crate::jolt::instruction::JoltInstructionSet;
use crate::jolt::vm::JoltTraceStep;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::utils::transcript::ProofTranscript;

use super::key::UniformSpartanKey;
use super::ops::ConstraintInput;
use super::spartan::{SpartanError, UniformSpartanProof};

use crate::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;
use common::rv_trace::NUM_CIRCUIT_FLAGS;

pub struct AuxPolynomials<F: JoltField> {
    pub left_lookup_operand: DensePolynomial<F>,
    pub right_lookup_operand: DensePolynomial<F>,
    pub imm_signed: DensePolynomial<F>,
    pub product: DensePolynomial<F>,
    pub relevant_y_chunks: Vec<DensePolynomial<F>>,
    pub write_lookup_output_to_rd: DensePolynomial<F>,
    pub write_pc_to_rd: DensePolynomial<F>,
    pub next_pc_jump: DensePolynomial<F>,
    pub should_branch: DensePolynomial<F>,
    pub next_pc: DensePolynomial<F>,
}

pub struct R1CSPolynomials<F: JoltField> {
    pub chunks_x: Vec<DensePolynomial<F>>,
    pub chunks_y: Vec<DensePolynomial<F>>,
    pub circuit_flags: [DensePolynomial<F>; NUM_CIRCUIT_FLAGS],
    pub aux: Option<AuxPolynomials<F>>,
}

impl<F: JoltField> R1CSPolynomials<F> {
    pub fn new<
        const C: usize,
        const M: usize,
        InstructionSet: JoltInstructionSet,
        I: ConstraintInput,
    >(
        trace: &[JoltTraceStep<InstructionSet>],
    ) -> Self {
        let log_M = log2(M) as usize;

        let mut chunks_x = vec![unsafe_allocate_zero_vec(trace.len()); C];
        let mut chunks_y = vec![unsafe_allocate_zero_vec(trace.len()); C];
        let mut circuit_flags = vec![unsafe_allocate_zero_vec(trace.len()); NUM_CIRCUIT_FLAGS];

        // TODO(moodlezoup): Can be parallelized
        for (step_index, step) in trace.iter().enumerate() {
            if let Some(instr) = &step.instruction_lookup {
                let (x, y) = instr.operand_chunks(C, log_M);
                for i in 0..C {
                    chunks_x[i][step_index] = F::from_u64(x[i]).unwrap();
                    chunks_y[i][step_index] = F::from_u64(y[i]).unwrap();
                }
            }

            for j in 0..NUM_CIRCUIT_FLAGS {
                if step.circuit_flags[j] {
                    circuit_flags[j][step_index] = F::one();
                }
            }
        }

        Self {
            chunks_x: chunks_x
                .into_iter()
                .map(|vals| DensePolynomial::new(vals))
                .collect(),
            chunks_y: chunks_y
                .into_iter()
                .map(|vals| DensePolynomial::new(vals))
                .collect(),
            circuit_flags: circuit_flags
                .into_iter()
                .map(|vals| DensePolynomial::new(vals))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            aux: None,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSProof<const C: usize, I: ConstraintInput, F: JoltField> {
    pub key: UniformSpartanKey<C, I, F>,
    pub proof: UniformSpartanProof<C, I, F>,
}

impl<const C: usize, I: ConstraintInput, F: JoltField> R1CSProof<C, I, F> {
    #[tracing::instrument(skip_all, name = "R1CSProof::verify")]
    pub fn verify(&self, transcript: &mut ProofTranscript) -> Result<(), SpartanError> {
        self.proof.verify_precommitted(&self.key, transcript)
    }
}
