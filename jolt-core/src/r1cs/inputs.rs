#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use crate::jolt::instruction::JoltInstructionSet;
use crate::jolt::vm::JoltTraceStep;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::r1cs::jolt_constraints::JoltIn;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::{
    jolt::vm::{rv32i_vm::RV32I, JoltCommitments},
    utils::transcript::ProofTranscript,
};

use super::key::UniformSpartanKey;
use super::spartan::{SpartanError, UniformSpartanProof};

use crate::field::JoltField;
use crate::r1cs::builder::CombinedUniformBuilder;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::log2;
use common::constants::MEMORY_OPS_PER_INSTRUCTION;
use common::rv_trace::NUM_CIRCUIT_FLAGS;
use strum::EnumCount;

pub struct R1CSPolynomials<F: JoltField> {
    pub chunks_x: Vec<DensePolynomial<F>>,
    pub chunks_y: Vec<DensePolynomial<F>>,
    pub circuit_flags: [DensePolynomial<F>; NUM_CIRCUIT_FLAGS],
    pub aux: Vec<DensePolynomial<F>>,
}

impl<F: JoltField> R1CSPolynomials<F> {
    pub fn new<const C: usize, const M: usize, InstructionSet: JoltInstructionSet>(
        trace: &[JoltTraceStep<InstructionSet>],
        builder: &CombinedUniformBuilder<F, JoltIn>,
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

        let aux = builder.compute_aux(todo!("polynomials"));
        // #[cfg(test)]
        // {
        //     let (az, bz, cz) = builder.compute_spartan_Az_Bz_Cz(todo!("polynomials"), &aux);
        //     builder.assert_valid(&az, &bz, &cz);
        // }

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
            aux: todo!(),
        }
    }
}

pub struct R1CSAuxVariables<F: JoltField> {
    x: DensePolynomial<F>,
    y: DensePolynomial<F>,
    imm_signed: DensePolynomial<F>,
    x_times_y: DensePolynomial<F>,
    relevant_chunk_y: [DensePolynomial<F>; 4],
    rd_nonzero_and_lookup_to_rd: DensePolynomial<F>,
    rd_nonzero_and_jmp: DensePolynomial<F>,
    next_pc_jump: DensePolynomial<F>,
    should_branch: DensePolynomial<F>,
    next_pc: DensePolynomial<F>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSProof<F: JoltField, C: CommitmentScheme<Field = F>> {
    pub key: UniformSpartanKey<F>,
    pub proof: UniformSpartanProof<F, C>,
}

impl<F: JoltField, C: CommitmentScheme<Field = F>> R1CSProof<F, C> {
    #[tracing::instrument(skip_all, name = "R1CSProof::verify")]
    pub fn verify(
        &self,
        generators: &C::Setup,
        jolt_commitments: JoltCommitments<C>,
        C: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(), SpartanError> {
        let witness_segment_commitments = Self::format_commitments(&jolt_commitments, C);
        self.proof.verify_precommitted(
            &self.key,
            witness_segment_commitments,
            generators,
            transcript,
        )
    }

    #[tracing::instrument(skip_all, name = "R1CSProof::format_commitments")]
    pub fn format_commitments(
        jolt_commitments: &JoltCommitments<C>,
        C: usize,
    ) -> Vec<&C::Commitment> {
        let r1cs_commitments = &jolt_commitments.r1cs;
        let bytecode_trace_commitments = &jolt_commitments.bytecode.trace_commitments;
        let memory_trace_commitments = &jolt_commitments.read_write_memory.trace_commitments
            [..1 + MEMORY_OPS_PER_INSTRUCTION + 5]; // a_read_write, v_read, v_write
        let instruction_lookup_indices_commitments =
            &jolt_commitments.instruction_lookups.trace_commitment[..C];
        let instruction_flag_commitments = &jolt_commitments.instruction_lookups.trace_commitment
            [jolt_commitments.instruction_lookups.trace_commitment.len() - RV32I::COUNT - 1
                ..jolt_commitments.instruction_lookups.trace_commitment.len() - 1];

        let mut combined_commitments: Vec<&C::Commitment> = Vec::new();

        combined_commitments.push(&bytecode_trace_commitments[0]); // "virtual" address
        combined_commitments.push(&bytecode_trace_commitments[2]); // "real" address
        combined_commitments.push(&bytecode_trace_commitments[3]); // op_flags_packed
        combined_commitments.push(&bytecode_trace_commitments[4]); // rd
        combined_commitments.push(&bytecode_trace_commitments[5]); // rs1
        combined_commitments.push(&bytecode_trace_commitments[6]); // rs2
        combined_commitments.push(&bytecode_trace_commitments[7]); // imm

        combined_commitments.extend(memory_trace_commitments.iter());

        combined_commitments.extend(instruction_lookup_indices_commitments.iter());
        combined_commitments.push(
            jolt_commitments
                .instruction_lookups
                .trace_commitment
                .last()
                .unwrap(),
        );

        combined_commitments.extend(r1cs_commitments.iter());
        combined_commitments.extend(instruction_flag_commitments.iter());

        combined_commitments
    }
}
