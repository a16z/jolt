#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use crate::poly::commitment::commitment_scheme::{CommitmentScheme};
use crate::utils::transcript::AppendToTranscript;
use crate::{
    jolt::vm::{rv32i_vm::RV32I, JoltCommitments},
    utils::{
        transcript::ProofTranscript,
    },
};

use super::key::UniformSpartanKey;
use super::spartan_3::{SpartanError, UniformSpartanProof};

use crate::poly::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::{constants::MEMORY_OPS_PER_INSTRUCTION};
use rayon::prelude::*;

use strum::EnumCount;

#[derive(Clone, Debug, Default)]
pub struct R1CSInputs<'a, F: JoltField> {
    padded_trace_len: usize,
    pub pc: Vec<F>,
    pub bytecode_a: Vec<F>,
    bytecode_v: Vec<F>,
    memreg_a_rw: &'a [F],
    memreg_v_reads: Vec<&'a F>,
    memreg_v_writes: Vec<&'a F>,
    pub chunks_x: Vec<F>,
    pub chunks_y: Vec<F>,
    pub chunks_query: Vec<F>,
    lookup_outputs: Vec<F>,
    pub circuit_flags_bits: Vec<F>,
    instruction_flags_bits: Vec<F>,
}

// #[derive(Clone, Debug, Default)]
// pub struct R1CSStepInputs<F: JoltField> {
//     pub padded_trace_len: usize,
//     pub input_pc: F,
//     pub bytecode_v: Vec<F>,
//     pub memreg_v_reads: Vec<F>,
//     pub memreg_v_writes: Vec<F>,
//     pub chunks_y: Vec<F>,
//     pub chunks_query: Vec<F>,
//     pub lookup_outputs: Vec<F>,
//     pub circuit_flags_bits: Vec<F>,
//     pub instruction_flags_bits: Vec<F>,
// }

impl<'a, F: JoltField> R1CSInputs<'a, F> {
    #[tracing::instrument(skip_all, name = "R1CSInputs::new")]
    pub fn new(
        padded_trace_len: usize,
        pc: Vec<F>,
        bytecode_a: Vec<F>,
        bytecode_v: Vec<F>,
        memreg_a_rw: &'a [F],
        memreg_v_reads: Vec<&'a F>,
        memreg_v_writes: Vec<&'a F>,
        chunks_x: Vec<F>,
        chunks_y: Vec<F>,
        chunks_query: Vec<F>,
        lookup_outputs: Vec<F>,
        circuit_flags_bits: Vec<F>,
        instruction_flags_bits: Vec<F>,
    ) -> Self {
        assert!(pc.len() % padded_trace_len == 0);
        assert!(bytecode_a.len() % padded_trace_len == 0);
        assert!(bytecode_v.len() % padded_trace_len == 0);
        assert!(memreg_a_rw.len() % padded_trace_len == 0);
        assert!(memreg_v_reads.len() % padded_trace_len == 0);
        assert!(memreg_v_writes.len() % padded_trace_len == 0);
        assert!(chunks_x.len() % padded_trace_len == 0);
        assert!(chunks_y.len() % padded_trace_len == 0);
        assert!(chunks_query.len() % padded_trace_len == 0);
        assert!(lookup_outputs.len() % padded_trace_len == 0);
        assert!(circuit_flags_bits.len() % padded_trace_len == 0);
        assert!(instruction_flags_bits.len() % padded_trace_len == 0);

        Self {
            padded_trace_len,
            pc,
            bytecode_a,
            bytecode_v,
            memreg_a_rw,
            memreg_v_reads,
            memreg_v_writes,
            chunks_x,
            chunks_y,
            chunks_query,
            lookup_outputs,
            circuit_flags_bits,
            instruction_flags_bits,
        }
    }

    // fn push_to_step<T: Borrow<F>>(&self, data: &Vec<T>, step: &mut Vec<F>, step_index: usize) {
    //     let num_vals = data.len() / self.padded_trace_len;
    //     for var_index in 0..num_vals {
    //         step.push(*data[var_index * self.padded_trace_len + step_index].borrow());
    //     }
    // }

    // pub fn clone_step(&self, step_index: usize) -> R1CSStepInputs<F> {
    //     let program_counter = if step_index > 0 && self.bytecode_a[step_index].is_zero() {
    //         F::zero()
    //     } else {
    //         self.bytecode_a[step_index]
    //     };

    //     let mut output = R1CSStepInputs {
    //         padded_trace_len: self.padded_trace_len,
    //         input_pc: program_counter,
    //         bytecode_v: Vec::with_capacity(6),
    //         memreg_v_reads: Vec::with_capacity(7),
    //         memreg_v_writes: Vec::with_capacity(7),
    //         chunks_y: Vec::with_capacity(4),
    //         chunks_query: Vec::with_capacity(4),
    //         lookup_outputs: Vec::with_capacity(2),
    //         circuit_flags_bits: Vec::with_capacity(NUM_CIRCUIT_FLAGS),
    //         instruction_flags_bits: Vec::with_capacity(RV32I::COUNT),
    //     };
    //     self.push_to_step(&self.bytecode_v, &mut output.bytecode_v, step_index);
    //     self.push_to_step(&self.memreg_v_reads, &mut output.memreg_v_reads, step_index);
    //     self.push_to_step(
    //         &self.memreg_v_writes,
    //         &mut output.memreg_v_writes,
    //         step_index,
    //     );
    //     self.push_to_step(&self.chunks_y, &mut output.chunks_y, step_index);
    //     self.push_to_step(&self.chunks_query, &mut output.chunks_query, step_index);
    //     self.push_to_step(&self.lookup_outputs, &mut output.lookup_outputs, step_index);
    //     self.push_to_step(
    //         &self.circuit_flags_bits,
    //         &mut output.circuit_flags_bits,
    //         step_index,
    //     );
    //     self.push_to_step(
    //         &self.instruction_flags_bits,
    //         &mut output.instruction_flags_bits,
    //         step_index,
    //     );

    //     output
    // }

    pub fn clone_to_trace_len_chunks(&self, padded_trace_len: usize) -> Vec<Vec<F>> {
        let mut chunks: Vec<Vec<F>> = Vec::new();

        let pc_chunks = self
            .pc
            .par_chunks(padded_trace_len)
            .map(|chunk| chunk.to_vec());
        chunks.par_extend(pc_chunks);

        let bytecode_a_chunks = self
            .bytecode_a
            .par_chunks(padded_trace_len)
            .map(|chunk| chunk.to_vec());
        chunks.par_extend(bytecode_a_chunks);

        let bytecode_v_chunks = self
            .bytecode_v
            .par_chunks(padded_trace_len)
            .map(|chunk| chunk.to_vec());
        chunks.par_extend(bytecode_v_chunks);

        let memreg_a_rw_chunks = self
            .memreg_a_rw
            .par_chunks(padded_trace_len)
            .map(|chunk| chunk.to_vec());
        chunks.par_extend(memreg_a_rw_chunks);

        let memreg_v_reads_chunks = self
            .memreg_v_reads
            .par_chunks(padded_trace_len)
            .map(|chunk| chunk.par_iter().map(|&elem| *elem).collect::<Vec<F>>());
        chunks.par_extend(memreg_v_reads_chunks);

        let memreg_v_writes_chunks = self
            .memreg_v_writes
            .par_chunks(padded_trace_len)
            .map(|chunk| chunk.par_iter().map(|&elem| *elem).collect::<Vec<F>>());
        chunks.par_extend(memreg_v_writes_chunks);

        let chunks_x_chunks = self
            .chunks_x
            .par_chunks(padded_trace_len)
            .map(|chunk| chunk.to_vec());
        chunks.par_extend(chunks_x_chunks);

        let chunks_y_chunks = self
            .chunks_y
            .par_chunks(padded_trace_len)
            .map(|chunk| chunk.to_vec());
        chunks.par_extend(chunks_y_chunks);

        let chunks_query_chunks = self
            .chunks_query
            .par_chunks(padded_trace_len)
            .map(|chunk| chunk.to_vec());
        chunks.par_extend(chunks_query_chunks);

        let lookup_outputs_chunks = self
            .lookup_outputs
            .par_chunks(padded_trace_len)
            .map(|chunk| chunk.to_vec());
        chunks.par_extend(lookup_outputs_chunks);

        let circuit_flags_bits_chunks = self
            .circuit_flags_bits
            .par_chunks(padded_trace_len)
            .map(|chunk| chunk.to_vec());
        chunks.par_extend(circuit_flags_bits_chunks);

        let instruction_flags_bits_chunks = self
            .instruction_flags_bits
            .par_chunks(padded_trace_len)
            .map(|chunk| chunk.to_vec());
        chunks.par_extend(instruction_flags_bits_chunks);
        chunks
    }
}

// TODO(sragss): Reprivatize.
/// Commitments unique to R1CS.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSCommitment<C: CommitmentScheme> {
    pub io: Vec<C::Commitment>,
    pub aux: Vec<C::Commitment>,
    /// Operand chunks { x, y }
    pub chunks: Vec<C::Commitment>,
    pub circuit_flags: Vec<C::Commitment>,
}

impl<C: CommitmentScheme> AppendToTranscript for R1CSCommitment<C> {
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript) {
        transcript.append_message(label, b"R1CSCommitment_begin");
        for commitment in &self.io {
            commitment.append_to_transcript(b"io", transcript);
        }
        for commitment in &self.aux {
            commitment.append_to_transcript(b"aux", transcript);
        }
        for commitment in &self.chunks {
            commitment.append_to_transcript(b"chunks_s", transcript);
        }
        for commitment in &self.circuit_flags {
            commitment.append_to_transcript(b"circuit_flags", transcript);
        }
        transcript.append_message(label, b"R1CSCommitment_end");
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSProof<F: JoltField, C: CommitmentScheme<Field = F>> {
    pub key: UniformSpartanKey<F>,
    pub proof: UniformSpartanProof<F, C>,
}

impl<F: JoltField, C: CommitmentScheme<Field = F>> R1CSProof<F, C> {
    fn format_commitments(jolt_commitments: &JoltCommitments<C>, C: usize) -> Vec<&C::Commitment> {
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
        combined_commitments.extend(r1cs_commitments.as_ref().unwrap().io.iter());

        combined_commitments.push(&bytecode_trace_commitments[0]); // "virtual" address
        combined_commitments.push(&bytecode_trace_commitments[2]); // "real" address
        combined_commitments.push(&bytecode_trace_commitments[3]); // op_flags_packed
        combined_commitments.push(&bytecode_trace_commitments[4]); // rd
        combined_commitments.push(&bytecode_trace_commitments[5]); // rs1
        combined_commitments.push(&bytecode_trace_commitments[6]); // rs2
        combined_commitments.push(&bytecode_trace_commitments[7]); // imm

        combined_commitments.extend(memory_trace_commitments.iter());

        combined_commitments.extend(r1cs_commitments.as_ref().unwrap().chunks.iter());

        combined_commitments.extend(instruction_lookup_indices_commitments.iter());

        combined_commitments.push(
            jolt_commitments
                .instruction_lookups
                .trace_commitment
                .last()
                .unwrap(),
        );

        combined_commitments.extend(r1cs_commitments.as_ref().unwrap().circuit_flags.iter());

        combined_commitments.extend(instruction_flag_commitments.iter());

        combined_commitments.extend(r1cs_commitments.as_ref().unwrap().aux.iter());

        combined_commitments
    }

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
}
