#![allow(
    clippy::len_without_is_empty,
    clippy::type_complexity,
    clippy::too_many_arguments
)]

use crate::poly::commitment::commitment_scheme::{BatchType, CommitmentScheme};
use crate::utils::transcript::AppendToTranscript;
use crate::{
    jolt::vm::{rv32i_vm::RV32I, JoltCommitments},
    r1cs::r1cs_shape::R1CSShape,
    utils::{
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
        transcript::ProofTranscript,
    },
};

use super::{
    constraints::R1CSBuilder,
    spartan::{SpartanError, UniformShapeBuilder, UniformSpartanKey, UniformSpartanProof},
};

use crate::poly::field::JoltField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::{constants::MEMORY_OPS_PER_INSTRUCTION, rv_trace::NUM_CIRCUIT_FLAGS};
use rayon::prelude::*;
use std::borrow::Borrow;
use strum::EnumCount;

#[tracing::instrument(name = "synthesize_witnesses", skip_all)]
/// Returns (io, aux) = (pc_out, pc, aux)
fn synthesize_witnesses<F: JoltField>(
    inputs: &R1CSInputs<F>,
    num_aux: usize,
) -> (Vec<F>, Vec<F>, Vec<Vec<F>>) {
    let span = tracing::span!(tracing::Level::TRACE, "synthesize_witnesses");
    let _enter = span.enter();
    let triples_stepwise: Vec<(Vec<F>, F, F)> = (0..inputs.padded_trace_len)
        .into_par_iter()
        .map(|i| {
            let step = inputs.clone_step(i);
            let pc_cur = step.input_pc;
            let aux = R1CSBuilder::calculate_jolt_aux(step, num_aux);
            (aux, pc_cur, F::zero())
        })
        .collect();
    drop(_enter);

    // TODO(sragss / arasuarun): Remove pc_out, pc from calculate_aux and triples_stepwise

    // Convert step-wise to variable-wise
    // [[aux_var_0, aux_var_1, ...], [aux_var_0, aux_var_1, ...], ...] => [[aux_var_0, aux_var_0, ...], [aux_var_1, aux_var_1, ...], ...]
    // Aux result shape: aux[num_vars][num_steps]

    let num_vars = triples_stepwise[0].0.len();
    let mut aux: Vec<Vec<F>> = (0..num_vars)
        .into_par_iter()
        .map(|_| unsafe_allocate_zero_vec::<F>(inputs.padded_trace_len))
        .collect();
    let mut pc_out: Vec<F> = unsafe_allocate_zero_vec(inputs.padded_trace_len);
    let mut pc: Vec<F> = unsafe_allocate_zero_vec(inputs.padded_trace_len);

    let other_span = tracing::span!(tracing::Level::TRACE, "aux_recombine");
    let _enter = other_span.enter();
    aux.par_iter_mut()
        .enumerate()
        .for_each(|(var_index, aux_varwise)| {
            for step_index in 0..inputs.padded_trace_len {
                aux_varwise[step_index] = triples_stepwise[step_index].0[var_index];
            }
        });
    drop(_enter);

    pc_out
        .par_iter_mut()
        .zip(pc.par_iter_mut())
        .enumerate()
        .for_each(|(step_index, (slot_out, slot))| {
            *slot_out = triples_stepwise[step_index].1;
            *slot = triples_stepwise[step_index].2;
        });

    drop_in_background_thread(triples_stepwise);

    (pc_out, pc, aux)
}

#[derive(Clone, Debug, Default)]
pub struct R1CSInputs<'a, F: JoltField> {
    padded_trace_len: usize,
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
}

#[derive(Clone, Debug, Default)]
pub struct R1CSStepInputs<F: JoltField> {
    pub padded_trace_len: usize,
    pub input_pc: F,
    pub bytecode_v: Vec<F>,
    pub memreg_v_reads: Vec<F>,
    pub memreg_v_writes: Vec<F>,
    pub chunks_y: Vec<F>,
    pub chunks_query: Vec<F>,
    pub lookup_outputs: Vec<F>,
    pub circuit_flags_bits: Vec<F>,
    pub instruction_flags_bits: Vec<F>,
}

impl<'a, F: JoltField> R1CSInputs<'a, F> {
    #[tracing::instrument(skip_all, name = "R1CSInputs::new")]
    pub fn new(
        padded_trace_len: usize,
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

    fn push_to_step<T: Borrow<F>>(&self, data: &Vec<T>, step: &mut Vec<F>, step_index: usize) {
        let num_vals = data.len() / self.padded_trace_len;
        for var_index in 0..num_vals {
            step.push(*data[var_index * self.padded_trace_len + step_index].borrow());
        }
    }

    pub fn clone_step(&self, step_index: usize) -> R1CSStepInputs<F> {
        let program_counter = if step_index > 0 && self.bytecode_a[step_index].is_zero() {
            F::zero()
        } else {
            self.bytecode_a[step_index]
        };

        let mut output = R1CSStepInputs {
            padded_trace_len: self.padded_trace_len,
            input_pc: program_counter,
            bytecode_v: Vec::with_capacity(6),
            memreg_v_reads: Vec::with_capacity(7),
            memreg_v_writes: Vec::with_capacity(7),
            chunks_y: Vec::with_capacity(4),
            chunks_query: Vec::with_capacity(4),
            lookup_outputs: Vec::with_capacity(2),
            circuit_flags_bits: Vec::with_capacity(NUM_CIRCUIT_FLAGS),
            instruction_flags_bits: Vec::with_capacity(RV32I::COUNT),
        };
        self.push_to_step(&self.bytecode_v, &mut output.bytecode_v, step_index);
        self.push_to_step(&self.memreg_v_reads, &mut output.memreg_v_reads, step_index);
        self.push_to_step(
            &self.memreg_v_writes,
            &mut output.memreg_v_writes,
            step_index,
        );
        self.push_to_step(&self.chunks_y, &mut output.chunks_y, step_index);
        self.push_to_step(&self.chunks_query, &mut output.chunks_query, step_index);
        self.push_to_step(&self.lookup_outputs, &mut output.lookup_outputs, step_index);
        self.push_to_step(
            &self.circuit_flags_bits,
            &mut output.circuit_flags_bits,
            step_index,
        );
        self.push_to_step(
            &self.instruction_flags_bits,
            &mut output.instruction_flags_bits,
            step_index,
        );

        output
    }

    pub fn trace_len(&self) -> usize {
        self.padded_trace_len
    }

    pub fn num_vars_per_step(&self) -> usize {
        let trace_len = self.trace_len();
        self.bytecode_a.len() / trace_len
            + self.bytecode_v.len() / trace_len
            + self.memreg_a_rw.len() / trace_len
            + self.memreg_v_reads.len() / trace_len
            + self.memreg_v_writes.len() / trace_len
            + self.chunks_x.len() / trace_len
            + self.chunks_y.len() / trace_len
            + self.chunks_query.len() / trace_len
            + self.lookup_outputs.len() / trace_len
            + self.circuit_flags_bits.len() / trace_len
            + self.instruction_flags_bits.len() / trace_len
    }

    #[tracing::instrument(skip_all, name = "R1CSInputs::trace_len_chunks")]
    pub fn clone_to_trace_len_chunks(&self, padded_trace_len: usize) -> Vec<Vec<F>> {
        // TODO(sragss / arasuarun): Explain why non-trace-len relevant stuff (ex: bytecode) gets chunked to padded_trace_len
        let mut chunks: Vec<Vec<F>> = Vec::new();
        chunks.par_extend(
            self.bytecode_a
                .par_chunks(padded_trace_len)
                .map(|chunk| chunk.to_vec()),
        );
        chunks.par_extend(
            self.bytecode_v
                .par_chunks(padded_trace_len)
                .map(|chunk| chunk.to_vec()),
        );
        chunks.par_extend(
            self.memreg_a_rw
                .par_chunks(padded_trace_len)
                .map(|chunk| chunk.to_vec()),
        );
        chunks.par_extend(
            self.memreg_v_reads
                .par_chunks(padded_trace_len)
                .map(|chunk| chunk.par_iter().map(|&elem| *elem).collect::<Vec<F>>()),
        );
        chunks.par_extend(
            self.memreg_v_writes
                .par_chunks(padded_trace_len)
                .map(|chunk| chunk.par_iter().map(|&elem| *elem).collect::<Vec<F>>()),
        );
        chunks.par_extend(
            self.chunks_x
                .par_chunks(padded_trace_len)
                .map(|chunk| chunk.to_vec()),
        );
        chunks.par_extend(
            self.chunks_y
                .par_chunks(padded_trace_len)
                .map(|chunk| chunk.to_vec()),
        );
        chunks.par_extend(
            self.chunks_query
                .par_chunks(padded_trace_len)
                .map(|chunk| chunk.to_vec()),
        );
        chunks.par_extend(
            self.lookup_outputs
                .par_chunks(padded_trace_len)
                .map(|chunk| chunk.to_vec()),
        );
        chunks.par_extend(
            self.circuit_flags_bits
                .par_chunks(padded_trace_len)
                .map(|chunk| chunk.to_vec()),
        );
        chunks.par_extend(
            self.instruction_flags_bits
                .par_chunks(padded_trace_len)
                .map(|chunk| chunk.to_vec()),
        );
        chunks
    }
}

/// Commitments unique to R1CS.
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct R1CSCommitment<C: CommitmentScheme> {
    io: Vec<C::Commitment>,
    aux: Vec<C::Commitment>,
    /// Operand chunks { x, y }
    chunks: Vec<C::Commitment>,
    circuit_flags: Vec<C::Commitment>,
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
    proof: UniformSpartanProof<F, C>,
}

impl<F: JoltField, C: CommitmentScheme<Field = F>> R1CSProof<F, C> {
    /// Computes the full witness in segments of len `padded_trace_len`, commits to new required intermediary variables.
    #[tracing::instrument(skip_all, name = "R1CSProof::compute_witness_commit")]
    pub fn compute_witness_commit(
        _W: usize,
        _C: usize,
        padded_trace_len: usize,
        memory_start: u64,
        inputs: &R1CSInputs<F>,
        generators: &C::Setup,
    ) -> Result<(UniformSpartanKey<F>, Vec<Vec<F>>, R1CSCommitment<C>), SpartanError> {
        let span = tracing::span!(tracing::Level::TRACE, "shape_stuff");
        let _enter = span.enter();
        let mut jolt_shape = R1CSBuilder::default();
        R1CSBuilder::jolt_r1cs_matrices(&mut jolt_shape, memory_start);
        let key = UniformSpartanProof::<F, C>::setup_precommitted(
            &jolt_shape,
            padded_trace_len,
            memory_start,
        )?;
        drop(_enter);
        drop(span);

        let (pc_out, pc, aux) = synthesize_witnesses(inputs, jolt_shape.num_internal);
        let io_segments = vec![pc_out, pc];
        let io_segments_ref = vec![io_segments[0].as_slice(), io_segments[1].as_slice()];
        let aux_ref: Vec<&[F]> = aux.iter().map(AsRef::as_ref).collect();
        let io_comms = C::batch_commit(io_segments_ref.as_slice(), generators, BatchType::Big);
        let aux_comms = C::batch_commit(aux_ref.as_slice(), generators, BatchType::Big);

        let span = tracing::span!(tracing::Level::INFO, "new_commitments");
        let _guard = span.enter();
        let chunk_batch_size =
            inputs.chunks_x.len() / padded_trace_len + inputs.chunks_y.len() / padded_trace_len;
        let mut chunk_batch_slices: Vec<&[F]> = Vec::with_capacity(chunk_batch_size);
        for batchee in [&inputs.chunks_x, &inputs.chunks_y].iter() {
            chunk_batch_slices.extend(batchee.chunks(padded_trace_len));
        }
        let chunks_comms =
            C::batch_commit(chunk_batch_slices.as_slice(), generators, BatchType::Big);

        let circuit_flag_slices: Vec<&[F]> =
            inputs.circuit_flags_bits.chunks(padded_trace_len).collect();
        let circuit_flags_comms =
            C::batch_commit(circuit_flag_slices.as_slice(), generators, BatchType::Big);
        drop(_guard);

        let r1cs_commitments = R1CSCommitment {
            io: io_comms,
            aux: aux_comms,
            chunks: chunks_comms,
            circuit_flags: circuit_flags_comms,
        };

        let cloning_stuff_span =
            tracing::span!(tracing::Level::TRACE, "cloning_to_witness_segments");
        let _enter = cloning_stuff_span.enter();
        let inputs_segments = inputs.clone_to_trace_len_chunks(padded_trace_len);

        let mut w_segments: Vec<Vec<F>> =
            Vec::with_capacity(io_segments.len() + inputs_segments.len() + aux.len());
        w_segments.extend(io_segments.into_iter());
        w_segments.par_extend(inputs_segments.into_par_iter());
        w_segments.par_extend(aux.into_par_iter());

        drop(_enter);
        drop(cloning_stuff_span);

        Ok((key, w_segments, r1cs_commitments))
    }

    #[tracing::instrument(skip_all, name = "R1CSProof::prove")]
    pub fn prove(
        key: UniformSpartanKey<F>,
        witness_segments: Vec<Vec<F>>,
        transcript: &mut ProofTranscript,
    ) -> Result<Self, SpartanError> {
        // TODO(sragss): Fiat shamir (relevant) commitments
        let proof = UniformSpartanProof::prove_precommitted(&key, witness_segments, transcript)?;
        Ok(R1CSProof::<F, C> { proof, key })
    }

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
        // TODO(sragss): Fiat shamir (relevant) commitments
        let witness_segment_commitments = Self::format_commitments(&jolt_commitments, C);
        self.proof.verify_precommitted(
            witness_segment_commitments,
            &self.key,
            &[],
            generators,
            transcript,
        )
    }
}

impl<F: JoltField> UniformShapeBuilder<F> for R1CSBuilder {
    fn single_step_shape(&self, memory_start: u64) -> R1CSShape<F> {
        let mut jolt_shape = R1CSBuilder::default();
        R1CSBuilder::jolt_r1cs_matrices(&mut jolt_shape, memory_start);
        let constraints_F = jolt_shape.convert_to_field();
        let shape_single = R1CSShape::<F> {
            A: constraints_F.0,
            B: constraints_F.1,
            C: constraints_F.2,
            num_cons: jolt_shape.num_constraints + 1, // +1 for the IO consistency constraint
            num_vars: jolt_shape.num_aux,             // shouldn't include 1 or IO
            num_io: jolt_shape.num_inputs,
        };

        shape_single.pad_vars()
    }
}
