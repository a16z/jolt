//! Shape audits for verifier-native ZK proofs.

use common::jolt_device::JoltDevice;
use jolt_blindfold::BlindFoldProtocol;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};

use jolt_verifier::{
    config::{validate_proof_config, JoltProtocolConfig},
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        stage1, stage2, stage3, stage4, stage5, stage6a, stage6b, stage7, stage8,
        zk::{blindfold, inputs::BlindFoldInputs, outputs::zk_stage_outputs},
    },
    verifier::{validate_inputs, validate_proof_consistency, CheckedInputs},
    VerifierError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZkBlindFoldProtocolShape {
    pub coefficient_rows: usize,
    pub output_claim_rows: usize,
    pub auxiliary_rows: usize,
    pub witness_row_count: usize,
    pub witness_row_len: usize,
    pub error_row_count: usize,
    pub error_row_len: usize,
    pub eval_commitments: usize,
}

impl ZkBlindFoldProtocolShape {
    fn from_protocol<F, C>(protocol: &BlindFoldProtocol<F, C>) -> Self
    where
        F: Field,
    {
        Self {
            coefficient_rows: protocol.dimensions.coefficient_rows,
            output_claim_rows: protocol.dimensions.output_claim_rows,
            auxiliary_rows: protocol.dimensions.auxiliary_rows,
            witness_row_count: protocol.dimensions.witness.row_count,
            witness_row_len: protocol.dimensions.witness.row_len,
            error_row_count: protocol.dimensions.error.row_count,
            error_row_len: protocol.dimensions.error.row_len,
            eval_commitments: protocol.eval_commitments.len(),
        }
    }
}

fn absorb_preamble<PCS, VC, ZkProof, T>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
) where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let public_io = &checked.public_io;
    absorb_labeled_bytes(
        transcript,
        b"preprocessing_digest",
        &checked.preprocessing_digest,
    );
    absorb_labeled_u64(
        transcript,
        b"max_input_size",
        public_io.memory_layout.max_input_size,
    );
    absorb_labeled_u64(
        transcript,
        b"max_output_size",
        public_io.memory_layout.max_output_size,
    );
    absorb_labeled_u64(transcript, b"heap_size", public_io.memory_layout.heap_size);
    absorb_labeled_bytes(transcript, b"inputs", &public_io.inputs);
    absorb_labeled_bytes(transcript, b"outputs", &public_io.outputs);
    absorb_labeled_u64(transcript, b"panic", public_io.panic as u64);
    absorb_labeled_u64(transcript, b"ram_K", checked.ram_K as u64);
    absorb_labeled_u64(transcript, b"trace_length", checked.trace_length as u64);
    absorb_labeled_u64(transcript, b"entry_address", checked.entry_address);
    absorb_labeled_u64(
        transcript,
        b"ram_rw_phase1_num_rounds",
        proof.rw_config.ram_rw_phase1_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"ram_rw_phase2_num_rounds",
        proof.rw_config.ram_rw_phase2_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"registers_rw_phase1_num_rounds",
        proof.rw_config.registers_rw_phase1_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"registers_rw_phase2_num_rounds",
        proof.rw_config.registers_rw_phase2_num_rounds as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"log_k_chunk",
        proof.one_hot_config.log_k_chunk as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"lookups_ra_virtual_log_k_chunk",
        proof.one_hot_config.lookups_ra_virtual_log_k_chunk as u64,
    );
    absorb_labeled_u64(
        transcript,
        b"dory_layout",
        proof.trace_polynomial_order.transcript_scalar(),
    );
}

fn absorb_commitments<PCS, VC, ZkProof, T>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
    transcript: &mut T,
) where
    PCS: CommitmentScheme,
    PCS::Output: AppendToTranscript,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let mut absorb_commitment = |commitment: &PCS::Output| {
        append_payload_label(transcript, b"commitment", commitment);
        transcript.append(commitment);
    };
    absorb_commitment(&proof.commitments.rd_inc);
    absorb_commitment(&proof.commitments.ram_inc);
    for commitment in &proof.commitments.ra.instruction {
        absorb_commitment(commitment);
    }
    for commitment in &proof.commitments.ra.ram {
        absorb_commitment(commitment);
    }
    for commitment in &proof.commitments.ra.bytecode {
        absorb_commitment(commitment);
    }
    if let Some(untrusted_advice_commitment) = &proof.untrusted_advice_commitment {
        append_payload_label(transcript, b"untrusted_advice", untrusted_advice_commitment);
        transcript.append(untrusted_advice_commitment);
    }
    if let Some(trusted_advice_commitment) = trusted_advice_commitment {
        append_payload_label(transcript, b"trusted_advice", trusted_advice_commitment);
        transcript.append(trusted_advice_commitment);
    }
    if let Some(committed) = preprocessing.program.committed() {
        for commitment in &committed.bytecode_chunk_commitments {
            append_payload_label(transcript, b"bytecode_chunk_commit", commitment);
            transcript.append(commitment);
        }
        append_payload_label(
            transcript,
            b"program_image_commitment",
            &committed.program_image_commitment,
        );
        transcript.append(&committed.program_image_commitment);
    }
}

fn append_payload_label<T, A>(transcript: &mut T, label: &'static [u8], payload: &A)
where
    T: Transcript,
    A: AppendToTranscript,
{
    if let Some(len) = payload.transcript_payload_len() {
        transcript.append(&LabelWithCount(label, len));
    } else {
        transcript.append(&Label(label));
    }
}

fn absorb_labeled_bytes<T: Transcript>(transcript: &mut T, label: &'static [u8], bytes: &[u8]) {
    transcript.append(&LabelWithCount(label, bytes.len() as u64));
    transcript.append_bytes(bytes);
}

fn absorb_labeled_u64<T: Transcript>(transcript: &mut T, label: &'static [u8], value: u64) {
    transcript.append(&Label(label));
    transcript.append(&U64Word(value));
}

pub fn audit_zk_blindfold_protocol_shape<F, PCS, VC, T, ZkProof>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS, VC, ZkProof>,
    trusted_advice_commitment: Option<&PCS::Output>,
) -> Result<ZkBlindFoldProtocolShape, VerifierError>
where
    F: Field + AppendToTranscript,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    let config = JoltProtocolConfig::for_zk(true);
    validate_proof_config(&config, proof)?;

    let checked = validate_inputs(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment.is_some(),
        true,
    )?;
    validate_proof_consistency(proof, true)?;

    let mut transcript = T::new(b"Jolt");
    absorb_preamble(&checked, proof, &mut transcript);
    absorb_commitments(
        preprocessing,
        proof,
        trusted_advice_commitment,
        &mut transcript,
    );

    let formula_dimensions = jolt_verifier::stages::build_formula_dimensions(
        proof,
        preprocessing,
        &checked,
        checked.trace_length.ilog2() as usize,
        JoltRelationId::InstructionReadRaf,
    )?;

    let stage1 = stage1::verify(&checked, proof, &mut transcript)?;
    let stage2 = stage2::verify(&checked, proof, &mut transcript, &stage1)?;
    let stage3 = stage3::verify(&checked, proof, &mut transcript, &stage1, &stage2)?;
    let stage4 = stage4::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        &stage2,
        &stage3,
    )?;
    let stage5 = stage5::verify(
        &checked,
        proof,
        &formula_dimensions,
        &mut transcript,
        &stage2,
        &stage4,
    )?;
    let stage6a = stage6a::verify(
        &checked,
        proof,
        &formula_dimensions,
        &mut transcript,
        &stage1,
        &stage2,
        &stage3,
        &stage4,
        &stage5,
    )?;
    let stage6b = stage6b::verify(
        &checked,
        preprocessing,
        proof,
        &formula_dimensions,
        &mut transcript,
        &stage1,
        &stage2,
        &stage3,
        &stage4,
        &stage5,
        &stage6a,
    )?;
    let stage7 = stage7::verify(
        &checked,
        proof,
        &formula_dimensions,
        &mut transcript,
        &stage4,
        &stage6b,
    )?;
    let stage8 = stage8::verify(
        &checked,
        preprocessing,
        proof,
        &formula_dimensions,
        trusted_advice_commitment,
        &mut transcript,
        &stage6b,
        &stage7,
    )?;

    let zk_stages = zk_stage_outputs::<PCS, VC>(
        &stage1, &stage2, &stage3, &stage4, &stage5, &stage6a, &stage6b, &stage7, &stage8,
    )?;
    let blindfold = blindfold::build(BlindFoldInputs {
        checked: &checked,
        preprocessing,
        proof,
        stage1: zk_stages.stage1,
        stage2: zk_stages.stage2,
        stage3: zk_stages.stage3,
        stage4: zk_stages.stage4,
        stage5: zk_stages.stage5,
        stage6a: zk_stages.stage6a,
        stage6b: zk_stages.stage6b,
        stage7: zk_stages.stage7,
        stage8: zk_stages.stage8,
    })?;

    Ok(ZkBlindFoldProtocolShape::from_protocol(&blindfold.protocol))
}
