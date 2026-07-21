//! Shape audits for verifier-native ZK proofs.

use common::jolt_device::JoltDevice;
use jolt_blindfold::BlindFoldProtocol;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_transcript::{AppendToTranscript, Transcript};

use jolt_verifier::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        stage1, stage2, stage3, stage4, stage5, stage6a, stage6b, stage7, stage8,
        zk::{blindfold, inputs::BlindFoldInputs},
    },
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
    let jolt_verifier::PreStage1VerifierState {
        checked,
        mut transcript,
    } = jolt_verifier::verify_until_stage1::<PCS, VC, T, ZkProof>(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        true,
    )?;

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
        preprocessing,
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

    let blindfold = blindfold::build(BlindFoldInputs {
        checked: &checked,
        preprocessing,
        proof,
        stage1: stage1.zk()?,
        stage2: stage2.zk()?,
        stage3: stage3.zk()?,
        stage4: stage4.zk()?,
        stage5: stage5.zk()?,
        stage6a: stage6a.zk()?,
        stage6b: stage6b.zk()?,
        stage7: stage7.zk()?,
        stage8: stage8.zk()?,
    })?;

    Ok(ZkBlindFoldProtocolShape::from_protocol(&blindfold))
}
