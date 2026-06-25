//! Compatibility audits for imported `jolt-core` ZK proof artifacts.

use common::jolt_device::JoltDevice;
use jolt_blindfold::BlindFoldProtocol;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::Field;
use jolt_openings::{
    BatchOpeningScheme, CommitmentLayoutDigest, CommitmentScheme, ZkBatchOpeningScheme,
};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::{
    config::{validate_proof_config, JoltProtocolConfig},
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        stage1, stage2, stage3, stage4, stage5, stage5_increment, stage6, stage7, stage8,
        zk::{blindfold, inputs::BlindFoldInputs, outputs::zk_stage_outputs},
    },
    verifier::{absorb_commitments, absorb_preamble, validate_inputs, validate_proof_consistency},
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
        + BatchOpeningScheme
        + ZkBatchOpeningScheme<HidingCommitment = VC::Output>,
    // Stage 8 still uses shared clear/ZK statement construction; splitting a
    // ZK-only builder would be broader than this audit helper.
    PCS::Output: AppendToTranscript + CommitmentLayoutDigest,
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
    )?;

    let stage1 = stage1::verify(&checked, preprocessing, proof, &mut transcript)?;
    let stage2 = stage2::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage2::deps(&stage1),
    )?;
    let stage3 = stage3::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage3::deps(&stage1, &stage2)?,
    )?;
    let stage4 = stage4::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage4::deps(&stage2, &stage3)?,
    )?;
    let stage5 = stage5::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage5::deps(&stage2, &stage4)?,
    )?;
    let stage5_increment = stage5_increment::verify(
        &checked,
        proof,
        &mut transcript,
        stage5_increment::deps(&stage2, &stage4, &stage5),
    )?;
    let stage6 = stage6::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage6::deps(
            &stage1,
            &stage2,
            &stage3,
            &stage4,
            &stage5,
            stage5_increment.as_ref(),
        )?,
    )?;
    let stage7 = stage7::verify(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        stage7::deps(&stage4, &stage6)?,
    )?;
    let stage8 = stage8::verify(
        &checked,
        preprocessing,
        proof,
        trusted_advice_commitment,
        &mut transcript,
        stage8::deps(&stage6, &stage7)?,
    )?;

    let zk_stages = zk_stage_outputs::<PCS, VC>(
        &stage1, &stage2, &stage3, &stage4, &stage5, &stage6, &stage7, &stage8,
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
        stage6: zk_stages.stage6,
        stage7: zk_stages.stage7,
        stage8: zk_stages.stage8,
    })?;

    Ok(ZkBlindFoldProtocolShape::from_protocol(&blindfold.protocol))
}
