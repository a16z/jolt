//! Shape audits for verifier-native ZK proofs.

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::JoltDevice;
use jolt_blindfold::BlindFoldProtocol;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::{Field, RingAccumulator, WithAccumulator};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_transcript::{DuplexSpongeInterface, VerifierState};

use jolt_verifier::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8,
        zk::{blindfold, inputs::BlindFoldInputs, outputs::zk_stage_outputs},
    },
    verifier::verify_until_stage1,
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

pub fn audit_zk_blindfold_protocol_shape<F, PCS, VC, H>(
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof: &JoltProof<PCS>,
    trusted_advice_commitment: Option<&PCS::Output>,
) -> Result<ZkBlindFoldProtocolShape, VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>
        + AdditivelyHomomorphic
        + ZkOpeningScheme<HidingCommitment = VC::Output>,
    PCS::Output: CanonicalDeserialize + CanonicalSerialize + HomomorphicCommitment<F>,
    VC: VectorCommitment<Field = F>,
    VC::Output: Copy + HomomorphicCommitment<F> + CanonicalSerialize + CanonicalDeserialize,
    H: DuplexSpongeInterface<U = u8> + Default,
    for<'a> VerifierState<'a, H>: jolt_transcript::FsTranscript<F>,
    <F as WithAccumulator>::Accumulator: RingAccumulator<Element = F>,
{
    let state = verify_until_stage1::<PCS, VC, H>(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
        true,
    )?;
    let checked = state.checked;
    let narg_commitments = state.narg_commitments;
    let mut transcript = state.transcript;

    let formula_dimensions = jolt_verifier::stages::build_formula_dimensions(
        proof,
        preprocessing,
        &checked,
        checked.trace_length.ilog2() as usize,
        JoltRelationId::InstructionReadRaf,
    )?;

    let stage1 = stage1::verify::<PCS, VC, _>(&checked, proof, &mut transcript)?;
    let stage2 = stage2::verify::<PCS, VC, _>(&checked, proof, &mut transcript, &stage1)?;
    let stage3 = stage3::verify::<PCS, VC, _>(&checked, proof, &mut transcript, &stage1, &stage2)?;
    let stage4 = stage4::verify::<PCS, VC, _>(
        &checked,
        preprocessing,
        proof,
        &mut transcript,
        &stage2,
        &stage3,
    )?;
    let stage5 = stage5::verify::<PCS, VC, _>(
        &checked,
        proof,
        &formula_dimensions,
        &mut transcript,
        &stage2,
        &stage4,
    )?;
    let stage6 = stage6::verify::<PCS, VC, _>(
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
    let stage7 = stage7::verify::<PCS, VC, _>(
        &checked,
        proof,
        &formula_dimensions,
        &mut transcript,
        &stage4,
        &stage6,
    )?;
    let stage8 = stage8::verify::<PCS::Field, PCS, VC, _>(
        &checked,
        preprocessing,
        proof,
        &formula_dimensions,
        &narg_commitments,
        trusted_advice_commitment,
        &mut transcript,
        &stage6,
        &stage7,
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
