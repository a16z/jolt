//! Shape audits for verifier-native ZK proofs.

use common::jolt_device::JoltDevice;
use jolt_blindfold::BlindFoldProtocol;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_transcript::{AppendToTranscript, Transcript};

use jolt_verifier::{
    preprocessing::JoltVerifierPreprocessing, proof::JoltProof, stages::zk::blindfold,
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
    let (stages, _transcript) = jolt_verifier::verify_stages::<F, PCS, VC, T, ZkProof>(
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment,
    )?;
    let blindfold = blindfold::build(stages.blindfold_inputs(preprocessing, proof)?)?;
    Ok(ZkBlindFoldProtocolShape::from_protocol(&blindfold))
}
