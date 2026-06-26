use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::CommittedOutputClaims;

use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        stage1::Stage1ZkOutput, stage2::Stage2ZkOutput, stage3::Stage3ZkOutput,
        stage4::Stage4ZkOutput, stage5::Stage5ZkOutput, stage6::Stage6ZkOutput,
        stage7::Stage7ZkOutput, stage8::Stage8ZkOutput,
    },
    verifier::CheckedInputs,
};

pub(crate) struct CommittedOutputClaimInputs<'a, C> {
    pub checked: &'a CheckedInputs,
    pub output_claims: &'a CommittedOutputClaims<C>,
    pub proof_label: &'static str,
    pub output_claim_count: usize,
    pub stage: JoltRelationId,
}

pub struct BlindFoldInputs<'a, PCS, VC, ZkProof>
where
    PCS: CommitmentScheme,
    VC: jolt_crypto::VectorCommitment<Field = PCS::Field>,
{
    pub checked: &'a CheckedInputs,
    pub preprocessing: &'a JoltVerifierPreprocessing<PCS, VC>,
    pub proof: &'a JoltProof<PCS, VC, ZkProof>,
    pub stage1: &'a Stage1ZkOutput<PCS::Field, VC::Output>,
    pub stage2: &'a Stage2ZkOutput<PCS::Field, VC::Output>,
    pub stage3: &'a Stage3ZkOutput<PCS::Field, VC::Output>,
    pub stage4: &'a Stage4ZkOutput<PCS::Field, VC::Output>,
    pub stage5: &'a Stage5ZkOutput<PCS::Field, VC::Output>,
    pub stage6: &'a Stage6ZkOutput<PCS::Field, VC::Output>,
    pub stage7: &'a Stage7ZkOutput<PCS::Field, VC::Output>,
    pub stage8: &'a Stage8ZkOutput<PCS::Field, PCS::Output, VC::Output>,
}
