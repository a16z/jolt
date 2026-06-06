use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig, JoltRelationId};
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::SumcheckProof;

use crate::{
    config::JoltProtocolConfig,
    preprocessing::JoltVerifierPreprocessing,
    proof::{JoltProof, TracePolynomialOrder},
    stages::{
        stage1::Stage1ZkOutput, stage2::Stage2ZkOutput, stage3::Stage3ZkOutput,
        stage4::Stage4ZkOutput, stage5::Stage5ZkOutput, stage6::Stage6ZkOutput,
        stage7::Stage7ZkOutput, stage8::Stage8ZkOutput,
    },
    verifier::CheckedInputs,
};

pub(crate) struct CommittedOutputClaimInputs<'a, F: Field, C> {
    pub checked: &'a CheckedInputs,
    pub proof: &'a SumcheckProof<F, C>,
    pub proof_label: &'static str,
    pub output_claim_count: usize,
    pub stage: JoltRelationId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlindFoldProofContext {
    pub protocol: JoltProtocolConfig,
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    pub trace_polynomial_order: TracePolynomialOrder,
    pub untrusted_advice_commitment_present: bool,
}

impl BlindFoldProofContext {
    pub fn from_proof<PCS, VC, ZkProof>(proof: &JoltProof<PCS, VC, ZkProof>) -> Self
    where
        PCS: CommitmentScheme,
        VC: jolt_crypto::VectorCommitment<Field = PCS::Field>,
    {
        Self {
            protocol: proof.protocol,
            rw_config: proof.rw_config,
            one_hot_config: proof.one_hot_config,
            trace_polynomial_order: proof.trace_polynomial_order,
            untrusted_advice_commitment_present: proof.untrusted_advice_commitment.is_some(),
        }
    }
}

pub struct BlindFoldInputs<'a, PCS, VC>
where
    PCS: CommitmentScheme,
    VC: jolt_crypto::VectorCommitment<Field = PCS::Field>,
{
    pub checked: &'a CheckedInputs,
    pub context: BlindFoldProofContext,
    pub preprocessing: &'a JoltVerifierPreprocessing<PCS, VC>,
    pub stage1: &'a Stage1ZkOutput<PCS::Field, VC::Output>,
    pub stage2: &'a Stage2ZkOutput<PCS::Field, VC::Output>,
    pub stage3: &'a Stage3ZkOutput<PCS::Field, VC::Output>,
    pub stage4: &'a Stage4ZkOutput<PCS::Field, VC::Output>,
    pub stage5: &'a Stage5ZkOutput<PCS::Field, VC::Output>,
    pub stage6: &'a Stage6ZkOutput<PCS::Field, VC::Output>,
    pub stage7: &'a Stage7ZkOutput<PCS::Field, VC::Output>,
    pub stage8: &'a Stage8ZkOutput<PCS::Field, PCS::Output, VC::Output>,
}
