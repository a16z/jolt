use jolt_blindfold::BlindFoldProtocol;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::CommittedOutputClaims;

use crate::{
    stages::{stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8},
    VerifierError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommittedOutputClaimShape {
    pub output_claim_count: usize,
    pub row_count: usize,
    pub row_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommittedOutputClaimOutput<C> {
    pub shape: CommittedOutputClaimShape,
    pub commitments: CommittedOutputClaims<C>,
}

#[derive(Clone, Debug)]
pub struct BlindFoldOutput<F: Field, C> {
    pub protocol: BlindFoldProtocol<F, C>,
}

pub struct ZkStageOutputs<'a, PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub stage1: &'a stage1::Stage1ZkOutput<PCS::Field, VC::Output>,
    pub stage2: &'a stage2::Stage2ZkOutput<PCS::Field, VC::Output>,
    pub stage3: &'a stage3::Stage3ZkOutput<PCS::Field, VC::Output>,
    pub stage4: &'a stage4::Stage4ZkOutput<PCS::Field, VC::Output>,
    pub stage5: &'a stage5::Stage5ZkOutput<PCS::Field, VC::Output>,
    pub stage6: &'a stage6::Stage6ZkOutput<PCS::Field, VC::Output>,
    pub stage7: &'a stage7::Stage7ZkOutput<PCS::Field, VC::Output>,
    pub stage8: &'a stage8::Stage8ZkOutput<PCS::Field, PCS::Output, VC::Output>,
}

#[expect(
    clippy::too_many_arguments,
    reason = "The top-level verifier explicitly threads each stage output in protocol order."
)]
pub fn zk_stage_outputs<'a, PCS, VC>(
    stage1: &'a stage1::Stage1Output<PCS::Field, VC::Output>,
    stage2: &'a stage2::Stage2Output<PCS::Field, VC::Output>,
    stage3: &'a stage3::Stage3Output<PCS::Field, VC::Output>,
    stage4: &'a stage4::Stage4Output<PCS::Field, VC::Output>,
    stage5: &'a stage5::Stage5Output<PCS::Field, VC::Output>,
    stage6: &'a stage6::Stage6Output<PCS::Field, VC::Output>,
    stage7: &'a stage7::Stage7Output<PCS::Field, VC::Output>,
    stage8: &'a stage8::Stage8Output<PCS::Field, PCS::Output, VC::Output>,
) -> Result<ZkStageOutputs<'a, PCS, VC>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    Ok(ZkStageOutputs {
        stage1: match stage1 {
            stage1::Stage1Output::Zk(stage) => stage,
            stage1::Stage1Output::Clear(_) => {
                return Err(VerifierError::ExpectedCommittedProof { field: "stage1" });
            }
        },
        stage2: match stage2 {
            stage2::Stage2Output::Zk(stage) => stage,
            stage2::Stage2Output::Clear(_) => {
                return Err(VerifierError::ExpectedCommittedProof { field: "stage2" });
            }
        },
        stage3: match stage3 {
            stage3::Stage3Output::Zk(stage) => stage,
            stage3::Stage3Output::Clear(_) => {
                return Err(VerifierError::ExpectedCommittedProof { field: "stage3" });
            }
        },
        stage4: match stage4 {
            stage4::Stage4Output::Zk(stage) => stage,
            stage4::Stage4Output::Clear(_) => {
                return Err(VerifierError::ExpectedCommittedProof { field: "stage4" });
            }
        },
        stage5: match stage5 {
            stage5::Stage5Output::Zk(stage) => stage,
            stage5::Stage5Output::Clear(_) => {
                return Err(VerifierError::ExpectedCommittedProof { field: "stage5" });
            }
        },
        stage6: match stage6 {
            stage6::Stage6Output::Zk(stage) => stage,
            stage6::Stage6Output::Clear(_) => {
                return Err(VerifierError::ExpectedCommittedProof { field: "stage6" });
            }
        },
        stage7: match stage7 {
            stage7::Stage7Output::Zk(stage) => stage,
            stage7::Stage7Output::Clear(_) => {
                return Err(VerifierError::ExpectedCommittedProof { field: "stage7" });
            }
        },
        stage8: match stage8 {
            stage8::Stage8Output::Zk(stage) => stage,
            stage8::Stage8Output::Clear(_) => {
                return Err(VerifierError::ExpectedCommittedProof { field: "stage8" });
            }
        },
    })
}
