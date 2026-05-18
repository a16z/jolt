//! Verifier-owned proof model types.

use jolt_blindfold::BlindFoldProof;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
use jolt_crypto::{Commitment, VectorCommitment};
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::SumcheckProof;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[expect(non_snake_case, reason = "Matches current jolt-core proof field name.")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    bound = "OpeningClaims: Serialize + DeserializeOwned, ZkProof: Serialize + DeserializeOwned"
)]
pub struct JoltProof<
    PCS,
    VC,
    OpeningClaims = (),
    ZkProof = BlindFoldProof<<PCS as CommitmentScheme>::Field, <VC as Commitment>::Output>,
> where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub commitments: Vec<PCS::Output>,
    pub stages: JoltStageProofs<PCS::Field, VC>,
    pub joint_opening_proof: PCS::Proof,
    pub untrusted_advice_commitment: Option<PCS::Output>,
    pub opening_claims: Option<OpeningClaims>,
    pub blindfold_proof: Option<ZkProof>,
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JoltStageProofs<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub stage1_uni_skip_first_round_proof: SumcheckProof<F, VC::Output>,
    pub stage1_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub stage2_uni_skip_first_round_proof: SumcheckProof<F, VC::Output>,
    pub stage2_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub stage3_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub stage4_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub stage5_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub stage6_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub stage7_sumcheck_proof: SumcheckProof<F, VC::Output>,
}
