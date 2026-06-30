//! Verifier-owned proof model types.

use jolt_blindfold::BlindFoldProof;
pub use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
use jolt_crypto::{Commitment, VectorCommitment};
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::SumcheckProof;
use serde::{Deserialize, Serialize};

use crate::{
    config::JoltProtocolConfig,
    stages::{stage1, stage2, stage3, stage4, stage5, stage6, stage7},
    VerifierError,
};

#[expect(
    non_snake_case,
    reason = "Matches current jolt-prover-legacy proof field name."
)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "ZkProof: Serialize + serde::de::DeserializeOwned")]
pub struct JoltProof<
    PCS,
    VC,
    ZkProof = BlindFoldProof<<PCS as CommitmentScheme>::Field, <VC as Commitment>::Output>,
> where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub protocol: JoltProtocolConfig,
    pub commitments: JoltCommitments<PCS::Output>,
    pub stages: JoltStageProofs<PCS::Field, VC>,
    pub joint_opening_proof: PCS::Proof,
    pub untrusted_advice_commitment: Option<PCS::Output>,
    pub claims: JoltProofClaims<PCS::Field, ZkProof>,
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    pub trace_polynomial_order: TracePolynomialOrder,
}

impl<PCS, VC, ZkProof> JoltProof<PCS, VC, ZkProof>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    #[expect(
        clippy::too_many_arguments,
        reason = "Constructor mirrors the proof payload while keeping internal verifier claims private."
    )]
    pub fn new(
        commitments: JoltCommitments<PCS::Output>,
        stages: JoltStageProofs<PCS::Field, VC>,
        joint_opening_proof: PCS::Proof,
        untrusted_advice_commitment: Option<PCS::Output>,
        claims: JoltProofClaims<PCS::Field, ZkProof>,
        trace_length: usize,
        ram_k: usize,
        rw_config: JoltReadWriteConfig,
        one_hot_config: JoltOneHotConfig,
        trace_polynomial_order: TracePolynomialOrder,
    ) -> Self {
        let protocol = JoltProtocolConfig::for_zk(claims.is_zk());
        Self {
            protocol,
            commitments,
            stages,
            joint_opening_proof,
            untrusted_advice_commitment,
            claims,
            trace_length,
            ram_K: ram_k,
            rw_config,
            one_hot_config,
            trace_polynomial_order,
        }
    }

    pub(crate) fn clear_claims(&self) -> Result<&ClearProofClaims<PCS::Field>, VerifierError> {
        match &self.claims {
            JoltProofClaims::Clear(claims) => Ok(claims),
            JoltProofClaims::Zk { .. } => Err(VerifierError::UnexpectedBlindFoldProof),
        }
    }

    pub(crate) fn blindfold_proof(&self) -> Result<&ZkProof, VerifierError> {
        match &self.claims {
            JoltProofClaims::Clear(_) => Err(VerifierError::MissingBlindFoldProof),
            JoltProofClaims::Zk { blindfold_proof } => Ok(blindfold_proof),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltCommitments<C> {
    pub rd_inc: C,
    pub ram_inc: C,
    pub ra: JoltRaCommitments<C>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltRaCommitments<C> {
    pub instruction: Vec<C>,
    pub ram: Vec<C>,
    pub bytecode: Vec<C>,
}

impl<C> JoltRaCommitments<C> {
    pub fn new(instruction: Vec<C>, ram: Vec<C>, bytecode: Vec<C>) -> Self {
        Self {
            instruction,
            ram,
            bytecode,
        }
    }
}

impl<C> JoltCommitments<C> {
    pub fn new(rd_inc: C, ram_inc: C, ra: JoltRaCommitments<C>) -> Self {
        Self {
            rd_inc,
            ram_inc,
            ra,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[expect(
    clippy::large_enum_variant,
    reason = "Clear claims are the verifier-owned standard proof payload; keeping them inline avoids heap indirection in the common clear path."
)]
#[serde(bound = "ZkProof: Serialize + serde::de::DeserializeOwned")]
pub enum JoltProofClaims<F, ZkProof>
where
    F: Field,
{
    Clear(ClearProofClaims<F>),
    Zk { blindfold_proof: ZkProof },
}

impl<F, ZkProof> JoltProofClaims<F, ZkProof>
where
    F: Field,
{
    pub const fn is_zk(&self) -> bool {
        matches!(self, Self::Zk { .. })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ClearProofClaims<F: Field> {
    pub stage1: stage1::outputs::Stage1OutputClaims<F>,
    pub stage2: stage2::outputs::Stage2OutputClaims<F>,
    pub stage3: stage3::outputs::Stage3OutputClaims<F>,
    pub stage4: stage4::outputs::Stage4OutputClaims<F>,
    pub stage5: stage5::outputs::Stage5OutputClaims<F, F>,
    pub stage6: stage6::outputs::Stage6OutputClaims<F>,
    pub stage7: stage7::outputs::Stage7OutputClaims<F>,
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
    pub stage6a_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub stage6b_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub stage7_sumcheck_proof: SumcheckProof<F, VC::Output>,
}
