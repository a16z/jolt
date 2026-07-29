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
    stages::{stage1, stage2, stage3, stage4, stage5, stage6a, stage6b, stage7},
    VerifierError,
};

/// The proof-carried polynomial commitments on the homomorphic build: one
/// commitment per committed polynomial.
#[cfg(not(feature = "akita"))]
pub type ProofCommitments<PCS> = JoltCommitments<<PCS as Commitment>::Output>;
/// The proof-carried polynomial commitments on the `akita` build: the single
/// packed `OneHotTrace` commitment carrying every per-proof column.
#[cfg(feature = "akita")]
pub type ProofCommitments<PCS> = <PCS as Commitment>::Output;

/// The final-opening discharge on the homomorphic build: one RLC-batched PCS
/// opening proof at the unified point.
#[cfg(not(feature = "akita"))]
pub type JointOpeningProof<PCS> = <PCS as CommitmentScheme>::Proof;
/// The Akita OneHotTrace opening is native and same-point. Only auxiliary packed
/// objects retain the generic reduction proof.
#[cfg(feature = "akita")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaJointOpeningProof<F, P> {
    pub one_hot_trace: P,
    pub auxiliary: Option<jolt_openings::PackedOpeningProof<F, P>>,
}

#[cfg(feature = "akita")]
pub type JointOpeningProof<PCS> =
    AkitaJointOpeningProof<<PCS as CommitmentScheme>::Field, <PCS as CommitmentScheme>::Proof>;

#[expect(
    non_snake_case,
    reason = "Matches current jolt-prover-legacy proof field name."
)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "PCS::Field: Serialize, ZkProof: Serialize",
    deserialize = "PCS::Field: for<'a> Deserialize<'a>, ZkProof: serde::de::DeserializeOwned"
))]
pub struct JoltProof<
    PCS,
    VC,
    ZkProof = BlindFoldProof<<PCS as CommitmentScheme>::Field, <VC as Commitment>::Output>,
> where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub protocol: JoltProtocolConfig,
    pub commitments: ProofCommitments<PCS>,
    pub stages: JoltStageProofs<PCS::Field, VC>,
    pub joint_opening_proof: JointOpeningProof<PCS>,
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
    pub(crate) fn clear_claims(&self) -> Result<&ClearProofClaims<PCS::Field>, VerifierError> {
        match &self.claims {
            JoltProofClaims::Clear(claims) => Ok(claims),
            JoltProofClaims::Zk { .. } => Err(VerifierError::UnexpectedBlindFoldProof),
        }
    }

    #[cfg(not(feature = "akita"))]
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
    pub instruction_ra: Vec<C>,
    pub ram_ra: Vec<C>,
    pub bytecode_ra: Vec<C>,
}

impl<C> JoltCommitments<C> {
    pub fn new(
        rd_inc: C,
        ram_inc: C,
        instruction_ra: Vec<C>,
        ram_ra: Vec<C>,
        bytecode_ra: Vec<C>,
    ) -> Self {
        Self {
            rd_inc,
            ram_inc,
            instruction_ra,
            ram_ra,
            bytecode_ra,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[expect(
    clippy::large_enum_variant,
    reason = "Clear claims are the verifier-owned standard proof payload; keeping them inline avoids heap indirection in the common clear path."
)]
#[serde(bound(
    serialize = "F: Serialize, ZkProof: Serialize",
    deserialize = "F: for<'a> Deserialize<'a>, ZkProof: serde::de::DeserializeOwned"
))]
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
#[serde(bound(serialize = "F: Serialize", deserialize = "F: for<'a> Deserialize<'a>"))]
pub struct ClearProofClaims<F: Field> {
    pub stage1: stage1::outputs::Stage1OutputClaims<F>,
    pub stage2: stage2::outputs::Stage2OutputClaims<F>,
    pub stage3: stage3::outputs::Stage3OutputClaims<F>,
    pub stage4: stage4::outputs::Stage4OutputClaims<F>,
    pub stage5: stage5::outputs::Stage5OutputClaims<F>,
    pub stage6a: stage6a::outputs::Stage6aOutputClaims<F>,
    pub stage6b: stage6b::outputs::Stage6bOutputClaims<F>,
    pub stage7: stage7::outputs::Stage7OutputClaims<F>,
    /// The reconstruction phase's claims, at the head of the stage-8 region;
    /// cells are present exactly when advice or a committed program is.
    #[cfg(feature = "akita")]
    pub reconstruction: crate::stages::stage8::reconstruction::ReconstructionOutputClaims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, <VC as Commitment>::Output: Serialize",
    deserialize = "F: for<'a> Deserialize<'a>, <VC as Commitment>::Output: serde::de::DeserializeOwned"
))]
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
    /// The reconstruction phase, at the head of the stage-8 region; present
    /// exactly when advice or a committed program is.
    #[cfg(feature = "akita")]
    pub reconstruction_sumcheck_proof: Option<SumcheckProof<F, VC::Output>>,
}
