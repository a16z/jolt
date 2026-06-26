//! Verifier-owned proof model types.

use ark_serialize::CanonicalSerialize;
pub use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_transcript::{serialize_slice, BytesMsg, Encoding};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

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
#[serde(bound = "")]
pub struct JoltProof<PCS, VC, ZkProof = ()>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub protocol: JoltProtocolConfig,
    /// Spongefish NARG frames consumed by the modular verifier. This carries
    /// witness commitments, prover-only sumcheck/uni-skip round payloads, and
    /// BlindFold payloads. Dory's joint opening proof and non-ZK opening
    /// claims remain structural because spongefish has no non-absorbing hint
    /// channel for those values.
    pub narg: Vec<u8>,
    pub commitments: JoltCommitments<PCS::Output>,
    pub joint_opening_proof: PCS::Proof,
    pub untrusted_advice_commitment: Option<PCS::Output>,
    pub claims: JoltProofClaims<PCS::Field>,
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    pub trace_polynomial_order: TracePolynomialOrder,
    #[serde(skip)]
    _proof_marker: PhantomData<fn() -> (VC, ZkProof)>,
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
        joint_opening_proof: PCS::Proof,
        untrusted_advice_commitment: Option<PCS::Output>,
        claims: JoltProofClaims<PCS::Field>,
        trace_length: usize,
        ram_k: usize,
        rw_config: JoltReadWriteConfig,
        one_hot_config: JoltOneHotConfig,
        trace_polynomial_order: TracePolynomialOrder,
    ) -> Self
    where
        PCS::Output: CanonicalSerialize + Clone,
    {
        let protocol = JoltProtocolConfig::for_zk(claims.is_zk());
        let narg = verifier_narg_prefix(&commitments, untrusted_advice_commitment.as_ref());
        Self {
            protocol,
            narg,
            commitments,
            joint_opening_proof,
            untrusted_advice_commitment,
            claims,
            trace_length,
            ram_K: ram_k,
            rw_config,
            one_hot_config,
            trace_polynomial_order,
            _proof_marker: PhantomData,
        }
    }

    pub(crate) fn clear_claims(&self) -> Result<&ClearProofClaims<PCS::Field>, VerifierError> {
        match &self.claims {
            JoltProofClaims::Clear(claims) => Ok(claims),
            JoltProofClaims::Zk => Err(VerifierError::UnexpectedBlindFoldProof),
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
#[serde(bound = "")]
pub enum JoltProofClaims<F>
where
    F: Field,
{
    Clear(ClearProofClaims<F>),
    Zk,
}

impl<F> JoltProofClaims<F>
where
    F: Field,
{
    pub const fn is_zk(&self) -> bool {
        matches!(self, Self::Zk)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ClearProofClaims<F: Field> {
    pub stage1: stage1::outputs::Stage1OutputClaims<F>,
    pub stage2: stage2::outputs::Stage2OutputClaims<F>,
    pub stage3: stage3::outputs::Stage3OutputClaims<F>,
    pub stage4: stage4::outputs::Stage4OutputClaims<F>,
    pub stage5: stage5::outputs::Stage5OutputClaims<F>,
    pub stage6: stage6::outputs::Stage6OutputClaims<F>,
    pub stage7: stage7::outputs::Stage7OutputClaims<F>,
}

fn append_narg_frame<T: CanonicalSerialize>(narg: &mut Vec<u8>, values: &[T]) {
    let frame = BytesMsg(serialize_slice(values));
    narg.extend_from_slice(frame.encode().as_ref());
}

pub(crate) fn verifier_narg_prefix<C>(
    commitments: &JoltCommitments<C>,
    untrusted_advice: Option<&C>,
) -> Vec<u8>
where
    C: CanonicalSerialize + Clone,
{
    let mut narg = Vec::new();
    append_narg_frame(&mut narg, &proof_commitments_payload_order(commitments));
    match untrusted_advice {
        Some(commitment) => append_narg_frame(&mut narg, std::slice::from_ref(commitment)),
        None => append_narg_frame::<C>(&mut narg, &[]),
    }
    narg
}

pub(crate) fn proof_commitments_payload_order<C: Clone>(
    commitments: &JoltCommitments<C>,
) -> Vec<C> {
    let mut ordered = Vec::with_capacity(
        2 + commitments.ra.instruction.len()
            + commitments.ra.ram.len()
            + commitments.ra.bytecode.len(),
    );
    ordered.push(commitments.rd_inc.clone());
    ordered.push(commitments.ram_inc.clone());
    ordered.extend(commitments.ra.instruction.iter().cloned());
    ordered.extend(commitments.ra.ram.iter().cloned());
    ordered.extend(commitments.ra.bytecode.iter().cloned());
    ordered
}
