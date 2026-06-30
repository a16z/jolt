//! Verifier-owned proof model types.

pub use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use jolt_openings::CommitmentScheme;
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
    /// witness commitments, optional untrusted-advice commitment, prover-only
    /// sumcheck/uni-skip round payloads, and BlindFold payloads. Dory's joint
    /// opening proof and non-ZK opening claims remain structural because
    /// spongefish has no non-absorbing hint channel for those values.
    pub narg: Vec<u8>,
    pub joint_opening_proof: PCS::Proof,
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
        narg: Vec<u8>,
        joint_opening_proof: PCS::Proof,
        claims: JoltProofClaims<PCS::Field>,
        trace_length: usize,
        ram_k: usize,
        rw_config: JoltReadWriteConfig,
        one_hot_config: JoltOneHotConfig,
        trace_polynomial_order: TracePolynomialOrder,
    ) -> Self {
        let protocol = JoltProtocolConfig::for_zk(claims.is_zk());
        Self {
            protocol,
            narg,
            joint_opening_proof,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NargProofCommitments<C> {
    pub commitments: JoltCommitments<C>,
    pub untrusted_advice_commitment: Option<C>,
}

impl<C> NargProofCommitments<C> {
    pub fn new(commitments: JoltCommitments<C>, untrusted_advice_commitment: Option<C>) -> Self {
        Self {
            commitments,
            untrusted_advice_commitment,
        }
    }

    pub const fn untrusted_advice_commitment_present(&self) -> bool {
        self.untrusted_advice_commitment.is_some()
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

pub(crate) fn proof_commitment_counts(
    one_hot_config: JoltOneHotConfig,
    ram_k: usize,
) -> Result<(usize, usize), VerifierError> {
    let committed_chunk_bits = one_hot_config.committed_chunk_bits();
    if committed_chunk_bits == 0 {
        return Err(VerifierError::InvalidCommitmentCount {
            expected: 2,
            got: 0,
        });
    }
    let instruction_ra_count = (2 * RISCV_XLEN).div_ceil(committed_chunk_bits);
    let ram_ra_count = ceil_log_2(ram_k).div_ceil(committed_chunk_bits);
    Ok((instruction_ra_count, ram_ra_count))
}

pub(crate) fn commitments_from_proof_payload_order<C>(
    commitments: Vec<C>,
    instruction_ra_count: usize,
    ram_ra_count: usize,
) -> Result<JoltCommitments<C>, VerifierError> {
    let minimum = 2 + instruction_ra_count + ram_ra_count;
    if commitments.len() < minimum {
        return Err(VerifierError::InvalidCommitmentCount {
            expected: minimum,
            got: commitments.len(),
        });
    }

    let mut commitments = commitments.into_iter();
    let Some(rd_inc) = commitments.next() else {
        return Err(VerifierError::InvalidCommitmentCount {
            expected: minimum,
            got: 0,
        });
    };
    let Some(ram_inc) = commitments.next() else {
        return Err(VerifierError::InvalidCommitmentCount {
            expected: minimum,
            got: 1,
        });
    };
    let instruction_ra = commitments
        .by_ref()
        .take(instruction_ra_count)
        .collect::<Vec<_>>();
    let ram_ra = commitments.by_ref().take(ram_ra_count).collect::<Vec<_>>();
    let bytecode_ra = commitments.collect::<Vec<_>>();

    Ok(JoltCommitments::new(
        rd_inc,
        ram_inc,
        JoltRaCommitments::new(instruction_ra, ram_ra, bytecode_ra),
    ))
}

fn ceil_log_2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}
