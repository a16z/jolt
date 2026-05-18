//! Verifier-owned proof model types.

use jolt_blindfold::BlindFoldProof;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltOpeningId, JoltReadWriteConfig};
use jolt_crypto::{Commitment, VectorCommitment};
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::SumcheckProof;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[expect(non_snake_case, reason = "Matches current jolt-core proof field name.")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "ZkProof: Serialize + DeserializeOwned")]
pub struct JoltProof<
    PCS,
    VC,
    ZkProof = BlindFoldProof<<PCS as CommitmentScheme>::Field, <VC as Commitment>::Output>,
> where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub commitments: Vec<PCS::Output>,
    pub stages: JoltStageProofs<PCS::Field, VC>,
    pub joint_opening_proof: PCS::Proof,
    pub untrusted_advice_commitment: Option<PCS::Output>,
    pub(crate) opening_claims: Option<Vec<(JoltOpeningId, PCS::Field)>>,
    pub blindfold_proof: Option<ZkProof>,
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
        commitments: Vec<PCS::Output>,
        stages: JoltStageProofs<PCS::Field, VC>,
        joint_opening_proof: PCS::Proof,
        untrusted_advice_commitment: Option<PCS::Output>,
        blindfold_proof: Option<ZkProof>,
        trace_length: usize,
        ram_k: usize,
        rw_config: JoltReadWriteConfig,
        one_hot_config: JoltOneHotConfig,
        trace_polynomial_order: TracePolynomialOrder,
    ) -> Self {
        Self {
            commitments,
            stages,
            joint_opening_proof,
            untrusted_advice_commitment,
            opening_claims: None,
            blindfold_proof,
            trace_length,
            ram_K: ram_k,
            rw_config,
            one_hot_config,
            trace_polynomial_order,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TracePolynomialOrder {
    #[default]
    CycleMajor,
    AddressMajor,
}

impl TracePolynomialOrder {
    pub fn address_cycle_to_index(
        self,
        address: usize,
        cycle: usize,
        num_addresses: usize,
        num_cycles: usize,
    ) -> usize {
        match self {
            Self::CycleMajor => address * num_cycles + cycle,
            Self::AddressMajor => cycle * num_addresses + address,
        }
    }

    pub fn index_to_address_cycle(
        self,
        index: usize,
        num_addresses: usize,
        num_cycles: usize,
    ) -> (usize, usize) {
        match self {
            Self::CycleMajor => (index / num_cycles, index % num_cycles),
            Self::AddressMajor => (index % num_addresses, index / num_addresses),
        }
    }
}

impl From<TracePolynomialOrder> for u8 {
    fn from(order: TracePolynomialOrder) -> Self {
        match order {
            TracePolynomialOrder::CycleMajor => 0,
            TracePolynomialOrder::AddressMajor => 1,
        }
    }
}

impl TryFrom<u8> for TracePolynomialOrder {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::CycleMajor),
            1 => Ok(Self::AddressMajor),
            _ => Err(()),
        }
    }
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
