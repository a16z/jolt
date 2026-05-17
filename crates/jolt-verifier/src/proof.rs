//! Verifier-owned proof model types.

use crate::compat::config::{OneHotConfig, ReadWriteConfig};

#[expect(non_snake_case, reason = "Matches current jolt-core proof field name.")]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct JoltProof<
    Commitment,
    UniSkipProof,
    SumcheckProof,
    OpeningProof,
    OpeningClaims,
    BlindFoldProof = (),
> {
    pub commitments: Vec<Commitment>,
    pub stages: JoltStageProofs<UniSkipProof, SumcheckProof>,
    pub joint_opening_proof: OpeningProof,
    pub untrusted_advice_commitment: Option<Commitment>,
    pub payload: ProofPayload<OpeningClaims, BlindFoldProof>,
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: ReadWriteConfig,
    pub one_hot_config: OneHotConfig,
    pub dory_layout: DoryLayout,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct JoltStageProofs<UniSkipProof, SumcheckProof> {
    pub stage1_uni_skip_first_round_proof: UniSkipProof,
    pub stage1_sumcheck_proof: SumcheckProof,
    pub stage2_uni_skip_first_round_proof: UniSkipProof,
    pub stage2_sumcheck_proof: SumcheckProof,
    pub stage3_sumcheck_proof: SumcheckProof,
    pub stage4_sumcheck_proof: SumcheckProof,
    pub stage5_sumcheck_proof: SumcheckProof,
    pub stage6_sumcheck_proof: SumcheckProof,
    pub stage7_sumcheck_proof: SumcheckProof,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ProofPayload<OpeningClaims, BlindFoldProof> {
    Clear { opening_claims: OpeningClaims },
    Zk { blindfold_proof: BlindFoldProof },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DoryLayout {
    #[default]
    CycleMajor,
    AddressMajor,
}

impl DoryLayout {
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

impl From<DoryLayout> for u8 {
    fn from(layout: DoryLayout) -> Self {
        match layout {
            DoryLayout::CycleMajor => 0,
            DoryLayout::AddressMajor => 1,
        }
    }
}

impl TryFrom<u8> for DoryLayout {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::CycleMajor),
            1 => Ok(Self::AddressMajor),
            _ => Err(()),
        }
    }
}
