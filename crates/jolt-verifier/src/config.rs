//! Verifier-selected protocol configuration.

use common::constants::MAX_BLINDFOLD_GENERATORS;
use jolt_claims::protocols::field_inline::FieldInlineConfig;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use serde::{Deserialize, Serialize};

use crate::{
    pcs_assist::{NoPcsAssistConfig, PcsProofAssist},
    proof::JoltProof,
    VerifierError,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkConfig {
    Transparent,
    BlindFold,
}

impl ZkConfig {
    pub const fn from_bool(zk: bool) -> Self {
        if zk {
            Self::BlindFold
        } else {
            Self::Transparent
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProtocolConfig<PcsAssistConfig = NoPcsAssistConfig> {
    pub zk: ZkConfig,
    pub field_inline: FieldInlineConfig,
    pub pcs_assist: Option<PcsAssistConfig>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltProtocolConfigSummary {
    pub zk: ZkConfig,
    pub field_inline: FieldInlineConfig,
    pub pcs_assist_enabled: bool,
}

impl JoltProtocolConfig<NoPcsAssistConfig> {
    pub const fn for_zk(zk: bool) -> Self {
        Self {
            zk: ZkConfig::from_bool(zk),
            field_inline: SELECTED_FIELD_INLINE_CONFIG,
            pcs_assist: None,
        }
    }
}

impl<PcsAssistConfig> JoltProtocolConfig<PcsAssistConfig> {
    pub fn selected_for_zk<PCS, PcsAssist>(zk: bool) -> Self
    where
        PCS: jolt_openings::CommitmentScheme,
        PcsAssist: PcsProofAssist<PCS, Config = PcsAssistConfig>,
    {
        Self {
            zk: ZkConfig::from_bool(zk),
            field_inline: SELECTED_FIELD_INLINE_CONFIG,
            pcs_assist: selected_pcs_assist_config::<PCS, PcsAssist>(),
        }
    }

    pub const fn summary(&self) -> JoltProtocolConfigSummary {
        JoltProtocolConfigSummary {
            zk: self.zk,
            field_inline: self.field_inline,
            pcs_assist_enabled: self.pcs_assist.is_some(),
        }
    }
}

#[cfg(feature = "pcs-assist")]
fn selected_pcs_assist_config<PCS, PcsAssist>() -> Option<PcsAssist::Config>
where
    PCS: jolt_openings::CommitmentScheme,
    PcsAssist: PcsProofAssist<PCS>,
{
    Some(PcsAssist::selected_config())
}

#[cfg(not(feature = "pcs-assist"))]
fn selected_pcs_assist_config<PCS, PcsAssist>() -> Option<PcsAssist::Config>
where
    PCS: jolt_openings::CommitmentScheme,
    PcsAssist: PcsProofAssist<PCS>,
{
    None
}

#[cfg(feature = "field-inline")]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::native_v1();

#[cfg(not(feature = "field-inline"))]
pub const SELECTED_FIELD_INLINE_CONFIG: FieldInlineConfig = FieldInlineConfig::disabled();

#[cfg(feature = "zk")]
pub const SELECTED_ZK_CONFIG: ZkConfig = ZkConfig::BlindFold;

#[cfg(not(feature = "zk"))]
pub const SELECTED_ZK_CONFIG: ZkConfig = ZkConfig::Transparent;

pub const JOLT_VERIFIER_CONFIG: JoltProtocolConfig = JoltProtocolConfig {
    zk: SELECTED_ZK_CONFIG,
    field_inline: SELECTED_FIELD_INLINE_CONFIG,
    pcs_assist: None,
};

pub const fn zk_vector_commitment_capacity_requirement() -> usize {
    let outer_uniskip_capacity = SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE + 1;
    let product_uniskip_capacity = SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE + 1;
    let committed_round_capacity = if outer_uniskip_capacity > product_uniskip_capacity {
        outer_uniskip_capacity
    } else {
        product_uniskip_capacity
    };

    if MAX_BLINDFOLD_GENERATORS > committed_round_capacity {
        next_power_of_two(MAX_BLINDFOLD_GENERATORS)
    } else {
        next_power_of_two(committed_round_capacity)
    }
}

const fn next_power_of_two(value: usize) -> usize {
    let mut power = 1;
    while power < value {
        power *= 2;
    }
    power
}

pub fn validate_proof_config<PCS, VC, ZkProof, PcsAssist>(
    config: &JoltProtocolConfig<PcsAssist::Config>,
    proof: &JoltProof<PCS, VC, ZkProof, PcsAssist>,
) -> Result<(), VerifierError>
where
    PCS: jolt_openings::CommitmentScheme,
    VC: jolt_crypto::VectorCommitment<Field = PCS::Field>,
    PcsAssist: PcsProofAssist<PCS>,
{
    if &proof.protocol != config {
        return Err(VerifierError::ProtocolConfigMismatch {
            expected: config.summary(),
            got: proof.protocol.summary(),
        });
    }

    #[cfg(feature = "pcs-assist")]
    {
        if proof.pcs_assist.is_none() {
            return Err(VerifierError::MissingPcsAssistProof);
        }
    }

    #[cfg(not(feature = "pcs-assist"))]
    {
        if proof.pcs_assist.is_some() {
            return Err(VerifierError::UnexpectedPcsAssistProof);
        }
    }

    Ok(())
}
