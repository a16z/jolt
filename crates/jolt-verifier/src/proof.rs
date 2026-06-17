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
    config::{validate_protocol_config, JoltProtocolConfig, PcsFamily},
    stages::{stage1, stage2, stage3, stage4, stage5, stage6, stage7},
    VerifierError,
};

#[expect(non_snake_case, reason = "Matches current jolt-core proof field name.")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "PCS::Field: Serialize, ZkProof: Serialize",
    deserialize = "PCS::Field: serde::de::DeserializeOwned, ZkProof: serde::de::DeserializeOwned"
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
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineCommitments<C>,
}

pub type DoryCommitmentPayload<C> = JoltCommitments<C>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentPayload<C> {
    Dory(DoryCommitmentPayload<C>),
    Akita(AkitaCommitmentPayload<C>),
}

impl<C> CommitmentPayload<C> {
    pub const fn family(&self) -> PcsFamily {
        match self {
            Self::Dory(_) => PcsFamily::Curve,
            Self::Akita(_) => PcsFamily::Lattice,
        }
    }

    pub fn as_dory(&self) -> Option<&DoryCommitmentPayload<C>> {
        match self {
            Self::Dory(payload) => Some(payload),
            Self::Akita(_) => None,
        }
    }

    pub fn as_akita(&self) -> Option<&AkitaCommitmentPayload<C>> {
        match self {
            Self::Dory(_) => None,
            Self::Akita(payload) => Some(payload),
        }
    }
}

impl<C> From<DoryCommitmentPayload<C>> for CommitmentPayload<C> {
    fn from(payload: DoryCommitmentPayload<C>) -> Self {
        Self::Dory(payload)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaCommitmentPayload<C> {
    pub packed_witness: C,
    pub layout_digest: [u8; 32],
    pub d_pack: usize,
}

impl<C> AkitaCommitmentPayload<C> {
    pub fn new(packed_witness: C, layout_digest: [u8; 32], d_pack: usize) -> Self {
        Self {
            packed_witness,
            layout_digest,
            d_pack,
        }
    }
}

pub fn validate_commitment_payload_family<C>(
    config: &JoltProtocolConfig,
    payload: &CommitmentPayload<C>,
) -> Result<(), VerifierError> {
    let expected = validate_protocol_config(config)?;
    let got = payload.family();
    if expected != got {
        return Err(VerifierError::CommitmentPayloadFamilyMismatch { expected, got });
    }
    Ok(())
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

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineCommitments<C> {
    pub field_registers: FieldRegistersCommitments<C>,
}

#[cfg(feature = "field-inline")]
impl<C> FieldInlineCommitments<C> {
    pub fn new(field_registers: FieldRegistersCommitments<C>) -> Self {
        Self { field_registers }
    }
}

#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldRegistersCommitments<C> {
    pub rd_inc: C,
}

#[cfg(feature = "field-inline")]
impl<C> FieldRegistersCommitments<C> {
    pub fn new(rd_inc: C) -> Self {
        Self { rd_inc }
    }
}

impl<C> JoltCommitments<C> {
    #[cfg(not(feature = "field-inline"))]
    pub fn new(rd_inc: C, ram_inc: C, ra: JoltRaCommitments<C>) -> Self {
        Self {
            rd_inc,
            ram_inc,
            ra,
        }
    }

    #[cfg(feature = "field-inline")]
    pub fn new(
        rd_inc: C,
        ram_inc: C,
        ra: JoltRaCommitments<C>,
        field_inline: FieldInlineCommitments<C>,
    ) -> Self {
        Self {
            rd_inc,
            ram_inc,
            ra,
            field_inline,
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
    deserialize = "F: serde::de::DeserializeOwned, ZkProof: serde::de::DeserializeOwned"
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
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct ClearProofClaims<F: Field> {
    pub stage1: stage1::inputs::Stage1Claims<F>,
    pub stage2: stage2::inputs::Stage2Claims<F>,
    pub stage3: stage3::inputs::Stage3Claims<F>,
    pub stage4: stage4::inputs::Stage4Claims<F>,
    pub stage5: stage5::inputs::Stage5Claims<F>,
    pub stage6: stage6::inputs::Stage6Claims<F>,
    pub stage7: stage7::inputs::Stage7Claims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize",
    deserialize = "F: serde::de::DeserializeOwned"
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        IncrementCommitmentMode, PackedWitnessConfig, PcsFamilyFlags, ProgramMode,
    };

    fn dory_payload() -> CommitmentPayload<u64> {
        CommitmentPayload::Dory(JoltCommitments::new(
            1,
            2,
            JoltRaCommitments::new(vec![3], vec![4], vec![5]),
        ))
    }

    fn akita_payload() -> CommitmentPayload<u64> {
        CommitmentPayload::Akita(AkitaCommitmentPayload::new(9, [7; 32], 43))
    }

    fn lattice_config() -> JoltProtocolConfig {
        let mut config = JoltProtocolConfig::for_zk(false);
        config.pcs = PcsFamilyFlags {
            curve: false,
            lattice: true,
        };
        config.lattice.program_mode = ProgramMode::Committed;
        config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
        config.lattice.packed_witness = PackedWitnessConfig {
            layout_digest: Some([7; 32]),
            d_pack: Some(43),
            field_rd_inc_family: false,
            trusted_advice_family: false,
            untrusted_advice_family: false,
        };
        config
    }

    #[test]
    fn commitment_payload_family_tracks_variant() {
        assert_eq!(dory_payload().family(), PcsFamily::Curve);
        assert_eq!(akita_payload().family(), PcsFamily::Lattice);
    }

    #[test]
    fn commitment_payload_validates_against_selected_pcs_family() {
        let curve = JoltProtocolConfig::for_zk(false);
        let lattice = lattice_config();

        assert!(validate_commitment_payload_family(&curve, &dory_payload()).is_ok());
        assert!(validate_commitment_payload_family(&lattice, &akita_payload()).is_ok());
    }

    #[test]
    fn commitment_payload_mode_mismatch_rejects() {
        let curve = JoltProtocolConfig::for_zk(false);
        let lattice = lattice_config();

        assert!(matches!(
            validate_commitment_payload_family(&curve, &akita_payload()),
            Err(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Curve,
                got: PcsFamily::Lattice,
            })
        ));
        assert!(matches!(
            validate_commitment_payload_family(&lattice, &dory_payload()),
            Err(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Lattice,
                got: PcsFamily::Curve,
            })
        ));
    }
}
