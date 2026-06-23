//! Verifier-owned proof model types.

use std::marker::PhantomData;

use jolt_blindfold::BlindFoldProof;
pub use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
use jolt_crypto::{Commitment, VectorCommitment};
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::{AppendToTranscript, Label, Transcript};
use serde::{Deserialize, Serialize};

use crate::{
    config::{validate_protocol_config, JoltProtocolConfig, PcsFamily},
    stages::{stage1, stage2, stage3, stage4, stage5, stage5_increment, stage6, stage7},
    VerifierError,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClearOnlyCommitment;

impl AppendToTranscript for ClearOnlyCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"clear_only_commitment"));
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClearOnlyVectorCommitment<F: Field>(PhantomData<F>);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClearOnlyVectorCommitmentSetup;

impl<F: Field> Commitment for ClearOnlyVectorCommitment<F> {
    type Output = ClearOnlyCommitment;
}

impl<F: Field> VectorCommitment for ClearOnlyVectorCommitment<F> {
    type Field = F;
    type Setup = ClearOnlyVectorCommitmentSetup;

    fn capacity(_setup: &Self::Setup) -> usize {
        0
    }

    fn commit(
        _setup: &Self::Setup,
        values: &[Self::Field],
        _blinding: &Self::Field,
    ) -> Self::Output {
        debug_assert!(values.is_empty());
        ClearOnlyCommitment
    }

    fn verify(
        _setup: &Self::Setup,
        _commitment: &Self::Output,
        values: &[Self::Field],
        _blinding: &Self::Field,
    ) -> bool {
        values.is_empty()
    }
}

#[expect(non_snake_case, reason = "Matches current jolt-core proof field name.")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    pub commitments: CommitmentPayload<PCS::Output>,
    pub stages: JoltStageProofs<PCS::Field, VC>,
    pub joint_opening_proof: PCS::Proof,
    #[serde(default)]
    pub lattice_packed_validity_opening_proof: Option<PCS::Proof>,
    #[serde(default)]
    pub lattice_precommitted_opening_proofs: Vec<PCS::Proof>,
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
        Self::new_with_payload(
            CommitmentPayload::Dory(commitments),
            stages,
            joint_opening_proof,
            untrusted_advice_commitment,
            claims,
            trace_length,
            ram_k,
            rw_config,
            one_hot_config,
            trace_polynomial_order,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Constructor mirrors the proof payload while keeping internal verifier claims private."
    )]
    pub fn new_with_payload(
        commitments: CommitmentPayload<PCS::Output>,
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
            lattice_packed_validity_opening_proof: None,
            lattice_precommitted_opening_proofs: Vec::new(),
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
#[serde(deny_unknown_fields)]
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
    Lattice(LatticeCommitmentPayload<C>),
}

impl<C> CommitmentPayload<C> {
    pub const fn family(&self) -> PcsFamily {
        match self {
            Self::Dory(_) => PcsFamily::Curve,
            Self::Lattice(_) => PcsFamily::Lattice,
        }
    }

    pub fn as_dory(&self) -> Option<&DoryCommitmentPayload<C>> {
        match self {
            Self::Dory(payload) => Some(payload),
            Self::Lattice(_) => None,
        }
    }

    pub fn as_lattice(&self) -> Option<&LatticeCommitmentPayload<C>> {
        match self {
            Self::Dory(_) => None,
            Self::Lattice(payload) => Some(payload),
        }
    }
}

impl<C> From<DoryCommitmentPayload<C>> for CommitmentPayload<C> {
    fn from(payload: DoryCommitmentPayload<C>) -> Self {
        Self::Dory(payload)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatticeCommitmentPayload<C> {
    pub packed_witness: C,
    pub layout_digest: [u8; 32],
    pub d_pack: usize,
}

impl<C> LatticeCommitmentPayload<C> {
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
    validate_commitment_payload_config(config, payload)
}

pub fn validate_commitment_payload_config<C>(
    config: &JoltProtocolConfig,
    payload: &CommitmentPayload<C>,
) -> Result<(), VerifierError> {
    let expected = validate_protocol_config(config)?;
    let got = payload.family();
    if expected != got {
        return Err(VerifierError::CommitmentPayloadFamilyMismatch { expected, got });
    }
    if let CommitmentPayload::Lattice(payload) = payload {
        validate_lattice_commitment_payload_config(config, payload)?;
    }
    Ok(())
}

pub fn validate_lattice_commitment_payload_config<C>(
    config: &JoltProtocolConfig,
    payload: &LatticeCommitmentPayload<C>,
) -> Result<(), VerifierError> {
    let expected = validate_protocol_config(config)?;
    if expected != PcsFamily::Lattice {
        return Err(VerifierError::CommitmentPayloadFamilyMismatch {
            expected,
            got: PcsFamily::Lattice,
        });
    }

    let Some(expected_digest) = config.lattice.packed_witness.layout_digest else {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice PCS mode requires a packed witness layout digest".to_owned(),
        });
    };
    if payload.layout_digest != expected_digest {
        return Err(VerifierError::LatticePayloadLayoutDigestMismatch {
            expected: expected_digest,
            got: payload.layout_digest,
        });
    }

    let Some(expected_d_pack) = config.lattice.packed_witness.d_pack else {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "lattice PCS mode requires D_pack".to_owned(),
        });
    };
    if payload.d_pack != expected_d_pack {
        return Err(VerifierError::LatticePayloadDimensionMismatch {
            expected: expected_d_pack,
            got: payload.d_pack,
        });
    }

    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct ClearProofClaims<F: Field> {
    pub stage1: stage1::inputs::Stage1Claims<F>,
    pub stage2: stage2::inputs::Stage2Claims<F>,
    pub stage3: stage3::inputs::Stage3Claims<F>,
    pub stage4: stage4::inputs::Stage4Claims<F>,
    pub stage5: stage5::inputs::Stage5Claims<F>,
    #[serde(default)]
    pub stage5_increment: Option<stage5_increment::Stage5IncrementClaims<F>>,
    pub stage6: stage6::inputs::Stage6Claims<F>,
    pub stage7: stage7::inputs::Stage7Claims<F>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    #[serde(default)]
    pub stage5_increment_sumcheck_proof: Option<SumcheckProof<F, VC::Output>>,
    pub stage6a_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub stage6b_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub stage7_sumcheck_proof: SumcheckProof<F, VC::Output>,
    #[serde(default)]
    pub lattice_packed_validity_sumcheck_proof: Option<SumcheckProof<F, VC::Output>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{IncrementCommitmentMode, PackedWitnessConfig, ProgramMode};

    fn dory_payload() -> CommitmentPayload<u64> {
        CommitmentPayload::Dory(JoltCommitments::new(
            1,
            2,
            JoltRaCommitments::new(vec![3], vec![4], vec![5]),
            #[cfg(feature = "field-inline")]
            FieldInlineCommitments::new(FieldRegistersCommitments::new(6)),
        ))
    }

    fn lattice_payload() -> CommitmentPayload<u64> {
        CommitmentPayload::Lattice(LatticeCommitmentPayload::new(9, [7; 32], 43))
    }

    fn lattice_config() -> JoltProtocolConfig {
        let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
        config.lattice.program_mode = ProgramMode::Committed;
        config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
        config.lattice.packed_witness = PackedWitnessConfig {
            layout_digest: Some([7; 32]),
            d_pack: Some(43),
            validity_digest: Some([11; 32]),
        };
        #[cfg(feature = "field-inline")]
        {
            config.lattice.field_inline.enabled = true;
        }
        config
    }

    #[test]
    fn commitment_payload_family_tracks_variant() {
        assert_eq!(dory_payload().family(), PcsFamily::Curve);
        assert_eq!(lattice_payload().family(), PcsFamily::Lattice);
    }

    #[test]
    #[expect(
        clippy::panic,
        reason = "test serialization failures should fail loudly"
    )]
    fn commitment_payload_serialization_is_variant_tagged() {
        let dory_value = serde_json::to_value(dory_payload())
            .unwrap_or_else(|error| panic!("Dory payload should serialize: {error}"));
        let lattice_value = serde_json::to_value(lattice_payload())
            .unwrap_or_else(|error| panic!("Lattice payload should serialize: {error}"));

        assert!(dory_value.get("Dory").is_some());
        assert!(dory_value.get("Lattice").is_none());
        assert!(lattice_value.get("Lattice").is_some());
        assert!(lattice_value.get("Dory").is_none());

        let dory_roundtrip: CommitmentPayload<u64> = serde_json::from_value(dory_value)
            .unwrap_or_else(|error| panic!("Dory payload should deserialize: {error}"));
        let lattice_roundtrip: CommitmentPayload<u64> = serde_json::from_value(lattice_value)
            .unwrap_or_else(|error| panic!("Lattice payload should deserialize: {error}"));

        assert!(matches!(dory_roundtrip, CommitmentPayload::Dory(_)));
        assert!(matches!(lattice_roundtrip, CommitmentPayload::Lattice(_)));
    }

    #[test]
    #[expect(
        clippy::panic,
        reason = "test serialization failures should fail loudly"
    )]
    fn dory_payload_rejects_extra_commitment_fields() {
        let mut value = serde_json::to_value(dory_payload())
            .unwrap_or_else(|error| panic!("Dory payload should serialize: {error}"));
        let dory = value
            .get_mut("Dory")
            .and_then(|payload| payload.as_object_mut())
            .unwrap_or_else(|| panic!("Dory payload should be an object"));
        let previous = dory.insert(
            "extra_commitment".to_string(),
            serde_json::Value::from(10_u64),
        );
        assert!(previous.is_none());

        assert!(
            serde_json::from_value::<CommitmentPayload<u64>>(value).is_err(),
            "Dory payloads must reject extra commitment fields"
        );

        let mut value = serde_json::to_value(dory_payload())
            .unwrap_or_else(|error| panic!("Dory payload should serialize: {error}"));
        let ra = value
            .get_mut("Dory")
            .and_then(|payload| payload.as_object_mut())
            .and_then(|payload| payload.get_mut("ra"))
            .and_then(|ra| ra.as_object_mut())
            .unwrap_or_else(|| panic!("Dory RA payload should be an object"));
        let previous = ra.insert("extra_ra".to_string(), serde_json::Value::from(11_u64));
        assert!(previous.is_none());

        assert!(
            serde_json::from_value::<CommitmentPayload<u64>>(value).is_err(),
            "Dory RA payloads must reject extra commitment fields"
        );

        #[cfg(feature = "field-inline")]
        {
            let mut value = serde_json::to_value(dory_payload())
                .unwrap_or_else(|error| panic!("Dory payload should serialize: {error}"));
            let field_inline = value
                .get_mut("Dory")
                .and_then(|payload| payload.as_object_mut())
                .and_then(|payload| payload.get_mut("field_inline"))
                .and_then(|field_inline| field_inline.as_object_mut())
                .unwrap_or_else(|| panic!("Dory field-inline payload should be an object"));
            let previous = field_inline.insert(
                "extra_field_inline".to_string(),
                serde_json::Value::from(12_u64),
            );
            assert!(previous.is_none());

            assert!(
                serde_json::from_value::<CommitmentPayload<u64>>(value).is_err(),
                "Dory field-inline payloads must reject extra commitment fields"
            );

            let mut value = serde_json::to_value(dory_payload())
                .unwrap_or_else(|error| panic!("Dory payload should serialize: {error}"));
            let field_registers = value
                .get_mut("Dory")
                .and_then(|payload| payload.as_object_mut())
                .and_then(|payload| payload.get_mut("field_inline"))
                .and_then(|field_inline| field_inline.as_object_mut())
                .and_then(|field_inline| field_inline.get_mut("field_registers"))
                .and_then(|field_registers| field_registers.as_object_mut())
                .unwrap_or_else(|| panic!("Dory field-register payload should be an object"));
            let previous = field_registers.insert(
                "extra_field_register".to_string(),
                serde_json::Value::from(13_u64),
            );
            assert!(previous.is_none());

            assert!(
                serde_json::from_value::<CommitmentPayload<u64>>(value).is_err(),
                "Dory field-register payloads must reject extra commitment fields"
            );
        }
    }

    #[test]
    #[expect(
        clippy::panic,
        reason = "test serialization failures should fail loudly"
    )]
    fn lattice_payload_rejects_extra_packed_commitments() {
        let mut value = serde_json::to_value(lattice_payload())
            .unwrap_or_else(|error| panic!("Lattice payload should serialize: {error}"));
        let lattice = value
            .get_mut("Lattice")
            .and_then(|payload| payload.as_object_mut())
            .unwrap_or_else(|| panic!("Lattice payload should be an object"));
        let previous = lattice.insert(
            "extra_packed_witness".to_string(),
            serde_json::Value::from(10_u64),
        );
        assert!(previous.is_none());

        let result = serde_json::from_value::<CommitmentPayload<u64>>(value);

        assert!(
            result.is_err(),
            "Lattice payloads must reject extra packed commitments"
        );
    }

    #[test]
    fn commitment_payload_validates_against_selected_pcs_family() {
        let curve = JoltProtocolConfig::for_zk(false);
        let lattice = lattice_config();

        assert!(validate_commitment_payload_family(&curve, &dory_payload()).is_ok());
        assert!(validate_commitment_payload_family(&lattice, &lattice_payload()).is_ok());
    }

    #[test]
    fn commitment_payload_mode_mismatch_rejects() {
        let curve = JoltProtocolConfig::for_zk(false);
        let lattice = lattice_config();

        assert!(matches!(
            validate_commitment_payload_family(&curve, &lattice_payload()),
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

    #[test]
    fn lattice_payload_layout_digest_mismatch_rejects() {
        let lattice = lattice_config();
        let payload = CommitmentPayload::Lattice(LatticeCommitmentPayload::new(9, [8; 32], 43));

        assert!(matches!(
            validate_commitment_payload_config(&lattice, &payload),
            Err(VerifierError::LatticePayloadLayoutDigestMismatch {
                expected,
                got,
            }) if expected == [7; 32] && got == [8; 32]
        ));
    }

    #[test]
    fn lattice_payload_dimension_mismatch_rejects() {
        let lattice = lattice_config();
        let payload = CommitmentPayload::Lattice(LatticeCommitmentPayload::new(9, [7; 32], 44));

        assert!(matches!(
            validate_commitment_payload_config(&lattice, &payload),
            Err(VerifierError::LatticePayloadDimensionMismatch {
                expected: 43,
                got: 44,
            })
        ));
    }

    #[test]
    fn direct_lattice_payload_validator_requires_lattice_config() {
        let curve = JoltProtocolConfig::for_zk(false);
        let payload = LatticeCommitmentPayload::new(9, [7; 32], 43);

        assert!(matches!(
            validate_lattice_commitment_payload_config(&curve, &payload),
            Err(VerifierError::CommitmentPayloadFamilyMismatch {
                expected: PcsFamily::Curve,
                got: PcsFamily::Lattice,
            })
        ));
    }
}
