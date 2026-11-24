use core::fmt::{Formatter, Result as FmtResult};
pub use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
use jolt_crypto::{Commitment, VectorCommitment};
use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::SumcheckProof;
use jolt_zk::BlindFoldProof;
use serde::{
    de::{value::SeqAccessDeserializer, Deserializer, Error as DeError, SeqAccess, Visitor},
    Deserialize, Serialize,
};

use crate::{
    config::JoltProtocolConfig,
    stages::{
        pow::StagePowNonces, stage1::outputs::Stage1OutputClaims,
        stage2::outputs::Stage2OutputClaims, stage3::outputs::Stage3OutputClaims,
        stage4::outputs::Stage4OutputClaims, stage5::outputs::Stage5OutputClaims,
        stage6a::outputs::Stage6aOutputClaims, stage6b::outputs::Stage6bOutputClaims,
        stage7::outputs::Stage7OutputClaims,
    },
    VerifierError,
};

/// The proof-carried polynomial commitments on the homomorphic build: one
/// commitment per committed polynomial.
#[cfg(not(feature = "akita"))]
pub type ProofCommitments<PCS> = JoltCommitments<<PCS as Commitment>::Output>;
/// The proof-carried polynomial commitments on the `akita` build: the single
/// native `OneHotTrace` group commitment carrying the RA columns.
#[cfg(feature = "akita")]
pub type ProofCommitments<PCS> = <PCS as Commitment>::Output;

/// The final-opening discharge on the homomorphic build: one RLC-batched PCS
/// opening proof at the unified point.
#[cfg(not(feature = "akita"))]
pub type JointOpeningProof<PCS> = <PCS as CommitmentScheme>::Proof;
#[cfg(feature = "akita")]
pub type JointOpeningProof<PCS> = <PCS as CommitmentScheme>::Proof;

#[expect(non_snake_case, reason = "Preserves the deployed proof field name.")]
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
    /// Required RV64 increment commitment in Akita proofs.
    #[cfg(feature = "akita")]
    pub fused_inc_commitment: Option<PCS::Output>,
    /// Direct commitment to the full field-register increment polynomial.
    /// Required for every Akita field-inline proof, including an all-zero trace.
    /// Producers without field-inline semantics leave this absent and are rejected
    /// by the protocol-config gate when field-inline is required.
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    pub field_inc_commitment: Option<PCS::Output>,
    pub claims: JoltProofClaims<PCS::Field, ZkProof>,
    pub trace_length: usize,
    pub ram_K: usize,
    pub rw_config: JoltReadWriteConfig,
    pub one_hot_config: JoltOneHotConfig,
    pub trace_polynomial_order: TracePolynomialOrder,
    /// The spongefish NARG string: the prover messages the verifier reads
    /// from its transcript. Messages that are fields of this struct are
    /// absorbed as public messages and are not repeated here; in Akita proofs
    /// the string carries the opening argument. The verifier requires it to be
    /// read to its end.
    #[serde(deserialize_with = "deserialize_argument")]
    pub narg: Vec<u8>,
}

/// The name under which [`JoltProof::narg`] asks a deserializer for its bytes.
///
/// The field is written as a plain sequence of bytes and read as a newtype
/// struct of this name around that sequence. bincode, postcard and serde_json
/// decode such a newtype as its content, and a deserializer that hands a
/// newtype struct on as a sequence is accepted too, so they all read the
/// argument one element at a time, exactly as they read a bare `Vec<u8>`.
///
/// A deserializer adapter may recognise the name and pass the argument to the
/// visitor as one byte string instead, provided the decoder under it borrows
/// from its input and so checks an announced length against the bytes that
/// remain before it hands them out. The recursion example's size-limiting
/// adapter does so, because a verifier running as a guest pays for every
/// element of the sequence. The name is public because that adapter lives
/// outside this crate; it is not a Rust identifier, so a derived type can
/// carry it only through an explicit rename.
pub const PROOF_ARGUMENT_NEWTYPE: &str = "$jolt_verifier::proof::Argument";

fn deserialize_argument<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
    deserializer.deserialize_newtype_struct(PROOF_ARGUMENT_NEWTYPE, ArgumentVisitor)
}

struct ArgumentVisitor;

impl<'de> Visitor<'de> for ArgumentVisitor {
    type Value = Vec<u8>;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.write_str("the proof argument bytes")
    }

    // bincode, postcard and serde_json reach this: the argument is decoded as
    // a `Vec<u8>`.
    fn visit_newtype_struct<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        Vec::deserialize(deserializer)
    }

    // A deserializer that forwards a newtype struct to `deserialize_any` or to
    // a sequence reaches this, and gets the same `Vec<u8>` decoding.
    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
        Vec::deserialize(SeqAccessDeserializer::new(seq))
    }

    // An adapter that recognised `PROOF_ARGUMENT_NEWTYPE` reaches this.
    fn visit_bytes<E: DeError>(self, bytes: &[u8]) -> Result<Self::Value, E> {
        Ok(bytes.to_vec())
    }
}

impl<PCS, VC, ZkProof> JoltProof<PCS, VC, ZkProof>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    /// Assemble a proof shell. Akita producers must attach `fused_inc_commitment`;
    /// field-inline producers also attach the field-increment payload with
    /// [`Self::with_field_inc_commitment`]. Verification rejects missing required payloads.
    #[expect(
        clippy::too_many_arguments,
        reason = "one argument per proof component, mirroring the wire struct"
    )]
    pub fn new(
        protocol: JoltProtocolConfig,
        commitments: ProofCommitments<PCS>,
        stages: JoltStageProofs<PCS::Field, VC>,
        joint_opening_proof: JointOpeningProof<PCS>,
        untrusted_advice_commitment: Option<PCS::Output>,
        claims: JoltProofClaims<PCS::Field, ZkProof>,
        trace_length: usize,
        ram_k: usize,
        rw_config: JoltReadWriteConfig,
        one_hot_config: JoltOneHotConfig,
        trace_polynomial_order: TracePolynomialOrder,
    ) -> Self {
        Self {
            protocol,
            commitments,
            stages,
            joint_opening_proof,
            untrusted_advice_commitment,
            #[cfg(feature = "akita")]
            fused_inc_commitment: None,
            #[cfg(all(feature = "akita", feature = "field-inline"))]
            field_inc_commitment: None,
            claims,
            trace_length,
            ram_K: ram_k,
            rw_config,
            one_hot_config,
            trace_polynomial_order,
            narg: Vec::new(),
        }
    }

    /// Attach the direct field-increment commitment required by Akita field-inline proofs.
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    pub fn with_field_inc_commitment(mut self, commitment: PCS::Output) -> Self {
        self.field_inc_commitment = Some(commitment);
        self
    }
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

    /// Replace the claims payload, retyping the `ZkProof` slot; every wire
    /// field carries over unchanged.
    pub fn with_claims<Z>(self, claims: JoltProofClaims<PCS::Field, Z>) -> JoltProof<PCS, VC, Z> {
        JoltProof {
            protocol: self.protocol,
            commitments: self.commitments,
            stages: self.stages,
            joint_opening_proof: self.joint_opening_proof,
            untrusted_advice_commitment: self.untrusted_advice_commitment,
            #[cfg(feature = "akita")]
            fused_inc_commitment: self.fused_inc_commitment,
            #[cfg(all(feature = "akita", feature = "field-inline"))]
            field_inc_commitment: self.field_inc_commitment,
            claims,
            trace_length: self.trace_length,
            ram_K: self.ram_K,
            rw_config: self.rw_config,
            one_hot_config: self.one_hot_config,
            trace_polynomial_order: self.trace_polynomial_order,
            narg: self.narg,
        }
    }
}

/// The field-register commitments of the field-inline extension. `FieldRdInc` is the
/// extension's single committed polynomial; the field-register access columns are virtual
/// (anchored through the bytecode read-RAF path), so this nest stays one deep until the
/// protocol commits more.
#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldRegistersCommitments<C> {
    pub rd_inc: C,
}

/// The field-inline extension's committed payload, grouped by component as the
/// protocol spec lays it out (`FieldInlineCommitments::field_registers`).
#[cfg(feature = "field-inline")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldInlineCommitments<C> {
    pub field_registers: FieldRegistersCommitments<C>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JoltCommitments<C> {
    pub rd_inc: C,
    pub ram_inc: C,
    pub instruction_ra: Vec<C>,
    pub ram_ra: Vec<C>,
    pub bytecode_ra: Vec<C>,
    /// Present on every field-inline proof (the build with field-inline enabled proves all
    /// guests under the composed protocol). Carried as an `Option` because this type is shared
    /// with producers that cannot supply field-inline commitments — the legacy prover and the
    /// Akita converter — whose proofs fail the protocol-config gate before this field is ever
    /// read; [`validate_proof_consistency`] rejects a missing payload fail-closed for
    /// everything else.
    ///
    /// [`validate_proof_consistency`]: crate::verifier::validate_proof_consistency
    #[cfg(feature = "field-inline")]
    pub field_inline: Option<FieldInlineCommitments<C>>,
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
            #[cfg(feature = "field-inline")]
            field_inline: None,
        }
    }

    /// Attach the field-inline committed payload (the modular prover's stage 0 sets this on
    /// every field-inline proof).
    #[cfg(feature = "field-inline")]
    pub fn with_field_inline(mut self, field_inline: FieldInlineCommitments<C>) -> Self {
        self.field_inline = Some(field_inline);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    not(feature = "akita"),
    expect(
        clippy::large_enum_variant,
        reason = "clear claims stay inline on the standard verifier's common path"
    )
)]
#[serde(bound(
    serialize = "F: Serialize, ZkProof: Serialize",
    deserialize = "F: for<'a> Deserialize<'a>, ZkProof: serde::de::DeserializeOwned"
))]
pub enum JoltProofClaims<F, ZkProof>
where
    F: JoltField,
{
    Clear(ClearProofClaims<F>),
    Zk { blindfold_proof: ZkProof },
}

impl<F, ZkProof> JoltProofClaims<F, ZkProof>
where
    F: JoltField,
{
    pub const fn is_zk(&self) -> bool {
        matches!(self, Self::Zk { .. })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: for<'a> Deserialize<'a>"))]
pub struct ClearProofClaims<F: JoltField> {
    pub stage1: Stage1OutputClaims<F>,
    pub stage2: Stage2OutputClaims<F>,
    pub stage3: Stage3OutputClaims<F>,
    pub stage4: Stage4OutputClaims<F>,
    pub stage5: Stage5OutputClaims<F>,
    pub stage6a: Stage6aOutputClaims<F>,
    pub stage6b: Stage6bOutputClaims<F>,
    pub stage7: Stage7OutputClaims<F>,
}

impl<F: JoltField> ClearProofClaims<F> {
    #[expect(
        clippy::too_many_arguments,
        reason = "one argument per stage, mirroring the wire struct"
    )]
    pub fn new(
        stage1: Stage1OutputClaims<F>,
        stage2: Stage2OutputClaims<F>,
        stage3: Stage3OutputClaims<F>,
        stage4: Stage4OutputClaims<F>,
        stage5: Stage5OutputClaims<F>,
        stage6a: Stage6aOutputClaims<F>,
        stage6b: Stage6bOutputClaims<F>,
        stage7: Stage7OutputClaims<F>,
    ) -> Self {
        Self {
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            stage6a,
            stage6b,
            stage7,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, <VC as Commitment>::Output: Serialize",
    deserialize = "F: for<'a> Deserialize<'a>, <VC as Commitment>::Output: serde::de::DeserializeOwned"
))]
pub struct JoltStageProofs<F, VC>
where
    F: JoltField,
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
    /// Nonces of the challenge bursts outside the sumcheck rounds. A slot is
    /// filled only where the site is visited and the challenge set needs proof
    /// of work, so a BN254 proof carries none.
    pub pow_nonces: StagePowNonces,
}

#[cfg(test)]
mod tests {
    use super::deserialize_argument;
    use serde::de::value::{Error as ValueError, SeqDeserializer};

    /// A deserializer that forwards a newtype struct to `deserialize_any`, as
    /// serde's own value deserializers do, presents the argument as a
    /// sequence.
    #[test]
    fn proof_argument_decodes_from_a_forwarded_sequence() {
        let argument: Vec<u8> = (0..=u8::MAX).collect();
        let forwarded = SeqDeserializer::<_, ValueError>::new(argument.iter().copied());
        assert_eq!(deserialize_argument(forwarded), Ok(argument));
    }
}
