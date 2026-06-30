use std::{collections::BTreeSet, io::Cursor};

use akita_config::CommitmentConfig;
use akita_field::PseudoMersenneField;
use akita_pcs::{AkitaCommitmentScheme, AkitaDeserialize, AkitaSerialize};
use akita_prover::{CpuPreparedSetup, DensePoly, SparseRingPoly};
use akita_transcript::Transcript as AkitaBackendTranscript;
use akita_types::{
    AkitaBatchedProof as AkitaBackendBatchProof, AkitaBatchedProofShape,
    AkitaCommitmentHint as AkitaBackendCommitmentHint,
    AkitaVerifierSetup as AkitaBackendVerifierSetup, RingCommitment as AkitaBackendRingCommitment,
};
use jolt_field::{CanonicalBytes, FixedByteSize};
use jolt_openings::{OpeningsError, VerifierOpeningClaim};
use jolt_poly::{MultilinearPoly, Polynomial};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use serde::{Deserialize, Serialize};

pub type AkitaField = akita_config::proof_optimized::fp128::Field;
pub(crate) type AkitaConfig = akita_config::proof_optimized::fp128::D64Full;
pub(crate) const AKITA_D: usize = AkitaConfig::D;
pub(crate) type AkitaBackendExtField = <AkitaConfig as CommitmentConfig>::ExtField;
pub const AKITA_FIELD_MODULUS: u128 =
    u128::MAX - (<AkitaField as PseudoMersenneField>::MODULUS_OFFSET - 1);

pub(crate) type AkitaBackendScheme = AkitaCommitmentScheme<AKITA_D, AkitaConfig>;
pub(crate) type AkitaBackendCommitment = AkitaBackendRingCommitment<AkitaField, AKITA_D>;
pub(crate) type AkitaBackendHint = AkitaBackendCommitmentHint<AkitaField, AKITA_D>;
pub(crate) type AkitaBackendProof = AkitaBackendBatchProof<AkitaField, AkitaBackendExtField>;
pub(crate) type AkitaBackendProofShape = AkitaBatchedProofShape;
pub(crate) type AkitaBackendVerifier = AkitaBackendVerifierSetup<AkitaField>;
pub(crate) type AkitaBackendDensePoly = DensePoly<AkitaField, AKITA_D>;
pub(crate) type AkitaBackendSparsePoly = SparseRingPoly<AkitaField, AKITA_D>;
pub(crate) type AkitaBackendPreparedSetup = CpuPreparedSetup<AkitaField, AKITA_D>;

pub(crate) type AkitaLayoutDigest = [u8; 32];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaSetupParams {
    pub(crate) max_num_vars: usize,
    pub(crate) max_num_polys_per_commitment_group: usize,
    pub(crate) default_layout_digest: AkitaLayoutDigest,
}

impl AkitaSetupParams {
    pub fn new(
        max_num_vars: usize,
        max_num_polys_per_commitment_group: usize,
        default_layout_digest: AkitaLayoutDigest,
    ) -> Self {
        Self {
            max_num_vars,
            max_num_polys_per_commitment_group,
            default_layout_digest,
        }
    }

    pub fn max_num_vars(&self) -> usize {
        self.max_num_vars
    }

    pub fn max_num_polys_per_commitment_group(&self) -> usize {
        self.max_num_polys_per_commitment_group
    }

    pub fn default_layout_digest(&self) -> [u8; 32] {
        self.default_layout_digest
    }
}

#[derive(Clone, Debug)]
pub struct AkitaProverSetup {
    pub(crate) max_num_vars: usize,
    pub(crate) max_num_polys_per_commitment_group: usize,
    pub(crate) default_layout_digest: AkitaLayoutDigest,
    pub(crate) backend_prover_setup: akita_prover::AkitaProverSetup<AkitaField, AKITA_D>,
    pub(crate) prepared_backend_setup: AkitaBackendPreparedSetup,
    pub(crate) verifier: AkitaVerifierSetup,
}

impl AkitaProverSetup {
    pub fn max_num_vars(&self) -> usize {
        self.max_num_vars
    }

    pub fn max_num_polys_per_commitment_group(&self) -> usize {
        self.max_num_polys_per_commitment_group
    }

    pub fn default_layout_digest(&self) -> [u8; 32] {
        self.default_layout_digest
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaVerifierSetup {
    pub(crate) max_num_vars: usize,
    pub(crate) max_num_polys_per_commitment_group: usize,
    pub(crate) default_layout_digest: AkitaLayoutDigest,
    pub(crate) serialized_backend_bytes: Vec<u8>,
}

impl AkitaVerifierSetup {
    pub fn max_num_vars(&self) -> usize {
        self.max_num_vars
    }

    pub fn max_num_polys_per_commitment_group(&self) -> usize {
        self.max_num_polys_per_commitment_group
    }

    pub fn default_layout_digest(&self) -> [u8; 32] {
        self.default_layout_digest
    }

    pub fn serialized_akita_bytes(&self) -> &[u8] {
        &self.serialized_backend_bytes
    }
}

impl AppendToTranscript for AkitaVerifierSetup {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"akita_setup_key"));
        transcript.append_bytes(b"akita/fp128/d64full");
        transcript.append(&U64Word(AKITA_D as u64));
        transcript.append(&U64Word(self.max_num_vars as u64));
        transcript.append(&U64Word(self.max_num_polys_per_commitment_group as u64));
        transcript.append_bytes(&self.default_layout_digest);
        transcript.append(&LabelWithCount(
            b"akita_verifier_setup",
            self.serialized_backend_bytes.len() as u64,
        ));
        transcript.append_bytes(&self.serialized_backend_bytes);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaCommitment {
    pub(crate) layout_digest: AkitaLayoutDigest,
    pub(crate) num_vars: usize,
    pub(crate) poly_count: usize,
    pub(crate) serialized_backend_bytes: Vec<u8>,
}

impl AkitaCommitment {
    pub fn layout_digest(&self) -> [u8; 32] {
        self.layout_digest
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn poly_count(&self) -> usize {
        self.poly_count
    }

    pub fn serialized_akita_bytes(&self) -> &[u8] {
        &self.serialized_backend_bytes
    }
}

impl AppendToTranscript for AkitaCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"akita_commitment"));
        transcript.append_bytes(&self.layout_digest);
        transcript.append(&U64Word(self.num_vars as u64));
        transcript.append(&U64Word(self.poly_count as u64));
        transcript.append(&LabelWithCount(
            b"akita_commitment_bytes",
            self.serialized_backend_bytes.len() as u64,
        ));
        transcript.append_bytes(&self.serialized_backend_bytes);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaBatchProof {
    pub(crate) commitment: AkitaCommitment,
    pub(crate) statement_bridge: Vec<u8>,
    pub(crate) serialized_akita_proof_shape: Vec<u8>,
    pub(crate) serialized_akita_proof: Vec<u8>,
}

impl AkitaBatchProof {
    pub fn commitment(&self) -> &AkitaCommitment {
        &self.commitment
    }

    pub fn statement_bridge(&self) -> &[u8] {
        &self.statement_bridge
    }

    pub fn proof_shape(&self) -> &[u8] {
        &self.serialized_akita_proof_shape
    }

    pub fn proof_bytes(&self) -> &[u8] {
        &self.serialized_akita_proof
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaHidingCommitment {
    pub(crate) eval: Vec<u8>,
}

impl AkitaHidingCommitment {
    pub(crate) fn new(eval: Vec<u8>) -> Self {
        Self { eval }
    }
}

impl AppendToTranscript for AkitaHidingCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"akita_hiding_commitment"));
        transcript.append(&LabelWithCount(
            b"akita_hiding_eval",
            self.eval.len() as u64,
        ));
        transcript.append_bytes(&self.eval);
    }
}

#[derive(Clone, Debug, Default)]
pub struct AkitaProverHint {
    pub(crate) commitment: AkitaCommitment,
    pub(crate) backend_hint: Option<AkitaBackendHint>,
    pub(crate) source_kind: AkitaSourceKind,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum AkitaSourceKind {
    #[default]
    Dense,
    SparseUnit,
}

pub(crate) struct AkitaSparsePolynomial {
    pub(crate) backend_polynomial: AkitaBackendSparsePoly,
}

impl AkitaSparsePolynomial {
    pub(crate) fn from_jolt_unit_indices(
        num_vars: usize,
        indices: impl IntoIterator<Item = usize>,
    ) -> Result<Self, OpeningsError> {
        if num_vars >= usize::BITS as usize {
            return Err(invalid_batch(format!(
                "Akita sparse polynomial dimension {num_vars} exceeds usize bit width"
            )));
        }
        let domain_size = 1usize << num_vars;
        if domain_size < AKITA_D {
            return Err(invalid_batch(format!(
                "Akita sparse polynomial domain {domain_size} is smaller than ring dimension {AKITA_D}"
            )));
        }

        let mut seen = BTreeSet::new();
        let mut coeffs = Vec::new();
        for index in indices {
            if index >= domain_size {
                return Err(invalid_batch(format!(
                    "Akita sparse polynomial index {index} outside domain size {domain_size}"
                )));
            }
            if !seen.insert(index) {
                return Err(invalid_batch(format!(
                    "Akita sparse polynomial index {index} appears more than once"
                )));
            }
            let akita_index = jolt_to_akita_index(num_vars, index);
            coeffs.push((akita_index / AKITA_D, akita_index % AKITA_D, 1i8));
        }

        AkitaBackendSparsePoly::from_signed_coeffs(num_vars, domain_size / AKITA_D, coeffs)
            .map(|backend_polynomial| Self { backend_polynomial })
            .map_err(|error| {
                invalid_batch(format!(
                    "Akita sparse polynomial construction failed: {error}"
                ))
            })
    }

    pub(crate) fn num_vars(&self) -> usize {
        self.backend_polynomial.num_vars()
    }
}

fn jolt_to_akita_index(num_vars: usize, index: usize) -> usize {
    if num_vars == 0 {
        return index;
    }
    index.reverse_bits() >> (usize::BITS as usize - num_vars)
}

pub(crate) fn dense_polynomials(
    polynomials: &[Polynomial<AkitaField>],
) -> Result<Vec<AkitaBackendDensePoly>, OpeningsError> {
    polynomials
        .iter()
        .map(|poly| {
            let evals = jolt_to_akita_evals(poly.num_vars(), poly.evals())?;
            AkitaBackendDensePoly::from_field_evals(poly.num_vars(), &evals).map_err(akita_error)
        })
        .collect()
}

pub(crate) fn jolt_to_akita_evals(
    num_vars: usize,
    jolt_evals: &[AkitaField],
) -> Result<Vec<AkitaField>, OpeningsError> {
    let Some(expected) = u32::try_from(num_vars)
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift))
    else {
        return Err(invalid_batch(format!(
            "Akita polynomial dimension {num_vars} exceeds usize bit width"
        )));
    };
    if jolt_evals.len() != expected {
        return Err(invalid_batch(format!(
            "Akita polynomial has {} evaluations but dimension {num_vars} requires {expected}",
            jolt_evals.len()
        )));
    }
    if num_vars == 0 {
        return Ok(jolt_evals.to_vec());
    }
    let mut akita_evals = vec![AkitaField::zero(); jolt_evals.len()];
    for (jolt_index, &eval) in jolt_evals.iter().enumerate() {
        let akita_index = jolt_to_akita_index(num_vars, jolt_index);
        akita_evals[akita_index] = eval;
    }
    Ok(akita_evals)
}

pub(crate) fn polynomial_evaluations<P>(polynomial: &P) -> Vec<AkitaField>
where
    P: MultilinearPoly<AkitaField> + ?Sized,
{
    let capacity = if polynomial.num_vars() < usize::BITS as usize {
        1usize << polynomial.num_vars()
    } else {
        0
    };
    let mut evals = Vec::with_capacity(capacity);
    polynomial.for_each_row(polynomial.num_vars(), &mut |_, row| {
        evals.extend_from_slice(row);
    });
    evals
}

pub(crate) fn serialize_akita<T>(value: &T) -> Result<Vec<u8>, OpeningsError>
where
    T: AkitaSerialize,
{
    let mut bytes = Vec::with_capacity(value.compressed_size());
    value
        .serialize_compressed(&mut bytes)
        .map_err(akita_error)?;
    Ok(bytes)
}

pub(crate) fn deserialize_akita<T>(bytes: &[u8], ctx: &T::Context) -> Result<T, OpeningsError>
where
    T: AkitaDeserialize,
{
    T::deserialize_compressed(Cursor::new(bytes), ctx).map_err(akita_error)
}

pub(crate) fn field_bytes(value: AkitaField) -> Vec<u8> {
    let mut bytes = vec![0u8; AkitaField::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    bytes
}

pub(crate) fn invalid_batch(message: impl Into<String>) -> OpeningsError {
    OpeningsError::InvalidBatch(message.into())
}

pub(crate) fn akita_error(error: impl ToString) -> OpeningsError {
    OpeningsError::InvalidBatch(error.to_string())
}

pub(crate) fn transparent_zk_error() -> OpeningsError {
    OpeningsError::InvalidBatch(
        "Akita backend adapter is transparent-only and does not support ZK openings yet".to_owned(),
    )
}

pub(crate) struct AkitaBlackBoxBatchStatementTranscript<'a> {
    statement: &'a [VerifierOpeningClaim<AkitaField, AkitaCommitment>],
    commitment: &'a AkitaCommitment,
    point: &'a [AkitaField],
}

impl<'a> AkitaBlackBoxBatchStatementTranscript<'a> {
    pub(crate) fn new(
        statement: &'a [VerifierOpeningClaim<AkitaField, AkitaCommitment>],
        commitment: &'a AkitaCommitment,
        point: &'a [AkitaField],
    ) -> Self {
        Self {
            statement,
            commitment,
            point,
        }
    }
}

impl AppendToTranscript for AkitaBlackBoxBatchStatementTranscript<'_> {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"akita_batch_statement"));
        self.commitment.append_to_transcript(transcript);
        transcript.append_values(b"akita_pcs_point", self.point);
        transcript.append(&LabelWithCount(
            b"akita_claims",
            self.statement.len() as u64,
        ));
        for claim in self.statement {
            claim.commitment.append_to_transcript(transcript);
            claim.evaluation.value.append_to_transcript(transcript);
        }
    }
}

impl AppendToTranscript for AkitaBatchProof {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&LabelWithCount(
            b"akita_stmt_bridge",
            self.statement_bridge.len() as u64,
        ));
        transcript.append_bytes(&self.statement_bridge);
        transcript.append(&LabelWithCount(
            b"akita_proof_shape",
            self.serialized_akita_proof_shape.len() as u64,
        ));
        transcript.append_bytes(&self.serialized_akita_proof_shape);
        transcript.append(&LabelWithCount(
            b"akita_proof",
            self.serialized_akita_proof.len() as u64,
        ));
        transcript.append_bytes(&self.serialized_akita_proof);
    }
}

pub(crate) fn bridge_jolt_statement_challenge<T>(
    jolt_transcript: &mut T,
    akita_transcript: &mut impl AkitaBackendTranscript<AkitaField>,
) -> Vec<u8>
where
    T: Transcript<Challenge = AkitaField>,
{
    let bridge = jolt_transcript.challenge_scalar();
    akita_transcript.append_field(b"jolt_statement_bridge", &bridge);
    field_bytes(bridge)
}
