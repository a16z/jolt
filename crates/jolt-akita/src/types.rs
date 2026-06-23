use std::collections::BTreeSet;

use akita_config::CommitmentConfig;
use akita_field::PseudoMersenneField;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{AkitaPolyOps, CpuPreparedSetup, DensePoly, SparseRingPoly};
use akita_types::{
    AkitaBatchedProof as NativeBatchProof, AkitaBatchedProofShape,
    AkitaCommitmentHint as NativeCommitmentHint, AkitaVerifierSetup as NativeVerifierSetup,
    RingCommitment as NativeRingCommitment,
};
use jolt_openings::{CommitmentLayoutDigest, OpeningsError};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use serde::{Deserialize, Serialize};

pub type AkitaField = akita_config::proof_optimized::fp128::Field;
pub type AkitaConfig = akita_config::proof_optimized::fp128::D64Full;
pub const AKITA_D: usize = AkitaConfig::D;
pub const AKITA_FIELD_MODULUS: u128 =
    u128::MAX - (<AkitaField as PseudoMersenneField>::MODULUS_OFFSET - 1);

pub(crate) type NativeScheme = AkitaCommitmentScheme<AKITA_D, AkitaConfig>;
pub(crate) type NativeCommitment = NativeRingCommitment<AkitaField, AKITA_D>;
pub(crate) type NativeHint = NativeCommitmentHint<AkitaField, AKITA_D>;
pub(crate) type NativeProof = NativeBatchProof<AkitaField, AkitaField>;
pub(crate) type NativeProofShape = AkitaBatchedProofShape;
pub(crate) type NativeVerifier = NativeVerifierSetup<AkitaField>;
pub(crate) type NativeDensePoly = DensePoly<AkitaField, AKITA_D>;
pub(crate) type NativeSparsePoly = SparseRingPoly<AkitaField, AKITA_D>;
pub(crate) type NativePreparedSetup = CpuPreparedSetup<AkitaField, AKITA_D>;

pub type AkitaLayoutDigest = [u8; 32];

pub struct AkitaSparsePolynomial {
    pub(crate) native: NativeSparsePoly,
}

impl AkitaSparsePolynomial {
    pub fn from_jolt_unit_indices(
        num_vars: usize,
        indices: impl IntoIterator<Item = usize>,
    ) -> Result<Self, OpeningsError> {
        if num_vars >= usize::BITS as usize {
            return Err(invalid_sparse_polynomial(format!(
                "Akita sparse polynomial dimension {num_vars} exceeds usize bit width"
            )));
        }
        let domain_size = 1usize << num_vars;
        if domain_size < AKITA_D {
            return Err(invalid_sparse_polynomial(format!(
                "Akita sparse polynomial domain {domain_size} is smaller than ring dimension {AKITA_D}"
            )));
        }

        let mut seen = BTreeSet::new();
        let mut coeffs = Vec::new();
        for index in indices {
            if index >= domain_size {
                return Err(invalid_sparse_polynomial(format!(
                    "Akita sparse polynomial index {index} outside domain size {domain_size}"
                )));
            }
            if !seen.insert(index) {
                return Err(invalid_sparse_polynomial(format!(
                    "Akita sparse polynomial index {index} appears more than once"
                )));
            }
            let akita_index = jolt_to_akita_index(num_vars, index);
            coeffs.push((akita_index / AKITA_D, akita_index % AKITA_D, 1i8));
        }

        NativeSparsePoly::from_signed_coeffs(num_vars, domain_size / AKITA_D, coeffs)
            .map(|native| Self { native })
            .map_err(|error| {
                invalid_sparse_polynomial(format!(
                    "Akita sparse polynomial construction failed: {error}"
                ))
            })
    }

    pub fn num_vars(&self) -> usize {
        self.native.num_vars()
    }
}

pub(crate) fn jolt_to_akita_index(num_vars: usize, index: usize) -> usize {
    if num_vars == 0 {
        return index;
    }
    index.reverse_bits() >> (usize::BITS as usize - num_vars)
}

fn invalid_sparse_polynomial(reason: impl Into<String>) -> OpeningsError {
    OpeningsError::InvalidBatch(reason.into())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaSetupParams {
    pub max_num_vars: usize,
    pub max_num_polys_per_commitment_group: usize,
    pub default_layout_digest: AkitaLayoutDigest,
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
}

#[derive(Clone, Debug)]
pub struct AkitaProverSetup {
    pub max_num_vars: usize,
    pub max_num_polys_per_commitment_group: usize,
    pub default_layout_digest: AkitaLayoutDigest,
    pub(crate) native: akita_prover::AkitaProverSetup<AkitaField, AKITA_D>,
    pub(crate) prepared: NativePreparedSetup,
    pub(crate) verifier: AkitaVerifierSetup,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaVerifierSetup {
    pub max_num_vars: usize,
    pub max_num_polys_per_commitment_group: usize,
    pub default_layout_digest: AkitaLayoutDigest,
    pub native: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaCommitment {
    pub layout_digest: AkitaLayoutDigest,
    pub num_vars: usize,
    pub poly_count: usize,
    pub native: Vec<u8>,
}

impl AppendToTranscript for AkitaCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"akita_commitment"));
        transcript.append_bytes(&self.layout_digest);
        transcript.append(&U64Word(self.num_vars as u64));
        transcript.append(&U64Word(self.poly_count as u64));
        transcript.append(&LabelWithCount(
            b"akita_commitment_bytes",
            self.native.len() as u64,
        ));
        transcript.append_bytes(&self.native);
    }
}

impl CommitmentLayoutDigest for AkitaCommitment {
    fn layout_digest(&self) -> Option<[u8; 32]> {
        Some(self.layout_digest)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaBatchProof {
    pub commitment: AkitaCommitment,
    pub statement_bridge: Vec<u8>,
    pub proof_shape: Vec<u8>,
    pub proof: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaHidingCommitment {
    pub eval: Vec<u8>,
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
    pub commitment: AkitaCommitment,
    pub(crate) native: Option<NativeHint>,
}

impl AkitaProverHint {
    pub fn matches_commitment(&self, commitment: &AkitaCommitment) -> bool {
        self.commitment == *commitment
    }
}

pub(crate) fn append_field_slice<T>(transcript: &mut T, label: &'static [u8], values: &[AkitaField])
where
    T: Transcript<Challenge = AkitaField>,
{
    transcript.append(&LabelWithCount(label, values.len() as u64));
    for value in values {
        value.append_to_transcript(transcript);
    }
}
