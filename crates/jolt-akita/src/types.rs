use akita_config::CommitmentConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CpuPreparedSetup, DensePoly};
use akita_types::{
    AkitaBatchedProof as NativeBatchProof, AkitaBatchedProofShape,
    AkitaCommitmentHint as NativeCommitmentHint, AkitaVerifierSetup as NativeVerifierSetup,
    RingCommitment as NativeRingCommitment,
};
use jolt_openings::{BatchOpeningStatement, PhysicalView};
use jolt_poly::Polynomial;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use serde::{Deserialize, Serialize};

pub type AkitaField = akita_config::proof_optimized::fp128::Field;
pub type AkitaConfig = akita_config::proof_optimized::fp128::D64Full;
pub const AKITA_D: usize = AkitaConfig::D;

pub(crate) type NativeScheme = AkitaCommitmentScheme<AKITA_D, AkitaConfig>;
pub(crate) type NativeCommitment = NativeRingCommitment<AkitaField, AKITA_D>;
pub(crate) type NativeHint = NativeCommitmentHint<AkitaField, AKITA_D>;
pub(crate) type NativeProof = NativeBatchProof<AkitaField, AkitaField>;
pub(crate) type NativeProofShape = AkitaBatchedProofShape;
pub(crate) type NativeVerifier = NativeVerifierSetup<AkitaField>;
pub(crate) type NativeDensePoly = DensePoly<AkitaField, AKITA_D>;
pub(crate) type NativePreparedSetup = CpuPreparedSetup<AkitaField, AKITA_D>;

pub type AkitaLayoutDigest = [u8; 32];
pub type AkitaViewFormula = PhysicalView<AkitaField>;
pub type AkitaPackedViewStatement<OpeningId = (), RelationId = ()> =
    BatchOpeningStatement<AkitaField, AkitaCommitment, OpeningId, RelationId>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
pub struct AkitaVerifierSetup {
    pub max_num_vars: usize,
    pub max_num_polys_per_commitment_group: usize,
    pub default_layout_digest: AkitaLayoutDigest,
    pub native: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaBatchProof {
    pub commitment: AkitaCommitment,
    pub proof_shape: Vec<u8>,
    pub proof: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AkitaCommitInput {
    pub layout_digest: AkitaLayoutDigest,
    pub polynomial: Polynomial<AkitaField>,
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
