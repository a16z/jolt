use jolt_field::Field;
use jolt_openings::{BatchOpeningStatement, PhysicalView};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use serde::{Deserialize, Serialize};

pub type AkitaLayoutDigest = [u8; 32];
pub type AkitaViewFormula<F> = PhysicalView<F>;
pub type AkitaPackedViewStatement<F, OpeningId = (), RelationId = ()> =
    BatchOpeningStatement<F, AkitaCommitment, OpeningId, RelationId>;

const AKITA_MOCK_BACKEND_ID: [u8; 32] = *b"jolt-akita-mock-backend-v000001!";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AkitaSetupMode {
    Exact,
    Universal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AkitaFieldMode {
    BaseField,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaSetupParams {
    pub security_bits: u32,
    pub d_setup: usize,
    pub setup_mode: AkitaSetupMode,
    pub field_mode: AkitaFieldMode,
    pub default_layout_digest: AkitaLayoutDigest,
}

impl AkitaSetupParams {
    pub fn exact(d_setup: usize, default_layout_digest: AkitaLayoutDigest) -> Self {
        Self {
            security_bits: 128,
            d_setup,
            setup_mode: AkitaSetupMode::Exact,
            field_mode: AkitaFieldMode::BaseField,
            default_layout_digest,
        }
    }

    pub fn universal(d_setup: usize, default_layout_digest: AkitaLayoutDigest) -> Self {
        Self {
            security_bits: 128,
            d_setup,
            setup_mode: AkitaSetupMode::Universal,
            field_mode: AkitaFieldMode::BaseField,
            default_layout_digest,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaSetupKey {
    pub security_bits: u32,
    pub d_setup: usize,
    pub setup_mode: AkitaSetupMode,
    pub field_mode: AkitaFieldMode,
    pub backend_id: [u8; 32],
}

impl AkitaSetupKey {
    pub fn accepts_dimension(&self, d_pack: usize) -> bool {
        match self.setup_mode {
            AkitaSetupMode::Exact => d_pack == self.d_setup,
            AkitaSetupMode::Universal => d_pack <= self.d_setup,
        }
    }
}

impl From<&AkitaSetupParams> for AkitaSetupKey {
    fn from(params: &AkitaSetupParams) -> Self {
        Self {
            security_bits: params.security_bits,
            d_setup: params.d_setup,
            setup_mode: params.setup_mode,
            field_mode: params.field_mode,
            backend_id: AKITA_MOCK_BACKEND_ID,
        }
    }
}

impl AppendToTranscript for AkitaSetupKey {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"akita_setup_key"));
        transcript.append(&U64Word(self.security_bits as u64));
        transcript.append(&U64Word(self.d_setup as u64));
        transcript.append_bytes(&[self.setup_mode as u8]);
        transcript.append_bytes(&[self.field_mode as u8]);
        transcript.append_bytes(&self.backend_id);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaSetup {
    pub key: AkitaSetupKey,
    pub default_layout_digest: AkitaLayoutDigest,
}

impl AkitaSetup {
    pub fn new(params: AkitaSetupParams) -> Self {
        let key = AkitaSetupKey::from(&params);
        Self {
            key,
            default_layout_digest: params.default_layout_digest,
        }
    }
}

pub type AkitaProverSetup = AkitaSetup;
pub type AkitaVerifierSetup = AkitaSetup;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaCommitment {
    pub layout_digest: AkitaLayoutDigest,
    pub commitment_digest: [u8; 32],
    pub d_pack: usize,
}

impl AppendToTranscript for AkitaCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"akita_commitment"));
        transcript.append_bytes(&self.layout_digest);
        transcript.append_bytes(&self.commitment_digest);
        transcript.append(&U64Word(self.d_pack as u64));
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AkitaBatchProof<F: Field> {
    pub setup_key: AkitaSetupKey,
    pub packed_commitment: AkitaCommitment,
    pub statement_digest: F,
    pub coefficients: Vec<F>,
    pub reduced_opening: F,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AkitaHidingCommitment<F: Field> {
    pub commitment: F,
}

impl<F: Field> AppendToTranscript for AkitaHidingCommitment<F> {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"akita_hiding_commitment"));
        self.commitment.append_to_transcript(transcript);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct AkitaCommitInput<F: Field> {
    pub layout_digest: AkitaLayoutDigest,
    pub d_pack: usize,
    pub evaluations: Vec<F>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AkitaProverHint {
    pub layout_digest: AkitaLayoutDigest,
    pub commitment_digest: [u8; 32],
    pub d_pack: usize,
}

impl AkitaProverHint {
    pub fn matches_commitment(&self, commitment: &AkitaCommitment) -> bool {
        self.layout_digest == commitment.layout_digest
            && self.commitment_digest == commitment.commitment_digest
            && self.d_pack == commitment.d_pack
    }
}

pub(crate) fn append_field_slice<F, T>(transcript: &mut T, label: &'static [u8], values: &[F])
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&LabelWithCount(label, values.len() as u64));
    for value in values {
        value.append_to_transcript(transcript);
    }
}

pub(crate) fn u64_field<F: Field>(value: u64) -> F {
    F::from_u64(value)
}
