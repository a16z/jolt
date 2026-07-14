use std::{collections::BTreeSet, fmt, io::Cursor, sync::Arc, sync::OnceLock};

use akita_config::CommitmentConfig;
use akita_pcs::{AkitaCommitmentScheme, AkitaDeserialize, AkitaSerialize};
use akita_prover::{CpuBackend, CpuPreparedSetup, DensePoly, OneHotPoly, SparseRingPoly};
use akita_transcript::Transcript as AkitaBackendTranscript;
use akita_types::{
    AkitaBatchedProof as AkitaBackendBatchProof, AkitaBatchedProofShape,
    AkitaCommitmentHint as AkitaBackendCommitmentHint,
    AkitaVerifierSetup as AkitaBackendVerifierSetup, Commitment as AkitaBackendRingCommitment,
};
use jolt_field::CanonicalBytes;
use jolt_openings::{OpeningsError, VerifierOpeningClaim};
use jolt_poly::{MultilinearPoly, OneHotIndexOrder, Polynomial};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use serde::{Deserialize, Serialize};
use tracing::info_span;

pub type AkitaField = akita_config::proof_optimized::fp128::Field;
pub(crate) type AkitaConfig = akita_config::proof_optimized::fp128::D64Full;
pub(crate) type AkitaOneHotConfig = akita_config::proof_optimized::fp128::D64OneHot;
pub(crate) const AKITA_D: usize = AkitaConfig::D;
/// One-hot chunk size baked into the `D64OneHot` preset.
pub const AKITA_ONE_HOT_K: usize = 256;
pub(crate) const AKITA_ONE_HOT_LOG_K: usize = AKITA_ONE_HOT_K.ilog2() as usize;
const _: () = assert!(AKITA_ONE_HOT_K.is_power_of_two());

pub(crate) type AkitaBackendExtField = <AkitaConfig as CommitmentConfig>::ExtField;

pub(crate) type AkitaBackendScheme = AkitaCommitmentScheme<AkitaConfig>;
pub(crate) type AkitaOneHotBackendScheme = AkitaCommitmentScheme<AkitaOneHotConfig>;
pub(crate) type AkitaBackendCommitment = AkitaBackendRingCommitment<AkitaField>;
pub(crate) type AkitaBackendHint = AkitaBackendCommitmentHint<AkitaField>;
pub(crate) type AkitaBackendProof = AkitaBackendBatchProof<AkitaField, AkitaBackendExtField>;
pub(crate) type AkitaBackendProofShape = AkitaBatchedProofShape;
pub(crate) type AkitaBackendVerifier = AkitaBackendVerifierSetup<AkitaField>;
pub(crate) type AkitaBackendDensePoly = DensePoly<AkitaField>;
pub(crate) type AkitaBackendOneHotPoly = OneHotPoly<AkitaField, u8>;
pub(crate) type AkitaBackendSparsePoly = SparseRingPoly<AkitaField>;
pub(crate) type AkitaBackendPreparedSetup = CpuPreparedSetup<AkitaField>;
pub(crate) type AkitaBackendProverSetup = akita_prover::AkitaProverSetup<AkitaField>;
pub(crate) type BackendStack<'a> = akita_prover::UniformProverStack<'a, AkitaField, CpuBackend>;

pub(crate) type AkitaLayoutDigest = [u8; 32];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaSetupParams {
    pub(crate) max_num_vars: usize,
    pub(crate) max_num_polys_per_commitment_group: usize,
    pub(crate) default_layout_digest: AkitaLayoutDigest,
    /// When set, only the one-hot flavor's backend setup is built — the
    /// full-flavor setup for the same shape is large and slow, and a packed
    /// one-hot commitment object never touches it.
    pub(crate) one_hot_only: bool,
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
            one_hot_only: false,
        }
    }

    /// Setup parameters for a commitment object that only ever commits and
    /// opens through the one-hot flavor (the packed `W_jolt` group): skips
    /// building the full-flavor backend setup of the same shape.
    pub fn one_hot_only(
        max_num_vars: usize,
        max_num_polys_per_commitment_group: usize,
        default_layout_digest: AkitaLayoutDigest,
    ) -> Self {
        Self {
            max_num_vars,
            max_num_polys_per_commitment_group,
            default_layout_digest,
            one_hot_only: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AkitaProverSetup {
    pub(crate) backend_prover_setup: Option<Arc<AkitaBackendProverSetup>>,
    pub(crate) prepared_backend_setup: Option<Arc<AkitaBackendPreparedSetup>>,
    pub(crate) one_hot_backend_prover_setup: Option<Arc<AkitaBackendProverSetup>>,
    pub(crate) prepared_one_hot_backend_setup: Option<Arc<AkitaBackendPreparedSetup>>,
    pub(crate) verifier: AkitaVerifierSetup,
}

impl AkitaProverSetup {
    pub fn max_num_vars(&self) -> usize {
        self.verifier.max_num_vars
    }

    pub fn max_num_polys_per_commitment_group(&self) -> usize {
        self.verifier.max_num_polys_per_commitment_group
    }

    pub fn default_layout_digest(&self) -> [u8; 32] {
        self.verifier.default_layout_digest
    }

    pub(crate) fn full_backend(
        &self,
    ) -> Result<(&AkitaBackendProverSetup, &AkitaBackendPreparedSetup), OpeningsError> {
        self.backend_prover_setup
            .as_deref()
            .zip(self.prepared_backend_setup.as_deref())
            .ok_or_else(|| {
                OpeningsError::InvalidSetup(
                    "this Akita setup was built without the full-flavor backend".to_string(),
                )
            })
    }

    pub(crate) fn one_hot_backend(
        &self,
    ) -> Result<(&AkitaBackendProverSetup, &AkitaBackendPreparedSetup), OpeningsError> {
        let backend = self
            .one_hot_backend_prover_setup
            .as_deref()
            .ok_or_else(|| invalid_batch("Akita setup has no one-hot backend"))?;
        let prepared = self
            .prepared_one_hot_backend_setup
            .as_deref()
            .ok_or_else(|| invalid_batch("Akita setup has no prepared one-hot backend"))?;
        Ok((backend, prepared))
    }
}

/// The verifier setup is pure shape: the backend keys are a deterministic
/// function of `(max_num_vars, max_num_polys_per_commitment_group)` over a
/// fixed internal seed, so they are never serialized or transcript-absorbed —
/// [`append_verifier_setup`] binds the shape and both sides derive the same
/// keys from it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaVerifierSetup {
    pub(crate) max_num_vars: usize,
    pub(crate) max_num_polys_per_commitment_group: usize,
    pub(crate) default_layout_digest: AkitaLayoutDigest,
    #[serde(skip)]
    pub(crate) backend_cache: BackendVerifierCache,
}

impl AkitaVerifierSetup {
    /// Primes the lazy key cache with freshly built backend keys, so
    /// in-process setups never pay the shape→key re-derivation.
    pub(crate) fn prime_backend_cache(
        &self,
        full: Option<AkitaBackendVerifier>,
        one_hot: Option<AkitaBackendVerifier>,
    ) {
        if let Some(full) = full {
            let _ = self.backend_cache.full.get_or_init(|| full);
        }
        if let Some(one_hot) = one_hot {
            let _ = self.backend_cache.one_hot.get_or_init(|| one_hot);
        }
    }

    /// Backend verifier key for `flavor`, cached after the first use.
    /// [`AkitaScheme::setup`](crate::AkitaScheme) primes the cache with the
    /// freshly built keys; a serde-transported setup re-derives them from the
    /// shape on first use (one-time, setup-class cost).
    pub(crate) fn backend_verifier(
        &self,
        flavor: AkitaBackendFlavor,
    ) -> Result<&AkitaBackendVerifier, OpeningsError> {
        let cache = match flavor {
            AkitaBackendFlavor::Full => &self.backend_cache.full,
            AkitaBackendFlavor::OneHot => &self.backend_cache.one_hot,
        };
        if let Some(verifier) = cache.get() {
            return Ok(verifier);
        }
        let verifier = self.build_backend_verifier(flavor)?;
        Ok(cache.get_or_init(|| verifier))
    }

    fn build_backend_verifier(
        &self,
        flavor: AkitaBackendFlavor,
    ) -> Result<AkitaBackendVerifier, OpeningsError> {
        let invalid_setup =
            |err: &dyn std::fmt::Display| OpeningsError::InvalidSetup(err.to_string());
        match flavor {
            AkitaBackendFlavor::Full => {
                let prover_setup = AkitaBackendScheme::setup_prover(
                    self.max_num_vars,
                    self.max_num_polys_per_commitment_group,
                )
                .map_err(|err| invalid_setup(&err))?;
                Ok(AkitaBackendScheme::setup_verifier(&prover_setup))
            }
            AkitaBackendFlavor::OneHot => {
                if self.max_num_vars < AKITA_ONE_HOT_LOG_K {
                    return Err(invalid_batch("Akita verifier setup has no one-hot backend"));
                }
                let prover_setup = AkitaOneHotBackendScheme::setup_prover(
                    self.max_num_vars,
                    self.max_num_polys_per_commitment_group,
                )
                .map_err(|err| invalid_setup(&err))?;
                Ok(AkitaOneHotBackendScheme::setup_verifier(&prover_setup))
            }
        }
    }
}

/// Lazily deserialized backend verifier keys. Derived state: ignored by
/// equality and skipped by serde; clones share the cache.
#[derive(Clone, Default)]
pub(crate) struct BackendVerifierCache {
    full: Arc<OnceLock<AkitaBackendVerifier>>,
    one_hot: Arc<OnceLock<AkitaBackendVerifier>>,
}

impl fmt::Debug for BackendVerifierCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BackendVerifierCache")
    }
}

impl PartialEq for BackendVerifierCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for BackendVerifierCache {}

/// Binds the setup key for one backend flavor into the transcript, by shape
/// only: the backend key is a deterministic function of the absorbed
/// dimensions over a fixed internal seed, so hashing the (large) serialized
/// key adds no binding.
pub(crate) fn append_verifier_setup<T: Transcript>(
    transcript: &mut T,
    setup: &AkitaVerifierSetup,
    flavor: AkitaBackendFlavor,
) {
    transcript.append(&Label(b"akita_setup_key"));
    transcript.append_bytes(b"akita/fp128/d64");
    transcript.append_bytes(flavor.transcript_label());
    transcript.append(&U64Word(AKITA_D as u64));
    transcript.append(&U64Word(setup.max_num_vars as u64));
    transcript.append(&U64Word(setup.max_num_polys_per_commitment_group as u64));
    transcript.append_bytes(&setup.default_layout_digest);
}

/// Binds the batch statement (commitment group, point, per-claim data) into
/// the transcript.
pub(crate) fn append_batch_statement<T: Transcript>(
    transcript: &mut T,
    statement: &[VerifierOpeningClaim<AkitaField, AkitaCommitment>],
    commitment: &AkitaCommitment,
    point: &[AkitaField],
) {
    transcript.append(&Label(b"akita_batch_statement"));
    commitment.append_to_transcript(transcript);
    transcript.append_values(b"akita_pcs_point", point);
    transcript.append(&LabelWithCount(b"akita_claims", statement.len() as u64));
    for claim in statement {
        claim.commitment.append_to_transcript(transcript);
        claim.evaluation.value.append_to_transcript(transcript);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AkitaBackendFlavor {
    #[default]
    Full,
    OneHot,
}

impl AkitaBackendFlavor {
    pub(crate) const fn transcript_label(self) -> &'static [u8] {
        match self {
            Self::Full => b"full",
            Self::OneHot => b"one_hot",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaCommitment {
    pub(crate) backend_flavor: AkitaBackendFlavor,
    pub(crate) layout_digest: AkitaLayoutDigest,
    pub(crate) num_vars: usize,
    pub(crate) poly_count: usize,
    /// Field-coefficient count of the serialized backend commitment — the
    /// deserialization context [`akita_types::Commitment`] requires.
    pub(crate) backend_coeff_len: usize,
    pub(crate) serialized_backend_bytes: Vec<u8>,
}

impl AkitaCommitment {
    pub fn backend_flavor(&self) -> AkitaBackendFlavor {
        self.backend_flavor
    }

    pub fn layout_digest(&self) -> [u8; 32] {
        self.layout_digest
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn poly_count(&self) -> usize {
        self.poly_count
    }
}

impl AppendToTranscript for AkitaCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"akita_commitment"));
        transcript.append_bytes(self.backend_flavor.transcript_label());
        transcript.append_bytes(&self.layout_digest);
        transcript.append(&U64Word(self.num_vars as u64));
        transcript.append(&U64Word(self.poly_count as u64));
        transcript.append(&U64Word(self.backend_coeff_len as u64));
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
    pub(crate) statement_bridge: Vec<u8>,
    pub(crate) serialized_akita_proof_shape: Vec<u8>,
    pub(crate) serialized_akita_proof: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    pub(crate) backend: Option<(AkitaBackendCommitment, AkitaBackendHint)>,
    pub(crate) polynomials: AkitaHintPolynomials,
}

/// Backend representation of the committed polynomials, produced at commit
/// time and reused when opening. The variant doubles as the source-kind
/// discriminator, so a hint can never pair one kind's metadata with another
/// kind's polynomials.
#[derive(Clone, Debug)]
pub(crate) enum AkitaHintPolynomials {
    Dense(Arc<[AkitaBackendDensePoly]>),
    OneHot(Arc<[AkitaBackendOneHotPoly]>),
    SparseUnit(Arc<[AkitaBackendSparsePoly]>),
}

impl Default for AkitaHintPolynomials {
    fn default() -> Self {
        Self::Dense(Vec::new().into())
    }
}

impl AkitaHintPolynomials {
    pub(crate) const fn backend_flavor(&self) -> AkitaBackendFlavor {
        match self {
            Self::Dense(_) | Self::SparseUnit(_) => AkitaBackendFlavor::Full,
            Self::OneHot(_) => AkitaBackendFlavor::OneHot,
        }
    }

    pub(crate) const fn kind(&self) -> &'static str {
        match self {
            Self::Dense(_) => "dense",
            Self::OneHot(_) => "one_hot",
            Self::SparseUnit(_) => "sparse_unit",
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Dense(polys) => polys.len(),
            Self::OneHot(polys) => polys.len(),
            Self::SparseUnit(polys) => polys.len(),
        }
    }
}

/// `2^num_vars`, or `None` when it does not fit in `usize`.
pub(crate) fn domain_size(num_vars: usize) -> Option<usize> {
    u32::try_from(num_vars)
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift))
}

#[doc(hidden)]
pub fn reverse_point(point: &[AkitaField]) -> Vec<AkitaField> {
    point.iter().rev().copied().collect()
}

pub(crate) fn backend_stack<'a>(
    backend_prover_setup: &'a AkitaBackendProverSetup,
    prepared_backend_setup: &'a AkitaBackendPreparedSetup,
) -> Result<BackendStack<'a>, OpeningsError> {
    let _span = info_span!("jolt_akita::make_backend_stack").entered();
    akita_prover::UniformProverStack::uniform(
        &CpuBackend,
        prepared_backend_setup,
        backend_prover_setup.expanded.as_ref(),
    )
    .map_err(|err| OpeningsError::InvalidSetup(err.to_string()))
}

pub(crate) fn one_hot_polynomial<P>(
    polynomial: &P,
) -> Result<Option<AkitaBackendOneHotPoly>, OpeningsError>
where
    P: MultilinearPoly<AkitaField> + ?Sized,
{
    if !polynomial.is_one_hot()
        || polynomial.one_hot_k() != Some(AKITA_ONE_HOT_K)
        || polynomial.one_hot_index_order() != Some(OneHotIndexOrder::RowMajor)
    {
        return Ok(None);
    }

    let indices = polynomial
        .one_hot_indices()
        .ok_or_else(|| invalid_batch("Jolt one-hot polynomial did not expose its indices"))?;
    AkitaBackendOneHotPoly::new(AKITA_ONE_HOT_K, AKITA_D, indices.to_vec())
        .map(Some)
        .map_err(akita_error)
}

pub(crate) fn sparse_unit_polynomial(
    num_vars: usize,
    indices: impl IntoIterator<Item = usize>,
) -> Result<AkitaBackendSparsePoly, OpeningsError> {
    let domain_size = domain_size(num_vars).ok_or_else(|| {
        invalid_batch(format!(
            "Akita sparse polynomial dimension {num_vars} exceeds usize bit width"
        ))
    })?;
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

    AkitaBackendSparsePoly::from_signed_coeffs(num_vars, AKITA_D, domain_size / AKITA_D, coeffs)
        .map_err(|error| {
            invalid_batch(format!(
                "Akita sparse polynomial construction failed: {error}"
            ))
        })
}

#[doc(hidden)]
pub fn jolt_to_akita_index(num_vars: usize, index: usize) -> usize {
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
            AkitaBackendDensePoly::from_field_evals(poly.num_vars(), AKITA_D, &evals)
                .map_err(akita_error)
        })
        .collect()
}

#[doc(hidden)]
pub fn jolt_to_akita_evals(
    num_vars: usize,
    jolt_evals: &[AkitaField],
) -> Result<Vec<AkitaField>, OpeningsError> {
    let Some(expected) = domain_size(num_vars) else {
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

/// Materializes a polynomial's evaluations directly in Akita's (bit-reversed)
/// index order, avoiding a second full-size buffer for the reorder pass.
pub(crate) fn akita_ordered_evaluations<P>(polynomial: &P) -> Result<Vec<AkitaField>, OpeningsError>
where
    P: MultilinearPoly<AkitaField> + ?Sized,
{
    let num_vars = polynomial.num_vars();
    let Some(len) = domain_size(num_vars) else {
        return Err(invalid_batch(format!(
            "Akita polynomial dimension {num_vars} exceeds usize bit width"
        )));
    };
    let mut evals = vec![AkitaField::zero(); len];
    let mut jolt_index = 0usize;
    polynomial.for_each_row(num_vars, &mut |_, row| {
        for &eval in row {
            evals[jolt_to_akita_index(num_vars, jolt_index)] = eval;
            jolt_index += 1;
        }
    });
    Ok(evals)
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

pub(crate) fn invalid_batch(message: impl Into<String>) -> OpeningsError {
    OpeningsError::InvalidBatch(message.into())
}

pub(crate) fn akita_error(error: impl ToString) -> OpeningsError {
    OpeningsError::InvalidBatch(error.to_string())
}

pub(crate) fn commit_failed(error: impl ToString) -> OpeningsError {
    OpeningsError::CommitFailed(error.to_string())
}

pub(crate) fn prove_failed(error: impl ToString) -> OpeningsError {
    OpeningsError::ProveFailed(error.to_string())
}

pub(crate) fn transparent_zk_error() -> OpeningsError {
    OpeningsError::InvalidBatch(
        "Akita backend adapter is transparent-only and does not support ZK openings yet".to_owned(),
    )
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
    bridge.to_bytes_le_vec()
}
