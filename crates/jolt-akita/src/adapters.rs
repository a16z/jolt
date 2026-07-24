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
use jolt_poly::{MultilinearPoly, OneHotIndexOrder, OneHotPolynomial, Polynomial};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use serde::{Deserialize, Serialize};
use tracing::info_span;

pub type AkitaField = akita_config::proof_optimized::fp128::Field;
pub(crate) type AkitaConfig = akita_config::proof_optimized::fp128::D64Dense;
pub(crate) type AkitaOneHotK16Config = crate::configs::JoltD64OneHotK16;
pub(crate) type AkitaOneHotK256Config = crate::configs::JoltD64OneHotK256;
pub(crate) const AKITA_D: usize = AkitaConfig::D;
pub const AKITA_ONE_HOT_K16: usize = 16;
pub const AKITA_ONE_HOT_K256: usize = 256;

pub(crate) type AkitaBackendExtField = <AkitaConfig as CommitmentConfig>::ExtField;

pub(crate) type AkitaBackendScheme = AkitaCommitmentScheme<AkitaConfig>;
pub(crate) type AkitaOneHotK16BackendScheme = AkitaCommitmentScheme<AkitaOneHotK16Config>;
pub(crate) type AkitaOneHotK256BackendScheme = AkitaCommitmentScheme<AkitaOneHotK256Config>;
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

/// Worker stack size for [`with_backend_pool`]. Stacks are lazily committed,
/// so oversizing costs virtual address space only.
const BACKEND_WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Runs `f` with rayon parallelism on a dedicated pool whose workers have
/// large stacks.
///
/// The Akita backend kernels recurse deeply inside rayon parallel iterators
/// (the bridge splitter re-splits whenever a job migrates to a stealing
/// worker, and the fold kernels carry large frames), which overflows rayon's
/// default 2 MiB worker stacks nondeterministically — observed as SIGABRT in
/// the packed prover at trace-scale shapes. Every backend setup/commit/
/// prove/verify entry funnels through this pool. Nested calls reuse it.
#[expect(
    clippy::expect_used,
    reason = "a pool that cannot spawn threads is an unrecoverable environment failure"
)]
pub(crate) fn with_backend_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .thread_name(|index| format!("jolt-akita-{index}"))
            .stack_size(BACKEND_WORKER_STACK_BYTES)
            .build()
            .expect("the Akita backend thread pool must build")
    })
    .install(f)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaSetupParams {
    pub(crate) max_num_vars: usize,
    pub(crate) max_num_polys_per_commitment_group: usize,
    pub(crate) default_layout_digest: AkitaLayoutDigest,
    pub(crate) one_hot_k: usize,
    /// When set, only the one-hot flavor's backend setup is built — the
    /// dense-flavor setup for the same shape is large and slow, and a packed
    /// one-hot commitment object never touches it.
    pub(crate) one_hot_only: bool,
    /// When set, only the dense flavor's backend setup is built — the one-hot
    /// flavor dominates the setup cost (~30x the dense flavor at advice
    /// shapes), and a sparse-unit or dense commitment object never touches it.
    pub(crate) dense_only: bool,
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
            one_hot_k: AKITA_ONE_HOT_K256,
            one_hot_only: false,
            dense_only: false,
        }
    }

    /// Setup parameters for a commitment object that only ever commits and
    /// opens through the one-hot flavor (the packed `OneHotTrace` group): skips
    /// building the dense-flavor backend setup of the same shape.
    pub fn one_hot_only(
        max_num_vars: usize,
        max_num_polys_per_commitment_group: usize,
        default_layout_digest: AkitaLayoutDigest,
        one_hot_k: usize,
    ) -> Self {
        Self {
            max_num_vars,
            max_num_polys_per_commitment_group,
            default_layout_digest,
            one_hot_k,
            one_hot_only: true,
            dense_only: false,
        }
    }

    /// Setup parameters for a commitment object that only ever commits and
    /// opens through the dense flavor (sparse-unit or dense polynomials, e.g.
    /// the advice byte columns and the precommitted program): skips building
    /// the one-hot backend setup of the same shape.
    pub fn dense_only(
        max_num_vars: usize,
        max_num_polys_per_commitment_group: usize,
        default_layout_digest: AkitaLayoutDigest,
    ) -> Self {
        Self {
            max_num_vars,
            max_num_polys_per_commitment_group,
            default_layout_digest,
            one_hot_k: AKITA_ONE_HOT_K256,
            one_hot_only: false,
            dense_only: true,
        }
    }

    pub fn one_hot_k(&self) -> usize {
        self.one_hot_k
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

    pub fn one_hot_k(&self) -> usize {
        self.verifier.one_hot_k
    }

    pub(crate) fn dense_backend(
        &self,
    ) -> Result<(&AkitaBackendProverSetup, &AkitaBackendPreparedSetup), OpeningsError> {
        self.backend_prover_setup
            .as_deref()
            .zip(self.prepared_backend_setup.as_deref())
            .ok_or_else(|| {
                OpeningsError::InvalidSetup(
                    "this Akita setup was built without the dense-flavor backend".to_string(),
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
/// function of `(max_num_vars, max_num_polys_per_commitment_group, one_hot_k)`
/// over a fixed internal seed, so they are never serialized or
/// transcript-absorbed — [`append_verifier_setup`] binds these parameters and
/// both sides derive the same keys from them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaVerifierSetup {
    pub(crate) max_num_vars: usize,
    pub(crate) max_num_polys_per_commitment_group: usize,
    pub(crate) default_layout_digest: AkitaLayoutDigest,
    pub(crate) one_hot_k: usize,
    #[serde(skip)]
    pub(crate) backend_cache: BackendVerifierCache,
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

    pub fn one_hot_k(&self) -> usize {
        self.one_hot_k
    }

    /// Primes the lazy key cache with freshly built backend keys, so
    /// in-process setups never pay the shape→key re-derivation.
    pub(crate) fn prime_backend_cache(
        &self,
        dense: Option<AkitaBackendVerifier>,
        one_hot: Option<AkitaBackendVerifier>,
    ) {
        if let Some(dense) = dense {
            let _ = self.backend_cache.dense.get_or_init(|| dense);
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
            AkitaBackendFlavor::Dense => &self.backend_cache.dense,
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
            AkitaBackendFlavor::Dense => {
                let prover_setup = with_backend_pool(|| {
                    AkitaBackendScheme::setup_prover(
                        self.max_num_vars,
                        self.max_num_polys_per_commitment_group,
                    )
                })
                .map_err(|err| invalid_setup(&err))?;
                with_backend_pool(|| AkitaBackendScheme::setup_verifier(&prover_setup))
                    .map_err(|err| invalid_setup(&err))
            }
            AkitaBackendFlavor::OneHot => {
                let log_k = validate_one_hot_k(self.one_hot_k)?;
                if self.max_num_vars < log_k {
                    return Err(invalid_batch("Akita verifier setup has no one-hot backend"));
                }
                let prover_setup = one_hot_setup_prover(
                    self.one_hot_k,
                    self.max_num_vars,
                    self.max_num_polys_per_commitment_group,
                )
                .map_err(|err| invalid_setup(&err))?;
                one_hot_setup_verifier(self.one_hot_k, &prover_setup)
            }
        }
    }
}

/// Lazily deserialized backend verifier keys. Derived state: ignored by
/// equality and skipped by serde; clones share the cache.
#[derive(Clone, Default)]
pub(crate) struct BackendVerifierCache {
    dense: Arc<OnceLock<AkitaBackendVerifier>>,
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
    transcript.append(&U64Word(setup.one_hot_k as u64));
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
    Dense,
    OneHot,
}

impl AkitaBackendFlavor {
    pub(crate) const fn transcript_label(self) -> &'static [u8] {
        match self {
            Self::Dense => b"dense",
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
    pub(crate) one_hot_k: usize,
    /// Field-coefficient count of the serialized backend commitment — the
    /// deserialization context [`akita_types::Commitment`] requires.
    pub(crate) backend_coeff_len: usize,
    pub(crate) serialized_backend_bytes: Vec<u8>,
}

impl jolt_openings::GroupCommitmentMetadata for AkitaCommitment {
    fn is_one_hot_backend(&self) -> bool {
        self.backend_flavor() == AkitaBackendFlavor::OneHot
    }

    fn layout_digest(&self) -> [u8; 32] {
        self.layout_digest()
    }

    fn num_vars(&self) -> usize {
        self.num_vars()
    }

    fn poly_count(&self) -> usize {
        self.poly_count()
    }

    fn one_hot_k(&self) -> usize {
        self.one_hot_k()
    }
}

impl jolt_openings::GroupSetupMetadata for AkitaVerifierSetup {
    fn max_num_vars(&self) -> usize {
        self.max_num_vars()
    }

    fn max_num_polys_per_commitment_group(&self) -> usize {
        self.max_num_polys_per_commitment_group()
    }

    fn default_layout_digest(&self) -> [u8; 32] {
        self.default_layout_digest()
    }

    fn one_hot_k(&self) -> usize {
        self.one_hot_k()
    }
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

    pub fn one_hot_k(&self) -> usize {
        self.one_hot_k
    }
}

impl AppendToTranscript for AkitaCommitment {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&Label(b"akita_commitment"));
        transcript.append_bytes(self.backend_flavor.transcript_label());
        transcript.append_bytes(&self.layout_digest);
        transcript.append(&U64Word(self.num_vars as u64));
        transcript.append(&U64Word(self.poly_count as u64));
        transcript.append(&U64Word(self.one_hot_k as u64));
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
            Self::Dense(_) | Self::SparseUnit(_) => AkitaBackendFlavor::Dense,
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

    pub(crate) fn one_hot_k(&self) -> Option<usize> {
        match self {
            Self::OneHot(polys) => polys
                .first()
                .and_then(akita_prover::RootPolyMeta::onehot_chunk_size),
            Self::Dense(_) | Self::SparseUnit(_) => None,
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
    one_hot_k: usize,
) -> Result<Option<AkitaBackendOneHotPoly>, OpeningsError>
where
    P: MultilinearPoly<AkitaField> + ?Sized,
{
    if !polynomial.is_one_hot()
        || polynomial.one_hot_k() != Some(one_hot_k)
        || polynomial.one_hot_index_order() != Some(OneHotIndexOrder::RowMajor)
    {
        return Ok(None);
    }

    let indices = polynomial
        .one_hot_indices()
        .ok_or_else(|| invalid_batch("Jolt one-hot polynomial did not expose its indices"))?;
    let _ = validate_one_hot_k(one_hot_k)?;
    AkitaBackendOneHotPoly::new(one_hot_k, AKITA_D, indices.to_vec())
        .map(Some)
        .map_err(akita_error)
}

pub(crate) fn owned_one_hot_polynomial(
    polynomial: OneHotPolynomial,
    one_hot_k: usize,
) -> Result<AkitaBackendOneHotPoly, OpeningsError> {
    if polynomial.k() != one_hot_k || polynomial.index_order() != OneHotIndexOrder::RowMajor {
        return Err(invalid_batch(format!(
            "Akita owned one-hot polynomial requires row-major K={one_hot_k}"
        )));
    }
    let _ = validate_one_hot_k(one_hot_k)?;
    AkitaBackendOneHotPoly::new(one_hot_k, AKITA_D, polynomial.into_indices()).map_err(akita_error)
}

pub(crate) fn validate_one_hot_k(one_hot_k: usize) -> Result<usize, OpeningsError> {
    match one_hot_k {
        AKITA_ONE_HOT_K16 => Ok(4),
        AKITA_ONE_HOT_K256 => Ok(8),
        _ => Err(invalid_batch(format!(
            "Akita one-hot chunk size must be 16 or 256, got {one_hot_k}"
        ))),
    }
}

pub(crate) fn one_hot_setup_prover(
    one_hot_k: usize,
    max_num_vars: usize,
    max_num_polys: usize,
) -> Result<AkitaBackendProverSetup, akita_pcs::AkitaError> {
    with_backend_pool(|| match one_hot_k {
        AKITA_ONE_HOT_K16 => AkitaOneHotK16BackendScheme::setup_prover(max_num_vars, max_num_polys),
        AKITA_ONE_HOT_K256 => {
            AkitaOneHotK256BackendScheme::setup_prover(max_num_vars, max_num_polys)
        }
        _ => unreachable!("one-hot K is validated before backend setup"),
    })
}

pub(crate) fn one_hot_setup_verifier(
    one_hot_k: usize,
    prover_setup: &AkitaBackendProverSetup,
) -> Result<AkitaBackendVerifier, OpeningsError> {
    let invalid_setup = |err: &dyn std::fmt::Display| OpeningsError::InvalidSetup(err.to_string());
    match one_hot_k {
        AKITA_ONE_HOT_K16 => {
            with_backend_pool(|| AkitaOneHotK16BackendScheme::setup_verifier(prover_setup))
                .map_err(|err| invalid_setup(&err))
        }
        AKITA_ONE_HOT_K256 => {
            with_backend_pool(|| AkitaOneHotK256BackendScheme::setup_verifier(prover_setup))
                .map_err(|err| invalid_setup(&err))
        }
        _ => Err(invalid_batch(format!(
            "Akita one-hot chunk size must be 16 or 256, got {one_hot_k}"
        ))),
    }
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
    let mut cursor = Cursor::new(bytes);
    let value = T::deserialize_compressed(&mut cursor, ctx).map_err(akita_error)?;
    if cursor.position() != bytes.len() as u64 {
        return Err(invalid_batch(
            "Akita payload has trailing bytes after deserialization",
        ));
    }
    Ok(value)
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

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        clippy::unwrap_used,
        reason = "tests assert successful conversions and exact error text"
    )]

    use super::*;

    fn af(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    /// Jolt indexes MLE evaluations big-endian (variable `j` carries index
    /// weight `2^(n-1-j)`, see the `eq_table` convention in `scheme.rs`);
    /// Akita indexes them little-endian (variable `j` carries weight `2^j`).
    /// The tables below are derived by hand from those weight conventions —
    /// e.g. for n = 3, jolt index 1 is the assignment (0, 0, 1), whose Akita
    /// index is 1 * 2^2 = 4 — not by re-running any bit arithmetic.
    #[test]
    fn jolt_to_akita_index_matches_hand_derived_tables() {
        let three_vars = [0, 4, 2, 6, 1, 5, 3, 7];
        for (jolt_index, &akita_index) in three_vars.iter().enumerate() {
            assert_eq!(
                jolt_to_akita_index(3, jolt_index),
                akita_index,
                "num_vars=3, jolt index {jolt_index}",
            );
        }

        let two_vars = [0, 2, 1, 3];
        for (jolt_index, &akita_index) in two_vars.iter().enumerate() {
            assert_eq!(
                jolt_to_akita_index(2, jolt_index),
                akita_index,
                "num_vars=2, jolt index {jolt_index}",
            );
        }

        assert_eq!(jolt_to_akita_index(1, 0), 0);
        assert_eq!(jolt_to_akita_index(1, 1), 1);
        assert_eq!(jolt_to_akita_index(0, 0), 0);
    }

    /// Reversing a reversal is the identity, and the map permutes the whole
    /// domain (every Akita index is hit exactly once).
    #[test]
    fn jolt_to_akita_index_is_a_self_inverse_permutation() {
        let num_vars = 4;
        let mut seen = [false; 16];
        for index in 0..16 {
            let mapped = jolt_to_akita_index(num_vars, index);
            assert!(mapped < 16);
            assert!(!seen[mapped], "akita index {mapped} hit twice");
            seen[mapped] = true;
            assert_eq!(jolt_to_akita_index(num_vars, mapped), index);
        }
    }

    #[test]
    fn jolt_to_akita_evals_permutes_an_explicit_two_var_vector() {
        let jolt = [af(10), af(20), af(30), af(40)];
        let akita = jolt_to_akita_evals(2, &jolt).expect("well-formed evaluations convert");
        // Jolt index 1 = assignment (0, 1) = Akita index 2, and vice versa;
        // the all-zero and all-one corners are fixed points.
        assert_eq!(akita, vec![af(10), af(30), af(20), af(40)]);
    }

    #[test]
    fn jolt_to_akita_evals_passes_zero_var_polynomials_through() {
        let jolt = [af(99)];
        assert_eq!(
            jolt_to_akita_evals(0, &jolt).expect("constant polynomial converts"),
            vec![af(99)],
        );
    }

    #[test]
    fn jolt_to_akita_evals_rejects_length_domain_mismatch() {
        let error = jolt_to_akita_evals(2, &[af(1), af(2), af(3)]).unwrap_err();
        assert!(matches!(error, OpeningsError::InvalidBatch(_)));
        assert_eq!(
            error.to_string(),
            "invalid batch opening: Akita polynomial has 3 evaluations but dimension 2 requires 4",
        );
    }

    #[test]
    fn jolt_to_akita_evals_rejects_dimension_beyond_usize_width() {
        let error = jolt_to_akita_evals(usize::BITS as usize, &[]).unwrap_err();
        assert!(matches!(error, OpeningsError::InvalidBatch(_)));
        assert_eq!(
            error.to_string(),
            format!(
                "invalid batch opening: Akita polynomial dimension {} exceeds usize bit width",
                usize::BITS
            ),
        );
    }

    #[test]
    fn reverse_point_reverses_coordinates_and_round_trips() {
        let point = vec![af(1), af(2), af(3)];
        assert_eq!(reverse_point(&point), vec![af(3), af(2), af(1)]);
        assert_eq!(reverse_point(&reverse_point(&point)), point);
        assert!(reverse_point(&[]).is_empty());
    }

    /// The identity the backend hand-off relies on: transforming the
    /// evaluations with `jolt_to_akita_evals` AND the opening point with
    /// `reverse_point` leaves the multilinear evaluation unchanged. Checked
    /// against a hand-rolled big-endian MLE evaluator, so a bug in either
    /// transform (or applying only one of them) fails this test.
    #[test]
    fn eval_and_point_transforms_together_preserve_mle_evaluation() {
        fn mle_big_endian(evals: &[AkitaField], point: &[AkitaField]) -> AkitaField {
            let one = af(1);
            let mut acc = af(0);
            for (index, &eval) in evals.iter().enumerate() {
                let mut weight = one;
                for (variable, &coordinate) in point.iter().enumerate() {
                    let bit = (index >> (point.len() - 1 - variable)) & 1;
                    weight *= if bit == 1 {
                        coordinate
                    } else {
                        one - coordinate
                    };
                }
                acc += weight * eval;
            }
            acc
        }

        let evals: Vec<AkitaField> = (0..8).map(|value| af(100 + 7 * value)).collect();
        let point = vec![af(3), af(17), af(29)];
        let transformed = jolt_to_akita_evals(3, &evals).expect("well-formed evaluations convert");

        assert_eq!(
            mle_big_endian(&transformed, &reverse_point(&point)),
            mle_big_endian(&evals, &point),
        );
        // Applying only the evaluation transform must NOT preserve the value
        // at this off-hypercube point — otherwise the check above is vacuous.
        assert_ne!(
            mle_big_endian(&transformed, &point),
            mle_big_endian(&evals, &point),
        );
    }

    #[test]
    fn sparse_unit_polynomial_rejects_malformed_index_sets() {
        let err = sparse_unit_polynomial(usize::BITS as usize, [0])
            .expect_err("2^64 domain must overflow");
        assert!(
            matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("bit width")),
            "unexpected error: {err}"
        );

        let err =
            sparse_unit_polynomial(3, [0]).expect_err("domain below the ring dimension rejects");
        assert!(
            matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("smaller than ring dimension")),
            "unexpected error: {err}"
        );

        let err = sparse_unit_polynomial(6, [64]).expect_err("out-of-domain index rejects");
        assert!(
            matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("outside domain size 64")),
            "unexpected error: {err}"
        );

        let err = sparse_unit_polynomial(6, [3, 3]).expect_err("duplicate index rejects");
        assert!(
            matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("more than once")),
            "unexpected error: {err}"
        );
    }

    struct HugeDimensionPoly;

    impl MultilinearPoly<AkitaField> for HugeDimensionPoly {
        fn num_vars(&self) -> usize {
            usize::BITS as usize
        }

        fn evaluate(&self, _point: &[AkitaField]) -> AkitaField {
            unreachable!("the dimension check rejects before any evaluation")
        }

        fn for_each_row(&self, _sigma: usize, _f: &mut dyn FnMut(usize, &[AkitaField])) {
            unreachable!("the dimension check rejects before any row is streamed")
        }
    }

    #[test]
    fn akita_ordered_evaluations_rejects_unrepresentable_domains() {
        let err = akita_ordered_evaluations(&HugeDimensionPoly)
            .expect_err("2^64 evaluation domain must overflow");
        assert!(
            matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("bit width")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn deserialize_akita_rejects_trailing_bytes() {
        let shape = akita_types::LevelProofShape {
            extension_opening_reduction: None,
            v_coeffs: 3,
            stage1_stages: Vec::new(),
            stage2_sumcheck_proof: vec![3, 3],
            stage3_sumcheck: None,
            next_witness_binding: akita_types::NextWitnessBindingShape::TerminalInnerState,
        };
        let mut bytes = serialize_akita(&shape).expect("shape serializes");
        let roundtrip: akita_types::LevelProofShape =
            deserialize_akita(&bytes, &()).expect("exact bytes deserialize");
        assert_eq!(roundtrip, shape);

        bytes.push(0);
        let err = deserialize_akita::<akita_types::LevelProofShape>(&bytes, &())
            .expect_err("trailing bytes must be rejected");
        assert!(
            matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("trailing bytes")),
            "unexpected error: {err}"
        );
    }
}
