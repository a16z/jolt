use std::{
    fmt,
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
    sync::OnceLock,
};

#[cfg(feature = "profiling")]
use std::{cell::Cell, num::NonZeroUsize};

use akita_config::CommitmentConfig;
use akita_pcs::{
    AkitaCommitmentScheme, AkitaDeserialize, AkitaError, AkitaSerialize, AkitaTranscript,
};
use akita_prover::{CpuBackend, CpuPreparedSetup, DensePoly, OneHotPoly};
use akita_schedules::TrustedScheduleCatalog;
use akita_types::{
    AkitaBatchedProof as AkitaBackendBatchProof, AkitaBatchedProofShape,
    AkitaCommitmentHint as AkitaBackendCommitmentHint,
    AkitaVerifierSetup as AkitaBackendVerifierSetup, Commitment as AkitaBackendRingCommitment,
    CommittedGroup as AkitaBackendCommittedGroup, OpeningScheduleSelection, ScheduleRowDigest,
};
use jolt_field::{CanonicalBytes, Zero};
use jolt_openings::{OpeningsError, VerifierOpeningClaim};
use jolt_poly::{MultilinearPoly, OneHotIndexOrder, OneHotPolynomial, Polynomial};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};
use tracing::info_span;

use crate::configs::{JoltDenseBounded, JoltOneHotK16, JoltOneHotK256};
use crate::schedule_registry::PrecommittedScheduleParams;
use crate::trace_onehot::TracePackedOneHot;

pub type AkitaField = akita_config::proof_optimized::fp128::Field;
pub(crate) type AkitaConfig = JoltDenseBounded;
pub(crate) type AkitaOneHotK16Config = JoltOneHotK16;
pub(crate) type AkitaOneHotK256Config = JoltOneHotK256;
/// Smallest A dimension accepted by the delegated adaptive policy. Source
/// objects use this only for dimension-independent flat storage metadata;
/// each generated schedule still selects its exact per-role dimensions.
pub(crate) const AKITA_SOURCE_RING_DIMENSION: usize =
    akita_config::proof_optimized::fp128::Dense::A_RING_DIMENSIONS[0];
const _: () = assert!(
    AKITA_SOURCE_RING_DIMENSION
        == akita_config::proof_optimized::fp128::OneHot::A_RING_DIMENSIONS[0]
);
pub const AKITA_ONE_HOT_K16: usize = 16;
pub const AKITA_ONE_HOT_K256: usize = 256;

/// Runtime bytes for Jolt's three base schedule families.
///
/// These bytes are ordinary input data. They are intentionally neither
/// generated Rust nor embedded with `include_bytes!`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaScheduleArtifacts {
    dense: Vec<u8>,
    one_hot_k16: Vec<u8>,
    one_hot_k256: Vec<u8>,
}

impl AkitaScheduleArtifacts {
    pub const DIRECTORY_ENV: &'static str = "JOLT_AKITA_SCHEDULE_DIR";

    pub fn new(dense: Vec<u8>, one_hot_k16: Vec<u8>, one_hot_k256: Vec<u8>) -> Self {
        Self {
            dense,
            one_hot_k16,
            one_hot_k256,
        }
    }

    /// Load Jolt's checked-in artifacts from a normal filesystem directory.
    pub fn from_directory(directory: impl AsRef<Path>) -> Result<Self, OpeningsError> {
        let directory = directory.as_ref();
        let read = |family: &str| {
            let path = directory.join(format!("{family}.aks"));
            std::fs::read(&path).map_err(|error| {
                OpeningsError::InvalidSetup(format!(
                    "read Akita schedule artifact {}: {error}",
                    path.display()
                ))
            })
        };
        Ok(Self::new(
            read(JoltDenseBounded::schedule_family_name())?,
            read(JoltOneHotK16::schedule_family_name())?,
            read(JoltOneHotK256::schedule_family_name())?,
        ))
    }

    /// Host/dev helper that loads from `JOLT_AKITA_SCHEDULE_DIR`, or from this
    /// crate's packaged `schedules/` directory when the variable is unset.
    ///
    /// Application preprocessing should call this (or [`Self::from_directory`])
    /// once and pass the resulting immutable bundle explicitly to every setup.
    /// Protocol setup and verification never consult the environment.
    pub fn from_default_directory() -> Result<Self, OpeningsError> {
        let directory = std::env::var_os(Self::DIRECTORY_ENV).map_or_else(
            || Path::new(env!("CARGO_MANIFEST_DIR")).join("schedules"),
            PathBuf::from,
        );
        Self::from_directory(directory)
    }

    pub fn dense_catalog(&self) -> Result<TrustedScheduleCatalog, AkitaError> {
        akita_config::trusted_schedule_catalog_from_bytes::<JoltDenseBounded>(&self.dense)
    }

    pub fn one_hot_catalog(&self, one_hot_k: usize) -> Result<TrustedScheduleCatalog, AkitaError> {
        match one_hot_k {
            AKITA_ONE_HOT_K16 => {
                akita_config::trusted_schedule_catalog_from_bytes::<JoltOneHotK16>(
                    &self.one_hot_k16,
                )
            }
            AKITA_ONE_HOT_K256 => {
                akita_config::trusted_schedule_catalog_from_bytes::<JoltOneHotK256>(
                    &self.one_hot_k256,
                )
            }
            other => Err(AkitaError::InvalidSetup(format!(
                "unsupported Akita one-hot K={other}"
            ))),
        }
    }
}

pub(crate) type AkitaBackendExtField = <AkitaConfig as CommitmentConfig>::ExtField;

pub(crate) type AkitaBackendScheme = AkitaCommitmentScheme<AkitaConfig>;
pub(crate) type AkitaOneHotK16BackendScheme = AkitaCommitmentScheme<AkitaOneHotK16Config>;
pub(crate) type AkitaOneHotK256BackendScheme = AkitaCommitmentScheme<AkitaOneHotK256Config>;
pub(crate) type AkitaBackendCommitment = AkitaBackendCommittedGroup<AkitaField>;
pub(crate) type AkitaBackendCommitmentPayload = AkitaBackendRingCommitment<AkitaField>;
pub(crate) type AkitaBackendHint = AkitaBackendCommitmentHint<AkitaField>;
pub(crate) type AkitaBackendProof = AkitaBackendBatchProof<AkitaField, AkitaBackendExtField>;
pub(crate) type AkitaBackendProofShape = AkitaBatchedProofShape;
pub(crate) type AkitaBackendVerifier = AkitaBackendVerifierSetup<AkitaField>;
pub(crate) type AkitaBackendDensePoly = DensePoly<AkitaField>;
pub(crate) type AkitaBackendOneHotPoly = OneHotPoly<AkitaField, u8>;
pub(crate) type AkitaBackendPreparedSetup = CpuPreparedSetup<AkitaField>;
pub(crate) type AkitaBackendProverSetup = akita_prover::AkitaProverSetup<AkitaField>;
pub(crate) type BackendStack<'a> = akita_prover::UniformProverStack<'a, AkitaField, CpuBackend>;

pub(crate) type AkitaLayoutDigest = [u8; 32];
const SCHEDULE_SELECTION_BYTES: usize = 32;

/// Worker stack size for [`with_backend_pool`]. Stacks are lazily committed,
/// so oversizing costs virtual address space only.
const BACKEND_WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

#[expect(
    clippy::expect_used,
    reason = "a pool that cannot spawn threads is an unrecoverable environment failure"
)]
fn build_backend_pool(name: &'static str, num_threads: Option<usize>) -> ThreadPool {
    let mut builder = ThreadPoolBuilder::new()
        .thread_name(move |index| format!("{name}-{index}"))
        .stack_size(BACKEND_WORKER_STACK_BYTES);
    if let Some(num_threads) = num_threads {
        builder = builder.num_threads(num_threads);
    }
    builder
        .build()
        .expect("the Akita backend thread pool must build")
}

fn backend_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| build_backend_pool("jolt-akita", None))
}

#[cfg(feature = "profiling")]
fn host_parallel_verifier_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let num_threads = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        build_backend_pool("jolt-akita-verify-parallel", Some(num_threads))
    })
}

#[cfg(feature = "profiling")]
fn single_threaded_verifier_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| build_backend_pool("jolt-akita-verify-single", Some(1)))
}

#[cfg(feature = "profiling")]
#[derive(Clone, Copy)]
enum ProfileBackendPool {
    Default,
    HostParallel,
    SingleThreaded,
}

#[cfg(feature = "profiling")]
thread_local! {
    static PROFILE_BACKEND_POOL: Cell<ProfileBackendPool> = const {
        Cell::new(ProfileBackendPool::Default)
    };
}

#[cfg(feature = "profiling")]
struct ProfileBackendPoolGuard(ProfileBackendPool);

#[cfg(feature = "profiling")]
impl Drop for ProfileBackendPoolGuard {
    fn drop(&mut self) {
        PROFILE_BACKEND_POOL.with(|pool| pool.set(self.0));
    }
}

#[cfg(feature = "profiling")]
fn with_profile_backend_pool<R>(selection: ProfileBackendPool, f: impl FnOnce() -> R) -> R {
    let previous = PROFILE_BACKEND_POOL.with(|pool| pool.replace(selection));
    let _guard = ProfileBackendPoolGuard(previous);
    f()
}

/// Runs verifier backend calls in `f` on an explicit host-sized pool.
#[cfg(feature = "profiling")]
#[doc(hidden)]
pub fn with_host_parallel_verifier_backend<R>(f: impl FnOnce() -> R) -> R {
    let _ = host_parallel_verifier_pool();
    with_profile_backend_pool(ProfileBackendPool::HostParallel, f)
}

/// Runs verifier backend calls in `f` on exactly one worker.
#[cfg(feature = "profiling")]
#[doc(hidden)]
pub fn with_single_threaded_verifier_backend<R>(f: impl FnOnce() -> R) -> R {
    let _ = single_threaded_verifier_pool();
    with_profile_backend_pool(ProfileBackendPool::SingleThreaded, f)
}

#[cfg(feature = "profiling")]
#[doc(hidden)]
pub fn host_parallel_verifier_threads() -> usize {
    host_parallel_verifier_pool().current_num_threads()
}

/// Runs `f` with rayon parallelism on a dedicated pool whose workers have
/// large stacks.
///
/// The Akita backend kernels recurse deeply inside rayon parallel iterators
/// (the bridge splitter re-splits whenever a job migrates to a stealing
/// worker, and the fold kernels carry large frames), which overflows rayon's
/// default 2 MiB worker stacks nondeterministically — observed as SIGABRT in
/// the packed prover at trace-scale shapes. Every backend setup/commit/
/// prove/verify entry funnels through this pool. Nested calls reuse it.
pub(crate) fn with_backend_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    #[cfg(feature = "profiling")]
    match PROFILE_BACKEND_POOL.with(Cell::get) {
        ProfileBackendPool::HostParallel => return host_parallel_verifier_pool().install(f),
        ProfileBackendPool::SingleThreaded => {
            return single_threaded_verifier_pool().install(f);
        }
        ProfileBackendPool::Default => {}
    }
    backend_pool().install(f)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaSetupParams {
    pub(crate) max_num_vars: usize,
    pub(crate) max_num_polys_per_commitment_group: usize,
    /// Capacity of the complete ordered group batch. This is passed to
    /// Akita's setup constructor; commitment entry points still enforce the
    /// separate group-local limit above.
    pub(crate) max_total_batch_polys: usize,
    pub(crate) default_layout_digest: AkitaLayoutDigest,
    pub(crate) one_hot_k: usize,
    pub(crate) flavor: AkitaSetupFlavor,
    /// Recipe for the dynamic grouped rows accepted by this setup.
    ///
    /// Replaying serialized setup parameters intentionally reruns guided
    /// preprocessing. Verifier transport serializes [`AkitaVerifierSetup`]
    /// instead, which contains the finalized catalog and never replans.
    #[serde(default, rename = "advice_schedule")]
    pub(crate) precommitted_schedule: Option<PrecommittedScheduleParams>,
    /// Immutable base catalogs loaded once by application preprocessing.
    pub(crate) schedule_artifacts: Arc<AkitaScheduleArtifacts>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum AkitaSetupFlavor {
    Both,
    OneHot,
    Dense,
}

impl AkitaSetupParams {
    pub fn new(
        max_num_vars: usize,
        max_num_polys_per_commitment_group: usize,
        default_layout_digest: AkitaLayoutDigest,
        schedule_artifacts: Arc<AkitaScheduleArtifacts>,
    ) -> Self {
        Self {
            max_num_vars,
            max_num_polys_per_commitment_group,
            max_total_batch_polys: max_num_polys_per_commitment_group,
            default_layout_digest,
            one_hot_k: AKITA_ONE_HOT_K256,
            flavor: AkitaSetupFlavor::Both,
            precommitted_schedule: None,
            schedule_artifacts,
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
        schedule_artifacts: Arc<AkitaScheduleArtifacts>,
    ) -> Self {
        Self {
            max_num_vars,
            max_num_polys_per_commitment_group,
            max_total_batch_polys: max_num_polys_per_commitment_group,
            default_layout_digest,
            one_hot_k,
            flavor: AkitaSetupFlavor::OneHot,
            precommitted_schedule: None,
            schedule_artifacts,
        }
    }

    /// Shape-exact one-hot final setup that can discharge a heterogeneous
    /// opening containing independently committed prefix groups.
    pub fn one_hot_only_grouped(
        max_num_vars: usize,
        max_num_polys_per_commitment_group: usize,
        max_total_batch_polys: usize,
        default_layout_digest: AkitaLayoutDigest,
        one_hot_k: usize,
        precommitted_schedule: Option<PrecommittedScheduleParams>,
        schedule_artifacts: Arc<AkitaScheduleArtifacts>,
    ) -> Self {
        Self {
            max_num_vars,
            max_num_polys_per_commitment_group,
            max_total_batch_polys,
            default_layout_digest,
            one_hot_k,
            flavor: AkitaSetupFlavor::OneHot,
            precommitted_schedule,
            schedule_artifacts,
        }
    }

    /// Setup parameters for objects that use only the dense flavor, omitting
    /// the one-hot backend setup.
    pub fn dense_only(
        max_num_vars: usize,
        max_num_polys_per_commitment_group: usize,
        default_layout_digest: AkitaLayoutDigest,
        schedule_artifacts: Arc<AkitaScheduleArtifacts>,
    ) -> Self {
        Self {
            max_num_vars,
            max_num_polys_per_commitment_group,
            max_total_batch_polys: max_num_polys_per_commitment_group,
            default_layout_digest,
            one_hot_k: AKITA_ONE_HOT_K256,
            flavor: AkitaSetupFlavor::Dense,
            precommitted_schedule: None,
            schedule_artifacts,
        }
    }

    pub fn one_hot_k(&self) -> usize {
        self.one_hot_k
    }

    pub fn max_total_batch_polys(&self) -> usize {
        self.max_total_batch_polys
    }
}

#[derive(Clone, Debug)]
pub struct AkitaProverSetup {
    pub(crate) backend_prover_setup: Option<Arc<AkitaBackendProverSetup>>,
    pub(crate) prepared_backend_setup: Option<Arc<AkitaBackendPreparedSetup>>,
    pub(crate) one_hot_backend_prover_setup: Option<Arc<AkitaBackendProverSetup>>,
    pub(crate) prepared_one_hot_backend_setup: Option<Arc<AkitaBackendPreparedSetup>>,
    pub(crate) schedule_artifacts: Arc<AkitaScheduleArtifacts>,
    pub(crate) verifier: AkitaVerifierSetup,
}

impl AkitaProverSetup {
    pub fn max_num_vars(&self) -> usize {
        self.verifier.max_num_vars
    }

    pub fn max_num_polys_per_commitment_group(&self) -> usize {
        self.verifier.max_num_polys_per_commitment_group
    }

    pub fn max_total_batch_polys(&self) -> usize {
        self.verifier.max_total_batch_polys
    }

    pub fn default_layout_digest(&self) -> [u8; 32] {
        self.verifier.default_layout_digest
    }

    pub fn one_hot_k(&self) -> usize {
        self.verifier.one_hot_k
    }

    /// Releases transformed setup slots after the trace commitment. Later
    /// opening work rebuilds the slots on first use.
    pub fn release_post_commit_ntt_residency(&self) -> Result<(), OpeningsError> {
        for prepared in [
            self.prepared_backend_setup.as_deref(),
            self.prepared_one_hot_backend_setup.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            let _ = prepared
                .drop_built_ntt_slots()
                .map_err(|error| OpeningsError::InvalidSetup(error.to_string()))?;
        }
        Ok(())
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

/// Serializable public inputs for deriving backend keys.
///
/// The exact validated schedule artifacts are serialized; derived scheme
/// objects, prepared keys, and caches are rebuilt lazily after transport.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaVerifierSetup {
    pub(crate) max_num_vars: usize,
    pub(crate) max_num_polys_per_commitment_group: usize,
    pub(crate) max_total_batch_polys: usize,
    pub(crate) default_layout_digest: AkitaLayoutDigest,
    pub(crate) one_hot_k: usize,
    /// Exact setup-owned catalogs, including any program-specific grouped rows.
    pub(crate) schedule_artifacts: AkitaVerifierScheduleArtifacts,
    #[serde(skip)]
    pub(crate) backend_cache: BackendVerifierCache,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub(crate) enum AkitaVerifierScheduleArtifacts {
    Dense { dense: Vec<u8> },
    OneHot { one_hot: Vec<u8> },
    Both { dense: Vec<u8>, one_hot: Vec<u8> },
}

impl AkitaVerifierScheduleArtifacts {
    fn dense(&self) -> Option<&[u8]> {
        match self {
            Self::Dense { dense } | Self::Both { dense, .. } => Some(dense),
            Self::OneHot { .. } => None,
        }
    }

    fn one_hot(&self) -> Option<&[u8]> {
        match self {
            Self::OneHot { one_hot } | Self::Both { one_hot, .. } => Some(one_hot),
            Self::Dense { .. } => None,
        }
    }
}

impl AkitaVerifierSetup {
    pub fn max_num_vars(&self) -> usize {
        self.max_num_vars
    }

    pub fn max_num_polys_per_commitment_group(&self) -> usize {
        self.max_num_polys_per_commitment_group
    }

    pub fn max_total_batch_polys(&self) -> usize {
        self.max_total_batch_polys
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

    pub(crate) fn dense_scheme(&self) -> Result<&AkitaBackendScheme, OpeningsError> {
        let result = self.backend_cache.dense_scheme.get_or_init(|| {
            self.schedule_artifacts
                .dense()
                .ok_or_else(|| "Akita verifier setup has no dense schedule artifact".to_string())
                .and_then(|bytes| {
                    AkitaBackendScheme::from_schedule_artifact(bytes)
                        .map_err(|error| error.to_string())
                })
        });
        result
            .as_ref()
            .map_err(|error| OpeningsError::InvalidSetup(error.clone()))
    }

    pub(crate) fn one_hot_k16_scheme(&self) -> Result<&AkitaOneHotK16BackendScheme, OpeningsError> {
        let result = self.backend_cache.one_hot_k16_scheme.get_or_init(|| {
            self.schedule_artifacts
                .one_hot()
                .ok_or_else(|| "Akita verifier setup has no one-hot schedule artifact".to_string())
                .and_then(|bytes| {
                    AkitaOneHotK16BackendScheme::from_schedule_artifact(bytes)
                        .map_err(|error| error.to_string())
                })
        });
        result
            .as_ref()
            .map_err(|error| OpeningsError::InvalidSetup(error.clone()))
    }

    pub(crate) fn one_hot_k256_scheme(
        &self,
    ) -> Result<&AkitaOneHotK256BackendScheme, OpeningsError> {
        let result = self.backend_cache.one_hot_k256_scheme.get_or_init(|| {
            self.schedule_artifacts
                .one_hot()
                .ok_or_else(|| "Akita verifier setup has no one-hot schedule artifact".to_string())
                .and_then(|bytes| {
                    AkitaOneHotK256BackendScheme::from_schedule_artifact(bytes)
                        .map_err(|error| error.to_string())
                })
        });
        result
            .as_ref()
            .map_err(|error| OpeningsError::InvalidSetup(error.clone()))
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
                let scheme = self.dense_scheme()?;
                let prover_setup = with_backend_pool(|| {
                    scheme.setup_prover(self.max_num_vars, self.max_total_batch_polys)
                })
                .map_err(|err| invalid_setup(&err))?;
                with_backend_pool(|| scheme.setup_verifier(&prover_setup))
                    .map_err(|err| invalid_setup(&err))
            }
            AkitaBackendFlavor::OneHot => {
                let log_k = validate_one_hot_k(self.one_hot_k)?;
                if self.max_num_vars < log_k {
                    return Err(invalid_batch("Akita verifier setup has no one-hot backend"));
                }
                let prover_setup =
                    one_hot_setup_prover(self, self.max_num_vars, self.max_total_batch_polys)
                        .map_err(|err| invalid_setup(&err))?;
                one_hot_setup_verifier(self, &prover_setup)
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
    dense_scheme: Arc<OnceLock<Result<AkitaBackendScheme, String>>>,
    one_hot_k16_scheme: Arc<OnceLock<Result<AkitaOneHotK16BackendScheme, String>>>,
    one_hot_k256_scheme: Arc<OnceLock<Result<AkitaOneHotK256BackendScheme, String>>>,
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

/// Binds one backend flavor's setup identity into the transcript. The backend
/// key is determined by the absorbed dimensions and admitted catalog; binding
/// the validated catalog digest avoids hashing the large serialized key while
/// preventing cross-catalog replay.
pub(crate) fn append_verifier_setup<T: Transcript>(
    transcript: &mut T,
    setup: &AkitaVerifierSetup,
    flavor: AkitaBackendFlavor,
) -> Result<(), OpeningsError> {
    transcript.append(&Label(b"akita_setup_key"));
    transcript.append_bytes(b"akita/fp128");
    transcript.append_bytes(flavor.transcript_label());
    transcript.append(&U64Word(setup.max_num_vars as u64));
    transcript.append(&U64Word(setup.max_num_polys_per_commitment_group as u64));
    transcript.append(&U64Word(setup.max_total_batch_polys as u64));
    transcript.append(&U64Word(setup.one_hot_k as u64));
    transcript.append_bytes(&setup.default_layout_digest);
    let catalog_digest = match flavor {
        AkitaBackendFlavor::Dense => setup.dense_scheme()?.schedules().catalog_digest(),
        AkitaBackendFlavor::OneHot => match setup.one_hot_k {
            AKITA_ONE_HOT_K16 => setup.one_hot_k16_scheme()?.schedules().catalog_digest(),
            AKITA_ONE_HOT_K256 => setup.one_hot_k256_scheme()?.schedules().catalog_digest(),
            other => {
                return Err(invalid_batch(format!(
                    "unsupported Akita one-hot K={other}"
                )))
            }
        },
    };
    transcript.append_bytes(&catalog_digest);
    Ok(())
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

    fn max_total_batch_polys(&self) -> usize {
        self.max_total_batch_polys()
    }

    fn default_layout_digest(&self) -> [u8; 32] {
        self.default_layout_digest()
    }

    fn one_hot_k(&self) -> usize {
        self.one_hot_k()
    }
}

impl jolt_openings::GroupSetupMetadata for AkitaProverSetup {
    fn max_num_vars(&self) -> usize {
        self.max_num_vars()
    }

    fn max_num_polys_per_commitment_group(&self) -> usize {
        self.max_num_polys_per_commitment_group()
    }

    fn max_total_batch_polys(&self) -> usize {
        self.max_total_batch_polys()
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
    /// Fixed-width public identity of the exact generated row selected by the
    /// prover. The verifier resolves this digest under its configured catalog;
    /// the backend proof body does not encode the selection itself.
    pub(crate) schedule_selection: [u8; SCHEDULE_SELECTION_BYTES],
    pub(crate) backend_proof: Vec<u8>,
}

impl AkitaBatchProof {
    pub(crate) fn new(selection: OpeningScheduleSelection, backend_proof: Vec<u8>) -> Self {
        Self {
            schedule_selection: *selection.row_digest.as_bytes(),
            backend_proof,
        }
    }

    pub(crate) fn selection(&self) -> OpeningScheduleSelection {
        OpeningScheduleSelection {
            row_digest: ScheduleRowDigest::from_bytes(self.schedule_selection),
        }
    }

    /// Headerless backend proof body produced by Akita's canonical encoder.
    pub fn backend_proof_body_size(&self) -> usize {
        self.backend_proof.len()
    }

    /// Sum of the raw component bytes before the enclosing Jolt serializer
    /// adds container tags or length prefixes.
    pub fn unframed_payload_size(&self) -> Option<usize> {
        SCHEDULE_SELECTION_BYTES.checked_add(self.backend_proof.len())
    }
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
    TraceOneHot(TracePackedOneHot),
}

impl Default for AkitaHintPolynomials {
    fn default() -> Self {
        Self::Dense(Vec::new().into())
    }
}

impl AkitaHintPolynomials {
    pub(crate) const fn backend_flavor(&self) -> AkitaBackendFlavor {
        match self {
            Self::Dense(_) => AkitaBackendFlavor::Dense,
            Self::OneHot(_) | Self::TraceOneHot(_) => AkitaBackendFlavor::OneHot,
        }
    }

    pub(crate) const fn kind(&self) -> &'static str {
        match self {
            Self::Dense(_) => "dense",
            Self::OneHot(_) => "one_hot",
            Self::TraceOneHot(_) => "trace_one_hot",
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Dense(polys) => polys.len(),
            Self::OneHot(polys) => polys.len(),
            Self::TraceOneHot(_) => 1,
        }
    }

    pub(crate) fn one_hot_k(&self) -> Option<usize> {
        match self {
            Self::OneHot(polys) => polys
                .first()
                .and_then(akita_prover::RootPolyMeta::onehot_chunk_size),
            Self::TraceOneHot(polynomial) => {
                akita_prover::RootPolyMeta::onehot_chunk_size(polynomial)
            }
            Self::Dense(_) => None,
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
        &CpuBackend::DEFAULT,
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
    AkitaBackendOneHotPoly::new(one_hot_k, indices.to_vec())
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
    AkitaBackendOneHotPoly::new(one_hot_k, polynomial.into_indices()).map_err(akita_error)
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
    setup: &AkitaVerifierSetup,
    max_num_vars: usize,
    max_num_polys: usize,
) -> Result<AkitaBackendProverSetup, AkitaError> {
    with_backend_pool(|| match setup.one_hot_k {
        AKITA_ONE_HOT_K16 => setup
            .one_hot_k16_scheme()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?
            .setup_prover(max_num_vars, max_num_polys),
        AKITA_ONE_HOT_K256 => setup
            .one_hot_k256_scheme()
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?
            .setup_prover(max_num_vars, max_num_polys),
        _ => unreachable!("one-hot K is validated before backend setup"),
    })
}

pub(crate) fn one_hot_setup_verifier(
    setup: &AkitaVerifierSetup,
    prover_setup: &AkitaBackendProverSetup,
) -> Result<AkitaBackendVerifier, OpeningsError> {
    let invalid_setup = |err: &dyn std::fmt::Display| OpeningsError::InvalidSetup(err.to_string());
    match setup.one_hot_k {
        AKITA_ONE_HOT_K16 => with_backend_pool(|| {
            setup
                .one_hot_k16_scheme()?
                .setup_verifier(prover_setup)
                .map_err(|err| invalid_setup(&err))
        }),
        AKITA_ONE_HOT_K256 => with_backend_pool(|| {
            setup
                .one_hot_k256_scheme()?
                .setup_verifier(prover_setup)
                .map_err(|err| invalid_setup(&err))
        }),
        _ => Err(invalid_batch(format!(
            "unsupported Akita one-hot K={}",
            setup.one_hot_k
        ))),
    }
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
            AkitaBackendDensePoly::from_field_evals(poly.num_vars(), evals).map_err(akita_error)
        })
        .collect()
}

#[doc(hidden)]
#[expect(
    clippy::indexing_slicing,
    reason = "jolt_to_akita_index keeps num_vars bits of the reversal, so the index is < 2^num_vars = akita_evals.len()"
)]
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
#[expect(
    clippy::indexing_slicing,
    reason = "jolt_to_akita_index keeps num_vars bits of the reversal, so the index is < 2^num_vars = evals.len(); for num_vars = 0 the single index for_each_row yields is 0"
)]
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

/// Ends outer Jolt challenge derivation at one statement-bound challenge and
/// uses it to domain-separate the nested Akita transcript. No subsequent Jolt
/// challenge consumes the terminal opening proof, so reabsorbing that proof
/// into the outer transcript could not affect acceptance.
pub(crate) fn bridged_akita_transcript<T>(
    jolt_transcript: &mut T,
    session_label: &[u8],
) -> AkitaTranscript<AkitaField>
where
    T: Transcript<Challenge = AkitaField>,
{
    let bridge = jolt_transcript.challenge_scalar();
    let bridge_bytes = bridge.to_bytes_le_vec();
    // Akita replaces its sponge state when it binds the concrete instance but
    // preserves the session label, so the cross-protocol bridge belongs here.
    let mut bridged_session_label = Vec::with_capacity(session_label.len() + bridge_bytes.len());
    bridged_session_label.extend_from_slice(session_label);
    bridged_session_label.extend_from_slice(&bridge_bytes);
    AkitaTranscript::new(&bridged_session_label)
}
