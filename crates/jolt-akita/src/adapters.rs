use std::{
    collections::BTreeMap,
    fmt,
    io::{Cursor, ErrorKind},
    path::{Path, PathBuf},
    sync::Arc,
    sync::OnceLock,
};

#[cfg(feature = "profiling")]
use std::{cell::Cell, num::NonZeroUsize};

use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_params::{OpeningScheduleSelection, ScheduleRowDigest};
use akita_pcs::{
    AkitaCommitmentScheme, AkitaDeserialize, AkitaError, AkitaProverSetup as BackendProverSetup,
    AkitaSerialize, AkitaVerifier, CommitmentHandle, CpuBackend, DensePoly, OneHotPoly,
};
use akita_schedules::ValidatedScheduleCatalog;
use akita_types::{
    AkitaVerifierSetup as BackendVerifierSetup, Commitment as AkitaBackendRingCommitment,
    CommittedGroup as AkitaBackendCommittedGroup,
};
use jolt_field::{CanonicalBytes, Zero};
use jolt_openings::{OpeningsError, VerifierOpeningClaim};
use jolt_poly::{MultilinearPoly, OneHotIndexOrder, OneHotPolynomial, Polynomial};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript, U64Word};
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::{Deserialize, Serialize};

use crate::configs::{AkitaChunkProfile, JoltDenseBounded, JoltDenseFull};
use crate::one_hot_family::{
    with_one_hot_family, AkitaOneHotBackendScheme, AkitaOneHotBackendVerifier, OneHotFamily,
};
pub use crate::one_hot_family::{AKITA_ONE_HOT_K16, AKITA_ONE_HOT_K256};
use crate::schedule_registry::GroupedScheduleParams;

pub type AkitaField = akita_config::proof_optimized::fp128::Field;
pub(crate) type AkitaConfig = JoltDenseBounded;
/// Smallest A dimension accepted by the delegated adaptive policy. Source
/// objects use this only for dimension-independent flat storage metadata;
/// each generated schedule still selects its exact per-role dimensions.
pub(crate) const AKITA_SOURCE_RING_DIMENSION: usize =
    akita_config::proof_optimized::fp128::Dense::A_RING_DIMENSIONS[0];
const _: () = assert!(
    AKITA_SOURCE_RING_DIMENSION
        == akita_config::proof_optimized::fp128::OneHot::A_RING_DIMENSIONS[0]
);
/// Runtime bytes for Jolt's base schedule families.
///
/// These bytes are ordinary input data. They are intentionally neither
/// generated Rust nor embedded with `include_bytes!`.
///
/// Serialized bundles are regenerable preprocessing inputs, not a stable binary
/// format. The multi-chunk fields break compatibility with older bincode bundles;
/// reload compatible `.aks` catalogs and rebuild cached setup parameters. Serde
/// defaults support omitted fields in map-based formats such as JSON, not older
/// bincode encodings. This is separate from [`AkitaVerifierSetup`] transport.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AkitaScheduleArtifacts {
    dense: Vec<u8>,
    full_dense: Vec<u8>,
    pub(crate) one_hot_k16: Vec<u8>,
    pub(crate) one_hot_k256: Vec<u8>,
    #[serde(default)]
    pub(crate) one_hot_k16_w2r2: Vec<u8>,
    #[serde(default)]
    pub(crate) one_hot_k256_w2r2: Vec<u8>,
    #[serde(default)]
    pub(crate) one_hot_k16_w4r2: Vec<u8>,
    #[serde(default)]
    pub(crate) one_hot_k256_w4r2: Vec<u8>,
    #[serde(default)]
    pub(crate) one_hot_k16_w8r2: Vec<u8>,
    #[serde(default)]
    pub(crate) one_hot_k256_w8r2: Vec<u8>,
}

impl AkitaScheduleArtifacts {
    const DIRECTORY_ENV: &'static str = "JOLT_AKITA_SCHEDULE_DIR";

    pub fn new(
        dense: Vec<u8>,
        full_dense: Vec<u8>,
        one_hot_k16: Vec<u8>,
        one_hot_k256: Vec<u8>,
    ) -> Self {
        Self {
            dense,
            full_dense,
            one_hot_k16,
            one_hot_k256,
            one_hot_k16_w2r2: Vec::new(),
            one_hot_k256_w2r2: Vec::new(),
            one_hot_k16_w4r2: Vec::new(),
            one_hot_k256_w4r2: Vec::new(),
            one_hot_k16_w8r2: Vec::new(),
            one_hot_k256_w8r2: Vec::new(),
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
        let read_optional = |family: &str| {
            let path = directory.join(format!("{family}.aks"));
            match std::fs::read(&path) {
                Ok(bytes) => Ok(bytes),
                Err(error) if error.kind() == ErrorKind::NotFound => Ok(Vec::new()),
                Err(error) => Err(OpeningsError::InvalidSetup(format!(
                    "read Akita schedule artifact {}: {error}",
                    path.display()
                ))),
            }
        };
        Ok(Self {
            dense: read(JoltDenseBounded::schedule_family_name())?,
            full_dense: read(JoltDenseFull::schedule_family_name())?,
            one_hot_k16: read(OneHotFamily::K16Single.family_name())?,
            one_hot_k256: read(OneHotFamily::K256Single.family_name())?,
            one_hot_k16_w2r2: read_optional(OneHotFamily::K16W2R2.family_name())?,
            one_hot_k256_w2r2: read_optional(OneHotFamily::K256W2R2.family_name())?,
            one_hot_k16_w4r2: read_optional(OneHotFamily::K16W4R2.family_name())?,
            one_hot_k256_w4r2: read_optional(OneHotFamily::K256W4R2.family_name())?,
            one_hot_k16_w8r2: read_optional(OneHotFamily::K16W8R2.family_name())?,
            one_hot_k256_w8r2: read_optional(OneHotFamily::K256W8R2.family_name())?,
        })
    }

    /// The `schedules/` directory packaged with this crate: the fallback the
    /// default loader uses when `JOLT_AKITA_SCHEDULE_DIR` is unset, and the
    /// fixed location the checked-in catalog guards read directly, because
    /// they validate the committed artifacts rather than whatever an override
    /// points at.
    pub fn packaged_directory() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("schedules")
    }

    /// Loads from `JOLT_AKITA_SCHEDULE_DIR`, or from the packaged `schedules/`
    /// directory when the variable is unset. Protocol setup and verification
    /// never consult the environment.
    fn from_default_directory() -> Result<Self, OpeningsError> {
        let directory = std::env::var_os(Self::DIRECTORY_ENV)
            .map_or_else(Self::packaged_directory, PathBuf::from);
        Self::from_directory(directory)
    }

    /// Host/dev loader in the shared-handle form every setup call takes: the
    /// one way hosts, benches, and tests load a bundle they intend to hand to
    /// [`AkitaSetupParams`]. Reads `JOLT_AKITA_SCHEDULE_DIR`, falling back to
    /// [`Self::packaged_directory`].
    ///
    /// The handle is what is shared, not the bytes: every call re-reads the
    /// four required `.aks` files and any present profile companions, so hosts
    /// still load once at preprocessing and pass
    /// the bundle to each setup. Callers that must compare setup provenance
    /// keep their own handle rather than calling this twice — the Akita
    /// prover's advice guards test bundle identity with `Arc::ptr_eq`.
    ///
    /// Panics rather than returning: the packaged `schedules/` directory is
    /// part of the installation, so a missing or unreadable catalog is a
    /// broken deployment, not a condition any caller can act on. This is host
    /// and deployment-side only; nothing on the verify path reaches it.
    /// Deployments that must handle a missing catalog load a versioned,
    /// deployment-owned path with [`Self::from_directory`] instead.
    #[expect(
        clippy::expect_used,
        reason = "an unreadable packaged schedule directory is a broken installation"
    )]
    pub fn shared_from_default_directory() -> Arc<Self> {
        Arc::new(
            Self::from_default_directory()
                .expect("the external Akita schedule artifacts must load"),
        )
    }

    pub fn dense_catalog(&self) -> Result<ValidatedScheduleCatalog, AkitaError> {
        TrustedScheduleCatalog::<JoltDenseBounded>::from_artifact_bytes(&self.dense)
            .map(|catalog| catalog.catalog().clone())
    }

    pub fn full_dense_catalog(&self) -> Result<ValidatedScheduleCatalog, AkitaError> {
        TrustedScheduleCatalog::<JoltDenseFull>::from_artifact_bytes(&self.full_dense)
            .map(|catalog| catalog.catalog().clone())
    }

    pub(crate) fn full_dense_scheme(
        &self,
    ) -> Result<AkitaCommitmentScheme<JoltDenseFull>, AkitaError> {
        AkitaCommitmentScheme::<JoltDenseFull>::from_schedule_artifact(&self.full_dense)
    }

    pub fn one_hot_catalog(
        &self,
        one_hot_k: usize,
    ) -> Result<ValidatedScheduleCatalog, AkitaError> {
        self.one_hot_catalog_for_profile(one_hot_k, AkitaChunkProfile::Single)
    }

    pub fn one_hot_catalog_for_profile(
        &self,
        one_hot_k: usize,
        profile: AkitaChunkProfile,
    ) -> Result<ValidatedScheduleCatalog, AkitaError> {
        let family = OneHotFamily::from_parts(one_hot_k, profile)?;
        with_one_hot_family!(family, |Cfg| {
            TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(family.artifact(self))
                .map(|catalog| catalog.catalog().clone())
        })
    }
}

pub(crate) type AkitaBackendExtField = <AkitaConfig as CommitmentConfig>::ExtField;

pub(crate) type AkitaBackendScheme = AkitaCommitmentScheme<AkitaConfig>;
pub(crate) type AkitaBackendCommitment = AkitaBackendCommittedGroup<AkitaField>;
pub(crate) type AkitaBackendCommitmentPayload = AkitaBackendRingCommitment<AkitaField>;
pub(crate) type AkitaBackendHint = CommitmentHandle<AkitaField, AkitaBackendExtField>;
pub(crate) type AkitaBackendVerifierSetup = BackendVerifierSetup<AkitaField>;
pub(crate) type AkitaBackendDensePoly = DensePoly<AkitaField>;
pub(crate) type AkitaBackendOneHotPoly = OneHotPoly<AkitaField, u8>;
/// The owning CPU backend: prepared setup transforms plus every commitment
/// handle's source. Handles only prove on the backend that committed them.
pub(crate) type AkitaBackend = CpuBackend<AkitaField, AkitaBackendExtField>;
pub(crate) type AkitaBackendProverSetup = BackendProverSetup<AkitaField>;

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
/// the Akita prover at trace-scale shapes. Every backend setup/commit/
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

/// Regenerable recipe for constructing an Akita setup.
///
/// Its bincode representation is tied to the implementation version. The
/// multi-chunk profile and expanded [`AkitaScheduleArtifacts`] make older
/// serialized recipes incompatible; discard those caches and rerun preprocessing
/// with the current code and compatible catalogs. Serde defaults do not provide
/// bincode backward compatibility. The legacy `Single` [`AkitaVerifierSetup`]
/// encoding is preserved separately.
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
    #[serde(default)]
    pub(crate) akita_chunk_profile: AkitaChunkProfile,
    pub(crate) flavor: AkitaSetupFlavor,
    /// Recipe for the dynamic grouped rows accepted by this setup.
    ///
    /// Replaying serialized setup parameters intentionally reruns schedule
    /// planning. Verifier transport serializes [`AkitaVerifierSetup`]
    /// instead, which contains the finalized catalog and never replans.
    #[serde(default, rename = "advice_schedule")]
    pub(crate) grouped_schedule: Option<GroupedScheduleParams>,
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
            akita_chunk_profile: AkitaChunkProfile::Single,
            flavor: AkitaSetupFlavor::Both,
            grouped_schedule: None,
            schedule_artifacts,
        }
    }

    /// Setup parameters for a commitment object that only ever commits and
    /// opens through the one-hot flavor (the native `OneHotTrace` group): skips
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
            akita_chunk_profile: AkitaChunkProfile::Single,
            flavor: AkitaSetupFlavor::OneHot,
            grouped_schedule: None,
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
        grouped_schedule: Option<GroupedScheduleParams>,
        schedule_artifacts: Arc<AkitaScheduleArtifacts>,
    ) -> Self {
        Self {
            max_num_vars,
            max_num_polys_per_commitment_group,
            max_total_batch_polys,
            default_layout_digest,
            one_hot_k,
            akita_chunk_profile: AkitaChunkProfile::Single,
            flavor: AkitaSetupFlavor::OneHot,
            grouped_schedule,
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
            akita_chunk_profile: AkitaChunkProfile::Single,
            flavor: AkitaSetupFlavor::Dense,
            grouped_schedule: None,
            schedule_artifacts,
        }
    }

    pub fn one_hot_k(&self) -> usize {
        self.one_hot_k
    }

    /// Selects 1, 2, 4, or 8 witness chunks for the one-hot trace backend.
    /// Constructors default to [`AkitaChunkProfile::Single`].
    pub fn with_akita_chunk_profile(mut self, profile: AkitaChunkProfile) -> Self {
        self.akita_chunk_profile = profile;
        self
    }

    pub fn akita_chunk_profile(&self) -> AkitaChunkProfile {
        self.akita_chunk_profile
    }

    pub fn max_total_batch_polys(&self) -> usize {
        self.max_total_batch_polys
    }
}

#[derive(Debug)]
pub(crate) struct FullWidthBackendSetup {
    pub(crate) scheme: AkitaCommitmentScheme<JoltDenseFull>,
    pub(crate) backend: AkitaBackend,
}

#[derive(Clone, Debug)]
pub struct AkitaProverSetup {
    pub(crate) full_width: BTreeMap<usize, Arc<FullWidthBackendSetup>>,
    pub(crate) backend_prover_setup: Option<Arc<AkitaBackendProverSetup>>,
    pub(crate) cpu_backend: Option<Arc<AkitaBackend>>,
    pub(crate) one_hot_backend_prover_setup: Option<Arc<AkitaBackendProverSetup>>,
    pub(crate) one_hot_cpu_backend: Option<Arc<AkitaBackend>>,
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
        for backend in [
            self.cpu_backend.as_deref(),
            self.one_hot_cpu_backend.as_deref(),
        ]
        .into_iter()
        .flatten()
        .chain(self.full_width.values().map(|setup| &setup.backend))
        {
            let _ = backend.trim_caches().map_err(invalid_setup)?;
        }
        Ok(())
    }

    pub(crate) fn dense_backend(
        &self,
    ) -> Result<(&AkitaBackendProverSetup, &AkitaBackend), OpeningsError> {
        self.backend_prover_setup
            .as_deref()
            .zip(self.cpu_backend.as_deref())
            .ok_or_else(|| {
                OpeningsError::InvalidSetup(
                    "this Akita setup was built without the dense-flavor backend".to_string(),
                )
            })
    }

    pub(crate) fn one_hot_backend(
        &self,
    ) -> Result<(&AkitaBackendProverSetup, &AkitaBackend), OpeningsError> {
        let prover_setup = self
            .one_hot_backend_prover_setup
            .as_deref()
            .ok_or_else(|| invalid_batch("Akita setup has no one-hot backend"))?;
        let backend = self
            .one_hot_cpu_backend
            .as_deref()
            .ok_or_else(|| invalid_batch("Akita setup has no prepared one-hot backend"))?;
        Ok((prover_setup, backend))
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
    pub(crate) schedule_artifacts: AkitaVerifierScheduleArtifacts,
    #[serde(skip)]
    pub(crate) backend_cache: BackendVerifierCache,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub(crate) enum AkitaVerifierScheduleArtifacts {
    Dense {
        dense: Vec<u8>,
    },
    OneHot {
        one_hot: Vec<u8>,
    },
    Both {
        dense: Vec<u8>,
        one_hot: Vec<u8>,
    },
    OneHotChunked {
        profile: AkitaChunkProfile,
        one_hot: Vec<u8>,
    },
    BothChunked {
        profile: AkitaChunkProfile,
        dense: Vec<u8>,
        one_hot: Vec<u8>,
    },
}

impl AkitaVerifierScheduleArtifacts {
    fn dense(&self) -> Option<&[u8]> {
        match self {
            Self::Dense { dense } | Self::Both { dense, .. } | Self::BothChunked { dense, .. } => {
                Some(dense)
            }
            Self::OneHot { .. } | Self::OneHotChunked { .. } => None,
        }
    }

    fn one_hot(&self) -> Option<&[u8]> {
        match self {
            Self::OneHot { one_hot }
            | Self::Both { one_hot, .. }
            | Self::OneHotChunked { one_hot, .. }
            | Self::BothChunked { one_hot, .. } => Some(one_hot),
            Self::Dense { .. } => None,
        }
    }

    fn akita_chunk_profile(&self) -> AkitaChunkProfile {
        match self {
            Self::OneHotChunked { profile, .. } | Self::BothChunked { profile, .. } => *profile,
            Self::Dense { .. } | Self::OneHot { .. } | Self::Both { .. } => {
                AkitaChunkProfile::Single
            }
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

    pub fn akita_chunk_profile(&self) -> AkitaChunkProfile {
        self.schedule_artifacts.akita_chunk_profile()
    }

    /// Primes the lazy verifier cache from freshly built backend keys, so
    /// in-process setups never pay the shape→key re-derivation.
    pub(crate) fn prime_backend_cache(
        &self,
        dense: Option<AkitaBackendVerifierSetup>,
        one_hot: Option<AkitaBackendVerifierSetup>,
    ) -> Result<(), OpeningsError> {
        if let Some(dense) = dense {
            let verifier =
                with_backend_pool(|| self.dense_scheme()?.verifier(dense).map_err(invalid_setup))?;
            let _ = self.backend_cache.dense.get_or_init(|| verifier);
        }
        if let Some(one_hot) = one_hot {
            let verifier = self.build_one_hot_verifier(one_hot)?;
            let _ = self.backend_cache.one_hot.get_or_init(|| verifier);
        }
        Ok(())
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

    pub(crate) fn one_hot_scheme(&self) -> Result<&AkitaOneHotBackendScheme, OpeningsError> {
        let result = self.backend_cache.one_hot_scheme.get_or_init(|| {
            self.schedule_artifacts
                .one_hot()
                .ok_or_else(|| "Akita verifier setup has no one-hot schedule artifact".to_string())
                .and_then(|bytes| {
                    OneHotFamily::from_parts(self.one_hot_k, self.akita_chunk_profile())
                        .and_then(|family| AkitaOneHotBackendScheme::from_artifact(family, bytes))
                        .map_err(|error| error.to_string())
                })
        });
        result
            .as_ref()
            .map_err(|error| OpeningsError::InvalidSetup(error.clone()))
    }

    /// Variables the one-hot backend setup covers: the exact final arity, or
    /// the largest precommitted group of this setup's grouped rows when that
    /// is larger. Akita sizes a setup only from the catalog rows whose every
    /// group fits its capacity (`SetupRequirements::from_catalog`), and an
    /// advice or committed-program object may exceed the trace group it is
    /// opened with.
    pub(crate) fn one_hot_backend_num_vars(&self) -> Result<usize, OpeningsError> {
        let largest_precommitted = with_one_hot_family!(scheme self.one_hot_scheme()?, |scheme| {
            largest_precommitted_num_vars(scheme.schedules())
        });
        Ok(self.max_num_vars.max(largest_precommitted))
    }

    /// Dense backend verifier, cached after the first use.
    /// [`AkitaScheme::setup`](crate::AkitaScheme) primes the cache with the
    /// freshly built key; a serde-transported setup re-derives it from the
    /// shape on first use (one-time, setup-class cost).
    pub(crate) fn dense_verifier(&self) -> Result<&AkitaVerifier<AkitaConfig>, OpeningsError> {
        if let Some(verifier) = self.backend_cache.dense.get() {
            return Ok(verifier);
        }
        let scheme = self.dense_scheme()?;
        let verifier = with_backend_pool(|| {
            let prover_setup =
                scheme.setup_prover(self.max_num_vars, self.max_total_batch_polys)?;
            scheme.verifier(scheme.setup_verifier(&prover_setup)?)
        })
        .map_err(invalid_setup)?;
        Ok(self.backend_cache.dense.get_or_init(|| verifier))
    }

    fn build_one_hot_verifier(
        &self,
        setup: AkitaBackendVerifierSetup,
    ) -> Result<AkitaOneHotBackendVerifier, OpeningsError> {
        let scheme = self.one_hot_scheme()?;
        with_backend_pool(|| scheme.verifier(setup)).map_err(invalid_setup)
    }

    pub(crate) fn one_hot_verifier(&self) -> Result<&AkitaOneHotBackendVerifier, OpeningsError> {
        if let Some(verifier) = self.backend_cache.one_hot.get() {
            return Ok(verifier);
        }
        let setup = self.one_hot_backend_verifier_setup()?;
        let verifier = self.build_one_hot_verifier(setup)?;
        Ok(self.backend_cache.one_hot.get_or_init(|| verifier))
    }

    fn one_hot_backend_verifier_setup(&self) -> Result<AkitaBackendVerifierSetup, OpeningsError> {
        let log_k = validate_one_hot_k(self.one_hot_k)?;
        if self.max_num_vars < log_k {
            return Err(invalid_batch("Akita verifier setup has no one-hot backend"));
        }
        let prover_setup = one_hot_setup_prover(self)?;
        one_hot_setup_verifier(self, &prover_setup)
    }
}

/// Lazily built backend verifiers (admitted rows plus their prepared
/// terminal matrices). Derived state: ignored by equality and skipped by
/// serde; clones share the cache.
#[derive(Clone, Default)]
pub(crate) struct BackendVerifierCache {
    dense: Arc<OnceLock<AkitaVerifier<AkitaConfig>>>,
    one_hot: Arc<OnceLock<AkitaOneHotBackendVerifier>>,
    dense_scheme: Arc<OnceLock<Result<AkitaBackendScheme, String>>>,
    one_hot_scheme: Arc<OnceLock<Result<AkitaOneHotBackendScheme, String>>>,
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
    let akita_chunk_profile = setup.akita_chunk_profile();
    transcript.append(&Label(b"akita_setup_key"));
    transcript.append_bytes(b"akita/fp128");
    transcript.append_bytes(flavor.transcript_label());
    transcript.append(&U64Word(setup.max_num_vars as u64));
    transcript.append(&U64Word(setup.max_num_polys_per_commitment_group as u64));
    transcript.append(&U64Word(setup.max_total_batch_polys as u64));
    transcript.append(&U64Word(setup.one_hot_k as u64));
    if flavor == AkitaBackendFlavor::OneHot && akita_chunk_profile != AkitaChunkProfile::Single {
        transcript.append(&Label(b"akita_chunk_profile"));
        transcript.append(&U64Word(akita_chunk_profile.num_chunks() as u64));
    }
    transcript.append_bytes(&setup.default_layout_digest);
    let catalog_digest = match flavor {
        AkitaBackendFlavor::Dense => setup.dense_scheme()?.schedules().catalog_digest(),
        AkitaBackendFlavor::OneHot => {
            with_one_hot_family!(scheme setup.one_hot_scheme()?, |scheme| scheme
                .schedules()
                .catalog_digest())
        }
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

    /// Headerless backend proof body: Akita's Spongefish argument bytes.
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
    /// The public committed group and the backend handle retaining its exact
    /// source, produced at commit time and consumed when opening.
    pub(crate) backend: Option<(AkitaBackendCommitment, AkitaBackendHint)>,
    pub(crate) source: AkitaHintSource,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum AkitaHintSource {
    Dense { poly_count: usize },
    OneHot { poly_count: usize, one_hot_k: usize },
    TraceOneHot { poly_count: usize, one_hot_k: usize },
}

impl Default for AkitaHintSource {
    fn default() -> Self {
        Self::Dense { poly_count: 0 }
    }
}

impl AkitaHintSource {
    pub(crate) const fn backend_flavor(self) -> AkitaBackendFlavor {
        match self {
            Self::Dense { .. } => AkitaBackendFlavor::Dense,
            Self::OneHot { .. } | Self::TraceOneHot { .. } => AkitaBackendFlavor::OneHot,
        }
    }

    pub(crate) const fn kind(self) -> &'static str {
        match self {
            Self::Dense { .. } => "dense",
            Self::OneHot { .. } => "one_hot",
            Self::TraceOneHot { .. } => "trace_one_hot",
        }
    }

    pub(crate) const fn len(self) -> usize {
        match self {
            Self::Dense { poly_count }
            | Self::OneHot { poly_count, .. }
            | Self::TraceOneHot { poly_count, .. } => poly_count,
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

/// The one-hot backend prover setup `setup` describes, sized by
/// [`AkitaVerifierSetup::one_hot_backend_num_vars`].
pub(crate) fn one_hot_setup_prover(
    setup: &AkitaVerifierSetup,
) -> Result<AkitaBackendProverSetup, OpeningsError> {
    let max_num_vars = setup.one_hot_backend_num_vars()?;
    let max_num_polys = setup.max_total_batch_polys;
    let scheme = setup.one_hot_scheme()?;
    with_backend_pool(|| {
        with_one_hot_family!(scheme scheme, |scheme| scheme
            .setup_prover(max_num_vars, max_num_polys))
    })
    .map_err(invalid_setup)
}

fn largest_precommitted_num_vars<Cfg: CommitmentConfig>(
    catalog: &TrustedScheduleCatalog<Cfg>,
) -> usize {
    catalog
        .rows()
        .flat_map(|row| &row.profiles().precommitteds)
        .map(|profile| profile.group.num_vars())
        .max()
        .unwrap_or(0)
}

pub(crate) fn one_hot_setup_verifier(
    setup: &AkitaVerifierSetup,
    prover_setup: &AkitaBackendProverSetup,
) -> Result<AkitaBackendVerifierSetup, OpeningsError> {
    let scheme = setup.one_hot_scheme()?;
    with_backend_pool(|| {
        with_one_hot_family!(scheme scheme, |scheme| scheme
            .setup_verifier(prover_setup)
            .map_err(invalid_setup))
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

pub(crate) fn invalid_setup(error: impl ToString) -> OpeningsError {
    OpeningsError::InvalidSetup(error.to_string())
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
/// uses it to domain-separate the nested Akita argument's session. No
/// subsequent Jolt challenge consumes the terminal opening proof, so
/// reabsorbing that proof into the outer transcript could not affect
/// acceptance.
pub(crate) fn bridged_akita_session<T>(jolt_transcript: &mut T, session_label: &[u8]) -> Vec<u8>
where
    T: Transcript<Challenge = AkitaField>,
{
    let bridge = jolt_transcript.challenge_scalar();
    let bridge_bytes = bridge.to_bytes_le_vec();
    // Akita binds the concrete instance into its own Fiat-Shamir state but
    // keeps the session bytes as domain separator, so the cross-protocol
    // bridge belongs here.
    let mut bridged_session = Vec::with_capacity(session_label.len() + bridge_bytes.len());
    bridged_session.extend_from_slice(session_label);
    bridged_session.extend_from_slice(&bridge_bytes);
    bridged_session
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        clippy::unwrap_used,
        reason = "tests assert successful conversions and exact error text"
    )]

    use super::*;
    use akita_types::RingVec;
    use jolt_field::Ring;

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
        // The wire form carries coefficients only, so the fixture stays D-free.
        let payload = AkitaBackendCommitmentPayload::new(RingVec::from_coeffs(
            (1..=64).map(AkitaField::from_u64).collect(),
        ));
        let coeff_len = payload.rows().coeff_len();
        let mut bytes = serialize_akita(&payload).expect("payload serializes");
        let roundtrip: AkitaBackendCommitmentPayload =
            deserialize_akita(&bytes, &coeff_len).expect("exact bytes deserialize");
        assert_eq!(roundtrip, payload);

        bytes.push(0);
        let err = deserialize_akita::<AkitaBackendCommitmentPayload>(&bytes, &coeff_len)
            .expect_err("trailing bytes must be rejected");
        assert!(
            matches!(&err, OpeningsError::InvalidBatch(message) if message.contains("trailing bytes")),
            "unexpected error: {err}"
        );
    }
}
