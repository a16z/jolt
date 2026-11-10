//! Whiteout masked openings and capacity-bounded Sigma matrix views.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;
#[cfg(feature = "zk-prover")]
use std::sync::OnceLock;
#[cfg(all(feature = "zk-prover", feature = "logging"))]
use std::sync::PoisonError;

use akita_config::proof_optimized::fp128::ResponseDigits;
use akita_config::{policy_of, CommitmentConfig, TrustedScheduleCatalog};
use akita_params::sis::sigma_binding::sigma_binding_min_rank;
use akita_params::sis::CommittedSourceContract;
use akita_params::zk::{AdmittedZkSchedule, ZkOpeningSchedule, ZK_CUT_LEVEL};
#[cfg(feature = "zk-prover")]
use akita_params::PrecommittedGroupProfiles;
use akita_params::{
    BasisMode, ChunkedWitnessCfg, CommittedGroupBatchProfile, FoldSchedule, GroupCommitPhaseParams,
    OpeningClaimsLayout, OpeningScheduleSelection, PolynomialGroupLayout, ScheduleLookupKey,
};
#[cfg(feature = "zk-prover")]
use akita_pcs::zk::{prove_masked_prefix, AkitaResponseProver, ResidualOpening};
use akita_pcs::zk::{
    AkitaResponseVerifier, CompressionShape, MaskedPrefix, ResidualMatrices, ResidualShape,
    ZkCommitmentConfig,
};
#[cfg(feature = "zk-prover")]
use akita_pcs::SelectedProverOpeningData;
use akita_pcs::{AkitaCommitmentScheme, AkitaError, AkitaVerifier, TrustedTerminalCache};
#[cfg(feature = "zk-prover")]
use akita_pcs::{CommitOutput, GroupContext, HidingRandomness, SourceHandle};
use akita_schedules::digit_commitment_min_rank;
#[cfg(feature = "zk-prover")]
use akita_types::Commitment;
use akita_types::{
    sample_akita_setup_seed, CommittedGroup, GroupBatchStatement, OpeningClaims,
    PolynomialGroupClaims,
};
use jolt_crypto::lattice::{
    derive_public_matrix_prefix, required_randomness_columns, PreparedMatrix, PublicMatrixSeed,
    CHALLENGE_RING_DIM,
};
use jolt_openings::OpeningsError;
use jolt_transcript::Transcript;
#[cfg(feature = "zk-prover")]
use jolt_zk::masking::CommittedPads;
use jolt_zk::masking::{AffineRow, CommittedPadsView, PadLayout, ResidualRelation, ZkError};
#[cfg(feature = "zk-prover")]
use jolt_zk::whiteout::{
    commit_lattice_pads, prove_sigma, ImageCompression, ResponseProver, ResponseStatement,
    ResponseWitness, WhiteoutPadOpening,
};
use jolt_zk::whiteout::{
    receive_lattice_pads, verify_sigma, BlockTarget, DenseSource, LatticePadCommitment,
    LatticePadPayload, MaskedOpeningShape, PadGeometry, PadPlacement, PrecommittedShape,
    SigmaGeometry, SigmaSetup, PAD_RANDOMNESS_LOG_BASIS, RESPONSE_LOG_BASIS,
};
#[cfg(feature = "zk-prover")]
use rand_core::{CryptoRng, RngCore};

#[cfg(feature = "zk-prover")]
use crate::adapters::AkitaBackendHint;
use crate::adapters::{
    with_backend_pool, AkitaBackendVerifierSetup, AkitaCommitment, AkitaField, WhiteoutSource,
};
#[cfg(feature = "zk-prover")]
use crate::adapters::{AkitaBackend, AkitaBackendProverSetup};
use crate::configs::{JoltDenseBounded, JoltDenseFull, JoltOneHotK16};
use crate::schedule_registry::GroupedScheduleParams;
#[cfg(feature = "zk-prover")]
use crate::scheme::AkitaProverTranscript;
use crate::scheme::AkitaVerifierTranscript;
use crate::verifier_resources::{generate_material, AkitaVerifierRequirements};
#[cfg(feature = "zk-prover")]
use akita_transcript::TranscriptSponge;

mod row;
#[cfg(feature = "zk-prover")]
pub use row::ProverGroup;
use row::RowMatrices;
pub use row::WhiteoutRow;
mod response;
#[cfg(all(test, feature = "zk-prover"))]
use response::ResponseRow;
use response::{PreparedResponse, ResponsePlans, ResponseRows};
mod setup;
pub(crate) use setup::PreparedWhiteout;
pub use setup::WhiteoutSetup;
#[cfg(all(test, feature = "zk-prover"))]
use setup::WhiteoutSetupWire;
mod transport;
use transport::WhiteoutPayloads;
mod capacity;
use capacity::CapacityStreams;
pub use capacity::{WhiteoutCapacity, WhiteoutCapacityError, WhiteoutStream};

type F = AkitaField;

/// The zero-knowledge configuration of Jolt's K=16 trace group.
pub type WhiteoutOneHot = ZkCommitmentConfig<JoltOneHotK16>;

/// The response proof's configuration (Akita `specs/whiteout-akita.md` §5). Direct-only:
/// recursive setup planning ranks setup envelope before proof bytes and keeps
/// a one-fold row at `ν = 24` (muldiv) whose proof is 2.3x larger.
pub type WhiteoutResponse = ResponseDigits;

/// Runs `$body` with `$D` the const ring dimension `$d` of a residual.
macro_rules! with_ring_dim {
    ($d:expr, $D:ident, $body:expr) => {
        match $d {
            64 => {
                const $D: usize = 64;
                $body
            }
            128 => {
                const $D: usize = 128;
                $body
            }
            256 => {
                const $D: usize = 256;
                $body
            }
            other => Err(ZkError::from(AkitaError::UnsupportedSchedule(format!(
                "residual ring dimension {other}"
            )))),
        }
    };
}

/// The Module-SIS rank of the pad commitment `A_p`, of `width` columns over
/// a ring of dimension `ring_dim`: Akita's Σ-binding table at jolt-zk's
/// `binding_collision_radius()`, the radius the residual's `A_a` and
/// `[B_a | B′_a]` are priced at (Akita `specs/whiteout-akita.md` §5.5.2). `A_p` binds the pads,
/// which the Σ-protocol certifies only inside its Gaussian box.
///
/// # Errors
///
/// [`AkitaError::UnsupportedSchedule`] if the table has no rank for the
/// matrix.
pub(crate) fn sigma_matrix_rank(ring_dim: usize, width: usize) -> Result<usize, ZkError> {
    sigma_binding_min_rank(WhiteoutOneHot::sis_modulus_profile(), ring_dim, width).ok_or_else(
        || {
            AkitaError::UnsupportedSchedule(format!(
                "no Σ-binding rank for A_p (d {ring_dim}, width {width})"
            ))
            .into()
        },
    )
}

/// The Module-SIS rank of `A_D`, of `width` challenge-ring columns of
/// range-checked radix-2^[`RESPONSE_LOG_BASIS`] digits: Akita's ordinary
/// pricing of a digit commitment at the digit-difference bound
/// (paper: security analysis, digit commitment), [`digit_commitment_min_rank`].
///
/// # Errors
///
/// [`AkitaError::UnsupportedSchedule`] if the table has no rank for the
/// matrix.
pub(crate) fn digit_matrix_rank(width: usize) -> Result<usize, ZkError> {
    let policy = policy_of::<WhiteoutOneHot>(ChunkedWitnessCfg::default_non_chunked());
    digit_commitment_min_rank(&policy, CHALLENGE_RING_DIM, width, RESPONSE_LOG_BASIS).ok_or_else(
        || {
            AkitaError::UnsupportedSchedule(format!(
                "no digit-commitment rank for A_D (width {width})"
            ))
            .into()
        },
    )
}

/// Jolt's public seeds for the streams `A_p` and `A_D` are cut from. They are
/// independent of Akita's setup stream, from which the residual blocks are
/// cut.
const PAD_STREAM_SEED: [u8; 32] = *b"jolt/whiteout/pad-commitment/v1\0";
const V_STREAM_SEED: [u8; 32] = *b"jolt/whiteout/sigma-a_d/v1\0\0\0\0\0\0";

fn public_stream(len: usize, seed: [u8; 32]) -> Result<Vec<F>, ZkError> {
    derive_public_matrix_prefix(len, &PublicMatrixSeed::shake256_paged_v1(seed))
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()).into())
}

/// The zero-knowledge configuration of a precommitted `u64`-bounded dense
/// group (advice words, committed programs).
pub type WhiteoutDenseBounded = ZkCommitmentConfig<JoltDenseBounded>;

/// The zero-knowledge configuration of a precommitted full-field dense group
/// (field-register increments).
pub type WhiteoutDenseFull = ZkCommitmentConfig<JoltDenseFull>;

/// A precommitted group's hiding commitment: the root profile of the
/// zero-knowledge row of its own shape, which carries the masked root's
/// `b_randomness_columns` (Akita `specs/whiteout-akita.md` §3.5), and that row's one-row catalog,
/// whose config admits the group's source at commit time.
struct PrecommittedRow {
    profile: GroupCommitPhaseParams,
    catalog: PrecommittedCatalog,
}

enum PrecommittedCatalog {
    Bounded(TrustedScheduleCatalog<WhiteoutDenseBounded>),
    FullField(TrustedScheduleCatalog<WhiteoutDenseFull>),
}

impl PrecommittedRow {
    fn plan(shape: PrecommittedShape) -> Result<Self, AkitaError> {
        match shape.source {
            DenseSource::Bounded => {
                let (profile, catalog) = plan_precommitted::<WhiteoutDenseBounded>(shape.num_vars)?;
                Ok(Self {
                    profile,
                    catalog: PrecommittedCatalog::Bounded(catalog),
                })
            }
            DenseSource::FullField => {
                let (profile, catalog) = plan_precommitted::<WhiteoutDenseFull>(shape.num_vars)?;
                Ok(Self {
                    profile,
                    catalog: PrecommittedCatalog::FullField(catalog),
                })
            }
        }
    }

    /// The row of `shape` from its persisted one-row catalog, admitted as
    /// [`plan`](Self::plan)'s is.
    fn load(shape: PrecommittedShape, artifact: &[u8]) -> Result<Self, WhiteoutPlanError> {
        let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(shape.num_vars));
        match shape.source {
            DenseSource::Bounded => {
                let (profile, catalog) = load_single_row::<WhiteoutDenseBounded>(artifact, &key)?;
                Ok(Self {
                    profile,
                    catalog: PrecommittedCatalog::Bounded(catalog),
                })
            }
            DenseSource::FullField => {
                let (profile, catalog) = load_single_row::<WhiteoutDenseFull>(artifact, &key)?;
                Ok(Self {
                    profile,
                    catalog: PrecommittedCatalog::FullField(catalog),
                })
            }
        }
    }

    fn contract(&self) -> Result<CommittedSourceContract, AkitaError> {
        match self.catalog {
            PrecommittedCatalog::Bounded(_) => WhiteoutDenseBounded::committed_source_contract(),
            PrecommittedCatalog::FullField(_) => WhiteoutDenseFull::committed_source_contract(),
        }
    }

    /// The row's one-row catalog, as persisted.
    fn artifact(&self) -> Result<Vec<u8>, AkitaError> {
        match &self.catalog {
            PrecommittedCatalog::Bounded(catalog) => catalog.to_artifact_bytes(),
            PrecommittedCatalog::FullField(catalog) => catalog.to_artifact_bytes(),
        }
    }

    /// Commits `source` under the hiding profile: the backend samples fresh
    /// randomness for every B slice and keeps it in the handle.
    #[cfg(feature = "zk-prover")]
    fn commit(
        &self,
        backend: &AkitaBackend,
        source: &SourceHandle<F, F>,
    ) -> Result<CommitOutput<F, F>, AkitaError> {
        let context = GroupContext::explicit(&self.profile);
        match &self.catalog {
            PrecommittedCatalog::Bounded(catalog) => backend.commit(catalog, source, context),
            PrecommittedCatalog::FullField(catalog) => backend.commit(catalog, source, context),
        }
    }

    /// Rebuilds the commitment `expected` of `source` under the hiding
    /// profile from the randomness it was made with; fails unless the
    /// rebuilt commitment is `expected`.
    #[cfg(feature = "zk-prover")]
    fn recommit(
        &self,
        backend: &AkitaBackend,
        source: &SourceHandle<F, F>,
        randomness: &HidingRandomness,
        expected: &Commitment<F>,
    ) -> Result<CommitOutput<F, F>, AkitaError> {
        let context = GroupContext::explicit(&self.profile);
        match &self.catalog {
            PrecommittedCatalog::Bounded(catalog) => {
                backend.recommit_hiding(catalog, source, context, randomness, expected)
            }
            PrecommittedCatalog::FullField(catalog) => {
                backend.recommit_hiding(catalog, source, context, randomness, expected)
            }
        }
    }
}

/// The root profile and one-row catalog of the zero-knowledge row of one
/// `num_vars`-variable dense polynomial under `Cfg`, planned unchunked: a
/// masked level spans one witness chunk, and the group is folded at the
/// single-chunk trace row's masked root.
fn plan_precommitted<Cfg: CommitmentConfig<Field = F, ExtField = F>>(
    num_vars: usize,
) -> Result<(GroupCommitPhaseParams, TrustedScheduleCatalog<Cfg>), AkitaError> {
    let chunking = ChunkedWitnessCfg::default_non_chunked();
    let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
    let planned = akita_planner::find_zk_schedule(
        &key,
        Cfg::committed_source_contract()?,
        &[],
        &policy_of::<Cfg>(chunking),
        Cfg::ring_challenge_config,
        ZkOpeningSchedule::new(ZK_CUT_LEVEL)?,
    )?;
    let profile =
        GroupCommitPhaseParams::try_from_params(key.final_group, &planned.schedule.root.params)?;
    let catalog = TrustedScheduleCatalog::<Cfg>::try_from_rows(
        [(
            CommittedGroupBatchProfile {
                final_group: profile,
                precommitteds: Vec::new(),
            },
            planned.schedule,
        )],
        chunking,
    )?;
    Ok((profile, catalog))
}

/// Decodes a persisted one-row catalog under `Cfg` through Akita's trusted
/// artifact path, which audits the row (`ValidatedScheduleCatalog`: SIS
/// keys and bounds against the policy's tables, digit and width geometry,
/// challenge hooks) and admits it as `Cfg` admits rows (for a
/// zero-knowledge `Cfg`, [`ZkOpeningSchedule::admit`] with its Σ-binding
/// ranks); then requires the row to be `key`'s. Returns the row's root
/// profile.
fn load_single_row<Cfg: CommitmentConfig<Field = F, ExtField = F>>(
    artifact: &[u8],
    key: &ScheduleLookupKey,
) -> Result<(GroupCommitPhaseParams, TrustedScheduleCatalog<Cfg>), WhiteoutPlanError> {
    let catalog = TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(
        artifact,
        ChunkedWitnessCfg::default_non_chunked(),
    )
    .map_err(WhiteoutPlanError::Inadmissible)?;
    if catalog.rows().len() != 1 {
        return Err(WhiteoutPlanError::Statement(AkitaError::InvalidSetup(
            format!("a one-row plan holds {} rows", catalog.rows().len()),
        )));
    }
    let profile = catalog
        .resolve_key(key)
        .map_err(WhiteoutPlanError::Statement)?
        .profiles()
        .final_group;
    Ok((profile, catalog))
}

/// The format tag and version of [`WhiteoutPlans`]. Bump the version when
/// the plans' statement changes (the Σ statement, the residual's
/// admission): a verifier then refuses plans of the old statement instead of
/// misreading them.
const WHITEOUT_PLANS_MAGIC: [u8; 4] = *b"WOPL";
const WHITEOUT_PLANS_VERSION: u32 = 2;

/// A Whiteout setup's persisted plans: its zero-knowledge catalog and,
/// optionally, a set of response schedules, each as Akita's canonical
/// trusted catalog artifact. A plan is a public parameter: a setup that
/// loads one re-audits and re-admits every row and checks it is the row of
/// the setup's own statement before any use ([`WhiteoutPlanError`]), and a
/// plan that fails is refused, never replaced by planning. The planner's
/// search is not re-run, so a loaded row is admissible but not necessarily
/// the planner's cheapest. The pads absorb the masked catalog's digest, the
/// selected row's digest and the response schedule's digest
/// ([`WhiteoutTail::receive_pads`]).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WhiteoutPlans {
    magic: [u8; 4],
    version: u32,
    catalog: CatalogPlan,
    /// Each response schedule's one-row catalog, by `e`'s variable count:
    /// when present, the setup refuses any length it lacks; when absent,
    /// the setup loads the packaged response catalog.
    responses: Option<BTreeMap<usize, Vec<u8>>>,
}

impl WhiteoutPlans {
    fn validate_header(&self) -> Result<(), WhiteoutPlanError> {
        if (self.magic, self.version) != (WHITEOUT_PLANS_MAGIC, WHITEOUT_PLANS_VERSION) {
            return Err(WhiteoutPlanError::Format {
                magic: self.magic,
                version: self.version,
            });
        }
        Ok(())
    }
}

/// [`WhiteoutPlans`]' catalog: the trace group shape it was planned for,
/// each precommitted shape's one-row catalog in the setup's shape order,
/// and the catalog of the masked opening rows.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogPlan {
    num_vars: usize,
    num_polys: usize,
    precommitted: Vec<Vec<u8>>,
    trace: Vec<u8>,
}

/// Why a setup refuses a persisted plan.
#[derive(Debug, thiserror::Error)]
pub enum WhiteoutPlanError {
    #[error("Whiteout setup has no response row for {num_vars} variables")]
    MissingResponse { num_vars: usize },
    /// Not this version's plan format.
    #[error("Whiteout plans {magic:?} version {version} are not this build's format")]
    Format { magic: [u8; 4], version: u32 },
    /// A row fails Akita's artifact decoding, audit or admission, or its
    /// residual shape at the cut cannot be derived.
    #[error("a persisted Whiteout plan is inadmissible: {0}")]
    Inadmissible(#[source] AkitaError),
    /// The plans are not this statement's: another trace group, rows for
    /// other precommitted groups or other row counts, a response row of
    /// another length, or no response row for a length the proof needs.
    #[error("a persisted Whiteout plan is for another statement: {0}")]
    Statement(#[source] AkitaError),
}

impl From<WhiteoutPlanError> for ZkError {
    fn from(error: WhiteoutPlanError) -> Self {
        Self::Whiteout(Box::new(error))
    }
}

/// A setup's zero-knowledge catalog: one row per masked opening shape the
/// setup admits (its trace group with each admitted combination of
/// precommitted groups), the precommitted groups' hiding profiles, and the
/// Akita verifier over the setup's verifier material: one prefix of Akita's
/// setup stream, derived from its seed, that also holds the residual matrices
/// and the response verifiers' matrices. A prover's setup and backend are
/// built only on the first prover call. One backend commits every hiding
/// group and proves every opening: Akita cannot import a hiding commitment
/// into another backend.
pub struct WhiteoutCatalog {
    scheme: AkitaCommitmentScheme<WhiteoutOneHot>,
    material: AkitaBackendVerifierSetup,
    precommitted: BTreeMap<PrecommittedShape, PrecommittedRow>,
    num_vars: usize,
    num_polys: usize,
    #[cfg(feature = "zk-prover")]
    prover: OnceLock<CatalogProver>,
}

/// A prover's setup over the catalog's capacity and the backend built on it.
#[cfg(feature = "zk-prover")]
struct CatalogProver {
    setup: AkitaBackendProverSetup,
    backend: AkitaBackend,
}

impl WhiteoutCatalog {
    /// Plans the zero-knowledge row of a trace group of `num_polys` K=16
    /// one-hot polynomials in `num_vars` variables beside each admitted
    /// list of precommitted groups, and builds the verifier, whose material
    /// also covers `responses`. Deterministic, so both roles derive the same
    /// catalog, whose digest the opening absorbs.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if a shape has no zero-knowledge row.
    fn plan(
        num_vars: usize,
        num_polys: usize,
        admitted: &BTreeSet<Vec<PrecommittedShape>>,
        responses: &ResponseRows,
    ) -> Result<(Self, PlannedRows), ZkError> {
        let chunking = ChunkedWitnessCfg::default_non_chunked();
        let mut precommitted = BTreeMap::new();
        for shape in admitted.iter().flatten() {
            if let Entry::Vacant(entry) = precommitted.entry(*shape) {
                let _ = entry.insert(PrecommittedRow::plan(*shape)?);
            }
        }
        let final_group =
            OpeningClaimsLayout::new(num_vars, num_polys)?.root_final_group_layout()?;
        let mut rows = Vec::with_capacity(admitted.len());
        for groups in admitted {
            let (key, contracts) = row_key(final_group, &precommitted, groups)?;
            let planned = tracing::info_span!("find_zk_schedule").in_scope(|| {
                akita_planner::find_zk_schedule(
                    &key,
                    WhiteoutOneHot::committed_source_contract()?,
                    &contracts,
                    &policy_of::<WhiteoutOneHot>(chunking),
                    WhiteoutOneHot::ring_challenge_config,
                    ZkOpeningSchedule::new(ZK_CUT_LEVEL)?,
                )
            })?;
            rows.push((
                CommittedGroupBatchProfile {
                    final_group: GroupCommitPhaseParams::try_from_params(
                        key.final_group,
                        &planned.schedule.root.params,
                    )?,
                    precommitteds: key.precommitteds,
                },
                planned.schedule,
            ));
        }
        let catalog = TrustedScheduleCatalog::<WhiteoutOneHot>::try_from_rows(rows, chunking)?;
        let residuals = residuals(
            &catalog,
            row_keys(&catalog, final_group, &precommitted, admitted)?,
        )?;
        Ok((
            Self::build(
                catalog,
                precommitted,
                (num_vars, num_polys),
                admitted,
                &residuals,
                responses,
                None,
            )?,
            residuals,
        ))
    }

    /// [`plan`](Self::plan)'s catalog from its persisted `plan`: each row
    /// audited and admitted by Akita ([`load_single_row`],
    /// `TrustedScheduleCatalog::from_artifact_bytes`), then required to be
    /// exactly the rows of this statement: the trace group's shape, one row
    /// per admitted list of precommitted groups keyed on their profiles, and
    /// one row per precommitted shape keyed on its variable count.
    fn load(
        plan: &CatalogPlan,
        num_vars: usize,
        num_polys: usize,
        admitted: &BTreeSet<Vec<PrecommittedShape>>,
        responses: &ResponseRows,
        material: Option<AkitaBackendVerifierSetup>,
    ) -> Result<(Self, PlannedRows), ZkError> {
        let statement = |what: String| WhiteoutPlanError::Statement(AkitaError::InvalidSetup(what));
        if (plan.num_vars, plan.num_polys) != (num_vars, num_polys) {
            return Err(statement(format!(
                "planned for a trace group of {} variables and {} polynomials, not {num_vars} and \
                 {num_polys}",
                plan.num_vars, plan.num_polys
            ))
            .into());
        }
        let shapes = admitted.iter().flatten().collect::<BTreeSet<_>>();
        if plan.precommitted.len() != shapes.len() {
            return Err(statement(format!(
                "{} precommitted rows for {} precommitted shapes",
                plan.precommitted.len(),
                shapes.len()
            ))
            .into());
        }
        let precommitted = shapes
            .into_iter()
            .zip(&plan.precommitted)
            .map(|(shape, artifact)| Ok((*shape, PrecommittedRow::load(*shape, artifact)?)))
            .collect::<Result<BTreeMap<_, _>, WhiteoutPlanError>>()?;
        let catalog = TrustedScheduleCatalog::<WhiteoutOneHot>::from_artifact_bytes(
            &plan.trace,
            ChunkedWitnessCfg::default_non_chunked(),
        )
        .map_err(WhiteoutPlanError::Inadmissible)?;
        let final_group =
            OpeningClaimsLayout::new(num_vars, num_polys)?.root_final_group_layout()?;
        let keys = row_keys(&catalog, final_group, &precommitted, admitted)
            .map_err(WhiteoutPlanError::Statement)?;
        let residuals = residuals(&catalog, keys).map_err(WhiteoutPlanError::Inadmissible)?;
        Ok((
            Self::build(
                catalog,
                precommitted,
                (num_vars, num_polys),
                admitted,
                &residuals,
                responses,
                material,
            )?,
            residuals,
        ))
    }

    /// The catalog of `catalog`'s rows and the `precommitted` rows, with its
    /// verifier over material derived from Akita's setup seed. The material
    /// covers what this catalog's verifier, the `residuals`' matrices and
    /// every row of `responses` read, so each is a view of it.
    fn build(
        catalog: TrustedScheduleCatalog<WhiteoutOneHot>,
        precommitted: BTreeMap<PrecommittedShape, PrecommittedRow>,
        (num_vars, num_polys): (usize, usize),
        admitted: &BTreeSet<Vec<PrecommittedShape>>,
        residuals: &PlannedRows,
        responses: &ResponseRows,
        material: Option<AkitaBackendVerifierSetup>,
    ) -> Result<Self, ZkError> {
        let scheme = AkitaCommitmentScheme::new(catalog);
        let max_batched_polys = admitted
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .checked_add(num_polys)
            .ok_or_else(|| AkitaError::InvalidSetup("batched polynomial count overflows".into()))?;
        // The setup's matrix covers the widest group: a precommitted group
        // (a committed program's bytecode) can be wider than the trace.
        let max_num_vars = precommitted
            .keys()
            .map(|shape| shape.num_vars)
            .fold(num_vars, usize::max);
        let mut requirements = AkitaVerifierRequirements::new(max_num_vars, max_batched_polys);
        requirements
            .include(scheme.schedules(), max_num_vars, max_batched_polys)
            .map_err(material_error)?;
        for (_, residual) in residuals.values() {
            requirements.retain(residual.stream_len()?);
        }
        responses.include(&mut requirements)?;
        let material = if let Some(material) = material {
            let descriptor = material.expanded().descriptor();
            if descriptor.num_field_elements < requirements.retained_field_elements
                || descriptor.setup_seed != sample_akita_setup_seed()
                || descriptor.max_num_vars < max_num_vars
                || descriptor.max_num_batched_polys < max_batched_polys
            {
                return Err(AkitaError::InvalidSetup(
                    "prepared Whiteout key does not cover the setup".into(),
                )
                .into());
            }
            material
        } else {
            let _span = tracing::info_span!("whiteout_verifier_material").entered();
            with_backend_pool(|| generate_material(&requirements)).map_err(material_error)?
        };
        Ok(Self {
            scheme,
            material,
            precommitted,
            num_vars,
            num_polys,
            #[cfg(feature = "zk-prover")]
            prover: OnceLock::new(),
        })
    }

    /// The catalog as persisted.
    fn plan_of(&self) -> Result<CatalogPlan, AkitaError> {
        Ok(CatalogPlan {
            num_vars: self.num_vars,
            num_polys: self.num_polys,
            precommitted: self
                .precommitted
                .values()
                .map(PrecommittedRow::artifact)
                .collect::<Result<_, _>>()?,
            trace: self.scheme.schedules().to_artifact_bytes()?,
        })
    }

    /// The prover's setup at the verifier's capacity and its backend, built
    /// on first use. Its matrix is a longer prefix of the stream the
    /// verifier's material is cut from.
    #[cfg(feature = "zk-prover")]
    fn prover(&self) -> Result<&CatalogProver, AkitaError> {
        if let Some(prover) = self.prover.get() {
            return Ok(prover);
        }
        // Rayon work stays outside the cell to avoid worker reentrancy deadlocks.
        let _span = tracing::info_span!("whiteout_catalog_prover_setup").entered();
        let descriptor = self.material.expanded().descriptor();
        let setup = self
            .scheme
            .setup_prover(descriptor.max_num_vars, descriptor.max_num_batched_polys)?;
        let backend = AkitaBackend::new(Arc::clone(&setup.expanded))?;
        Ok(self.prover.get_or_init(|| CatalogProver { setup, backend }))
    }

    /// The backend every hiding source is imported into, built on first
    /// use.
    ///
    /// # Errors
    ///
    /// [`AkitaError`] if the backend cannot be built.
    #[cfg(feature = "zk-prover")]
    pub fn backend(&self) -> Result<&AkitaBackend, AkitaError> {
        Ok(&self.prover()?.backend)
    }

    /// Commits `source`, one dense polynomial, hiding as a precommitted group
    /// of `shape`.
    ///
    /// # Errors
    ///
    /// [`AkitaError::UnsupportedSchedule`] if no admitted opening folds a
    /// group of `shape`, and the backend's errors.
    #[cfg(feature = "zk-prover")]
    pub fn commit_precommitted(
        &self,
        shape: PrecommittedShape,
        source: &SourceHandle<F, F>,
    ) -> Result<(CommittedGroup<F>, AkitaBackendHint), AkitaError> {
        let row = self.precommitted.get(&shape).ok_or_else(|| {
            AkitaError::UnsupportedSchedule(format!(
                "no admitted Whiteout opening folds a precommitted group of {shape:?}"
            ))
        })?;
        let output = row.commit(self.backend()?, source)?;
        Ok((output.committed_group, output.private_handle))
    }

    /// Rebuilds the hiding commitment `expected` of `source`, a precommitted
    /// group of `shape`, from the randomness it was made with (`specs/whiteout-jolt-integration.md` §5).
    ///
    /// # Errors
    ///
    /// [`AkitaError::InvalidInput`] unless (`source`, `randomness`) rebuild
    /// exactly `expected`, and [`commit_precommitted`](Self::commit_precommitted)'s
    /// errors.
    #[cfg(feature = "zk-prover")]
    pub fn recommit_precommitted(
        &self,
        shape: PrecommittedShape,
        source: &SourceHandle<F, F>,
        randomness: &HidingRandomness,
        expected: &Commitment<F>,
    ) -> Result<(CommittedGroup<F>, AkitaBackendHint), AkitaError> {
        let row = self.precommitted.get(&shape).ok_or_else(|| {
            AkitaError::UnsupportedSchedule(format!(
                "no admitted Whiteout opening folds a precommitted group of {shape:?}"
            ))
        })?;
        let output = row.recommit(self.backend()?, source, randomness, expected)?;
        Ok((output.committed_group, output.private_handle))
    }
}

/// Each admitted list of precommitted groups' row key and residual shape.
type PlannedRows = BTreeMap<Vec<PrecommittedShape>, (ScheduleLookupKey, ResidualShape)>;

/// A verifier-material sizing or derivation failure as Akita's setup error.
fn material_error(error: OpeningsError) -> AkitaError {
    AkitaError::InvalidSetup(error.to_string())
}

/// The lookup key of the row folding `groups` beside the trace group
/// `final_group`, and the groups' producer contracts.
fn row_key(
    final_group: PolynomialGroupLayout,
    precommitted: &BTreeMap<PrecommittedShape, PrecommittedRow>,
    groups: &[PrecommittedShape],
) -> Result<(ScheduleLookupKey, Vec<CommittedSourceContract>), AkitaError> {
    let rows = groups
        .iter()
        .map(|shape| {
            precommitted
                .get(shape)
                .ok_or_else(|| AkitaError::InvalidSetup("a precommitted shape has no row".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let contracts = rows
        .iter()
        .map(|row| row.contract())
        .collect::<Result<Vec<_>, _>>()?;
    Ok((
        ScheduleLookupKey {
            root_accumulated_setup_prefix_len: 0,
            final_group,
            precommitteds: rows.iter().map(|row| row.profile).collect(),
        },
        contracts,
    ))
}

/// Each admitted list's row key, each resolving in `catalog`, which must
/// hold exactly those rows.
fn row_keys(
    catalog: &TrustedScheduleCatalog<WhiteoutOneHot>,
    final_group: PolynomialGroupLayout,
    precommitted: &BTreeMap<PrecommittedShape, PrecommittedRow>,
    admitted: &BTreeSet<Vec<PrecommittedShape>>,
) -> Result<Vec<(Vec<PrecommittedShape>, ScheduleLookupKey)>, AkitaError> {
    if catalog.rows().len() != admitted.len() {
        return Err(AkitaError::InvalidSetup(format!(
            "the Whiteout catalog holds {} rows, its setup admits {} shapes",
            catalog.rows().len(),
            admitted.len()
        )));
    }
    admitted
        .iter()
        .map(|groups| {
            let (key, _) = row_key(final_group, precommitted, groups)?;
            let _ = catalog.resolve_key(&key)?;
            Ok((groups.clone(), key))
        })
        .collect()
}

/// Each keyed row's residual shape at its cut.
fn residuals(
    catalog: &TrustedScheduleCatalog<WhiteoutOneHot>,
    keys: Vec<(Vec<PrecommittedShape>, ScheduleLookupKey)>,
) -> Result<PlannedRows, AkitaError> {
    keys.into_iter()
        .map(|(groups, key)| {
            let residual = ResidualShape::new(AdmittedZkSchedule::new(
                catalog.resolve_key(&key)?.schedule(),
                WhiteoutOneHot::committed_source_contract()?,
                <WhiteoutOneHot as CommitmentConfig>::EXT_DEGREE,
            )?)?;
            Ok((groups, (key, residual)))
        })
        .collect()
}

macro_rules! with_matrices {
    ($matrices:expr, $streams:expr, $D:ident, $m:ident, $stream:ident, $body:expr) => {
        match &$matrices {
            RowMatrices::D64($m) => {
                const $D: usize = 64;
                let $stream = &$streams.pad64;
                $body
            }
            RowMatrices::D128($m) => {
                const $D: usize = 128;
                let $stream = &$streams.pad128;
                $body
            }
            RowMatrices::D256($m) => {
                const $D: usize = 256;
                let $stream = &$streams.pad256;
                $body
            }
        }
    };
}

/// One masked opening row and its prepared coefficient capacity.
pub struct WhiteoutTail {
    row: WhiteoutRow,
    responses: Arc<ResponsePlans>,
    streams: Arc<CapacityStreams>,
}

/// The public state one pad layout fixes under a row: the pad commitment
/// `A_p` (with `L`), `A_D`, and the response verifier of `e`'s length. Each is
/// a deterministic function of the row, the layout and the setup's plans,
/// never of a proof.
struct LayoutViews<const D: usize> {
    pads: LatticePadCommitment<F, D>,
    v_matrix: Arc<PreparedMatrix<F, CHALLENGE_RING_DIM>>,
    response: Arc<PreparedResponse>,
}

impl WhiteoutTail {
    fn new(row: WhiteoutRow, responses: Arc<ResponsePlans>, streams: Arc<CapacityStreams>) -> Self {
        Self {
            row,
            responses,
            streams,
        }
    }

    fn geometry<const D: usize>(
        &self,
        layout: &PadLayout,
    ) -> Result<(PadGeometry<D>, SigmaGeometry, usize), ZkError> {
        let placement = PadPlacement::new(layout)?;
        let (rank, randomness) = Self::pad_rank::<D>(&placement)?;
        let pads = PadGeometry::new(placement, rank, randomness)?;
        let sigma = SigmaGeometry::new(
            self.row.shape.segments(),
            self.row.shape.block_shapes(),
            &pads,
        )?;
        let rank = digit_matrix_rank(sigma.v_shape(1)?.width)?;
        Ok((pads, sigma, rank))
    }

    fn layout_state<const D: usize>(
        &self,
        layout: &PadLayout,
        pad_stream: &PreparedMatrix<F, D>,
    ) -> Result<Arc<LayoutViews<D>>, ZkError> {
        let _span = tracing::info_span!("whiteout_response_lookup").entered();
        let (pads, sigma, rank) = self.geometry::<D>(layout)?;
        let v_shape = sigma.v_shape(rank)?;
        let required = WhiteoutCapacity {
            pad: pads.required_coefficients(),
            v: v_shape
                .num_field_elements()
                .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?,
        };
        self.streams.capacity.check(required)?;
        let num_vars = AkitaResponseVerifier::<WhiteoutResponse>::num_vars(
            &sigma.response(rank, &CompressionShape::<WhiteoutResponse>::default())?,
        )?;
        let response = self.responses.get(num_vars)?;
        Ok(Arc::new(LayoutViews {
            pads: tracing::info_span!("whiteout_pad_commitment_setup")
                .in_scope(|| LatticePadCommitment::from_geometry(pads, pad_stream))?,
            v_matrix: Arc::new(
                tracing::info_span!("whiteout_v_matrix_setup")
                    .in_scope(|| self.streams.v.view(v_shape, 0))
                    .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?,
            ),
            response,
        }))
    }

    pub fn requirements(&self, layout: &PadLayout) -> Result<WhiteoutCapacity, ZkError> {
        with_ring_dim!(self.row.ring_dim(), D, {
            let (pads, sigma, rank) = self.geometry::<D>(layout)?;
            Ok(WhiteoutCapacity {
                pad: pads.required_coefficients(),
                v: sigma
                    .v_shape(rank)?
                    .num_field_elements()
                    .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?,
            })
        })
    }

    /// Prices `A_p` and its hiding randomness together: the randomness width
    /// depends on the rank and the rank on the width, so both are iterated
    /// to a fixed point, as Akita's planner prices a hiding commitment
    /// (`price_with_hiding_randomness`, at most 8 rounds).
    fn pad_rank<const D: usize>(placement: &PadPlacement) -> Result<(usize, usize), ZkError> {
        const MAX_ROUNDS: usize = 8;
        let mut randomness_columns = 0;
        for _ in 0..MAX_ROUNDS {
            let rank = sigma_matrix_rank(
                D,
                LatticePadCommitment::<F, D>::width(placement, randomness_columns)?,
            )?;
            let image_coeffs = rank.checked_mul(D).ok_or_else(|| {
                AkitaError::InvalidSetup("hidden image size overflows usize".into())
            })?;
            // r_p must also hide L's R_64 output (Whiteout Σ §1.5).
            let needed = required_randomness_columns::<F>(
                image_coeffs,
                CHALLENGE_RING_DIM,
                D,
                PAD_RANDOMNESS_LOG_BASIS,
                8,
            )
            .map_err(|error| AkitaError::InvalidSetup(error.to_string()))?;
            if needed == randomness_columns {
                return Ok((rank, randomness_columns));
            }
            randomness_columns = needed;
        }
        Err(AkitaError::InvalidSetup(
            "the pad commitment's rank and randomness found no fixed point".into(),
        )
        .into())
    }

    /// The Σ setup over the row's residual with commitment `cm(u_a)` =
    /// `u_a_payload` (and u_a, on the prover's side), and `state`'s pad
    /// commitment with image `pad_image`.
    fn sigma<const D: usize>(
        &self,
        state: &LayoutViews<D>,
        matrices: &ResidualMatrices<D>,
        u_a_payload: &[F],
        u_a: Option<&[F]>,
        pad_image: BlockTarget<F, D>,
    ) -> Result<SigmaSetup<F, D>, ZkError> {
        let _span = tracing::info_span!("whiteout_sigma_setup").entered();
        let blocks = self.row.shape.blocks::<D>(matrices, u_a_payload, u_a)?;
        Ok(SigmaSetup::new(
            self.row.shape.segments(),
            blocks,
            &state.pads,
            pad_image,
            Arc::clone(&state.v_matrix),
        )?)
    }

    /// The public parameters the pads bind before any pad is committed: the
    /// masked catalog's digest and the selected row's, which fix the
    /// masked levels' shapes and the pad commitment's (also absorbed at
    /// Stage 8, `WhiteoutRow::absorb_selection`), and the response
    /// schedule's catalog digest, under whose setup the pads are compressed.
    fn pad_statement<const D: usize>(
        &self,
        layout: &PadLayout,
        stream: &PreparedMatrix<F, D>,
    ) -> Result<([[u8; 32]; 3], Arc<LayoutViews<D>>), ZkError> {
        let state = self.layout_state::<D>(layout, stream)?;
        let digests = [
            self.row.catalog.scheme.schedules().catalog_digest(),
            *self.row.selection()?.row_digest.as_bytes(),
            state.response.row.digest,
        ];
        Ok((digests, state))
    }

    #[cfg(feature = "zk-prover")]
    /// The response prover for the row and `layout`, as the image compression
    /// the masked prefix sends u_a under: the chain the response proof
    /// checks.
    ///
    /// # Errors
    ///
    /// As [`Self::commit_pads`].
    pub fn compression(
        &self,
        layout: &PadLayout,
    ) -> Result<Arc<AkitaResponseProver<WhiteoutResponse>>, ZkError> {
        with_matrices!(self.row.matrices, self.streams, D, _matrices, stream, {
            Ok(self.layout_state::<D>(layout, stream)?.response.prover()?)
        })
    }

    #[cfg(feature = "zk-prover")]
    /// Absorbs the pads' public parameters ([`Self::pad_statement`]), then
    /// samples, commits and sends the pads of `layout` (row 2).
    ///
    /// # Errors
    ///
    /// [`ZkError`] if the layout does not fit or a rank is not priced.
    pub fn commit_pads<R: RngCore + CryptoRng>(
        &self,
        layout: PadLayout,
        transcript: &mut AkitaProverTranscript,
        rng: &mut R,
    ) -> Result<CommittedPads<F, WhiteoutPadOpening<F>>, ZkError> {
        with_matrices!(self.row.matrices, self.streams, D, _matrices, stream, {
            let (digests, state) = self.pad_statement::<D>(&layout, stream)?;
            absorb_pad_statement(transcript, &digests);
            commit_lattice_pads(
                &state.pads,
                state.response.prover()?.as_ref() as &dyn ImageCompression<F>,
                layout,
                transcript,
                rng,
            )
        })
    }

    /// Reads the pad commitment [`Self::commit_pads`] sends, after
    /// absorbing the pads' public parameters ([`Self::pad_statement`]):
    /// persisted plans are public parameters the transcript binds before
    /// the first challenge that depends on them.
    ///
    /// # Errors
    ///
    /// As [`Self::commit_pads`], or a truncated payload.
    pub fn receive_pads(
        &self,
        layout: PadLayout,
        transcript: &mut AkitaVerifierTranscript<'_>,
    ) -> Result<CommittedPadsView<LatticePadPayload<F>>, ZkError> {
        let _span = tracing::info_span!("whiteout_receive_pads").entered();
        with_matrices!(self.row.matrices, self.streams, D, _matrices, stream, {
            let (digests, state) = self.pad_statement::<D>(&layout, stream)?;
            absorb_pad_statement(transcript, &digests);
            receive_lattice_pads(
                &state.pads,
                &CompressionShape::<WhiteoutResponse>::default(),
                layout,
                transcript,
            )
        })
    }

    #[cfg(feature = "zk-prover")]
    /// Proves `relation` over the committed pads and the residual opening:
    /// the Σ-protocol, then the response proof (rows 7 and 8).
    ///
    /// # Errors
    ///
    /// [`ZkError`] if the relation does not hold, or the Σ attempts are
    /// exhausted (the caller reruns).
    pub fn prove<R: RngCore + CryptoRng + Send>(
        &self,
        relation: &ResidualRelation<F>,
        residual: &ResidualOpening<F, F>,
        pads: &CommittedPads<F, WhiteoutPadOpening<F>>,
        transcript: &mut AkitaProverTranscript,
        rng: &mut R,
    ) -> Result<(), ZkError> {
        let layout = pads.layout().clone();
        with_matrices!(self.row.matrices, self.streams, D, _matrices, stream, {
            let opening = pads.opening();
            let state = self.layout_state::<D>(&layout, stream)?;
            let sigma = self.sigma::<D>(
                &state,
                _matrices,
                &residual.residual_payload,
                Some(&residual.residual_commitment),
                BlockTarget::Compressed {
                    payload: opening.payload().to_vec(),
                    image: Some(opening.image::<D>()?),
                },
            )?;
            let lifted = sigma.lift(relation, &self.row.shape.claims(&residual.point)?)?;
            let s = self.row.shape.residual(residual)?;
            let attempts = prove_sigma(
                &sigma,
                &lifted,
                &s,
                opening.opening(),
                transcript,
                rng,
                &ProfiledResponse(state.response.prover()?.as_ref()),
            )?;
            tracing::info!(attempts, "Whiteout Σ accepted");
            Ok(())
        })
    }

    /// Verifies the Σ-protocol and the response proof of `relation` against
    /// the masked prefix's residual claim.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if either rejects.
    pub fn verify(
        &self,
        relation: &ResidualRelation<F>,
        prefix: &MaskedPrefix<F, F>,
        pads: &CommittedPadsView<LatticePadPayload<F>>,
        transcript: &mut AkitaVerifierTranscript<'_>,
    ) -> Result<(), ZkError> {
        let _span = tracing::info_span!("whiteout_verify_tail").entered();
        let layout = pads.layout().clone();
        let claim = prefix.residual_claim();
        with_matrices!(self.row.matrices, self.streams, D, _matrices, stream, {
            let state = self.layout_state::<D>(&layout, stream)?;
            let sigma = self.sigma::<D>(
                &state,
                _matrices,
                claim.commitment(),
                None,
                BlockTarget::Compressed {
                    payload: pads.commitment().payload.clone(),
                    image: None,
                },
            )?;
            let claims = self.row.shape.claims(claim.point())?;
            let lifted =
                tracing::info_span!("whiteout_lift").in_scope(|| sigma.lift(relation, &claims))?;
            tracing::info_span!("whiteout_verify_sigma")
                .in_scope(|| verify_sigma(&sigma, &lifted, transcript, &state.response.verifier))
        })
    }
}

/// One masked opening shape: its row and the Σ-protocol's commitments,
/// which own the row.
pub struct WhiteoutShape {
    tail: WhiteoutTail,
}

impl WhiteoutShape {
    pub fn row(&self) -> &WhiteoutRow {
        &self.tail.row
    }

    pub fn tail(&self) -> &WhiteoutTail {
        &self.tail
    }
}

/// The response prover under a `whiteout_prove_response` span that logs the
/// response proof's bytes, so a profile splits the tail into the Σ-protocol
/// and the response proof.
#[cfg(feature = "zk-prover")]
struct ProfiledResponse<'a>(&'a AkitaResponseProver<WhiteoutResponse>);

#[cfg(feature = "zk-prover")]
impl<const D: usize> ResponseProver<F, D> for ProfiledResponse<'_> {
    type Sponge = TranscriptSponge;

    fn compression(&self) -> &dyn ImageCompression<F> {
        self.0
    }

    fn prove_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        witness: &ResponseWitness,
        transcript: &mut AkitaProverTranscript,
    ) -> Result<(), ZkError> {
        let _span = tracing::info_span!("whiteout_prove_response").entered();
        let before = transcript.narg_string().len();
        #[cfg(feature = "logging")]
        {
            *census::RESPONSE_START
                .lock()
                .unwrap_or_else(PoisonError::into_inner) = Some(before);
        }
        ResponseProver::<F, D>::prove_response(self.0, statement, witness, transcript)?;
        tracing::info!(
            response_bytes = transcript.narg_string().len().saturating_sub(before),
            "Whiteout response proof"
        );
        Ok(())
    }

    fn verify_response(
        &self,
        statement: &ResponseStatement<'_, F, D>,
        transcript: &mut AkitaVerifierTranscript<'_>,
    ) -> Result<(), ZkError> {
        ResponseProver::<F, D>::verify_response(self.0, statement, transcript)
    }
}

/// Absorbs the pads' public parameters (the catalog digest, the selected
/// row's digest and the response schedule's digest) before t_p. A plain
/// function, outside `with_ring_dim!`, so the Fiat-Shamir inventory sees
/// these absorb sites.
fn absorb_pad_statement<T: Transcript>(transcript: &mut T, digests: &[[u8; 32]; 3]) {
    for digest in digests {
        transcript.append(digest);
    }
}

mod opening;

pub(crate) use opening::opens_whiteout;
#[cfg(feature = "zk-prover")]
pub use opening::WhiteoutResidual;
#[cfg(feature = "zk-prover")]
pub(crate) use opening::{opening_shape, whiteout};

#[cfg(all(test, feature = "zk-prover"))]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::string_slice,
    reason = "tests fail loudly"
)]
mod tests;

/// The Whiteout census's seam (`logging`): where the last response proof
/// began, which splits Σ from the response proof (Akita's response prover
/// writes no message record). Process-wide: Akita's pool may run the
/// response prover off the proving thread.
#[cfg(all(feature = "zk-prover", feature = "logging"))]
pub mod census {
    use std::sync::{Mutex, PoisonError};

    pub(super) static RESPONSE_START: Mutex<Option<usize>> = Mutex::new(None);

    /// Takes the argument-string offset at which the last response proof
    /// began.
    pub fn take_response_start() -> Option<usize> {
        RESPONSE_START
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }
}
