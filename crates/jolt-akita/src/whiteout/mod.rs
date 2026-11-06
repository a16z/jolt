//! Whiteout: Jolt's zero-knowledge proofs over Akita
//! (`specs/whiteout-jolt-integration.md`). The trace group is committed
//! hiding under a zero-knowledge row, opened through Akita's masked levels to
//! the cut, and the residual relation is proved by jolt-zk's Σ-protocol with
//! Akita's response proof.
//!
//! Scope:
//! - Zero-knowledge rows are planned on first use for the exact shape, in
//!   either role. Response schedules come from the packaged response catalog
//!   and are planned only for a length it lacks. Both are persisted with the
//!   setup ([`WhiteoutPlans`]); a setup that carries them validates and
//!   loads them instead of planning.
//! - One trace group under `JoltOneHotK16`, beside the precommitted dense
//!   groups its setup admits (advice, field increments, committed programs),
//!   each committed hiding under the profile of its own shape's
//!   zero-knowledge row (Akita `specs/whiteout-akita.md` §3.5).
//! - `A_p` is priced at the Σ collision radius ([`sigma_matrix_rank`]),
//!   `A_D` at its digits' difference bound ([`digit_matrix_rank`]).

use std::any::Any;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use akita_config::proof_optimized::fp128::ResponseDigits;
use akita_config::{policy_of, CommitmentConfig, TrustedScheduleCatalog};
use akita_params::sis::sigma_binding::sigma_binding_min_rank;
use akita_params::sis::CommittedSourceContract;
use akita_params::zk::{
    hiding_randomness_columns, AdmittedZkSchedule, DigitAlphabet, ImageShape, ZkOpeningSchedule,
    ZK_CUT_LEVEL,
};
#[cfg(feature = "zk-prover")]
use akita_params::PrecommittedGroupProfiles;
use akita_params::{
    BasisMode, ChunkedWitnessCfg, CommittedGroupBatchProfile, FoldSchedule, GroupCommitPhaseParams,
    OpeningClaimsLayout, OpeningScheduleSelection, PolynomialGroupLayout, ScheduleLookupKey,
};
#[cfg(feature = "zk-prover")]
use akita_pcs::zk::{prove_masked_prefix, ResidualOpening};
use akita_pcs::zk::{
    AkitaResponseProver, CompressionShape, MaskedPrefix, ResidualMatrices, ResidualShape,
    ZkCommitmentConfig,
};
#[cfg(feature = "zk-prover")]
use akita_pcs::SelectedProverOpeningData;
use akita_pcs::{AkitaCommitmentScheme, AkitaError, AkitaVerifier};
#[cfg(feature = "zk-prover")]
use akita_pcs::{CommitOutput, GroupContext, HidingRandomness, SourceHandle};
use akita_schedules::digit_commitment_min_rank;
#[cfg(feature = "zk-prover")]
use akita_types::Commitment;
use akita_types::{CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_crypto::lattice::{
    derive_public_matrix_prefix, PreparedMatrix, PublicMatrixSeed, CHALLENGE_RING_DIM,
};
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
    LatticePadPayload, MaskedOpeningShape, PadPlacement, PrecommittedShape, SigmaSetup,
    PAD_RANDOMNESS_LOG_BASIS, RESPONSE_LOG_BASIS,
};
#[cfg(feature = "zk-prover")]
use rand_core::{CryptoRng, RngCore};

#[cfg(feature = "zk-prover")]
use crate::adapters::AkitaBackendHint;
#[cfg(feature = "zk-prover")]
use crate::adapters::AkitaBackendProverSetup;
use crate::adapters::{AkitaBackend, AkitaCommitment, AkitaField, WhiteoutSource};
use crate::configs::{JoltDenseBounded, JoltDenseFull, JoltOneHotK16};
use crate::schedule_registry::GroupedScheduleParams;
#[cfg(feature = "zk-prover")]
use crate::scheme::AkitaProverTranscript;
use crate::scheme::AkitaVerifierTranscript;
#[cfg(feature = "zk-prover")]
use akita_transcript::TranscriptSponge;

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
const WHITEOUT_PLANS_VERSION: u32 = 1;

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
    /// the setup plans each length on first use.
    responses: Option<BTreeMap<usize, Vec<u8>>>,
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
/// precommitted groups), the precommitted groups' hiding profiles, the Akita
/// verifier, and, for a prover, the prover's setup (the one the verifier's is
/// derived from, so both share one public matrix) and its backend, built only
/// on the first prover call (a verifier never builds a backend). One backend
/// commits every hiding group and proves every opening: Akita cannot import a
/// hiding commitment into another backend.
pub struct WhiteoutCatalog {
    scheme: AkitaCommitmentScheme<WhiteoutOneHot>,
    verifier: AkitaVerifier<WhiteoutOneHot>,
    precommitted: BTreeMap<PrecommittedShape, PrecommittedRow>,
    num_vars: usize,
    num_polys: usize,
    #[cfg(feature = "zk-prover")]
    setup: AkitaBackendProverSetup,
    backend: OnceLock<AkitaBackend>,
}

impl WhiteoutCatalog {
    /// Plans the zero-knowledge row of a trace group of `num_polys` K=16
    /// one-hot polynomials in `num_vars` variables beside each admitted
    /// list of precommitted groups, and builds the verifier. Deterministic,
    /// so both roles derive the same catalog, whose digest the opening
    /// absorbs.
    ///
    /// # Errors
    ///
    /// [`AkitaError`] if a shape has no zero-knowledge row.
    fn plan(
        num_vars: usize,
        num_polys: usize,
        admitted: &BTreeSet<Vec<PrecommittedShape>>,
    ) -> Result<(Self, PlannedRows), AkitaError> {
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
            Self::build(catalog, precommitted, num_vars, num_polys, admitted)?,
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
            Self::build(catalog, precommitted, num_vars, num_polys, admitted)?,
            residuals,
        ))
    }

    /// The catalog of `catalog`'s rows and the `precommitted` rows, with
    /// its verifier.
    fn build(
        catalog: TrustedScheduleCatalog<WhiteoutOneHot>,
        precommitted: BTreeMap<PrecommittedShape, PrecommittedRow>,
        num_vars: usize,
        num_polys: usize,
        admitted: &BTreeSet<Vec<PrecommittedShape>>,
    ) -> Result<Self, AkitaError> {
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
        // Akita derives a verifier setup only from a prover setup, which
        // only a prover keeps.
        let setup = scheme.setup_prover(max_num_vars, max_batched_polys)?;
        let verifier = scheme.verifier(scheme.setup_verifier(&setup)?)?;
        Ok(Self {
            scheme,
            verifier,
            precommitted,
            num_vars,
            num_polys,
            #[cfg(feature = "zk-prover")]
            setup,
            backend: OnceLock::new(),
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

    /// The backend every hiding source is imported into, built on first
    /// use.
    ///
    /// # Errors
    ///
    /// [`AkitaError`] if the backend cannot be built.
    pub fn backend(&self) -> Result<&AkitaBackend, AkitaError> {
        if let Some(backend) = self.backend.get() {
            return Ok(backend);
        }
        let _span = tracing::info_span!("whiteout_catalog_prover_setup").entered();
        // The prover setup's matrix, which `prove_masked` pairs with that
        // setup's descriptor and prefix slots.
        #[cfg(feature = "zk-prover")]
        let expanded = &self.setup.expanded;
        #[cfg(not(feature = "zk-prover"))]
        let expanded = self.verifier.setup().expanded();
        let backend = AkitaBackend::new(Arc::clone(expanded))?;
        Ok(self.backend.get_or_init(|| backend))
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

    /// Akita's setup stream, from which the residual blocks are cut.
    fn stream(&self) -> &[F] {
        self.verifier
            .setup()
            .expanded()
            .shared_matrix()
            .as_field_slice()
    }
}

/// Each admitted list of precommitted groups' row key and residual shape.
type PlannedRows = BTreeMap<Vec<PrecommittedShape>, (ScheduleLookupKey, ResidualShape)>;

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

/// One masked opening shape's row: its catalog, the row's lookup key, the
/// residual shape at its cut and, once built, its opening matrices.
pub struct WhiteoutRow {
    catalog: Arc<WhiteoutCatalog>,
    key: ScheduleLookupKey,
    shape: ResidualShape,
    /// `ResidualMatrices<D>` at the row's ring dimension: a function of the
    /// row and the catalog's setup stream, so cut once
    /// ([`Self::residual_matrices`]).
    matrices: OnceLock<Arc<dyn Any + Send + Sync>>,
}

impl WhiteoutRow {
    /// The residual commitment's ring dimension, which the pad commitment
    /// and the Σ-protocol share.
    pub fn ring_dim(&self) -> usize {
        self.shape.ring_dim()
    }

    /// The schedule row both roles bind.
    pub fn selection(&self) -> Result<OpeningScheduleSelection, AkitaError> {
        Ok(self
            .catalog
            .scheme
            .schedules()
            .resolve_key(&self.key)?
            .selection())
    }

    /// The setup's catalog.
    pub fn catalog(&self) -> &WhiteoutCatalog {
        &self.catalog
    }

    /// Commits the trace group `source` hiding, under this row's profile,
    /// keyed on its precommitted groups: the backend draws fresh randomness
    /// for every masked-level slice.
    ///
    /// # Errors
    ///
    /// The backend's errors.
    #[cfg(feature = "zk-prover")]
    pub fn commit_hiding(
        &self,
        source: &SourceHandle<F, F>,
    ) -> Result<(CommittedGroup<F>, AkitaBackendHint), AkitaError> {
        let backend = self.catalog.backend()?;
        let schedules = self.catalog.scheme.schedules();
        let output = if self.key.precommitteds.is_empty() {
            backend.commit(
                schedules,
                source,
                GroupContext::scheduler_without_precommitted_groups(),
            )?
        } else {
            let profiles =
                PrecommittedGroupProfiles::from_profiles(self.key.precommitteds.clone())?;
            backend.commit(
                schedules,
                source,
                GroupContext::scheduler_with_precommitted_groups(&profiles),
            )?
        };
        Ok((output.committed_group, output.private_handle))
    }

    /// Proves the masked levels through the cut for `groups` (precommitted,
    /// then the trace group; each its commitment, handle, point in Akita's
    /// order and true evaluations), whose claims `forms` state in the same
    /// order, sending u_a compressed under `compression`
    /// ([`WhiteoutTail::compression`]). Returns the residual opening and
    /// Akita's auxiliary values.
    ///
    /// # Errors
    ///
    /// [`AkitaError::FoldOverflow`] on a one-shot fold overflow (the caller
    /// reruns), and the masked prover's errors.
    #[cfg(feature = "zk-prover")]
    pub fn prove_masked(
        &self,
        groups: Vec<ProverGroup>,
        forms: &[AffineRow<F>],
        pads: &mut CommittedPads<F, WhiteoutPadOpening<F>>,
        transcript: &mut AkitaProverTranscript,
        compression: &dyn ImageCompression<F>,
    ) -> Result<(ResidualOpening<F, F>, Vec<F>), AkitaError> {
        let backend = self.catalog.backend()?;
        let setup = &self.catalog.setup;
        let mut claims = Vec::with_capacity(groups.len());
        let mut handles = Vec::with_capacity(groups.len());
        for group in groups {
            claims.push(PolynomialGroupClaims::new(
                group.point,
                group.evaluations,
                group.commitment,
            )?);
            handles.push(group.handle);
        }
        let opening = SelectedProverOpeningData::from_committed_claims::<WhiteoutOneHot>(
            OpeningClaims::from_groups(claims)?,
            handles,
            self.catalog.scheme.schedules(),
        )?;
        if opening.selection() != self.selection()? {
            return Err(AkitaError::InvalidInput(
                "the opening's groups select another Whiteout row".into(),
            ));
        }
        let schedules = self.catalog.scheme.schedules();
        transcript.append(&schedules.catalog_digest());
        transcript.append(opening.selection().row_digest.as_bytes());
        let resolved = schedules.resolve_selection(opening.selection())?;
        let prefix_ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
            resolved.schedule(),
            opening.opening_layout(),
        )?;
        let prefixes = backend.import_setup_prefixes(&setup.prefix_slots, &prefix_ids)?;
        let prefix = prove_masked_prefix::<WhiteoutOneHot, _, _>(
            setup.expanded.descriptor(),
            schedules,
            backend,
            &prefixes,
            opening,
            &forms.to_vec(),
            pads,
            transcript,
            compression,
            BasisMode::Lagrange,
        )?;
        // The record closes a message where the next begins; Jolt writes Σ
        // next, so close the cut's last message here.
        #[cfg(feature = "logging")]
        akita_transcript::finish_proof_ranges(transcript);
        Ok(prefix)
    }

    /// Replays the masked levels through the cut on `pads`; `groups` are the
    /// decoded commitments at their points (Akita's order), precommitted
    /// first.
    ///
    /// # Errors
    ///
    /// The masked verifier's errors, or a row other than this shape's.
    pub fn verify_masked<C>(
        &self,
        groups: &[(CommittedGroup<F>, Vec<F>)],
        forms: Vec<AffineRow<F>>,
        pads: &mut CommittedPadsView<C>,
        transcript: &mut AkitaVerifierTranscript<'_>,
    ) -> Result<MaskedPrefix<F, F>, AkitaError> {
        let selection = self.absorb_selection(transcript)?;
        self.catalog.verifier.verify_masked_prefix(
            transcript,
            pads,
            masked_statement(selection, groups)?,
            forms,
            &CompressionShape::<WhiteoutResponse>::default(),
            BasisMode::Lagrange,
        )
    }

    /// The pad-layout replay of [`Self::verify_masked`]: counts the pads of
    /// the masked messages.
    ///
    /// # Errors
    ///
    /// The counting verifier's errors.
    pub fn count_masked(
        &self,
        groups: &[(CommittedGroup<F>, Vec<F>)],
        forms: Vec<AffineRow<F>>,
        view: &mut CommittedPadsView<()>,
        transcript: &mut AkitaVerifierTranscript<'_>,
    ) -> Result<MaskedPrefix<F, F>, AkitaError> {
        let selection = self.absorb_selection(transcript)?;
        self.catalog.verifier.count_masked_prefix(
            transcript,
            view,
            masked_statement(selection, groups)?,
            forms,
            &CompressionShape::<WhiteoutResponse>::default(),
            BasisMode::Lagrange,
        )
    }

    /// Absorbs the catalog and the selected row, both derived from the
    /// statement: the prover sends no selection.
    fn absorb_selection(
        &self,
        transcript: &mut AkitaVerifierTranscript<'_>,
    ) -> Result<OpeningScheduleSelection, AkitaError> {
        let selection = self.selection()?;
        transcript.append(&self.catalog.scheme.schedules().catalog_digest());
        transcript.append(selection.row_digest.as_bytes());
        Ok(selection)
    }

    /// Decodes the precommitted and trace commitments under this row's
    /// profiles, in the row's group order.
    pub(crate) fn backend_commitments(
        &self,
        precommitted: &[&AkitaCommitment],
        trace: &AkitaCommitment,
    ) -> Result<(Vec<CommittedGroup<F>>, CommittedGroup<F>), AkitaError> {
        if precommitted.len() != self.key.precommitteds.len() {
            return Err(AkitaError::InvalidInput(format!(
                "the Whiteout row opens {} precommitted groups, not {}",
                self.key.precommitteds.len(),
                precommitted.len()
            )));
        }
        crate::shape_guard::deserialize_checked_grouped_backend_payload(
            self.catalog.scheme.schedules(),
            precommitted,
            trace,
            self.selection()?,
        )
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))
    }

    /// The residual's opening matrices, cut from Akita's setup stream on
    /// first use and kept: they depend on the row and the setup alone.
    fn residual_matrices<const D: usize>(&self) -> Result<Arc<ResidualMatrices<D>>, ZkError> {
        let matrices = if let Some(matrices) = self.matrices.get() {
            Arc::clone(matrices)
        } else {
            // Built outside the cell: the build runs on Rayon, and a worker
            // waiting inside `get_or_init` could steal a job that waits on it.
            let built: Arc<dyn Any + Send + Sync> = Arc::new(
                tracing::info_span!("whiteout_residual_matrices")
                    .in_scope(|| self.shape.matrices::<D>(self.catalog.stream()))?,
            );
            Arc::clone(self.matrices.get_or_init(|| built))
        };
        downcast(matrices)
    }
}

/// `state` as the `T` its cache holds at this ring dimension.
fn downcast<T: Any + Send + Sync>(state: Arc<dyn Any + Send + Sync>) -> Result<Arc<T>, ZkError> {
    state.downcast::<T>().map_err(|_| {
        AkitaError::Internal("a Whiteout cache holds another ring dimension's state".into()).into()
    })
}

/// The verifier's statement for `groups` at their points (Akita's
/// order): its evaluations are never read, the masked forms replace
/// them.
fn masked_statement(
    selection: OpeningScheduleSelection,
    groups: &[(CommittedGroup<F>, Vec<F>)],
) -> Result<GroupBatchStatement<'_, F, F>, AkitaError> {
    let claims = groups
        .iter()
        .map(|(commitment, point)| {
            PolynomialGroupClaims::new(
                point.clone(),
                vec![F::default(); commitment.profile().group.num_polynomials()],
                commitment,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    GroupBatchStatement::new(selection, OpeningClaims::from_groups(claims)?)
}

/// One group the prover opens: its commitment, the handle retaining its
/// source, its point in Akita's order and its true evaluations.
#[cfg(feature = "zk-prover")]
pub struct ProverGroup {
    pub commitment: CommittedGroup<F>,
    pub handle: AkitaBackendHint,
    pub point: Vec<F>,
    pub evaluations: Vec<F>,
}

/// The Σ-protocol's commitments for one masked opening shape: its row and,
/// per pad layout, `A_p`, `A_D` and the response proof ([`LayoutState`]).
/// The tail owns its row, so a layout's state is always that row's.
pub struct WhiteoutTail {
    row: WhiteoutRow,
    responses: Arc<ResponsePlans>,
    /// Each pad layout's `LayoutState<D>` at the row's ring dimension, keyed
    /// by the whole layout, at most [`MAX_CACHED_LAYOUTS`] of them.
    layouts: Mutex<Vec<(PadLayout, Arc<dyn Any + Send + Sync>)>>,
}

/// The most pad layouts a tail keeps state for. The row fixes the trace
/// group and the precommitted groups, but the layout also counts the masked
/// messages of stages 1–7, which depend on dimensions each proof's header
/// carries (trace length, RAM size, read-write and one-hot configuration),
/// so a verifier can see as many layouts as headers it is sent. Past the cap a layout's state is built for its
/// verify and dropped: correct, only slower.
const MAX_CACHED_LAYOUTS: usize = 8;

/// The public state one pad layout fixes under a row: the pad commitment
/// `A_p` (with `L`), `A_D`, and the response prover of `e`'s length. Each is
/// a deterministic function of the row, the layout and the setup's plans,
/// never of a proof.
struct LayoutState<const D: usize> {
    pads: LatticePadCommitment<F, D>,
    v_matrix: Arc<PreparedMatrix<F, CHALLENGE_RING_DIM>>,
    response: Arc<PlannedResponse>,
}

impl WhiteoutTail {
    fn new(row: WhiteoutRow, responses: Arc<ResponsePlans>) -> Self {
        Self {
            row,
            responses,
            layouts: Mutex::default(),
        }
    }

    /// `layout`'s public state under the tail's row, built on first use and
    /// kept, for at most [`MAX_CACHED_LAYOUTS`] layouts.
    fn layout_state<const D: usize>(
        &self,
        layout: &PadLayout,
    ) -> Result<Arc<LayoutState<D>>, ZkError> {
        let _span = tracing::info_span!("whiteout_response_lookup").entered();
        let find = |layouts: &[(PadLayout, Arc<dyn Any + Send + Sync>)]| {
            layouts
                .iter()
                .find(|(seen, _)| seen == layout)
                .map(|(_, state)| Arc::clone(state))
        };
        let cached = find(&self.layouts.lock().unwrap_or_else(PoisonError::into_inner));
        let state = if let Some(state) = cached {
            state
        } else {
            // Built outside the lock, as `WhiteoutRow::residual_matrices`.
            let built: Arc<dyn Any + Send + Sync> = Arc::new(self.build_layout_state::<D>(layout)?);
            let mut layouts = self.layouts.lock().unwrap_or_else(PoisonError::into_inner);
            find(&layouts).unwrap_or_else(|| {
                if layouts.len() < MAX_CACHED_LAYOUTS {
                    layouts.push((layout.clone(), Arc::clone(&built)));
                }
                built
            })
        };
        downcast(state)
    }

    fn build_layout_state<const D: usize>(
        &self,
        layout: &PadLayout,
    ) -> Result<LayoutState<D>, ZkError> {
        let pads = Self::pad_commitment::<D>(layout)?;
        let blocks = self
            .row
            .shape
            .blocks::<D>(&*self.row.residual_matrices::<D>()?, &[], None)?;
        let v_matrix = {
            let _span = tracing::info_span!("whiteout_v_matrix_setup").entered();
            let v_rank = digit_matrix_rank(SigmaSetup::<F, D>::v_width(&blocks, &pads))?;
            let v_stream = public_stream(
                SigmaSetup::<F, D>::v_stream_len(&blocks, &pads, v_rank),
                V_STREAM_SEED,
            )?;
            SigmaSetup::<F, D>::v_matrix_of(&blocks, &pads, v_rank, &v_stream)?
        };
        // e's layout is a function of the public Σ setup, whose compressed
        // targets it reads only by length: the setup over empty payloads
        // sizes it.
        let sigma = SigmaSetup::new(
            self.row.shape.segments(),
            blocks,
            &pads,
            BlockTarget::Compressed {
                payload: Vec::new(),
                image: None,
            },
            Arc::clone(&v_matrix),
        )?;
        let num_vars = AkitaResponseProver::<WhiteoutResponse>::num_vars(
            &sigma.response_layout(&CompressionShape::<WhiteoutResponse>::default())?,
        )?;
        Ok(LayoutState {
            pads,
            v_matrix,
            response: self.responses.get(num_vars)?,
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
            let needed = hiding_randomness_columns(
                ImageShape {
                    rows: rank,
                    ring_dim: D,
                    modulus: WhiteoutOneHot::sis_modulus_profile(),
                },
                DigitAlphabet {
                    log_basis: PAD_RANDOMNESS_LOG_BASIS,
                },
            )?;
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

    fn pad_commitment<const D: usize>(
        layout: &PadLayout,
    ) -> Result<LatticePadCommitment<F, D>, ZkError> {
        let _span = tracing::info_span!("whiteout_pad_commitment_setup").entered();
        let placement = PadPlacement::new(layout)?;
        let (rank, randomness_columns) = Self::pad_rank::<D>(&placement)?;
        let stream = public_stream(
            LatticePadCommitment::<F, D>::stream_len(&placement, rank, randomness_columns)?,
            PAD_STREAM_SEED,
        )?;
        Ok(LatticePadCommitment::new(
            placement,
            &stream,
            rank,
            randomness_columns,
        )?)
    }

    /// The Σ setup over the row's residual with commitment `cm(u_a)` =
    /// `u_a_payload` (and u_a, on the prover's side), and `state`'s pad
    /// commitment with image `pad_image`.
    fn sigma<const D: usize>(
        &self,
        state: &LayoutState<D>,
        u_a_payload: &[F],
        u_a: Option<&[F]>,
        pad_image: BlockTarget<F, D>,
    ) -> Result<SigmaSetup<F, D>, ZkError> {
        let _span = tracing::info_span!("whiteout_sigma_setup").entered();
        let blocks =
            self.row
                .shape
                .blocks::<D>(&*self.row.residual_matrices::<D>()?, u_a_payload, u_a)?;
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
    ) -> Result<([[u8; 32]; 3], Arc<LayoutState<D>>), ZkError> {
        let state = self.layout_state::<D>(layout)?;
        let digests = [
            self.row.catalog.scheme.schedules().catalog_digest(),
            *self.row.selection()?.row_digest.as_bytes(),
            state.response.digest,
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
        with_ring_dim!(self.row.ring_dim(), D, {
            Ok(Arc::clone(&self.layout_state::<D>(layout)?.response.prover))
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
        with_ring_dim!(self.row.ring_dim(), D, {
            let (digests, state) = self.pad_statement::<D>(&layout)?;
            absorb_pad_statement(transcript, &digests);
            commit_lattice_pads(
                &state.pads,
                state.response.prover.as_ref() as &dyn ImageCompression<F>,
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
        with_ring_dim!(self.row.ring_dim(), D, {
            let (digests, state) = self.pad_statement::<D>(&layout)?;
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
        with_ring_dim!(self.row.ring_dim(), D, {
            let opening = pads.opening();
            let state = self.layout_state::<D>(&layout)?;
            let sigma = self.sigma::<D>(
                &state,
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
                &ProfiledResponse(state.response.prover.as_ref()),
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
        with_ring_dim!(self.row.ring_dim(), D, {
            let state = self.layout_state::<D>(&layout)?;
            let sigma = self.sigma::<D>(
                &state,
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
            tracing::info_span!("whiteout_verify_sigma").in_scope(|| {
                verify_sigma(&sigma, &lifted, transcript, state.response.prover.as_ref())
            })
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

/// A setup's Whiteout state: the grouped parameters that fix the masked
/// opening shapes it admits (its trace group beside each admitted list of
/// precommitted groups, `GroupedScheduleParams::whiteout_admitted`), the
/// plans it was constructed with, and derived state shared by clones (the
/// catalog, each shape's row, the response provers).
///
/// A setup is constructed either fresh, planning its catalog at
/// construction ([`Self::new`]), or from persisted [`WhiteoutPlans`] (serde,
/// or [`Self::new`] with plans), which it validates on first use and never
/// replaces by planning. Its serde form is its plans: the persisted ones as
/// given, or the fresh catalog's plan, so it is a function of the
/// construction, never of the proofs made since.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(into = "WhiteoutSetupWire", from = "WhiteoutSetupWire")]
pub struct WhiteoutSetup {
    grouped: Option<GroupedScheduleParams>,
    /// The persisted plans the setup was constructed with.
    persisted: Option<WhiteoutPlans>,
    /// The packaged response catalog ([`ResponsePlans`]). `None` when the
    /// persisted plans carry a response set, which then fixes every row
    /// ([`Self::from_parts`]), so the wire form carries it only for a setup
    /// that may still derive a response row.
    response_catalog: Option<Arc<Vec<u8>>>,
    planned: Arc<OnceLock<PlannedSetup>>,
}

/// [`WhiteoutSetup`]'s wire form.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WhiteoutSetupWire {
    grouped: Option<GroupedScheduleParams>,
    plans: Option<WhiteoutPlans>,
    response_catalog: Option<Vec<u8>>,
}

impl From<WhiteoutSetup> for WhiteoutSetupWire {
    fn from(setup: WhiteoutSetup) -> Self {
        let plans = setup.plans();
        Self {
            grouped: setup.grouped,
            plans,
            response_catalog: setup.response_catalog.map(Arc::unwrap_or_clone),
        }
    }
}

impl From<WhiteoutSetupWire> for WhiteoutSetup {
    fn from(wire: WhiteoutSetupWire) -> Self {
        Self::from_parts(wire.grouped, wire.plans, wire.response_catalog)
    }
}

/// [`WhiteoutSetup`]'s derived state.
struct PlannedSetup {
    catalog: Arc<WhiteoutCatalog>,
    shapes: BTreeMap<Vec<PrecommittedShape>, Arc<WhiteoutShape>>,
    /// The catalog as persisted.
    plan: CatalogPlan,
    responses: Arc<ResponsePlans>,
}

impl PlannedSetup {
    fn new(
        catalog: WhiteoutCatalog,
        residuals: PlannedRows,
        plan: CatalogPlan,
        responses: ResponsePlans,
    ) -> Self {
        let catalog = Arc::new(catalog);
        let responses = Arc::new(responses);
        let shapes = residuals
            .into_iter()
            .map(|(precommitted, (key, shape))| {
                let row = WhiteoutRow {
                    catalog: Arc::clone(&catalog),
                    key,
                    shape,
                    matrices: OnceLock::new(),
                };
                (
                    precommitted,
                    Arc::new(WhiteoutShape {
                        tail: WhiteoutTail::new(row, Arc::clone(&responses)),
                    }),
                )
            })
            .collect();
        Self {
            catalog,
            shapes,
            plan,
            responses,
        }
    }
}

impl Debug for WhiteoutSetup {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("WhiteoutSetup")
            .field("grouped", &self.grouped)
            .finish_non_exhaustive()
    }
}

/// Setups are equal when they admit the same shapes under the same plans and
/// carry the same packaged response catalog.
impl PartialEq for WhiteoutSetup {
    fn eq(&self, other: &Self) -> bool {
        self.grouped == other.grouped
            && self.plans() == other.plans()
            && self.response_catalog == other.response_catalog
    }
}

impl Eq for WhiteoutSetup {}

impl WhiteoutSetup {
    /// A setup admitting the trace group alone, or beside the precommitted
    /// groups `grouped` admits, whose plans come from `source`. A setup that
    /// plans does so now, for the trace group `trace` (`num_vars`,
    /// `num_polys`) of a setup with a masked opening; persisted plans are
    /// validated on first use.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if an admitted shape has no zero-knowledge row.
    pub(crate) fn new(
        grouped: Option<GroupedScheduleParams>,
        source: WhiteoutSource,
        trace: Option<(usize, usize)>,
        response_catalog: Option<Vec<u8>>,
    ) -> Result<Self, ZkError> {
        let persisted = match source {
            WhiteoutSource::Plan => None,
            WhiteoutSource::Load(plans) => Some(plans),
        };
        let setup = Self::from_parts(grouped, persisted, response_catalog);
        if let (None, Some((num_vars, num_polys))) = (&setup.persisted, trace) {
            let _ = setup.planned(num_vars, num_polys)?;
        }
        Ok(setup)
    }

    /// The one constructor: drops `response_catalog` when `persisted`
    /// carries a response set, which refuses any length it lacks.
    fn from_parts(
        grouped: Option<GroupedScheduleParams>,
        persisted: Option<WhiteoutPlans>,
        response_catalog: Option<Vec<u8>>,
    ) -> Self {
        let carries_responses = persisted
            .as_ref()
            .is_some_and(|plans| plans.responses.is_some());
        Self {
            grouped,
            persisted,
            response_catalog: response_catalog
                .filter(|_| !carries_responses)
                .map(Arc::new),
            planned: Arc::default(),
        }
    }

    fn admitted(&self) -> BTreeSet<Vec<PrecommittedShape>> {
        self.grouped.as_ref().map_or_else(
            || std::iter::once(Vec::new()).collect(),
            GroupedScheduleParams::whiteout_admitted,
        )
    }

    /// The plans the setup persists: those it was constructed with, else
    /// its catalog's plan with no response plans (responses are planned on
    /// demand and not persisted). `None` for a fresh setup not yet planned.
    fn plans(&self) -> Option<WhiteoutPlans> {
        if let Some(plans) = &self.persisted {
            return Some(plans.clone());
        }
        self.planned.get().map(|planned| WhiteoutPlans {
            magic: WHITEOUT_PLANS_MAGIC,
            version: WHITEOUT_PLANS_VERSION,
            catalog: planned.plan.clone(),
            responses: None,
        })
    }

    /// Every plan the setup holds or has derived, the response schedules
    /// included, as a setup constructed from them would persist them: its
    /// response set is then exhaustive.
    pub(crate) fn export(&self) -> Option<WhiteoutPlans> {
        let mut plans = self.plans()?;
        if let Some(planned) = self.planned.get() {
            plans.responses = Some(planned.responses.plans());
        }
        Some(plans)
    }

    /// Loads or plans the catalog of a trace group of `num_polys` K=16
    /// one-hot polynomials in `num_vars` variables over every admitted
    /// shape, once. Persisted plans that fail validation are refused, never
    /// replaced by planning.
    fn planned(&self, num_vars: usize, num_polys: usize) -> Result<&PlannedSetup, ZkError> {
        if let Some(planned) = self.planned.get() {
            if (planned.catalog.num_vars, planned.catalog.num_polys) != (num_vars, num_polys) {
                return Err(AkitaError::InvalidSetup(
                    "the Whiteout catalog was planned for another trace group".into(),
                )
                .into());
            }
            return Ok(planned);
        }
        let admitted = self.admitted();
        let planned = if let Some(plans) = &self.persisted {
            let _span = tracing::info_span!("whiteout_load_catalog").entered();
            if (plans.magic, plans.version) != (WHITEOUT_PLANS_MAGIC, WHITEOUT_PLANS_VERSION) {
                return Err(WhiteoutPlanError::Format {
                    magic: plans.magic,
                    version: plans.version,
                }
                .into());
            }
            let (catalog, residuals) =
                WhiteoutCatalog::load(&plans.catalog, num_vars, num_polys, &admitted)?;
            PlannedSetup::new(
                catalog,
                residuals,
                plans.catalog.clone(),
                ResponsePlans::new(plans.responses.clone(), self.response_catalog.clone()),
            )
        } else {
            let _span = tracing::info_span!("whiteout_plan_catalog").entered();
            let (catalog, residuals) = WhiteoutCatalog::plan(num_vars, num_polys, &admitted)?;
            let plan = catalog.plan_of()?;
            PlannedSetup::new(
                catalog,
                residuals,
                plan,
                ResponsePlans::new(None, self.response_catalog.clone()),
            )
        };
        Ok(self.planned.get_or_init(|| planned))
    }

    /// The setup's catalog, for a trace group of `num_polys` polynomials in
    /// `num_vars` variables.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if an admitted shape has no zero-knowledge row, or a
    /// persisted plan is refused ([`WhiteoutPlanError`]).
    #[cfg(feature = "zk-prover")]
    pub(crate) fn catalog(
        &self,
        num_vars: usize,
        num_polys: usize,
    ) -> Result<&WhiteoutCatalog, ZkError> {
        Ok(&self.planned(num_vars, num_polys)?.catalog)
    }

    /// The masked opening of `groups`.
    ///
    /// # Errors
    ///
    /// [`AkitaError::UnsupportedSchedule`] if the setup does not admit the
    /// shape, and [`ZkError`] if an admitted shape has no zero-knowledge row
    /// or a persisted plan is refused ([`WhiteoutPlanError`]).
    pub(crate) fn shape(&self, groups: &MaskedOpeningShape) -> Result<Arc<WhiteoutShape>, ZkError> {
        self.planned(groups.num_vars, groups.num_polys)?
            .shapes
            .get(&groups.precommitted)
            .cloned()
            .ok_or_else(|| {
                AkitaError::UnsupportedSchedule(format!(
                    "the Whiteout setup does not admit the precommitted groups {:?}",
                    groups.precommitted
                ))
                .into()
            })
    }
}

/// A setup's response provers, shared by its shapes, by `e`'s variable
/// count. With a persisted set, each is loaded from its plan and a length
/// the set lacks is refused. Without one, each is taken from the packaged
/// response catalog's row for its length, and planned on first use only for
/// a length the catalog lacks; the packaged row and the planned one are the
/// same row, so either gives the same one-row plan and digest.
struct ResponsePlans {
    persisted: Option<BTreeMap<usize, Vec<u8>>>,
    packaged: Option<Arc<Vec<u8>>>,
    /// `packaged`, decoded and admitted on first use.
    catalog: OnceLock<TrustedScheduleCatalog<WhiteoutResponse>>,
    derived: Mutex<BTreeMap<usize, Arc<PlannedResponse>>>,
}

impl ResponsePlans {
    fn new(persisted: Option<BTreeMap<usize, Vec<u8>>>, packaged: Option<Arc<Vec<u8>>>) -> Self {
        Self {
            persisted,
            packaged,
            catalog: OnceLock::new(),
            derived: Mutex::default(),
        }
    }

    /// The packaged catalog's row for `e` of `num_vars` variables, if the
    /// setup carries a catalog with that row. A catalog that fails
    /// admission is refused, never replaced by planning.
    fn packaged_row(
        &self,
        num_vars: usize,
    ) -> Result<Option<(CommittedGroupBatchProfile, FoldSchedule)>, ZkError> {
        let Some(bytes) = &self.packaged else {
            return Ok(None);
        };
        let catalog = if let Some(catalog) = self.catalog.get() {
            catalog
        } else {
            let _span = tracing::info_span!("whiteout_load_response_catalog").entered();
            let catalog = TrustedScheduleCatalog::<WhiteoutResponse>::from_artifact_bytes(
                bytes,
                ChunkedWitnessCfg::default_non_chunked(),
            )
            .map_err(WhiteoutPlanError::Inadmissible)?;
            self.catalog.get_or_init(|| catalog)
        };
        let group = PolynomialGroupLayout::singleton(num_vars);
        Ok(catalog
            .rows()
            .find(|row| {
                row.profiles().final_group.group == group && row.profiles().precommitteds.is_empty()
            })
            .map(|row| (row.profiles().clone(), row.schedule().clone())))
    }

    /// The response prover of `e` with `num_vars` variables.
    fn get(&self, num_vars: usize) -> Result<Arc<PlannedResponse>, ZkError> {
        if let Some(response) = self
            .derived
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&num_vars)
        {
            return Ok(Arc::clone(response));
        }
        // Planned outside the lock: planning runs on Rayon, and a worker
        // waiting in it could steal a job that waits on this lock. Two
        // callers may plan one length; they plan the same row, and the first
        // inserted is kept.
        let response = Arc::new(match &self.persisted {
            Some(persisted) => {
                let artifact = persisted.get(&num_vars).ok_or_else(|| {
                    WhiteoutPlanError::Statement(AkitaError::UnsupportedSchedule(format!(
                        "no persisted response plan for e of {num_vars} variables"
                    )))
                })?;
                PlannedResponse::load(num_vars, artifact)?
            }
            None => match self.packaged_row(num_vars)? {
                Some((profile, schedule)) => {
                    PlannedResponse::from_row(num_vars, profile, schedule)?
                }
                None => PlannedResponse::plan(num_vars)?,
            },
        });
        Ok(Arc::clone(
            self.derived
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(num_vars)
                .or_insert(response),
        ))
    }

    /// The persisted plans with the derived ones.
    fn plans(&self) -> BTreeMap<usize, Vec<u8>> {
        let mut plans = self.persisted.clone().unwrap_or_default();
        let derived = self.derived.lock().unwrap_or_else(PoisonError::into_inner);
        for (num_vars, response) in derived.iter() {
            let _ = plans.insert(*num_vars, response.plan.clone());
        }
        plans
    }
}

/// A response prover with its plan as persisted and the plan's catalog
/// digest, which the pads absorb.
struct PlannedResponse {
    prover: Arc<AkitaResponseProver<WhiteoutResponse>>,
    plan: Vec<u8>,
    digest: [u8; 32],
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

impl PlannedResponse {
    /// Plans the response row of `e` with `num_vars` variables.
    fn plan(num_vars: usize) -> Result<Self, ZkError> {
        let _span = tracing::info_span!("whiteout_plan_response", num_vars).entered();
        let chunking = ChunkedWitnessCfg::default_non_chunked();
        let group = PolynomialGroupLayout::singleton(num_vars);
        let key = ScheduleLookupKey::single(group);
        let planned = tracing::info_span!("find_schedule").in_scope(|| {
            akita_planner::find_schedule(
                &key,
                WhiteoutResponse::committed_source_contract()?,
                &[],
                &policy_of::<WhiteoutResponse>(chunking),
                WhiteoutResponse::ring_challenge_config,
            )
        })?;
        let profile = CommittedGroupBatchProfile {
            final_group: GroupCommitPhaseParams::try_from_params(
                group,
                &planned.schedule.root.params,
            )?,
            precommitteds: Vec::new(),
        };
        Self::from_row(num_vars, profile, planned.schedule)
    }

    /// The response row of `e` with `num_vars` variables as its one-row
    /// catalog, admitted as [`Self::plan`] admits a planned row.
    fn from_row(
        num_vars: usize,
        profile: CommittedGroupBatchProfile,
        schedule: FoldSchedule,
    ) -> Result<Self, ZkError> {
        let catalog = TrustedScheduleCatalog::try_from_rows(
            [(profile, schedule)],
            ChunkedWitnessCfg::default_non_chunked(),
        )?;
        let plan = catalog.to_artifact_bytes()?;
        Ok(Self::new(num_vars, catalog, plan)?)
    }

    /// The response row of `e` with `num_vars` variables from its persisted
    /// one-row catalog, audited by Akita and required to be that length's
    /// row ([`load_single_row`]).
    fn load(num_vars: usize, artifact: &[u8]) -> Result<Self, ZkError> {
        let _span = tracing::info_span!("whiteout_load_response", num_vars).entered();
        let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
        let (_, catalog) = load_single_row::<WhiteoutResponse>(artifact, &key)?;
        Ok(Self::new(num_vars, catalog, artifact.to_vec())?)
    }

    fn new(
        num_vars: usize,
        catalog: TrustedScheduleCatalog<WhiteoutResponse>,
        plan: Vec<u8>,
    ) -> Result<Self, AkitaError> {
        let digest = catalog.catalog_digest();
        let scheme = AkitaCommitmentScheme::new(catalog);
        let setup = AkitaResponseProver::setup(&scheme, num_vars)?;
        Ok(Self {
            prover: Arc::new(AkitaResponseProver::new(scheme, setup)?),
            plan,
            digest,
        })
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
