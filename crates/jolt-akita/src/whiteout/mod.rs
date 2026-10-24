//! Whiteout: Jolt's zero-knowledge proofs over Akita
//! (`specs/whiteout-jolt-integration.md`). The trace group is committed
//! hiding under a zero-knowledge row, opened through Akita's masked levels to
//! the cut, and the residual relation is proved by jolt-zk's Σ-protocol with
//! Akita's response proof.
//!
//! PROTOTYPE (port map in the spec's prototype deltas):
//! - Zero-knowledge rows are planned on first use for the exact shape, in
//!   both roles, instead of shipped as artifacts.
//! - One trace group under `JoltOneHotK16`, beside the precommitted dense
//!   groups its setup admits (advice, field increments, committed programs),
//!   each committed hiding under the profile of its own shape's
//!   zero-knowledge row (Akita §3.5, R6).
//! - `A_p` is priced at the Σ collision radius ([`sigma_matrix_rank`]),
//!   `A_D` at its digits' difference bound ([`digit_matrix_rank`]).

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use akita_config::proof_optimized::fp128::ResponseDigits;
use akita_config::{
    policy_of, CommitmentConfig, RecursiveCommitmentConfig, TrustedScheduleCatalog,
};
use akita_params::sis::sigma_binding::sigma_binding_min_rank;
use akita_params::sis::CommittedSourceContract;
use akita_params::zk::{
    hiding_randomness_columns, DigitAlphabet, ImageShape, ZkOpeningSchedule, ZK_CUT_LEVEL,
};
use akita_params::{
    BasisMode, CommittedGroupBatchProfile, GroupCommitPhaseParams, OpeningClaimsLayout,
    OpeningScheduleSelection, PolynomialGroupLayout, PrecommittedGroupProfiles, ScheduleLookupKey,
};
use akita_pcs::zk::{
    prove_masked_prefix, AkitaResponseProver, CompressionShape, MaskedPrefix, ResidualOpening,
    ResidualShape, ZkCommitmentConfig,
};
use akita_pcs::{
    AkitaCommitmentScheme, AkitaError, AkitaVerifier, CommitOutput, GroupContext,
    SelectedProverOpeningData,
};
use akita_schedules::digit_commitment_min_rank;
use akita_types::{CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_crypto::lattice::CHALLENGE_RING_DIM;
use jolt_crypto::lattice::{derive_public_matrix_prefix, PublicMatrixSeed};
use jolt_transcript::{Channel, ProverTranscript, Sponge, VerifierTranscript};
use jolt_zk::masking::{
    AffineRow, CommittedPads, CommittedPadsView, PadLayout, ResidualRelation, ZkError,
};
use jolt_zk::whiteout::{
    commit_lattice_pads, prove_sigma, receive_lattice_pads, verify_sigma, BlockTarget, DenseSource,
    ImageCompression, LatticePadCommitment, LatticePadPayload, MaskedOpeningShape, PadPlacement,
    PrecommittedShape, SigmaSetup, WhiteoutPadOpening, PAD_RANDOMNESS_LOG_BASIS,
    RESPONSE_LOG_BASIS,
};
use rand_core::{CryptoRng, RngCore};

use crate::adapters::{AkitaBackend, AkitaBackendHint, AkitaBackendProverSetup, AkitaField};
use crate::configs::{CatalogChunking, JoltDenseBounded, JoltDenseFull, JoltOneHotK16};
use crate::schedule_registry::GroupedScheduleParams;

type F = AkitaField;

/// The zero-knowledge configuration of Jolt's K=16 trace group.
pub type WhiteoutOneHot = ZkCommitmentConfig<JoltOneHotK16>;

/// The response proof's configuration (`specs/whiteout-akita.md` §5).
pub type WhiteoutResponse = RecursiveCommitmentConfig<ResponseDigits>;

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
/// `[B_a | B′_a]` are priced at (Akita A41, §5.5.2; `A_p` is A40). `A_p` binds the pads,
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
/// (`05_security.tex:31-33`), [`digit_commitment_min_rank`].
///
/// # Errors
///
/// [`AkitaError::UnsupportedSchedule`] if the table has no rank for the
/// matrix.
pub(crate) fn digit_matrix_rank(width: usize) -> Result<usize, ZkError> {
    let policy = policy_of::<WhiteoutOneHot>(<JoltOneHotK16 as CatalogChunking>::CHUNKING);
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
/// `b_randomness_columns` (Akita §3.5, R6), and that row's one-row catalog,
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

    fn contract(&self) -> Result<CommittedSourceContract, AkitaError> {
        match self.catalog {
            PrecommittedCatalog::Bounded(_) => WhiteoutDenseBounded::committed_source_contract(),
            PrecommittedCatalog::FullField(_) => WhiteoutDenseFull::committed_source_contract(),
        }
    }

    /// Commits `source` under the hiding profile: the backend samples fresh
    /// randomness for every B slice and keeps it in the handle.
    fn commit(
        &self,
        backend: &AkitaBackend,
        source: &akita_pcs::SourceHandle<F, F>,
    ) -> Result<CommitOutput<F, F>, AkitaError> {
        let context = GroupContext::explicit(&self.profile);
        match &self.catalog {
            PrecommittedCatalog::Bounded(catalog) => backend.commit(catalog, source, context),
            PrecommittedCatalog::FullField(catalog) => backend.commit(catalog, source, context),
        }
    }
}

/// The root profile and one-row catalog of the zero-knowledge row of one
/// `num_vars`-variable dense polynomial under `Cfg`, planned unchunked: a
/// masked level spans one witness chunk (D4), and the group is folded at the
/// single-chunk trace row's masked root.
fn plan_precommitted<Cfg: CommitmentConfig<Field = F, ExtField = F>>(
    num_vars: usize,
) -> Result<(GroupCommitPhaseParams, TrustedScheduleCatalog<Cfg>), AkitaError> {
    let chunking = akita_params::ChunkedWitnessCfg::default_non_chunked();
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

/// A setup's zero-knowledge catalog: one row per masked opening shape the
/// setup admits (its trace group with each admitted combination of
/// precommitted groups), the precommitted groups' hiding profiles, the Akita
/// verifier, and the prover's setup and backend, built only on the first
/// prover call (a verifier never builds prover state). One backend commits
/// every hiding group and proves every opening: Akita cannot import a hiding
/// commitment into another backend.
pub struct WhiteoutCatalog {
    scheme: AkitaCommitmentScheme<WhiteoutOneHot>,
    verifier: AkitaVerifier<WhiteoutOneHot>,
    precommitted: BTreeMap<PrecommittedShape, PrecommittedRow>,
    num_vars: usize,
    num_polys: usize,
    max_num_vars: usize,
    max_batched_polys: usize,
    prover: OnceLock<WhiteoutProver>,
}

/// The prover's half of a [`WhiteoutCatalog`].
struct WhiteoutProver {
    setup: AkitaBackendProverSetup,
    backend: AkitaBackend,
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
        let chunking = <JoltOneHotK16 as CatalogChunking>::CHUNKING;
        let mut precommitted = BTreeMap::new();
        for shape in admitted.iter().flatten() {
            if let Entry::Vacant(entry) = precommitted.entry(*shape) {
                let _ = entry.insert(PrecommittedRow::plan(*shape)?);
            }
        }
        let final_group =
            OpeningClaimsLayout::new(num_vars, num_polys)?.root_final_group_layout()?;
        let mut rows = Vec::with_capacity(admitted.len());
        let mut residuals = BTreeMap::new();
        for groups in admitted {
            let rows_of = groups
                .iter()
                .map(|shape| {
                    precommitted.get(shape).ok_or_else(|| {
                        AkitaError::InvalidSetup("a precommitted shape was not planned".into())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let key = ScheduleLookupKey {
                final_group,
                precommitteds: rows_of.iter().map(|row| row.profile).collect(),
            };
            let contracts = rows_of
                .iter()
                .map(|row| row.contract())
                .collect::<Result<Vec<_>, _>>()?;
            let planned = akita_planner::find_zk_schedule(
                &key,
                WhiteoutOneHot::committed_source_contract()?,
                &contracts,
                &policy_of::<WhiteoutOneHot>(chunking),
                WhiteoutOneHot::ring_challenge_config,
                ZkOpeningSchedule::new(ZK_CUT_LEVEL)?,
            )?;
            let residual = ResidualShape::new(akita_params::zk::AdmittedZkSchedule::new(
                &planned.schedule,
                WhiteoutOneHot::committed_source_contract()?,
                <WhiteoutOneHot as CommitmentConfig>::EXT_DEGREE,
            )?)?;
            rows.push((
                CommittedGroupBatchProfile {
                    final_group: GroupCommitPhaseParams::try_from_params(
                        key.final_group,
                        &planned.schedule.root.params,
                    )?,
                    precommitteds: key.precommitteds.clone(),
                },
                planned.schedule,
            ));
            let _ = residuals.insert(groups.clone(), (key, residual));
        }
        let catalog = TrustedScheduleCatalog::<WhiteoutOneHot>::try_from_rows(rows, chunking)?;
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
        // Akita derives a verifier setup only from a prover setup; this one
        // is dropped here.
        let verifier = scheme.verifier(
            scheme.setup_verifier(&scheme.setup_prover(max_num_vars, max_batched_polys)?)?,
        )?;
        Ok((
            Self {
                scheme,
                verifier,
                precommitted,
                num_vars,
                num_polys,
                max_num_vars,
                max_batched_polys,
                prover: OnceLock::new(),
            },
            residuals,
        ))
    }

    /// The prover's setup and backend, built on first use.
    fn prover(&self) -> Result<&WhiteoutProver, AkitaError> {
        if let Some(prover) = self.prover.get() {
            return Ok(prover);
        }
        let setup = self
            .scheme
            .setup_prover(self.max_num_vars, self.max_batched_polys)?;
        let backend = AkitaBackend::new(Arc::clone(&setup.expanded))?;
        Ok(self
            .prover
            .get_or_init(|| WhiteoutProver { setup, backend }))
    }

    /// The backend every hiding source is imported into.
    ///
    /// # Errors
    ///
    /// [`AkitaError`] if the prover's setup cannot be built.
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
    pub fn commit_precommitted(
        &self,
        shape: PrecommittedShape,
        source: &akita_pcs::SourceHandle<F, F>,
    ) -> Result<(CommittedGroup<F>, AkitaBackendHint), AkitaError> {
        let row = self.precommitted.get(&shape).ok_or_else(|| {
            AkitaError::UnsupportedSchedule(format!(
                "no admitted Whiteout opening folds a precommitted group of {shape:?}"
            ))
        })?;
        let output = row.commit(self.backend()?, source)?;
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

/// One masked opening shape's row: its catalog, the row's lookup key and the
/// residual shape at its cut.
pub struct WhiteoutRow {
    catalog: Arc<WhiteoutCatalog>,
    key: ScheduleLookupKey,
    shape: ResidualShape,
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
    pub fn commit_hiding(
        &self,
        source: &akita_pcs::SourceHandle<F, F>,
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
    /// order. Returns the residual opening and Akita's auxiliary values.
    ///
    /// # Errors
    ///
    /// [`AkitaError::FoldOverflow`] on a one-shot fold overflow (the caller
    /// reruns), and the masked prover's errors.
    pub fn prove_masked<S: Sponge>(
        &self,
        groups: Vec<ProverGroup>,
        forms: &[AffineRow<F>],
        pads: &mut CommittedPads<F, WhiteoutPadOpening<F>>,
        transcript: &mut ProverTranscript<S>,
    ) -> Result<(ResidualOpening<F, F>, Vec<F>), AkitaError> {
        let prover = self.catalog.prover()?;
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
        transcript.public(&schedules.catalog_digest());
        transcript.public(opening.selection().row_digest.as_bytes());
        let resolved = schedules.resolve_selection(opening.selection())?;
        let prefix_ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
            resolved.schedule(),
            opening.opening_layout(),
        )?;
        let prefixes = prover
            .backend
            .import_setup_prefixes(&prover.setup.prefix_slots, &prefix_ids)?;
        prove_masked_prefix::<WhiteoutOneHot, _, _, _>(
            prover.setup.expanded.descriptor(),
            schedules,
            &prover.backend,
            &prefixes,
            opening,
            &forms.to_vec(),
            pads,
            transcript,
            BasisMode::Lagrange,
        )
    }

    /// Replays the masked levels through the cut on `pads`; `groups` are the
    /// decoded commitments at their points (Akita's order), precommitted
    /// first.
    ///
    /// # Errors
    ///
    /// The masked verifier's errors, or a row other than this shape's.
    pub fn verify_masked<S: Sponge, C>(
        &self,
        groups: &[(CommittedGroup<F>, Vec<F>)],
        forms: Vec<AffineRow<F>>,
        pads: &mut CommittedPadsView<C>,
        transcript: &mut VerifierTranscript<'_, S>,
    ) -> Result<MaskedPrefix<F, F>, AkitaError> {
        let selection = self.absorb_selection(transcript)?;
        self.catalog.verifier.verify_masked_prefix(
            transcript,
            pads,
            masked_statement(selection, groups)?,
            forms,
            BasisMode::Lagrange,
        )
    }

    /// The pad-layout replay of [`Self::verify_masked`]: counts the pads of
    /// the masked messages.
    ///
    /// # Errors
    ///
    /// The counting verifier's errors.
    pub fn count_masked<S: Sponge>(
        &self,
        groups: &[(CommittedGroup<F>, Vec<F>)],
        forms: Vec<AffineRow<F>>,
        view: &mut CommittedPadsView<()>,
        transcript: &mut VerifierTranscript<'_, S>,
    ) -> Result<MaskedPrefix<F, F>, AkitaError> {
        let selection = self.absorb_selection(transcript)?;
        self.catalog.verifier.count_masked_prefix(
            transcript,
            view,
            masked_statement(selection, groups)?,
            forms,
            BasisMode::Lagrange,
        )
    }

    /// Absorbs the catalog and the selected row, both derived from the
    /// statement: the prover sends no selection.
    fn absorb_selection<S: Sponge>(
        &self,
        transcript: &mut VerifierTranscript<'_, S>,
    ) -> Result<OpeningScheduleSelection, AkitaError> {
        let selection = self.selection()?;
        transcript.public(&self.catalog.scheme.schedules().catalog_digest());
        transcript.public(selection.row_digest.as_bytes());
        Ok(selection)
    }

    /// Decodes the precommitted and trace commitments under this row's
    /// profiles, in the row's group order.
    pub(crate) fn backend_commitments(
        &self,
        precommitted: &[&crate::AkitaCommitment],
        trace: &crate::AkitaCommitment,
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

    /// Akita's setup stream, from which the residual blocks are cut.
    fn stream(&self) -> &[F] {
        self.catalog.stream()
    }
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
                vec![F::default(); commitment.profile.group.num_polynomials()],
                commitment,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    GroupBatchStatement::new(selection, OpeningClaims::from_groups(claims)?)
}

/// One group the prover opens: its commitment, the handle retaining its
/// source, its point in Akita's order and its true evaluations.
pub struct ProverGroup {
    pub commitment: CommittedGroup<F>,
    pub handle: AkitaBackendHint,
    pub point: Vec<F>,
    pub evaluations: Vec<F>,
}

/// The Σ-protocol's commitments for one pad layout: `A_p` and `A_D` with
/// their streams, the residual shape, and the response proof.
pub struct WhiteoutTail {
    response: Mutex<BTreeMap<usize, Arc<AkitaResponseProver<WhiteoutResponse>>>>,
}

impl WhiteoutTail {
    pub(crate) fn new() -> Self {
        Self {
            response: Mutex::new(BTreeMap::new()),
        }
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

    /// The Σ setup over `row`'s residual with commitment `u_a`, the pad
    /// commitment of `layout` with image `pad_image`.
    fn sigma<const D: usize>(
        row: &WhiteoutRow,
        layout: &PadLayout,
        u_a: &[F],
        pad_image: BlockTarget<F, D>,
    ) -> Result<SigmaSetup<F, D>, ZkError> {
        let pads = Self::pad_commitment::<D>(layout)?;
        let blocks = row.shape.blocks::<D>(row.stream(), u_a)?;
        let v_rank = digit_matrix_rank(SigmaSetup::<F, D>::v_width(&blocks, &pads))?;
        let v_stream = public_stream(
            SigmaSetup::<F, D>::v_stream_len(&blocks, &pads, v_rank),
            V_STREAM_SEED,
        )?;
        Ok(SigmaSetup::new(
            row.shape.segments(),
            blocks,
            &pads,
            pad_image,
            v_rank,
            &v_stream,
        )?)
    }

    /// The response prover for `e`'s length under `row` and `layout`,
    /// planned on first use.
    fn response<const D: usize>(
        &self,
        row: &WhiteoutRow,
        layout: &PadLayout,
    ) -> Result<Arc<AkitaResponseProver<WhiteoutResponse>>, ZkError> {
        let u_a = vec![F::default(); row.shape.geometry().profile.outer.matrix.output_rank() * D];
        let sigma = Self::sigma::<D>(
            row,
            layout,
            &u_a,
            BlockTarget::Compressed {
                payload: Vec::new(),
                image: None,
            },
        )?;
        let num_vars = AkitaResponseProver::<WhiteoutResponse>::num_vars(
            &sigma.response_layout(&CompressionShape::<WhiteoutResponse>::default())?,
        )?;
        let mut cache = self.response.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(prover) = cache.get(&num_vars) {
            return Ok(Arc::clone(prover));
        }
        let prover = Arc::new(plan_response(num_vars)?);
        let _ = cache.insert(num_vars, Arc::clone(&prover));
        Ok(prover)
    }

    /// Samples, commits and sends the pads of `layout` (row 2).
    ///
    /// # Errors
    ///
    /// [`ZkError`] if the layout does not fit or a rank is not priced.
    pub fn commit_pads<S: Sponge, R: RngCore + CryptoRng>(
        &self,
        row: &WhiteoutRow,
        layout: PadLayout,
        transcript: &mut ProverTranscript<S>,
        rng: &mut R,
    ) -> Result<CommittedPads<F, WhiteoutPadOpening<F>>, ZkError> {
        with_ring_dim!(row.ring_dim(), D, {
            let commitment = Self::pad_commitment::<D>(&layout)?;
            let response = self.response::<D>(row, &layout)?;
            commit_lattice_pads(
                &commitment,
                response.as_ref() as &dyn ImageCompression<F>,
                layout,
                transcript,
                rng,
            )
        })
    }

    /// Reads the pad commitment [`Self::commit_pads`] sends.
    ///
    /// # Errors
    ///
    /// As [`Self::commit_pads`], or a truncated payload.
    pub fn receive_pads<S: Sponge>(
        row: &WhiteoutRow,
        layout: PadLayout,
        transcript: &mut VerifierTranscript<'_, S>,
    ) -> Result<CommittedPadsView<LatticePadPayload<F>>, ZkError> {
        with_ring_dim!(row.ring_dim(), D, {
            let commitment = Self::pad_commitment::<D>(&layout)?;
            receive_lattice_pads(
                &commitment,
                &CompressionShape::<WhiteoutResponse>::default(),
                layout,
                transcript,
            )
        })
    }

    /// Proves `relation` over the committed pads and the residual opening:
    /// the Σ-protocol, then the response proof (rows 7 and 8).
    ///
    /// # Errors
    ///
    /// [`ZkError`] if the relation does not hold, or the Σ attempts are
    /// exhausted (the caller reruns).
    pub fn prove<S: Sponge, R: RngCore + CryptoRng + Send>(
        &self,
        row: &WhiteoutRow,
        relation: &ResidualRelation<F>,
        residual: &ResidualOpening<F, F>,
        pads: &CommittedPads<F, WhiteoutPadOpening<F>>,
        transcript: &mut ProverTranscript<S>,
        rng: &mut R,
    ) -> Result<(), ZkError> {
        let layout = pads.layout().clone();
        with_ring_dim!(row.ring_dim(), D, {
            let opening = pads.opening();
            let sigma = Self::sigma::<D>(
                row,
                &layout,
                &residual.residual_commitment,
                BlockTarget::Compressed {
                    payload: opening.payload().to_vec(),
                    image: Some(opening.image::<D>()?),
                },
            )?;
            let lifted = sigma.lift(relation, &row.shape.claims(&residual.point)?)?;
            let s = row.shape.residual(residual)?;
            let response = self.response::<D>(row, &layout)?;
            prove_sigma(
                &sigma,
                &lifted,
                &s,
                opening.opening(),
                transcript,
                rng,
                response.as_ref(),
            )
            .map(|_attempts| ())
        })
    }

    /// Verifies the Σ-protocol and the response proof of `relation` against
    /// the masked prefix's residual claim.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if either rejects.
    pub fn verify<S: Sponge>(
        &self,
        row: &WhiteoutRow,
        relation: &ResidualRelation<F>,
        prefix: &MaskedPrefix<F, F>,
        pads: &CommittedPadsView<LatticePadPayload<F>>,
        transcript: &mut VerifierTranscript<'_, S>,
    ) -> Result<(), ZkError> {
        let layout = pads.layout().clone();
        let claim = prefix.residual_claim();
        with_ring_dim!(row.ring_dim(), D, {
            let sigma = Self::sigma::<D>(
                row,
                &layout,
                claim.commitment(),
                BlockTarget::Compressed {
                    payload: pads.commitment().payload.clone(),
                    image: None,
                },
            )?;
            let lifted = sigma.lift(relation, &row.shape.claims(claim.point())?)?;
            let response = self.response::<D>(row, &layout)?;
            verify_sigma(&sigma, &lifted, transcript, response.as_ref())
        })
    }
}

/// One masked opening shape: its row and the Σ-protocol's commitments.
pub struct WhiteoutShape {
    row: WhiteoutRow,
    tail: WhiteoutTail,
}

impl WhiteoutShape {
    pub fn row(&self) -> &WhiteoutRow {
        &self.row
    }

    pub fn tail(&self) -> &WhiteoutTail {
        &self.tail
    }
}

/// A setup's Whiteout state: the grouped parameters that fix the masked
/// opening shapes it admits (its trace group beside each admitted list of
/// precommitted groups, `GroupedScheduleParams::whiteout_admitted`), and
/// derived state planned on first use: the catalog and each shape's row
/// (shared by clones and skipped by serde).
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WhiteoutSetup {
    grouped: Option<GroupedScheduleParams>,
    #[serde(skip)]
    planned: Arc<OnceLock<PlannedSetup>>,
}

/// [`WhiteoutSetup`]'s derived state.
struct PlannedSetup {
    catalog: Arc<WhiteoutCatalog>,
    shapes: BTreeMap<Vec<PrecommittedShape>, Arc<WhiteoutShape>>,
}

impl std::fmt::Debug for WhiteoutSetup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhiteoutSetup")
            .field("grouped", &self.grouped)
            .finish_non_exhaustive()
    }
}

/// Derived state aside, setups are equal when they admit the same shapes.
impl PartialEq for WhiteoutSetup {
    fn eq(&self, other: &Self) -> bool {
        self.grouped == other.grouped
    }
}

impl Eq for WhiteoutSetup {}

impl WhiteoutSetup {
    /// A setup admitting the trace group alone, or beside the precommitted
    /// groups `grouped` admits.
    pub(crate) fn new(grouped: Option<GroupedScheduleParams>) -> Self {
        Self {
            grouped,
            planned: Arc::default(),
        }
    }

    fn admitted(&self) -> BTreeSet<Vec<PrecommittedShape>> {
        self.grouped.as_ref().map_or_else(
            || std::iter::once(Vec::new()).collect(),
            GroupedScheduleParams::whiteout_admitted,
        )
    }

    /// Plans the catalog of a trace group of `num_polys` K=16 one-hot
    /// polynomials in `num_vars` variables over every admitted shape, once.
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
        let (catalog, residuals) = WhiteoutCatalog::plan(num_vars, num_polys, &self.admitted())?;
        let catalog = Arc::new(catalog);
        let shapes = residuals
            .into_iter()
            .map(|(precommitted, (key, shape))| {
                let row = WhiteoutRow {
                    catalog: Arc::clone(&catalog),
                    key,
                    shape,
                };
                (
                    precommitted,
                    Arc::new(WhiteoutShape {
                        row,
                        tail: WhiteoutTail::new(),
                    }),
                )
            })
            .collect();
        Ok(self
            .planned
            .get_or_init(|| PlannedSetup { catalog, shapes }))
    }

    /// The setup's catalog, for a trace group of `num_polys` polynomials in
    /// `num_vars` variables.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if an admitted shape has no zero-knowledge row.
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
    /// shape, and [`ZkError`] if an admitted shape has no zero-knowledge row.
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

/// The response prover over a catalog holding the response row of `e` with
/// `num_vars` variables, planned on first use.
fn plan_response(num_vars: usize) -> Result<AkitaResponseProver<WhiteoutResponse>, ZkError> {
    let chunking = akita_params::ChunkedWitnessCfg::default_non_chunked();
    let group = PolynomialGroupLayout::singleton(num_vars);
    let key = ScheduleLookupKey::single(group);
    let planned = akita_planner::find_schedule(
        &key,
        WhiteoutResponse::committed_source_contract()?,
        &[],
        &policy_of::<WhiteoutResponse>(chunking),
        WhiteoutResponse::ring_challenge_config,
    )?;
    let profile = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(group, &planned.schedule.root.params)?,
        precommitteds: Vec::new(),
    };
    let scheme = AkitaCommitmentScheme::new(TrustedScheduleCatalog::try_from_rows(
        [(profile, planned.schedule)],
        chunking,
    )?);
    let setup = AkitaResponseProver::setup(&scheme, num_vars)?;
    Ok(AkitaResponseProver::new(scheme, setup)?)
}

mod opening;

pub use opening::WhiteoutResidual;
pub(crate) use opening::{opening_shape, whiteout};

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests fail loudly"
)]
mod tests;
