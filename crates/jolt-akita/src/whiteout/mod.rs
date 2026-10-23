//! Whiteout: Jolt's zero-knowledge proofs over Akita
//! (`specs/whiteout-jolt-integration.md`). The trace group is committed
//! hiding under a zero-knowledge row, opened through Akita's masked levels to
//! the cut, and the residual relation is proved by jolt-zk's Σ-protocol with
//! Akita's response proof.
//!
//! PROTOTYPE (port map in the spec's prototype deltas):
//! - Zero-knowledge rows are planned on first use for the exact shape, in
//!   both roles, instead of shipped as artifacts.
//! - One trace group under `JoltOneHotK16`, no precommitted groups.
//! - `A_p` is priced at the Σ collision radius ([`sigma_matrix_rank`]),
//!   `A_D` at its digits' difference bound ([`digit_matrix_rank`]).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use akita_config::proof_optimized::fp128::ResponseDigits;
use akita_config::{
    policy_of, CommitmentConfig, RecursiveCommitmentConfig, TrustedScheduleCatalog,
};
use akita_params::sis::sigma_binding::sigma_binding_min_rank;
use akita_params::sis::{min_secure_rank, SisMatrixRole};
use akita_params::zk::{
    hiding_randomness_columns, DigitAlphabet, ImageShape, ZkOpeningSchedule, ZK_CUT_LEVEL,
};
use akita_params::{
    BasisMode, CommittedGroupBatchProfile, GroupCommitPhaseParams, OpeningClaimsLayout,
    OpeningScheduleSelection, PolynomialGroupLayout, ScheduleLookupKey,
};
use akita_pcs::zk::{
    prove_masked_prefix, AkitaResponseProver, CompressionShape, MaskedPrefix, ResidualOpening,
    ResidualShape, ZkCommitmentConfig,
};
use akita_pcs::{
    AkitaCommitmentScheme, AkitaError, AkitaVerifier, GroupContext, SelectedProverOpeningData,
};
use akita_schedules::planner_support::projected_collision_role_price;
use akita_types::{CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_crypto::lattice::CHALLENGE_RING_DIM;
use jolt_crypto::lattice::{derive_public_matrix_prefix, PublicMatrixSeed};
use jolt_transcript::{Channel, ProverTranscript, Sponge, VerifierTranscript};
use jolt_zk::masking::{
    AffineRow, CommittedPads, CommittedPadsView, PadLayout, ResidualRelation, ZkError,
};
use jolt_zk::whiteout::{
    commit_lattice_pads, prove_sigma, receive_lattice_pads, verify_sigma, BlockTarget,
    ImageCompression, LatticePadCommitment, LatticePadPayload, PadPlacement, SigmaSetup,
    WhiteoutPadOpening, PAD_RANDOMNESS_LOG_BASIS, RESPONSE_LOG_BASIS,
};
use rand_core::{CryptoRng, RngCore};

use crate::adapters::{AkitaBackend, AkitaBackendHint, AkitaBackendProverSetup, AkitaField};
use crate::configs::{CatalogChunking, JoltOneHotK16};

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
/// pricing of a digit commitment (its D role at the digit-difference bound,
/// `05_security.tex:31-33`), with the key Akita's planner builds for one.
///
/// # Errors
///
/// [`AkitaError::UnsupportedSchedule`] if the table has no rank for the
/// matrix.
pub(crate) fn digit_matrix_rank(width: usize) -> Result<usize, ZkError> {
    let policy = policy_of::<WhiteoutOneHot>(<JoltOneHotK16 as CatalogChunking>::CHUNKING);
    projected_collision_role_price(
        &policy,
        SisMatrixRole::Open,
        CHALLENGE_RING_DIM,
        CHALLENGE_RING_DIM,
        width,
        RESPONSE_LOG_BASIS,
    )
    .and_then(|(key, width)| min_secure_rank(key, u64::try_from(width).ok()?))
    .ok_or_else(|| {
        AkitaError::UnsupportedSchedule(format!("no digit-commitment rank for A_D (width {width})"))
            .into()
    })
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

/// One trace shape's zero-knowledge opening: its row, the Akita verifier,
/// the residual shape at its cut, and the prover's setup and backend, built
/// only on the first prover call (a verifier never builds prover state).
pub struct WhiteoutRow {
    scheme: AkitaCommitmentScheme<WhiteoutOneHot>,
    verifier: AkitaVerifier<WhiteoutOneHot>,
    shape: ResidualShape,
    num_vars: usize,
    num_polys: usize,
    prover: OnceLock<WhiteoutProver>,
}

/// The prover's half of a [`WhiteoutRow`].
struct WhiteoutProver {
    setup: AkitaBackendProverSetup,
    backend: AkitaBackend,
}

impl WhiteoutRow {
    /// Plans the zero-knowledge row of a trace group of `num_polys` K=16
    /// one-hot polynomials in `num_vars` variables and builds its setups.
    /// Deterministic, so both roles derive the same row.
    ///
    /// # Errors
    ///
    /// [`AkitaError`] if no zero-knowledge row exists for the shape.
    pub fn plan(num_vars: usize, num_polys: usize) -> Result<Self, AkitaError> {
        let chunking = <JoltOneHotK16 as CatalogChunking>::CHUNKING;
        let final_group =
            OpeningClaimsLayout::new(num_vars, num_polys)?.root_final_group_layout()?;
        let key = ScheduleLookupKey::single(final_group);
        let planned = akita_planner::find_zk_schedule(
            &key,
            WhiteoutOneHot::committed_source_contract()?,
            &[],
            &policy_of::<WhiteoutOneHot>(chunking),
            WhiteoutOneHot::ring_challenge_config,
            ZkOpeningSchedule::new(ZK_CUT_LEVEL)?,
        )?;
        let profile = CommittedGroupBatchProfile {
            final_group: GroupCommitPhaseParams::try_from_params(
                key.final_group,
                &planned.schedule.root.params,
            )?,
            precommitteds: Vec::new(),
        };
        let shape = ResidualShape::new(akita_params::zk::AdmittedZkSchedule::new(
            &planned.schedule,
            WhiteoutOneHot::committed_source_contract()?,
            <WhiteoutOneHot as CommitmentConfig>::EXT_DEGREE,
        )?)?;
        let catalog = TrustedScheduleCatalog::<WhiteoutOneHot>::try_from_rows(
            [(profile.clone(), planned.schedule)],
            chunking,
        )?;
        let scheme = AkitaCommitmentScheme::new(catalog);
        // Akita derives a verifier setup only from a prover setup; this one
        // is dropped here.
        let verifier =
            scheme.verifier(scheme.setup_verifier(&scheme.setup_prover(num_vars, num_polys)?)?)?;
        Ok(Self {
            scheme,
            verifier,
            shape,
            num_vars,
            num_polys,
            prover: OnceLock::new(),
        })
    }

    /// The prover's setup and backend, built on first use.
    fn prover(&self) -> Result<&WhiteoutProver, AkitaError> {
        if let Some(prover) = self.prover.get() {
            return Ok(prover);
        }
        let setup = self.scheme.setup_prover(self.num_vars, self.num_polys)?;
        let backend = AkitaBackend::new(Arc::clone(&setup.expanded))?;
        Ok(self
            .prover
            .get_or_init(|| WhiteoutProver { setup, backend }))
    }

    /// The residual commitment's ring dimension, which the pad commitment
    /// and the Σ-protocol share.
    pub fn ring_dim(&self) -> usize {
        self.shape.ring_dim()
    }

    /// The schedule row both roles bind.
    pub fn selection(&self) -> Result<OpeningScheduleSelection, AkitaError> {
        let key = ScheduleLookupKey::single(
            OpeningClaimsLayout::new(self.num_vars, self.num_polys)?.root_final_group_layout()?,
        );
        Ok(self.scheme.schedules().resolve_key(&key)?.selection())
    }

    /// Commits `source` hiding, under this row's profile: the backend draws
    /// fresh randomness for every masked-level slice.
    ///
    /// # Errors
    ///
    /// The backend's errors.
    pub fn commit_hiding(
        &self,
        source: &akita_pcs::SourceHandle<F, F>,
    ) -> Result<(CommittedGroup<F>, AkitaBackendHint), AkitaError> {
        let output = self.prover()?.backend.commit(
            self.scheme.schedules(),
            source,
            GroupContext::scheduler_without_precommitted_groups(),
        )?;
        Ok((output.committed_group, output.private_handle))
    }

    /// The backend the trace source is imported into.
    ///
    /// # Errors
    ///
    /// [`AkitaError`] if the prover's setup cannot be built.
    pub fn backend(&self) -> Result<&AkitaBackend, AkitaError> {
        Ok(&self.prover()?.backend)
    }

    /// Proves the masked levels through the cut for the trace group
    /// `commitment` at `point` (Akita's order), with true `evaluations` and
    /// their masked `forms`. Returns the residual opening and Akita's
    /// auxiliary values.
    ///
    /// # Errors
    ///
    /// [`AkitaError::FoldOverflow`] on a one-shot fold overflow (the caller
    /// reruns), and the masked prover's errors.
    #[expect(
        clippy::too_many_arguments,
        reason = "the group's commitment, handle, point and claims, their forms, and the pads"
    )]
    pub fn prove_masked<S: Sponge>(
        &self,
        commitment: CommittedGroup<F>,
        handle: AkitaBackendHint,
        point: Vec<F>,
        evaluations: Vec<F>,
        forms: &[AffineRow<F>],
        pads: &mut CommittedPads<F, WhiteoutPadOpening<F>>,
        transcript: &mut ProverTranscript<S>,
    ) -> Result<(ResidualOpening<F, F>, Vec<F>), AkitaError> {
        let prover = self.prover()?;
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point,
            evaluations,
            commitment,
        )?])?;
        let opening = SelectedProverOpeningData::from_committed_claims::<WhiteoutOneHot>(
            claims,
            vec![handle],
            self.scheme.schedules(),
        )?;
        transcript.public(&self.scheme.schedules().catalog_digest());
        transcript.public(opening.selection().row_digest.as_bytes());
        let resolved = self
            .scheme
            .schedules()
            .resolve_selection(opening.selection())?;
        let prefix_ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
            resolved.schedule(),
            opening.opening_layout(),
        )?;
        let prefixes = prover
            .backend
            .import_setup_prefixes(&prover.setup.prefix_slots, &prefix_ids)?;
        prove_masked_prefix::<WhiteoutOneHot, _, _, _>(
            prover.setup.expanded.descriptor(),
            self.scheme.schedules(),
            &prover.backend,
            &prefixes,
            opening,
            &forms.to_vec(),
            pads,
            transcript,
            BasisMode::Lagrange,
        )
    }

    /// The verifier's statement for `commitment` at `point`: its evaluations
    /// are never read, the masked forms replace them.
    fn statement<'a>(
        &self,
        selection: OpeningScheduleSelection,
        commitment: &'a CommittedGroup<F>,
        point: Vec<F>,
    ) -> Result<GroupBatchStatement<'a, F, F>, AkitaError> {
        GroupBatchStatement::new(
            selection,
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point,
                vec![F::default(); self.num_polys],
                commitment,
            )?])?,
        )
    }

    /// Replays the masked levels through the cut on `pads`.
    ///
    /// # Errors
    ///
    /// The masked verifier's errors, or a row other than this shape's.
    pub fn verify_masked<S: Sponge, C>(
        &self,
        commitment: &CommittedGroup<F>,
        point: Vec<F>,
        forms: Vec<AffineRow<F>>,
        pads: &mut CommittedPadsView<C>,
        transcript: &mut VerifierTranscript<'_, S>,
    ) -> Result<MaskedPrefix<F, F>, AkitaError> {
        let selection = self.absorb_selection(transcript)?;
        self.verifier.verify_masked_prefix(
            transcript,
            pads,
            self.statement(selection, commitment, point)?,
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
        commitment: &CommittedGroup<F>,
        point: Vec<F>,
        forms: Vec<AffineRow<F>>,
        view: &mut CommittedPadsView<()>,
        transcript: &mut VerifierTranscript<'_, S>,
    ) -> Result<MaskedPrefix<F, F>, AkitaError> {
        let selection = self.absorb_selection(transcript)?;
        self.verifier.count_masked_prefix(
            transcript,
            view,
            self.statement(selection, commitment, point)?,
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
        transcript.public(&self.scheme.schedules().catalog_digest());
        transcript.public(selection.row_digest.as_bytes());
        Ok(selection)
    }

    /// Decodes a trace commitment under this row's profile.
    pub(crate) fn backend_commitment(
        &self,
        commitment: &crate::AkitaCommitment,
    ) -> Result<CommittedGroup<F>, AkitaError> {
        crate::shape_guard::deserialize_checked_backend_payload(
            self.scheme.schedules(),
            commitment,
            self.selection()?,
            self.num_polys,
            &vec![F::default(); self.num_vars],
        )
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))
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

/// One masked opening shape: the trace group's row and the Σ-protocol's
/// commitments.
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

/// A setup's Whiteout state: its masked opening shape, planned on first use
/// (derived state, shared by clones and skipped by serde). A setup opens one
/// shape (its trace group's), so the cache holds one.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct WhiteoutSetup {
    #[serde(skip)]
    shape: Arc<Mutex<Option<CachedShape>>>,
}

/// The planned shape and its `(num_vars, num_polys)`.
type CachedShape = ((usize, usize), Arc<WhiteoutShape>);

impl std::fmt::Debug for WhiteoutSetup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhiteoutSetup").finish_non_exhaustive()
    }
}

/// Derived state only: every Whiteout setup is equal.
impl PartialEq for WhiteoutSetup {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for WhiteoutSetup {}

impl WhiteoutSetup {
    /// The masked opening of a K=16 trace group of `num_polys` polynomials
    /// in `num_vars` variables.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if no zero-knowledge row exists for the shape.
    pub(crate) fn shape(
        &self,
        num_vars: usize,
        num_polys: usize,
    ) -> Result<Arc<WhiteoutShape>, ZkError> {
        let mut cached = self.shape.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((key, shape)) = cached.as_ref() {
            if *key == (num_vars, num_polys) {
                return Ok(Arc::clone(shape));
            }
        }
        let shape = Arc::new(WhiteoutShape {
            row: WhiteoutRow::plan(num_vars, num_polys)?,
            tail: WhiteoutTail::new(),
        });
        *cached = Some(((num_vars, num_polys), Arc::clone(&shape)));
        Ok(shape)
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

pub(crate) use opening::prover_trace_shape;
pub use opening::WhiteoutResidual;

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests fail loudly"
)]
mod tests;
