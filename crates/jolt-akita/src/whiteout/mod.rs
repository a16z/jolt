//! Whiteout: Jolt's zero-knowledge proofs over Akita
//! (`specs/whiteout-jolt-integration.md`). The trace group is committed
//! hiding under a zero-knowledge row, opened through Akita's masked levels to
//! the cut, and the residual relation is proved by jolt-zk's Σ-protocol with
//! Akita's response proof.
//!
//! PROTOTYPE (port map in the spec's prototype deltas):
//! - Zero-knowledge rows are planned on first use for the exact shape, in
//!   both roles, instead of shipped as artifacts: the zk families are
//!   test-only until W5e re-prices them (A41).
//! - One trace group under `JoltOneHotK16`, no precommitted groups.
//! - `A_p` and `A_D` ranks come from [`sigma_matrix_rank`], which waits on
//!   W5e's Σ-binding table (A40).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError};

use akita_config::proof_optimized::fp128::ResponseDigits;
use akita_config::{
    policy_of, CommitmentConfig, RecursiveCommitmentConfig, TrustedScheduleCatalog,
};
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
use akita_types::{CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use jolt_crypto::lattice::{derive_public_matrix_prefix, PublicMatrixSeed};
use jolt_transcript::{Channel, ProverTranscript, Sponge, VerifierTranscript};
use jolt_zk::masking::{
    AffineRow, CommittedPads, CommittedPadsView, PadLayout, ResidualRelation, ZkError,
};
use jolt_zk::whiteout::{
    commit_lattice_pads, prove_sigma, receive_lattice_pads, verify_sigma, BlockTarget,
    ImageCompression, LatticePadCommitment, LatticePadPayload, PadPlacement, SigmaSetup,
    WhiteoutPadOpening, PAD_RANDOMNESS_LOG_BASIS,
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

/// The ranks of the pad commitment `A_p` and of `A_D`. Both bind openings
/// the Σ-protocol certifies only inside its Gaussian box, so both must be
/// Module-SIS hard at the Σ collision radius `4Γ_Δ·B_M` (Akita A40).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SigmaRanks {
    /// Priced by [`sigma_matrix_rank`] at each matrix's ring dimension and
    /// width.
    Priced,
    /// Fixed ranks for tests, NOT priced at the Σ radius (A40): the Akita
    /// end-to-end harness's 2 and 1, until W5e's table prices them.
    #[cfg(any(test, feature = "test-support"))]
    A40TestOnly { pad: usize, v: usize },
}

/// The Module-SIS rank of a matrix that binds a Σ-certified opening, of
/// `width` columns over a ring of dimension `ring_dim`, at the Σ collision
/// radius: the one owner of `A_p` and `A_D` sizing.
///
/// # Errors
///
/// Always, until Akita's Σ-binding table lands with W5e (A40):
/// `akita_params::sis::sigma_binding::sigma_binding_min_rank` prices it.
pub fn sigma_matrix_rank(ring_dim: usize, width: usize) -> Result<usize, ZkError> {
    Err(AkitaError::UnsupportedSchedule(format!(
        "A40: the Σ-binding rank of A_p/A_D (d {ring_dim}, width {width}) is priced by W5e's \
         sigma_binding_min_rank, which Akita dev does not have yet"
    ))
    .into())
}

impl SigmaRanks {
    fn pad(self, ring_dim: usize, width: usize) -> Result<usize, ZkError> {
        match self {
            Self::Priced => sigma_matrix_rank(ring_dim, width),
            #[cfg(any(test, feature = "test-support"))]
            Self::A40TestOnly { pad, .. } => Ok(pad),
        }
    }

    fn v(self, width: usize) -> Result<usize, ZkError> {
        match self {
            Self::Priced => sigma_matrix_rank(jolt_crypto::lattice::CHALLENGE_RING_DIM, width),
            #[cfg(any(test, feature = "test-support"))]
            Self::A40TestOnly { v, .. } => Ok(v),
        }
    }
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

/// One trace shape's zero-knowledge opening: its row, the Akita setups that
/// prove and verify it, and the residual shape at its cut.
pub struct WhiteoutRow {
    scheme: AkitaCommitmentScheme<WhiteoutOneHot>,
    prover: AkitaBackendProverSetup,
    backend: AkitaBackend,
    verifier: AkitaVerifier<WhiteoutOneHot>,
    shape: ResidualShape,
    num_vars: usize,
    num_polys: usize,
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
        let prover = scheme.setup_prover(num_vars, num_polys)?;
        let backend = AkitaBackend::new(Arc::clone(&prover.expanded))?;
        let verifier = scheme.verifier(scheme.setup_verifier(&prover)?)?;
        Ok(Self {
            scheme,
            prover,
            backend,
            verifier,
            shape,
            num_vars,
            num_polys,
        })
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn num_polys(&self) -> usize {
        self.num_polys
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
        let output = self.backend.commit(
            self.scheme.schedules(),
            source,
            GroupContext::scheduler_without_precommitted_groups(),
        )?;
        Ok((output.committed_group, output.private_handle))
    }

    /// The backend the trace source is imported into.
    pub fn backend(&self) -> &AkitaBackend {
        &self.backend
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
        transcript.send(opening.selection().row_digest.as_bytes());
        let resolved = self
            .scheme
            .schedules()
            .resolve_selection(opening.selection())?;
        let prefix_ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
            resolved.schedule(),
            opening.opening_layout(),
        )?;
        let prefixes = self
            .backend
            .import_setup_prefixes(&self.prover.prefix_slots, &prefix_ids)?;
        prove_masked_prefix::<WhiteoutOneHot, _, _, _>(
            self.prover.expanded.descriptor(),
            self.scheme.schedules(),
            &self.backend,
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
        transcript.public(&self.scheme.schedules().catalog_digest());
        let selection = OpeningScheduleSelection {
            row_digest: akita_params::ScheduleRowDigest::from_bytes(transcript.receive()?),
        };
        if selection != self.selection()? {
            return Err(AkitaError::InvalidProof);
        }
        self.verifier.verify_masked_prefix(
            transcript,
            pads,
            self.statement(selection, commitment, point)?,
            forms,
            BasisMode::Lagrange,
        )
    }

    /// The pad-layout replay of [`Self::verify_masked`]: counts the pads of
    /// the masked messages. The selection is read but not checked (the
    /// replay runs over zero bytes).
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
        transcript.public(&self.scheme.schedules().catalog_digest());
        let _: [u8; 32] = transcript.receive()?;
        self.verifier.count_masked_prefix(
            transcript,
            view,
            self.statement(self.selection()?, commitment, point)?,
            forms,
            BasisMode::Lagrange,
        )
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
    ranks: SigmaRanks,
    response: Mutex<BTreeMap<usize, Arc<AkitaResponseProver<WhiteoutResponse>>>>,
}

impl WhiteoutTail {
    pub fn new(ranks: SigmaRanks) -> Self {
        Self {
            ranks,
            response: Mutex::new(BTreeMap::new()),
        }
    }

    fn pad_commitment<const D: usize>(
        &self,
        layout: &PadLayout,
    ) -> Result<LatticePadCommitment<F, D>, ZkError> {
        let placement = PadPlacement::new(layout)?;
        // The randomness width depends on the rank, and the rank on the
        // width; the rank is priced at the width without randomness, then
        // checked at the full width.
        let rank = self
            .ranks
            .pad(D, LatticePadCommitment::<F, D>::width(&placement, 0)?)?;
        let randomness_columns = hiding_randomness_columns(
            ImageShape {
                rows: rank,
                ring_dim: D,
                modulus: WhiteoutOneHot::sis_modulus_profile(),
            },
            DigitAlphabet {
                log_basis: PAD_RANDOMNESS_LOG_BASIS,
            },
        )?;
        if self.ranks.pad(
            D,
            LatticePadCommitment::<F, D>::width(&placement, randomness_columns)?,
        )? > rank
        {
            return Err(AkitaError::InvalidSetup(
                "the pad commitment's rank does not cover its randomness".into(),
            )
            .into());
        }
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
        &self,
        row: &WhiteoutRow,
        layout: &PadLayout,
        u_a: &[F],
        pad_image: BlockTarget<F, D>,
    ) -> Result<SigmaSetup<F, D>, ZkError> {
        let pads = self.pad_commitment::<D>(layout)?;
        let blocks = row.shape.blocks::<D>(row.stream(), u_a)?;
        let v_rank = self.ranks.v(SigmaSetup::<F, D>::v_width(&blocks, &pads))?;
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
        let sigma = self.sigma::<D>(
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
    /// [`ZkError`] if the layout does not fit or a rank is not priced (A40).
    pub fn commit_pads<S: Sponge, R: RngCore + CryptoRng>(
        &self,
        row: &WhiteoutRow,
        layout: PadLayout,
        transcript: &mut ProverTranscript<S>,
        rng: &mut R,
    ) -> Result<CommittedPads<F, WhiteoutPadOpening<F>>, ZkError> {
        with_ring_dim!(row.ring_dim(), D, {
            let commitment = self.pad_commitment::<D>(&layout)?;
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
        &self,
        row: &WhiteoutRow,
        layout: PadLayout,
        transcript: &mut VerifierTranscript<'_, S>,
    ) -> Result<CommittedPadsView<LatticePadPayload<F>>, ZkError> {
        with_ring_dim!(row.ring_dim(), D, {
            let commitment = self.pad_commitment::<D>(&layout)?;
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
            let sigma = self.sigma::<D>(
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
            let sigma = self.sigma::<D>(
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

/// The planned shapes, keyed by `(num_vars, num_polys)`.
type ShapeCache = BTreeMap<(usize, usize), Arc<WhiteoutShape>>;

/// A setup's Whiteout parameters, with its masked opening shapes planned on
/// first use (derived state, shared by clones and skipped by serde).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct WhiteoutSetup {
    ranks: SigmaRanks,
    #[serde(skip)]
    shapes: Arc<Mutex<ShapeCache>>,
}

impl std::fmt::Debug for WhiteoutSetup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhiteoutSetup")
            .field("ranks", &self.ranks)
            .finish_non_exhaustive()
    }
}

impl PartialEq for WhiteoutSetup {
    fn eq(&self, other: &Self) -> bool {
        self.ranks == other.ranks
    }
}

impl Eq for WhiteoutSetup {}

impl WhiteoutSetup {
    pub(crate) fn new(ranks: SigmaRanks) -> Self {
        Self {
            ranks,
            shapes: Arc::default(),
        }
    }

    /// The masked opening of a K=16 trace group of `num_polys` polynomials
    /// in `num_vars` variables.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if no zero-knowledge row exists for the shape.
    pub fn shape(&self, num_vars: usize, num_polys: usize) -> Result<Arc<WhiteoutShape>, ZkError> {
        let mut shapes = self.shapes.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(shape) = shapes.get(&(num_vars, num_polys)) {
            return Ok(Arc::clone(shape));
        }
        let shape = Arc::new(WhiteoutShape {
            row: WhiteoutRow::plan(num_vars, num_polys)?,
            tail: WhiteoutTail::new(self.ranks),
        });
        let _ = shapes.insert((num_vars, num_polys), Arc::clone(&shape));
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

pub(crate) use opening::trace_shape;
pub use opening::WhiteoutResidual;

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests fail loudly"
)]
mod tests;
