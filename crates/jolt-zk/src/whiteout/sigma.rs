//! The Σ-protocol for the residual relation (Whiteout Protocol
//! `wo:prot:sigma`; `specs/whiteout-sigma.md` §2).
//!
//! After the masked transcript, the prover holds `s` (laid out by the
//! segment table) with `N·s = t` and every lifted row satisfied. It:
//!
//! 1. checks, before any challenge, every row and `ct(g)`, `N·s = t`, and the
//!    segment alphabets;
//! 2. squeezes ℓ aggregation challenges `ρ` and asserts `ct(h) = 0` for
//!    `h = Q_μ(1, s)`;
//! 3. squeezes a fork seed `σ`, then per attempt samples a Gaussian mask `y`,
//!    commits to `V = (N y, L y + Q_μ(0,y), L s + B_μ((1,s),(0,y)), h)` as
//!    `t_D`, draws a secret 256-bit nonce `n`, derives `c` from the attempt
//!    fork `(σ, n, t_D)`, and applies the joint rejection rule to
//!    `z = y + c·s` on the Gaussian coordinates;
//! 4. sends only the accepted attempt's `t_D` and `n`, then proves the
//!    response statement.
//!
//! Everything an attempt computes before its accept bit is constant time in
//! the attempt's secrets (y, n, c and what derives from them). A rejected
//! attempt leaves nothing in the transcript.

use std::ops::Range;
use std::sync::Arc;

#[cfg(feature = "whiteout-prover")]
use jolt_algebra::CyclotomicRing;
use jolt_algebra::JoltField;
use jolt_crypto::challenge::{
    sample_indexed_challenges_from_seed, OperatorNormRejection, D64_SELECTIVE_L2_CHALLENGE_CONFIG,
};
use jolt_crypto::lattice::{
    AjtaiLevel, GadgetDigits, MatrixShape, MatrixView, PreparedMatrix, CHALLENGE_RING_DIM,
};
use jolt_transcript::{
    Channel, Fork, Sponge, VerifierTranscript, ATTEMPT_NONCE_LEN, FORK_SEED_LEN,
};

use super::compression::ImageCompression;
use super::lift::{EvaluationClaims, LiftedRelation};
use super::linear::{BlockTarget, GadgetTerm, LRow, LinearBlock, ResidualBlock, RingLinearMap};
use super::pad_commitment::LatticePadCommitment;
#[cfg(feature = "whiteout-prover")]
use super::pad_commitment::LatticePadOpening;
use super::params::{SigmaParams, MAX_RESPONSE_NUM_VARS, RESPONSE_LOG_BASIS};
use super::placement::PadPlacement;
use super::response::{
    ResponseLayout, ResponseProver, ResponseSegment, ResponseStatement, FIELD_LEVELS,
};
use super::segments::{Segment, SegmentRole, SegmentTable};
use super::WhiteoutError;
use crate::masking::{ResidualRelation, ZkError};
#[cfg(feature = "whiteout-prover")]
use crate::secret::SecretVec;

/// The public setup of one Σ-protocol shape. Every position in `s` is
/// derived here from the segment table and the pad placement; callers never
/// pass offsets.
pub struct SigmaSetup<F: JoltField, const D: usize> {
    table: SegmentTable,
    params: SigmaParams,
    placement: PadPlacement,
    /// `N` and its targets: the commitment openings.
    linear: RingLinearMap<F, D>,
    l_row: LRow<F, D>,
    /// `A_D`, binding on the radix-16 digits of `V`, over the challenge ring.
    v_level: AjtaiLevel,
    v_matrix: Arc<PreparedMatrix<F, CHALLENGE_RING_DIM>>,
    /// Native coefficients of the residual segments, which precede the pads.
    #[cfg(feature = "whiteout-prover")]
    residual_len: usize,
}

impl<F: JoltField, const D: usize> SigmaSetup<F, D> {
    /// The setup for a residual opening of `residual` segments (witness,
    /// inner digits, randomness) with opening rows `blocks`, followed by the
    /// pad segments of `pads` with image `pad_image`, and `A_D` =
    /// `v_matrix` ([`Self::v_matrix_of`]), whose width must be these rows'.
    /// Every segment must have native dimension `D`, so that `N` is linear
    /// over the challenge ring.
    pub fn new(
        residual: Vec<Segment>,
        blocks: Vec<ResidualBlock<F, D>>,
        pads: &LatticePadCommitment<F, D>,
        pad_image: BlockTarget<F, D>,
        v_matrix: Arc<PreparedMatrix<F, CHALLENGE_RING_DIM>>,
    ) -> Result<Self, WhiteoutError> {
        if residual.iter().any(|segment| {
            matches!(
                segment.role,
                SegmentRole::PadDigits | SegmentRole::OperandDigits | SegmentRole::PadRandomness
            )
        }) {
            return Err(WhiteoutError::InvalidRelation(
                "residual segments may not hold pads",
            ));
        }
        let num_residual = residual.len();
        let mut segments = residual;
        segments.extend(pads.segments());
        let table = SegmentTable::new(segments)?;
        if !D.is_multiple_of(CHALLENGE_RING_DIM)
            || table.segments().any(|segment| segment.native_dim != D)
        {
            return Err(WhiteoutError::InvalidRelation(
                "every segment must have the setup's native ring dimension",
            ));
        }
        let placed = table.placed();
        let range = |segments: Range<usize>| -> Result<Range<usize>, WhiteoutError> {
            let first = segments.start;
            let last = segments
                .end
                .checked_sub(1)
                .filter(|&last| first <= last && last < num_residual)
                .ok_or(WhiteoutError::InvalidRelation(
                    "block reads outside the residual",
                ))?;
            let start = placed
                .get(first)
                .ok_or(WhiteoutError::InvalidRelation("segment missing"))?
                .coefficient;
            let end = placed
                .get(last)
                .ok_or(WhiteoutError::InvalidRelation("segment missing"))?
                .coefficients()
                .end;
            Ok(start..end)
        };
        let outputs = Self::output_rows(&blocks, pads);
        let mut linear = Vec::with_capacity(blocks.len() + 1);
        for block in blocks {
            let gadget = match block.gadget {
                Some(gadget) => Some(GadgetTerm {
                    input: range(gadget.segment..gadget.segment + 1)?,
                    log_basis: gadget.log_basis,
                    levels: gadget.levels,
                }),
                None => None,
            };
            linear.push(LinearBlock {
                matrix: block.matrix,
                input: range(block.input)?,
                gadget,
                target: block.target,
            });
        }
        let pad_start = table
            .unique(SegmentRole::PadDigits)
            .ok_or(WhiteoutError::InvalidRelation("pad digit segment missing"))?
            .coefficient;
        let randomness = table
            .unique(SegmentRole::PadRandomness)
            .ok_or(WhiteoutError::InvalidRelation(
                "pad randomness segment missing",
            ))?
            .coefficients();
        linear.push(LinearBlock {
            matrix: pads.matrix(),
            input: pad_start..randomness.end,
            gadget: None,
            target: pad_image,
        });
        let linear = RingLinearMap::new(linear)?;
        if linear.num_outputs() != outputs {
            return Err(WhiteoutError::InvariantViolation(
                "N's output rows differ from the blocks' and the pad commitment's",
            ));
        }

        let message_rings = Self::v_message_rings(outputs);
        let v_level = Self::v_level_of(message_rings, v_matrix.shape().rows);
        if v_matrix.shape() != v_level.matrix {
            return Err(WhiteoutError::InvalidRelation(
                "A_D's shape differs from the one V's rows need",
            ));
        }
        Ok(Self {
            params: SigmaParams::derive(&table)?,
            #[cfg(feature = "whiteout-prover")]
            residual_len: pad_start,
            placement: pads.placement().clone(),
            l_row: LRow {
                matrix: pads.l_matrix(),
                input: randomness,
            },
            table,
            linear,
            v_level,
            v_matrix,
        })
    }

    /// Challenge rings of `V`: the Σ messages of `num_outputs` rows of `N`
    /// over `R_D`, plus three.
    fn v_message_rings(num_outputs: usize) -> usize {
        (num_outputs * D) / CHALLENGE_RING_DIM + 3
    }

    /// The output ring rows of `N`: each opening block's rows, then the pad
    /// commitment's (the order [`Self::new`] stacks them in).
    fn output_rows(blocks: &[ResidualBlock<F, D>], pads: &LatticePadCommitment<F, D>) -> usize {
        blocks
            .iter()
            .map(|block| block.matrix.shape().rows)
            .sum::<usize>()
            + pads.matrix().shape().rows
    }

    /// `A_D`'s level: `message_rings` challenge rings of `V` in radix-16
    /// digits, at rank `v_rank`, binding only.
    fn v_level_of(message_rings: usize, v_rank: usize) -> AjtaiLevel {
        AjtaiLevel {
            matrix: MatrixShape {
                rows: v_rank,
                width: message_rings * FIELD_LEVELS,
                ring_dim: CHALLENGE_RING_DIM,
            },
            digits: GadgetDigits {
                log_basis: RESPONSE_LOG_BASIS,
                levels: FIELD_LEVELS,
            },
            message_rings,
            randomness_columns: 0,
        }
    }

    /// The width in challenge-ring columns of `A_D` for opening rows
    /// `blocks` and the pad commitment `pads`.
    pub fn v_width(blocks: &[ResidualBlock<F, D>], pads: &LatticePadCommitment<F, D>) -> usize {
        Self::v_message_rings(Self::output_rows(blocks, pads)) * FIELD_LEVELS
    }

    /// The public stream length [`Self::v_matrix_of`] reads for `A_D` of
    /// rank `v_rank`.
    pub fn v_stream_len(
        blocks: &[ResidualBlock<F, D>],
        pads: &LatticePadCommitment<F, D>,
        v_rank: usize,
    ) -> usize {
        v_rank * Self::v_width(blocks, pads) * CHALLENGE_RING_DIM
    }

    /// `A_D` of rank `v_rank` for opening rows `blocks` and the pad
    /// commitment `pads`, cut from `v_stream`. A function of their shapes
    /// alone, so one matrix serves every Σ setup of those shapes.
    pub fn v_matrix_of(
        blocks: &[ResidualBlock<F, D>],
        pads: &LatticePadCommitment<F, D>,
        v_rank: usize,
        v_stream: &[F],
    ) -> Result<Arc<PreparedMatrix<F, CHALLENGE_RING_DIM>>, WhiteoutError> {
        let level = Self::v_level_of(
            Self::v_message_rings(Self::output_rows(blocks, pads)),
            v_rank,
        );
        Ok(Arc::new(PreparedMatrix::new(&MatrixView::new(
            level.matrix,
            v_stream,
        )?)?))
    }

    /// This setup at another rejection parameter α (tests of the abort rate).
    #[cfg(all(test, feature = "whiteout-prover"))]
    pub(crate) fn with_alpha(mut self, alpha: u32) -> Result<Self, WhiteoutError> {
        self.params = SigmaParams::derive_with_alpha(&self.table, alpha)?;
        Ok(self)
    }

    /// Lifts `relation` over this setup's table and pad placement.
    pub fn lift(
        &self,
        relation: &ResidualRelation<F>,
        claims: &EvaluationClaims<F>,
    ) -> Result<LiftedRelation<F>, WhiteoutError> {
        LiftedRelation::new(relation, &self.placement, &self.table, claims)
    }

    pub fn params(&self) -> &SigmaParams {
        &self.params
    }

    pub(crate) fn table(&self) -> &SegmentTable {
        &self.table
    }

    pub(crate) fn linear(&self) -> &RingLinearMap<F, D> {
        &self.linear
    }

    pub(crate) fn l_row(&self) -> &LRow<F, D> {
        &self.l_row
    }

    #[cfg(feature = "whiteout-prover")]
    pub(crate) fn v_level(&self) -> &AjtaiLevel {
        &self.v_level
    }

    pub(crate) fn v_matrix(&self) -> &PreparedMatrix<F, CHALLENGE_RING_DIM> {
        &self.v_matrix
    }

    #[cfg(feature = "whiteout-prover")]
    /// `s`: the residual opening followed by the pad opening, zeroed when
    /// dropped.
    pub(crate) fn private_vector(
        &self,
        residual: &[i64],
        pads: &LatticePadOpening,
    ) -> Result<SecretVec<i64>, WhiteoutError> {
        if residual.len() != self.residual_len {
            return Err(WhiteoutError::WitnessLength {
                expected: self.residual_len,
                actual: residual.len(),
            });
        }
        let len = residual.len() + pads.digits.len();
        if len != self.table.num_coefficients() {
            return Err(WhiteoutError::WitnessLength {
                expected: self.table.num_coefficients(),
                actual: len,
            });
        }
        let mut s = SecretVec::from(vec![0i64; len]);
        let (head, tail) = s.split_at_mut(residual.len());
        head.copy_from_slice(residual);
        tail.copy_from_slice(&pads.digits);
        Ok(s)
    }

    /// Release check that `relation` was lifted over this setup's table.
    fn check_relation(&self, relation: &LiftedRelation<F>) -> Result<(), WhiteoutError> {
        if *relation.table() == self.table {
            Ok(())
        } else {
            Err(WhiteoutError::InvalidRelation(
                "the relation was lifted over another setup's table",
            ))
        }
    }

    /// Absorbs every compressed target's payload, in block order, as a public
    /// value: the first transcript operation of both roles, so `ρ` and every
    /// later challenge bind the payloads.
    fn absorb_targets<C: Channel>(&self, channel: &mut C) {
        for (_, payload) in self.compressed_targets() {
            channel.public_all(payload);
        }
    }

    /// The compressed targets of `N`, in block order: each image's length
    /// and payload.
    pub(crate) fn compressed_targets(&self) -> impl Iterator<Item = (usize, &[F])> {
        self.linear
            .targets()
            .filter_map(|(rows, target)| match target {
                BlockTarget::Compressed { payload, .. } => Some((rows * D, payload.as_slice())),
                BlockTarget::Public(_) => None,
            })
    }

    /// Every target of `N` as an image: the prover's view.
    #[cfg(feature = "whiteout-prover")]
    pub(crate) fn target_images(&self) -> Result<Vec<CyclotomicRing<F, D>>, WhiteoutError> {
        let mut images = Vec::with_capacity(self.linear.num_outputs());
        for (_, target) in self.linear.targets() {
            match target {
                BlockTarget::Public(image)
                | BlockTarget::Compressed {
                    image: Some(image), ..
                } => images.extend_from_slice(image),
                BlockTarget::Compressed { image: None, .. } => {
                    return Err(WhiteoutError::InvalidRelation(
                        "the prover's setup lacks a compressed target's image",
                    ))
                }
            }
        }
        Ok(images)
    }

    /// The length of `t_D`.
    pub(crate) fn v_image_len(&self) -> usize {
        self.v_level.matrix.rows * CHALLENGE_RING_DIM
    }

    /// The layout of `e`: the response, `V` and `θ`, then the compression
    /// witness of each compressed target of `N` and of `t_D`. A function of
    /// the public setup, so a response prover's setup can be sized for `e`
    /// before any proof.
    ///
    /// # Errors
    ///
    /// The compression's witness-length errors, and
    /// [`WhiteoutError::ResponseTooLarge`] above [`MAX_RESPONSE_NUM_VARS`].
    pub fn response_layout(
        &self,
        compression: &dyn ImageCompression<F>,
    ) -> Result<ResponseLayout, WhiteoutError> {
        let mut segments: Vec<ResponseSegment> = self
            .table
            .segments()
            .zip(self.params.response_digits())
            .enumerate()
            .map(|(segment, (entry, &digits))| ResponseSegment::Response {
                segment,
                coefficients: entry.num_coefficients(),
                digits,
            })
            .collect();
        segments.push(ResponseSegment::V {
            coefficients: self.v_level.message_rings * CHALLENGE_RING_DIM,
        });
        segments.push(ResponseSegment::Theta);
        let images = self
            .compressed_targets()
            .map(|(len, _)| len)
            .chain(std::iter::once(self.v_image_len()));
        for (image, coefficients) in images.enumerate() {
            segments.push(ResponseSegment::Compression {
                image,
                coefficients,
                digits: compression.witness_len(coefficients)?,
            });
        }
        let layout = ResponseLayout { segments };
        let _ = response_num_vars(layout.len())?;
        Ok(layout)
    }
}

/// `ν = ⌈log₂ len⌉` of a response vector of `len` digits, refused above
/// [`MAX_RESPONSE_NUM_VARS`] (D27's priced bound).
pub(crate) fn response_num_vars(len: usize) -> Result<u32, WhiteoutError> {
    let num_vars = len.next_power_of_two().trailing_zeros();
    if num_vars > MAX_RESPONSE_NUM_VARS {
        return Err(WhiteoutError::ResponseTooLarge { num_vars });
    }
    Ok(num_vars)
}

/// The attempt seed: `(σ, n, cm(t_D))` through the attempt fork.
pub(crate) fn attempt_seed<H: Sponge, F: JoltField>(
    sigma: &[u8; FORK_SEED_LEN],
    nonce: &[u8; ATTEMPT_NONCE_LEN],
    t_d: &[F],
) -> [u8; 32] {
    let mut fork = Fork::<H>::attempt(sigma, nonce);
    let mut bytes = vec![0u8; F::NUM_BYTES];
    for coefficient in t_d {
        coefficient.to_bytes_le(&mut bytes);
        fork.absorb(&bytes);
    }
    fork.squeeze()
}

/// Verifies the Σ-protocol: absorbs the compressed targets' payloads,
/// squeezes `ρ` and `σ`, reads `cm(t_D)` and the nonce, rebuilds `c`, and
/// has `response` verify the response statement.
pub fn verify_sigma<F: JoltField, const D: usize, H: Sponge, P: ResponseProver<F, D>>(
    setup: &SigmaSetup<F, D>,
    relation: &LiftedRelation<F>,
    transcript: &mut VerifierTranscript<'_, H>,
    response: &P,
) -> Result<(), ZkError> {
    setup.check_relation(relation)?;
    setup.absorb_targets(transcript);
    let rho: Vec<F> = (0..relation.labels().num_challenges())
        .map(|_| transcript.challenge::<F>())
        .collect();
    let sigma: [u8; FORK_SEED_LEN] = transcript.challenge_bytes();
    let compression = response.compression();
    let v_payload: Vec<F> = transcript.receive_n(compression.payload_len(setup.v_image_len())?)?;
    let nonce: [u8; ATTEMPT_NONCE_LEN] = transcript
        .receive_bytes(ATTEMPT_NONCE_LEN)?
        .try_into()
        .map_err(|_| WhiteoutError::ResponseRejected("nonce length"))?;
    let seed = attempt_seed::<H, F>(&sigma, &nonce, &v_payload);
    let challenge = sample_indexed_challenges_from_seed(
        &seed,
        CHALLENGE_RING_DIM,
        1,
        &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
        Some(OperatorNormRejection::D64_SELECTIVE_L2),
    )
    .map_err(WhiteoutError::from)?
    .into_iter()
    .next()
    .ok_or(WhiteoutError::ResponseRejected("no challenge sampled"))?;
    let statement = ResponseStatement {
        setup,
        relation,
        compression,
        challenge,
        rho,
        v_payload,
    };
    response.verify_response(&statement, transcript)
}

#[cfg(feature = "whiteout-prover")]
pub use prover::prove_sigma;
#[cfg(feature = "whiteout-test-support")]
pub use prover::prove_sigma_unchecked;
#[cfg(all(test, feature = "whiteout-prover"))]
pub(crate) use prover::take_rejected;
#[cfg(any(
    all(test, feature = "whiteout-prover"),
    feature = "whiteout-diagnostics",
    feature = "whiteout-test-support"
))]
pub(crate) use prover::{prove_sigma_with, SelfChecks};

#[cfg(feature = "whiteout-prover")]
mod prover {
    use std::ops::Range;

    use jolt_algebra::{CyclotomicRing, JoltField};
    use jolt_crypto::challenge::{sample_challenge_ct, SparseChallenge};
    use jolt_crypto::lattice::{commit, CHALLENGE_RING_DIM};
    use jolt_crypto::sample::{GaussianSampler, GaussianWidth, JointRejection, SampleError};
    use jolt_transcript::{Channel, ProverTranscript, Sponge, ATTEMPT_NONCE_LEN, FORK_SEED_LEN};
    use rand_core::{CryptoRng, RngCore};
    use zeroize::Zeroize;

    use super::super::challenge::{challenge_ring, challenge_times_i64_into};
    use super::super::compression::{CompressedImage, ImageCompression};
    use super::super::dit::with_dit;
    use super::super::lift::{
        constant_ring, AggregatedRelation, ChallengeRing, ClaimTables, LiftedRelation, LinearParts,
    };
    use super::super::pad_commitment::LatticePadOpening;
    use super::super::response::{ResponseProver, ResponseStatement, ResponseWitness};
    use super::super::WhiteoutError;
    use super::{attempt_seed, SigmaSetup};
    use crate::masking::ZkError;
    use crate::secret::SecretVec;

    /// Which of the prover's own checks run before the challenges.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum SelfChecks {
        Run,
        /// Test-only: skip the per-row, opening and alphabet checks, so
        /// tests can show the verifier rejects what they catch.
        #[cfg(any(test, feature = "whiteout-test-support"))]
        Skip,
    }

    /// A rejected attempt's `t_D` bytes and nonce, for substitution tests.
    #[cfg(test)]
    type Rejected = (Vec<u8>, [u8; ATTEMPT_NONCE_LEN]);

    /// Rejected attempts, keyed by the thread that called
    /// `prove_sigma_with`: the attempts run on the DIT pool's workers, and
    /// tests may share the process.
    #[cfg(test)]
    static REJECTED: std::sync::Mutex<Vec<(std::thread::ThreadId, Rejected)>> =
        std::sync::Mutex::new(Vec::new());

    #[cfg(test)]
    thread_local! {
        /// On the worker running a proof: the thread that called it.
        static CALLER: std::cell::Cell<Option<std::thread::ThreadId>> =
            const { std::cell::Cell::new(None) };
    }

    /// The rejected attempts of this thread's proofs, cleared.
    #[cfg(test)]
    pub(crate) fn take_rejected() -> Vec<Rejected> {
        let caller = std::thread::current().id();
        let Ok(mut rejected) = REJECTED.lock() else {
            return Vec::new();
        };
        let (mine, others) = std::mem::take(&mut *rejected)
            .into_iter()
            .partition::<Vec<_>, _>(|(thread, _)| *thread == caller);
        *rejected = others;
        mine.into_iter().map(|(_, attempt)| attempt).collect()
    }

    /// What an accepted attempt hands on.
    struct Accepted<F: JoltField> {
        /// `cm(t_D)` and its witness.
        t_d: CompressedImage<F>,
        nonce: [u8; ATTEMPT_NONCE_LEN],
        challenge: SparseChallenge,
        v: SecretVec<F>,
    }

    /// Per-proof state shared by the attempts, and the attempts' reused
    /// buffers, which are zeroed after a rejection and when dropped.
    struct Prover<'a, F: JoltField, const D: usize> {
        setup: &'a SigmaSetup<F, D>,
        compression: &'a dyn ImageCompression<F>,
        aggregated: AggregatedRelation<'a, F>,
        tables: ClaimTables<F>,
        s: &'a [i64],
        parts_s: LinearParts<F>,
        l_s: ChallengeRing<F>,
        h: ChallengeRing<F>,
        /// `(coefficient range, native dimension)` per Gaussian block.
        blocks: Vec<(Range<usize>, usize)>,
        /// The one sampler: every Gaussian coordinate has the same width.
        sampler: GaussianSampler,
        tail_bound: u64,
        sigma: [u8; FORK_SEED_LEN],
        /// The mask, zero on public coordinates.
        y: SecretVec<i64>,
        /// `z` and the shift `v = c·s` on the Gaussian coordinates, block-major.
        z_gaussian: SecretVec<i64>,
        v_gaussian: SecretVec<i64>,
    }

    /// Proves the residual relation for the residual opening `residual` and
    /// the pad opening `pads`, returning the number of attempts (independent
    /// of the witness; the only attempt-dependent value that may leave).
    /// Runs on a fresh rayon pool whose workers all have DIT set (A11).
    /// The compressed targets' payloads are absorbed first, as
    /// [`verify_sigma`] does.
    pub fn prove_sigma<F, const D: usize, H, R, P>(
        setup: &SigmaSetup<F, D>,
        relation: &LiftedRelation<F>,
        residual: &[i64],
        pads: &LatticePadOpening,
        transcript: &mut ProverTranscript<H>,
        rng: &mut R,
        response: &P,
    ) -> Result<usize, ZkError>
    where
        F: JoltField,
        H: Sponge,
        R: RngCore + CryptoRng + Send,
        P: ResponseProver<F, D> + Sync,
    {
        with_dit(|| {
            let s = setup.private_vector(residual, pads)?;
            prove_under_dit(
                setup,
                relation,
                &s,
                transcript,
                rng,
                response,
                SelfChecks::Run,
            )
        })?
    }

    /// [`prove_sigma`] with the prover's per-row, opening and alphabet
    /// checks skipped, so a test can show that the response statement and
    /// the verifier reject a false relation on their own (Akita's
    /// consistent-cheat tests, `specs/whiteout-akita.md` §5.5 HT4). Test
    /// support only.
    #[cfg(feature = "whiteout-test-support")]
    pub fn prove_sigma_unchecked<F, const D: usize, H, R, P>(
        setup: &SigmaSetup<F, D>,
        relation: &LiftedRelation<F>,
        residual: &[i64],
        pads: &LatticePadOpening,
        transcript: &mut ProverTranscript<H>,
        rng: &mut R,
        response: &P,
    ) -> Result<usize, ZkError>
    where
        F: JoltField,
        H: Sponge,
        R: RngCore + CryptoRng + Send,
        P: ResponseProver<F, D> + Sync,
    {
        with_dit(|| {
            let s = setup.private_vector(residual, pads)?;
            prove_under_dit(
                setup,
                relation,
                &s,
                transcript,
                rng,
                response,
                SelfChecks::Skip,
            )
        })?
    }

    #[cfg(any(
        test,
        feature = "whiteout-diagnostics",
        feature = "whiteout-test-support"
    ))]
    pub(crate) fn prove_sigma_with<F, const D: usize, H, R, P>(
        setup: &SigmaSetup<F, D>,
        relation: &LiftedRelation<F>,
        s: &[i64],
        transcript: &mut ProverTranscript<H>,
        rng: &mut R,
        response: &P,
        checks: SelfChecks,
    ) -> Result<usize, ZkError>
    where
        F: JoltField,
        H: Sponge,
        R: RngCore + CryptoRng + Send,
        P: ResponseProver<F, D> + Sync,
    {
        #[cfg(test)]
        let caller = std::thread::current().id();
        with_dit(|| {
            #[cfg(test)]
            CALLER.set(Some(caller));
            prove_under_dit(setup, relation, s, transcript, rng, response, checks)
        })?
    }

    fn prove_under_dit<F, const D: usize, H, R, P>(
        setup: &SigmaSetup<F, D>,
        relation: &LiftedRelation<F>,
        s: &[i64],
        transcript: &mut ProverTranscript<H>,
        rng: &mut R,
        response: &P,
        checks: SelfChecks,
    ) -> Result<usize, ZkError>
    where
        F: JoltField,
        H: Sponge,
        R: RngCore + CryptoRng,
        P: ResponseProver<F, D>,
    {
        setup.check_relation(relation)?;
        setup.absorb_targets(transcript);
        let table = setup.table();
        if s.len() != table.num_coefficients() {
            return Err(WhiteoutError::WitnessLength {
                expected: table.num_coefficients(),
                actual: s.len(),
            }
            .into());
        }
        let tables = relation.claim_tables()?;
        // Before any challenge: every row and ct(g), N·s = t, the alphabets.
        let digit_bound = table
            .segments()
            .map(|segment| segment.digit_bound())
            .max()
            .unwrap_or(0);
        if checks == SelfChecks::Run {
            if let Some(row) = relation.first_violation(&tables, s)? {
                return Err(WhiteoutError::ResidualUnsatisfied { row }.into());
            }
            if *setup.linear().apply(s, digit_bound)? != *setup.target_images()? {
                return Err(WhiteoutError::ResidualOpeningMismatch.into());
            }
            table.check_private_vector(s)?;
        }

        let rho: Vec<F> = (0..relation.labels().num_challenges())
            .map(|_| transcript.challenge::<F>())
            .collect();
        let aggregated = relation.aggregate(&rho)?;
        let parts_s = aggregated.linear_parts(&tables, s, false)?;
        // ct(h) = Σ μ_row · row(s) + ct(g), so the per-row check above
        // already guarantees ct(h) = 0.
        let h = aggregated.quadratic(&constant_ring(F::one()), &parts_s);
        let blocks = setup
            .params()
            .blocks()
            .iter()
            .map(|block| {
                let placed = table
                    .placed()
                    .get(block.segment)
                    .ok_or(WhiteoutError::InvalidRelation("block segment missing"))?;
                Ok((placed.coefficients(), placed.segment.native_dim))
            })
            .collect::<Result<Vec<_>, WhiteoutError>>()?;
        let width =
            GaussianWidth::from_multiplier(setup.params().width()).map_err(WhiteoutError::from)?;
        let gaussian_len = blocks.iter().map(|(range, _)| range.len()).sum();
        let mut prover = Prover {
            setup,
            compression: response.compression(),
            l_s: setup.l_row().apply(s, digit_bound)?,
            aggregated,
            tables,
            s,
            parts_s,
            h,
            blocks,
            sampler: GaussianSampler::new(width),
            tail_bound: width.tail_bound(),
            sigma: transcript.challenge_bytes(),
            y: SecretVec::from(vec![0i64; table.num_coefficients()]),
            z_gaussian: SecretVec::from(vec![0i64; gaussian_len]),
            v_gaussian: SecretVec::from(vec![0i64; gaussian_len]),
        };
        let rejection = setup.params().rejection()?;
        let max_attempts = setup.params().max_attempts()?;

        for attempt in 1..=max_attempts {
            let Some(accepted) = prover.attempt::<H, R>(&rejection, rng)? else {
                continue;
            };
            transcript.send_all(&accepted.t_d.payload);
            transcript.send_bytes(&accepted.nonce);
            let witness = prover.witness(&accepted)?;
            // The attempt's buffers are wiped and freed before the response
            // proof, the prover's largest step.
            let compression = prover.compression;
            drop(prover);
            let statement = ResponseStatement {
                setup,
                relation,
                compression,
                challenge: accepted.challenge,
                rho,
                v_payload: accepted.t_d.payload,
            };
            response.prove_response(&statement, &witness, transcript)?;
            return Ok(attempt);
        }
        Err(WhiteoutError::AttemptsExhausted.into())
    }

    impl<F: JoltField, const D: usize> Prover<'_, F, D> {
        /// One attempt. Every step before the accept bit is constant time in
        /// the attempt's secrets; only that bit is branched on. Public
        /// coordinates are never touched: y is zero there and their shift
        /// does not enter the rejection rule.
        #[inline(never)]
        fn attempt<H: Sponge, R: RngCore + CryptoRng>(
            &mut self,
            rejection: &JointRejection,
            rng: &mut R,
        ) -> Result<Option<Accepted<F>>, WhiteoutError> {
            for (range, _) in &self.blocks {
                self.sampler.fill_parallel(
                    rng,
                    self.y
                        .get_mut(range.clone())
                        .ok_or(WhiteoutError::InvalidRelation("block outside s"))?,
                );
            }
            let parts_y = self
                .aggregated
                .linear_parts(&self.tables, &self.y[..], true)?;
            let f = self.setup.linear().apply(&self.y, self.tail_bound)?;
            let e0 = self.setup.l_row().apply(&self.y, self.tail_bound)?
                + self.aggregated.quadratic_at_zero(&parts_y);
            let e1 = self.l_s + self.aggregated.polar(&self.parts_s, &parts_y);
            let mut v = Vec::with_capacity(self.setup.v_level().message_rings * CHALLENGE_RING_DIM);
            v.extend(
                f.iter()
                    .flat_map(|ring| ring.coefficients().iter().copied()),
            );
            for ring in [e0, e1, self.h] {
                v.extend_from_slice(ring.coefficients());
            }
            let v = SecretVec::from(v);
            let v_rings = SecretVec::from(
                v.chunks_exact(CHALLENGE_RING_DIM)
                    .map(CyclotomicRing::from_slice)
                    .collect::<Vec<ChallengeRing<F>>>(),
            );
            // A rejected attempt's t_D, cm(t_D) and witness are never
            // published.
            let t_d = SecretVec::from(
                commit(self.setup.v_level(), self.setup.v_matrix(), &v_rings, &[])?
                    .iter()
                    .flat_map(|ring| ring.coefficients().iter().copied())
                    .collect::<Vec<F>>(),
            );
            let compressed = self.compression.compress(&t_d)?;
            let (payload, witness) = (
                SecretVec::from(compressed.payload),
                SecretVec::from(compressed.witness),
            );

            let mut nonce = [0u8; ATTEMPT_NONCE_LEN];
            rng.fill_bytes(&mut nonce);
            let challenge =
                sample_challenge_ct(&attempt_seed::<H, F>(&self.sigma, &nonce, &payload))?;
            let mut c = challenge.to_dense_ct()?;

            let mut offset = 0;
            for (range, native_dim) in &self.blocks {
                let (z, v_shift) = (
                    self.z_gaussian
                        .get_mut(offset..offset + range.len())
                        .ok_or(WhiteoutError::InvalidRelation("Gaussian buffer"))?,
                    self.v_gaussian
                        .get_mut(offset..offset + range.len())
                        .ok_or(WhiteoutError::InvalidRelation("Gaussian buffer"))?,
                );
                offset += range.len();
                challenge_times_i64_into(
                    &c,
                    *native_dim,
                    self.s
                        .get(range.clone())
                        .ok_or(WhiteoutError::InvalidRelation("segment outside s"))?,
                    v_shift,
                )?;
                let y = self
                    .y
                    .get(range.clone())
                    .ok_or(WhiteoutError::InvalidRelation("segment outside s"))?;
                for ((dst, &y), &shift) in z.iter_mut().zip(y).zip(v_shift.iter()) {
                    *dst = y + shift;
                }
            }
            let accept = match rejection.accept(&self.z_gaussian, &self.v_gaussian, rng) {
                Ok(accept) => accept,
                Err(SampleError::ShiftOutOfRange) => {
                    return Err(WhiteoutError::InvariantViolation(
                        "shift outside its bound: malformed witness",
                    ))
                }
                Err(error) => return Err(error.into()),
            };
            if !bool::from(accept) {
                #[cfg(test)]
                if let (Some(caller), Ok(mut rejected)) = (CALLER.get(), REJECTED.lock()) {
                    let mut bytes = vec![0u8; F::NUM_BYTES];
                    let mut flat = Vec::new();
                    for coefficient in &payload {
                        coefficient.to_bytes_le(&mut bytes);
                        flat.extend_from_slice(&bytes);
                    }
                    rejected.push((caller, (flat, nonce)));
                }
                // Wiped now rather than when the next attempt overwrites
                // them. The attempt's own buffers are zeroed as they drop.
                // In place: y stays zero on public coordinates.
                self.y.zeroize();
                self.z_gaussian.zeroize();
                self.v_gaussian.zeroize();
                nonce.zeroize();
                c.zeroize();
                return Ok(None);
            }
            Ok(Some(Accepted {
                t_d: CompressedImage {
                    payload: payload.to_vec(),
                    witness: witness.to_vec(),
                },
                nonce,
                challenge,
                v,
            }))
        }

        /// `e` for the accepted attempt: `z = y + c·s` on every coordinate
        /// (public ones are computed only now), `V`, and `θ`.
        fn witness(&self, accepted: &Accepted<F>) -> Result<ResponseWitness, WhiteoutError> {
            let table = self.setup.table();
            let c = accepted.challenge.to_dense_ct()?;
            let mut z = SecretVec::from(self.y.to_vec());
            for placed in table.placed() {
                let range = placed.coefficients();
                let mut shift = SecretVec::from(vec![0i64; range.len()]);
                challenge_times_i64_into(
                    &c,
                    placed.segment.native_dim,
                    self.s
                        .get(range.clone())
                        .ok_or(WhiteoutError::InvalidRelation("segment outside s"))?,
                    &mut shift,
                )?;
                for (dst, value) in z
                    .get_mut(range)
                    .ok_or(WhiteoutError::InvalidRelation("segment outside s"))?
                    .iter_mut()
                    .zip(shift.iter())
                {
                    *dst += *value;
                }
            }
            let c_ring: ChallengeRing<F> = challenge_ring(&c);
            let parts_z = self.aggregated.linear_parts(&self.tables, &z[..], false)?;
            let product = SecretVec::from(self.aggregated.product_polynomial(&c_ring, &parts_z));
            let theta = product.get(CHALLENGE_RING_DIM..).unwrap_or_default();
            ResponseWitness::new(
                self.setup,
                self.compression,
                &z,
                &accepted.v,
                theta,
                &accepted.t_d.witness,
            )
        }
    }
}
