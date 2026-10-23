//! The back-end seam: a zero-knowledge argument for the residual relation
//! over pads committed before the first challenge.

use std::error::Error;

use jolt_algebra::Field;
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{
    ClaimMismatch, ProverTranscript, Sponge, TranscriptError, VerifierTranscript,
};
use rand_core::{CryptoRng, RngCore};
use thiserror::Error as ThisError;

use crate::blindfold::RelaxedError;
#[cfg(feature = "whiteout")]
use crate::whiteout::WhiteoutError;

use super::{CommittedPads, CommittedPadsView, PadLayout, ResidualRelation};

/// A field-generic source error (a sumcheck or BlindFold error), erased so
/// that [`ZkError`] stays field-independent like the verifier's error type.
/// Callers recover the typed error with `downcast_ref`.
pub type ErasedSource = Box<dyn Error + Send + Sync + 'static>;

/// Why masking or a back end failed. Every variant is an invariant violation
/// or a rejected proof, except [`ZkError::FoldOverflow`], after which the
/// caller may rerun the whole proof with fresh randomness.
#[derive(Debug, ThisError)]
pub enum ZkError {
    #[error("ZK transcript: {0}")]
    Transcript(#[from] TranscriptError),
    #[error("pad schedule exhausted: {requested} pads requested, {remaining} remaining")]
    PadsExhausted { requested: usize, remaining: usize },
    #[error("pad schedule has {scheduled} pads but {used} were used")]
    PadCountMismatch { scheduled: usize, used: usize },
    #[error("invalid pad layout: {reason}")]
    InvalidLayout { reason: &'static str },
    /// The schedule's product operand pads differ from the committed
    /// layout's, with equal pad counts.
    #[error("product operand pads differ from the committed layout")]
    OperandPadsMismatch,
    #[error("invalid residual relation: {reason}")]
    InvalidRelation { reason: &'static str },
    /// The prover's decoded values fail a check its lowering states.
    #[error("decoded values: {0}")]
    ClaimMismatch(#[from] ClaimMismatch),
    /// A private message's field is not the pad field: a pad vector of one
    /// field masks only messages of that field.
    #[error("private message field is not the pad field")]
    MessageFieldMismatch,
    /// A masked round's wire length differs from its degree schedule.
    #[error("masked round {round} has {got} values, its schedule {expected}")]
    RoundWidthMismatch {
        round: usize,
        expected: usize,
        got: usize,
    },
    /// A masked sumcheck has a different round count than its schedule.
    #[error("masked sumcheck has {got} rounds, its schedule {expected}")]
    RoundCountMismatch { expected: usize, got: usize },
    /// A `SumcheckError<F>`.
    #[error("masked sumcheck: {0}")]
    Sumcheck(#[source] ErasedSource),
    #[error("BlindFold relation shape: {0}")]
    BlindFoldShape(#[from] RelaxedError),
    /// A BlindFold `ProverError<F>`.
    #[error("BlindFold prover: {0}")]
    BlindFoldProver(#[source] ErasedSource),
    /// A BlindFold `VerificationError<F>` other than a transcript failure.
    #[error("BlindFold proof rejected: {0}")]
    BlindFoldRejected(#[source] ErasedSource),
    /// A Whiteout `WhiteoutError`, erased so the variant exists whether or
    /// not the `whiteout` feature is on (feature additivity: dependents match
    /// `ZkError` exhaustively).
    #[error("Whiteout: {0}")]
    Whiteout(#[source] ErasedSource),
    /// The lattice PCS's one-shot fold response at masked level `level`
    /// fell outside its digit range (Akita `AkitaError::FoldOverflow`). The
    /// proof is aborted; a rerun draws all of its randomness fresh. Akita
    /// maps its error to this variant, since jolt-zk cannot name Akita's
    /// error type.
    #[error("fold response overflowed its digit range at masked level {level}")]
    FoldOverflow { level: u32 },
}

#[cfg(feature = "whiteout")]
impl ZkError {
    /// Whether a fresh rerun of the whole proof may succeed: a fold overflow
    /// or exhausted Σ attempts, both independent of the witness
    /// (`specs/whiteout-jolt-integration.md` §6). Every other error is final.
    pub fn is_rerunnable(&self) -> bool {
        match self {
            Self::FoldOverflow { .. } => true,
            Self::Whiteout(source) => matches!(
                source.downcast_ref::<WhiteoutError>(),
                Some(WhiteoutError::AttemptsExhausted)
            ),
            Self::Transcript(_)
            | Self::PadsExhausted { .. }
            | Self::PadCountMismatch { .. }
            | Self::InvalidLayout { .. }
            | Self::OperandPadsMismatch
            | Self::MessageFieldMismatch
            | Self::RoundWidthMismatch { .. }
            | Self::RoundCountMismatch { .. }
            | Self::InvalidRelation { .. }
            | Self::ClaimMismatch(_)
            | Self::Sumcheck(_)
            | Self::BlindFoldShape(_)
            | Self::BlindFoldProver(_)
            | Self::BlindFoldRejected(_) => false,
        }
    }
}

#[cfg(feature = "whiteout")]
impl From<WhiteoutError> for ZkError {
    fn from(error: WhiteoutError) -> Self {
        Self::Whiteout(Box::new(error))
    }
}

/// A transcript failure surfaces as [`ZkError::Transcript`], and a
/// [`ZkError`] that crossed the sumcheck engine as
/// [`SumcheckError::PrivateChannel`] comes back as itself.
impl<F: Field + 'static> From<SumcheckError<F>> for ZkError {
    fn from(error: SumcheckError<F>) -> Self {
        if let SumcheckError::Transcript(error) = error {
            return Self::Transcript(error);
        }
        if let SumcheckError::PrivateChannel(source) = error {
            return source
                .downcast::<Self>()
                .map_or_else(Self::Sumcheck, |error| *error);
        }
        Self::Sumcheck(Box::new(error))
    }
}

impl<F: Field> From<ZkError> for SumcheckError<F> {
    fn from(error: ZkError) -> Self {
        Self::PrivateChannel(Box::new(error))
    }
}

/// A zero-knowledge back end: commits the pads, then proves the residual
/// relation over them and the residual witness. BlindFold (curve) is the
/// first implementation; Whiteout (lattice) is `Whiteout<R: ResponseProver<F>>`
/// with `R` reached through `Setup`, since `ResponseProver` is not
/// dyn-compatible.
pub trait ZkBackend<F> {
    type Setup;
    /// What the verifier holds after the pad commitment is sent.
    type PadCommitment;
    /// Prover-side data that opens the pad commitment.
    type PadOpening;
    /// The commitment to the residual witness.
    type ResidualCommitment;
    type Witness;

    /// Samples pads, commits to them and sends the commitment. Called before
    /// the first challenge.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if the layout does not fit the setup.
    fn commit_pads<H: Sponge, R: RngCore + CryptoRng>(
        setup: &Self::Setup,
        layout: &PadLayout,
        transcript: &mut ProverTranscript<H>,
        rng: &mut R,
    ) -> Result<CommittedPads<F, Self::PadOpening>, ZkError>;

    /// Reads the pad commitment at the same transcript position. The
    /// verifier's masked channel can only be built from the returned view.
    ///
    /// # Errors
    ///
    /// [`ZkError`] on a malformed commitment.
    fn receive_pad_commitment<H: Sponge>(
        setup: &Self::Setup,
        layout: &PadLayout,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<CommittedPadsView<Self::PadCommitment>, ZkError>;

    /// Proves that the committed pads and `witness` satisfy `relation`.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if they do not, or the back end fails.
    fn prove<H: Sponge, R: RngCore + CryptoRng>(
        setup: &Self::Setup,
        relation: &ResidualRelation<F>,
        pads: CommittedPads<F, Self::PadOpening>,
        residual: &Self::ResidualCommitment,
        witness: &Self::Witness,
        transcript: &mut ProverTranscript<H>,
        rng: &mut R,
    ) -> Result<(), ZkError>;

    /// Verifies the back end's proof of `relation` over the committed pads.
    ///
    /// # Errors
    ///
    /// [`ZkError`] if the proof is rejected.
    fn verify<H: Sponge>(
        setup: &Self::Setup,
        relation: &ResidualRelation<F>,
        pads: &CommittedPadsView<Self::PadCommitment>,
        residual: &Self::ResidualCommitment,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), ZkError>;
}
