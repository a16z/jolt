//! The back-end seam: a zero-knowledge argument for the residual relation
//! over pads committed before the first challenge.

use std::error::Error;

use jolt_field::Field;
use jolt_sumcheck::SumcheckError;
#[cfg(feature = "prover")]
use jolt_transcript::ProverTranscript;
use jolt_transcript::{ClaimMismatch, FieldTranscript, Transcript, VerificationError};
#[cfg(feature = "prover")]
use rand_core::{CryptoRng, RngCore};
use thiserror::Error as ThisError;

#[cfg(feature = "prover")]
use super::CommittedPads;
use super::{CommittedPadsView, PadLayout, ResidualRelation};
use crate::blindfold::RelaxedError;
#[cfg(feature = "whiteout")]
use crate::whiteout::WhiteoutError;

/// What [`ZkBackend::commit_pads`] returns: the prover's pads and the
/// commitment the proof carries.
#[cfg(feature = "prover")]
pub type CommittedPadsWith<F, B> = (
    CommittedPads<F, <B as ZkBackend<F>>::PadOpening>,
    <B as ZkBackend<F>>::PadCommitment,
);

/// A field-generic source error (a sumcheck, BlindFold or Whiteout error), erased so
/// that [`ZkError`] stays field-independent like the verifier's error type.
/// Callers recover the typed error with `downcast_ref`.
pub type ErasedSource = Box<dyn Error + Send + Sync + 'static>;

/// Why masking or a back end failed. Every variant is an invariant violation
/// or a rejected proof, except [`ZkError::FoldOverflow`], after which the
/// caller may rerun the whole proof with fresh randomness.
#[derive(Debug, ThisError)]
pub enum ZkError {
    #[error("the ZK transcript rejected a read, a challenge or a proof of work")]
    TranscriptRejected,
    #[error("a challenge draw was rejected")]
    ChallengeRejected,
    #[error("the masked tape ended before the verifier's schedule")]
    MaskedTapeExhausted,
    #[error("the masked tape holds messages past the verifier's schedule")]
    MaskedTapeTrailing,
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
    /// or exhausted Σ attempts, both independent of the witness. Every other
    /// error is final.
    pub fn is_rerunnable(&self) -> bool {
        match self {
            Self::FoldOverflow { .. } => true,
            Self::Whiteout(source) => matches!(
                source.downcast_ref::<WhiteoutError>(),
                Some(WhiteoutError::AttemptsExhausted)
            ),
            Self::TranscriptRejected
            | Self::ChallengeRejected
            | Self::MaskedTapeExhausted
            | Self::MaskedTapeTrailing
            | Self::PadsExhausted { .. }
            | Self::PadCountMismatch { .. }
            | Self::InvalidLayout { .. }
            | Self::OperandPadsMismatch
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

/// A transcript failure surfaces as [`ZkError::TranscriptRejected`], and a
/// [`ZkError`] that crossed the sumcheck engine as
/// [`SumcheckError::PrivateChannel`] comes back as itself.
impl From<VerificationError> for ZkError {
    fn from(_: VerificationError) -> Self {
        Self::TranscriptRejected
    }
}

impl<F: Field + 'static> From<SumcheckError<F>> for ZkError {
    fn from(error: SumcheckError<F>) -> Self {
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
    /// The back end's public setup.
    type Setup;
    /// The pad commitment, carried in the proof and absorbed before the
    /// first challenge.
    type PadCommitment;
    /// The prover's opening data for the pad commitment.
    type PadOpening;
    /// The residual commitment the relation's witness opens to.
    type ResidualCommitment;
    /// The prover's residual witness.
    type Witness;
    /// The back end's proof of the residual relation, carried in the proof.
    type Proof;

    /// Samples and commits the pads of `layout`, absorbs the commitment, and
    /// returns the pads with the commitment the proof carries.
    ///
    /// # Errors
    ///
    /// A layout the back end cannot commit.
    #[cfg(feature = "prover")]
    fn commit_pads<T: Transcript, R: RngCore + CryptoRng>(
        setup: &Self::Setup,
        layout: &PadLayout,
        transcript: &mut T,
        rng: &mut R,
    ) -> Result<CommittedPadsWith<F, Self>, ZkError>
    where
        Self: Sized;

    /// Checks the proof's pad commitment against `layout` and absorbs it,
    /// at the position [`commit_pads`](Self::commit_pads) absorbed it.
    ///
    /// # Errors
    ///
    /// A commitment of the wrong shape.
    fn receive_pad_commitment<T: Transcript>(
        setup: &Self::Setup,
        layout: &PadLayout,
        commitment: Self::PadCommitment,
        transcript: &mut T,
    ) -> Result<CommittedPadsView<Self::PadCommitment>, ZkError>;

    /// Proves the residual relation over the committed pads.
    ///
    /// # Errors
    ///
    /// Unused pads, a relation the pads do not satisfy, or a prover failure.
    #[cfg(feature = "prover")]
    fn prove<T: FieldTranscript<F> + ProverTranscript, R: RngCore + CryptoRng>(
        setup: &Self::Setup,
        relation: &ResidualRelation<F>,
        pads: CommittedPads<F, Self::PadOpening>,
        residual: &Self::ResidualCommitment,
        witness: &Self::Witness,
        transcript: &mut T,
        rng: &mut R,
    ) -> Result<Self::Proof, ZkError>;

    /// Verifies `proof` of the residual relation over the pad commitment.
    ///
    /// # Errors
    ///
    /// Unused pads or a rejected proof.
    fn verify<T: FieldTranscript<F>>(
        setup: &Self::Setup,
        relation: &ResidualRelation<F>,
        pads: &CommittedPadsView<Self::PadCommitment>,
        residual: &Self::ResidualCommitment,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), ZkError>;
}
