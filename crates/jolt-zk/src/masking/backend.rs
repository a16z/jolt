//! The back-end seam: a zero-knowledge argument for the residual relation
//! over pads committed before the first challenge.

use std::error::Error;

use jolt_field::Field;
use jolt_sumcheck::SumcheckError;
#[cfg(feature = "masking-prover")]
use jolt_transcript::ProverTranscript;
use jolt_transcript::{ClaimMismatch, Encoding, FieldTranscript, Transcript, VerificationError};
#[cfg(feature = "masking-prover")]
use rand_core::{CryptoRng, RngCore};
use thiserror::Error as ThisError;

#[cfg(feature = "masking-prover")]
use super::CommittedPads;
use super::{CommittedPadsView, PadLayout, ResidualRelation};

/// What [`ZkBackendProver::commit_pads`] returns: the prover's pads and the
/// commitment the proof carries.
#[cfg(feature = "masking-prover")]
pub type CommittedPadsWith<F, B> = (
    CommittedPads<F, <B as ZkBackendProver<F>>::PadOpening>,
    <B as ZkBackend<F>>::PadCommitment,
);

/// A field-generic source error (a sumcheck error), erased so
/// that [`ZkError`] stays field-independent like the verifier's error type.
/// Callers recover the typed error with `downcast_ref`.
pub type ErasedSource = Box<dyn Error + Send + Sync + 'static>;

/// Why masking or a back end failed: an invariant violation or a rejected
/// proof.
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

/// The verifier half of a zero-knowledge back end: receives the pad
/// commitment, then verifies the back end's proof of the residual relation
/// over it. This crate defines the seam only; no back end in it implements
/// it. A back end whose prover is not dyn-compatible reaches it through
/// `Setup`. The prover half is `ZkBackendProver`, so a verifier crate
/// implements this trait whatever features its build turns on.
pub trait ZkBackend<F> {
    /// The back end's public setup.
    type Setup;
    /// The pad commitment, carried in the proof and absorbed before the
    /// first challenge.
    type PadCommitment: Encoding;
    /// The residual commitment the relation's witness opens to.
    type ResidualCommitment;
    /// The back end's proof of the residual relation, carried in the proof.
    type Proof;

    /// Checks the proof's pad commitment against `layout` and absorbs it,
    /// at the position `ZkBackendProver::commit_pads` absorbed it, with
    /// [`CommittedPadsView::receive`].
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

    /// Verifies `proof` of the residual relation over the pad commitment,
    /// after [`verify`](Self::verify) checked that `relation` and `pads`
    /// share one layout. Callers use `verify`.
    ///
    /// # Errors
    ///
    /// Unused pads or a rejected proof.
    fn verify_residual<T: FieldTranscript<F>>(
        setup: &Self::Setup,
        relation: &ResidualRelation<F>,
        pads: &CommittedPadsView<Self::PadCommitment>,
        residual: &Self::ResidualCommitment,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), ZkError>;

    /// Verifies `proof` of the residual relation over the pad commitment.
    ///
    /// # Errors
    ///
    /// [`ZkError::InvalidLayout`] if the relation's layout differs from the
    /// committed one, then as [`verify_residual`](Self::verify_residual).
    fn verify<T: FieldTranscript<F>>(
        setup: &Self::Setup,
        relation: &ResidualRelation<F>,
        pads: &CommittedPadsView<Self::PadCommitment>,
        residual: &Self::ResidualCommitment,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), ZkError> {
        if relation.layout != *pads.layout() {
            return Err(ZkError::InvalidLayout {
                reason: "the relation's pad layout differs from the committed one",
            });
        }
        Self::verify_residual(setup, relation, pads, residual, proof, transcript)
    }
}

/// The prover half of a zero-knowledge back end: commits the pads, then
/// proves the residual relation over them and the residual witness.
#[cfg(feature = "masking-prover")]
pub trait ZkBackendProver<F>: ZkBackend<F> {
    /// The prover's opening data for the pad commitment.
    type PadOpening;
    /// The prover's residual witness.
    type Witness;

    /// Samples and commits the pads of `layout`, absorbs the commitment
    /// with [`CommittedPads::commit`], and returns the pads with the
    /// commitment the proof carries.
    ///
    /// # Errors
    ///
    /// A layout the back end cannot commit.
    fn commit_pads<T: Transcript, R: RngCore + CryptoRng>(
        setup: &Self::Setup,
        layout: &PadLayout,
        transcript: &mut T,
        rng: &mut R,
    ) -> Result<CommittedPadsWith<F, Self>, ZkError>
    where
        Self: Sized;

    /// Proves the residual relation over the committed pads.
    ///
    /// # Errors
    ///
    /// Unused pads, a relation the pads do not satisfy, or a prover failure.
    fn prove<T: FieldTranscript<F> + ProverTranscript, R: RngCore + CryptoRng>(
        setup: &Self::Setup,
        relation: &ResidualRelation<F>,
        pads: CommittedPads<F, Self::PadOpening>,
        residual: &Self::ResidualCommitment,
        witness: &Self::Witness,
        transcript: &mut T,
        rng: &mut R,
    ) -> Result<Self::Proof, ZkError>;
}
