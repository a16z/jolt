//! The back-end seam: a zero-knowledge argument for the residual relation
//! over pads committed before the first challenge.

use jolt_field::Field;
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{ProverTranscript, Sponge, TranscriptError, VerifierTranscript};
use rand_core::{CryptoRng, RngCore};
use thiserror::Error as ThisError;

use super::{CommittedPads, CommittedPadsView, PadLayout, ResidualRelation};

/// Why masking or a back end failed. Every variant is an invariant violation
/// or a rejected proof; none is retryable.
#[derive(Clone, Debug, ThisError, PartialEq, Eq)]
pub enum ZkError {
    #[error("ZK transcript: {0}")]
    Transcript(#[from] TranscriptError),
    #[error("pad schedule exhausted: {requested} pads requested, {remaining} remaining")]
    PadsExhausted { requested: usize, remaining: usize },
    #[error("pad schedule has {scheduled} pads but {used} were used")]
    PadCountMismatch { scheduled: usize, used: usize },
    #[error("invalid pad layout: {reason}")]
    InvalidLayout { reason: &'static str },
    #[error("invalid residual relation: {reason}")]
    InvalidRelation { reason: &'static str },
    #[error("masked sumcheck: {reason}")]
    Sumcheck { reason: String },
    #[error("{backend} back end: {reason}")]
    Backend {
        backend: &'static str,
        reason: String,
    },
}

impl<F: Field> From<SumcheckError<F>> for ZkError {
    fn from(error: SumcheckError<F>) -> Self {
        Self::Sumcheck {
            reason: error.to_string(),
        }
    }
}

impl<F: Field> From<ZkError> for SumcheckError<F> {
    fn from(error: ZkError) -> Self {
        Self::PrivateChannel {
            reason: error.to_string(),
        }
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
