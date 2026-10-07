//! Typed transcript failures.

use thiserror::Error;

/// Why a transcript operation failed.
///
/// Every verifier-side variant means the proof is rejected. A
/// [`VerifierTranscript`](crate::VerifierTranscript) that has returned an error
/// is poisoned: later receives fail with [`TranscriptError::Poisoned`] and
/// [`finish`](crate::VerifierTranscript::finish) fails, so a caller that
/// swallows an error cannot go on to accept.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum TranscriptError {
    /// The argument string ended before a prover message.
    #[error("proof ended before a prover message")]
    Truncated,
    /// Prover-message bytes are not the canonical encoding of any value.
    #[error("prover message is not canonically encoded")]
    NonCanonical,
    /// A length, count, or nonce exceeds the bound the protocol states for it.
    #[error("prover message exceeds its public bound")]
    OutOfBounds,
    /// Bytes remain after the protocol's last prover message.
    #[error("proof has trailing bytes")]
    TrailingBytes,
    /// The bytes a read returned differ from the bytes the sponge absorbed:
    /// an atom's decoder consumed other than its `NUM_BYTES`.
    #[error("the transcript's byte cursor disagrees with the sponge's")]
    CursorMismatch,
    /// The proof-of-work predicate rejected the received nonce.
    #[error("proof-of-work predicate rejected the nonce")]
    GrindingRejected,
    /// The requested proof-of-work difficulty is outside the supported range.
    #[error("unsupported proof-of-work difficulty")]
    UnsupportedGrinding,
    /// No nonce in the bounded search range satisfied the predicate.
    #[error("proof-of-work search exhausted its nonce range")]
    GrindingExhausted,
    /// An earlier receive failed, so this transcript can no longer accept.
    #[error("transcript is poisoned by an earlier failure")]
    Poisoned,
}
