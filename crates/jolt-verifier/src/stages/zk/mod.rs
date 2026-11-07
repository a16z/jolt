//! The masked (ZK) verifier: what the stages record from a masked transcript,
//! and the residual relation that states their checks.

use jolt_field::JoltField;
use jolt_sumcheck::{BatchedMaskedRounds, MaskedRoundReader};
use jolt_transcript::{FieldTranscript, PadSlots};
use jolt_zk::masking::{MaskedSession, MaskedVerifier, ZkError};

use crate::stages::relations::MaskedClaims;
use crate::VerifierError;

/// The residual relation over Jolt's stages 1–7.
#[cfg(any(not(feature = "akita"), feature = "zk"))]
#[doc(hidden)]
pub mod residual;

#[cfg(all(feature = "akita", feature = "zk"))]
pub use residual::MaskedMember;

/// A masked stage batch as the ZK verifier reads it: the masked rounds with
/// their batching data, the output points derived at the batch point, and the
/// masked output claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskedBatch<F, P> {
    pub rounds: BatchedMaskedRounds<F>,
    pub output_points: P,
    pub output_claims: MaskedClaims<F>,
}

/// A masked verifier's channel as the stage verifiers read it: masked
/// rounds, private messages placed on the pad schedule, and the challenges
/// the stages draw between them.
pub trait MaskedChannel<F: JoltField>: MaskedRoundReader<F, Error = ZkError> {
    /// Reads, absorbs and places one private message of `len` values.
    ///
    /// # Errors
    ///
    /// A truncated masked tape or an exhausted pad schedule.
    fn receive_private(&mut self, len: usize) -> Result<(Vec<F>, PadSlots), ZkError>;

    /// One challenge from the underlying transcript.
    ///
    /// # Errors
    ///
    /// A rejected challenge draw.
    fn challenge(&mut self) -> Result<F, VerifierError>;
}

impl<F, C, T> MaskedChannel<F> for MaskedVerifier<'_, '_, F, C, T>
where
    F: JoltField + 'static,
    T: FieldTranscript<F>,
{
    fn receive_private(&mut self, len: usize) -> Result<(Vec<F>, PadSlots), ZkError> {
        MaskedVerifier::receive_private(self, len)
    }

    fn challenge(&mut self) -> Result<F, VerifierError> {
        Ok(self.transcript().challenge()?)
    }
}

/// The masked channel over `transcript`, for a stage's ZK branch: the
/// masked session exists exactly when the proof is masked.
pub fn masked_channel<'a, 'm, F, P, T>(
    masked: Option<&'a mut MaskedSession<'m, F, P>>,
    transcript: &'a mut T,
) -> Result<MaskedVerifier<'a, 'm, F, P, T>, VerifierError>
where
    F: JoltField,
{
    masked
        .map(|session| session.channel(transcript))
        .ok_or(VerifierError::MissingProofPayload {
            field: "pad commitment",
        })
}
