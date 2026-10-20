//! The masked (ZK) verifier: what the stages record from a masked transcript,
//! and the residual relation that states their checks.

use jolt_sumcheck::BatchedMaskedRounds;
use jolt_transcript::{Sponge, VerifierTranscript};
use jolt_zk::masking::{CommittedPadsView, MaskedVerifier};

use crate::stages::relations::MaskedClaims;
use crate::VerifierError;

/// The residual relation over Jolt's stages. ZK applies only to the
/// homomorphic build: no zk protocol exists over the Akita axis (`akita` and
/// `zk` are mutually exclusive features).
#[cfg(not(feature = "akita"))]
#[doc(hidden)]
pub mod residual;

/// A masked stage batch as the ZK verifier reads it: the masked rounds with
/// their batching data, the output points derived at the batch point, and the
/// masked output claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskedBatch<F, P> {
    pub rounds: BatchedMaskedRounds<F>,
    pub output_points: P,
    pub output_claims: MaskedClaims<F>,
}

/// The masked channel over `transcript`, for a stage's ZK branch: the pad
/// view exists exactly when the proof is masked.
pub fn masked_channel<'a, 'p, H: Sponge, P>(
    pads: Option<&'a mut CommittedPadsView<P>>,
    transcript: &'a mut VerifierTranscript<'p, H>,
) -> Result<MaskedVerifier<'a, 'p, H>, VerifierError> {
    pads.map(|pads| pads.channel(transcript))
        .ok_or(VerifierError::MissingProofPayload {
            field: "pad commitment",
        })
}
