//! Batch challenges of the final opening.
//!
//! WARNING: the transcript interaction here must stay byte-identical to
//! `jolt_openings::HomomorphicBatch::verify_batch`'s statement absorption
//! (the `rlc_claims` label + per-claim values, then one
//! `challenge_scalar_powers` draw).

use crate::field::JoltField;
use crate::transcripts::Transcript;

/// Batch challenges of the homomorphic (RLC) final opening.
pub struct HomomorphicBatchChallenges<F: JoltField> {
    pub gamma_powers: Vec<F>,
    pub joint_claim: F,
}

/// Absorbs the final-opening claims and draws the RLC batch challenges.
///
/// In ZK mode the claims are secret, so nothing is absorbed before the draw;
/// binding comes from the BlindFold constraints instead.
pub fn homomorphic_batch_challenges<F: JoltField, T: Transcript>(
    transcript: &mut T,
    claims: &[F],
) -> HomomorphicBatchChallenges<F> {
    #[cfg(not(feature = "zk"))]
    transcript.append_scalars(b"rlc_claims", claims);
    let gamma_powers: Vec<F> = transcript.challenge_scalar_powers(claims.len());
    let joint_claim: F = gamma_powers
        .iter()
        .zip(claims.iter())
        .map(|(gamma, claim)| *gamma * *claim)
        .sum();
    HomomorphicBatchChallenges {
        gamma_powers,
        joint_claim,
    }
}
