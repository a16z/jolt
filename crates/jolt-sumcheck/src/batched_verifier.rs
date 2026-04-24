//! Batched sumcheck verification: reduces multiple claims into one via random
//! linear combination.
//!
//! Supports claims with **different** `num_vars` and `degree` bounds via
//! front-loaded batching: shorter instances are active only in the last
//! `num_vars` rounds and are padded with constant dummy polynomials in
//! earlier rounds. Each claim is scaled by $2^{N - n_i}$ where $N$ is the
//! maximum `num_vars` across all claims.

use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::claim::{EvaluationClaim, SumcheckClaim};
use crate::error::SumcheckError;
use crate::round::RoundVerifier;

/// Batched sumcheck verifier.
///
/// Recomputes the combined claim with the same scaling and batching
/// coefficients as the prover, then delegates to the single-instance
/// verifier.
pub struct BatchedSumcheckVerifier;

impl BatchedSumcheckVerifier {
    /// Verifies a batched sumcheck proof with a pluggable round verifier.
    ///
    /// Returns an [`EvaluationClaim`] `{ point: r, value: v }` on success,
    /// where `v` is the combined final evaluation and `r` is the full
    /// challenge vector of length `max(num_vars)`.
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if verification fails.
    #[tracing::instrument(skip_all, name = "BatchedSumcheckVerifier::verify")]
    pub fn verify<F, T, V>(
        claims: &[SumcheckClaim<F>],
        round_proofs: &[V::RoundProof],
        transcript: &mut T,
        verifier: &V,
    ) -> Result<EvaluationClaim<F>, SumcheckError>
    where
        F: Field,
        T: Transcript<Challenge = F>,
        V: RoundVerifier<F>,
    {
        let (first, rest) = claims.split_first().ok_or(SumcheckError::EmptyClaims)?;
        let max_num_vars = rest
            .iter()
            .fold(first.num_vars, |acc, c| acc.max(c.num_vars));
        let max_degree = rest.iter().fold(first.degree, |acc, c| acc.max(c.degree));

        // Fiat-Shamir: absorb claimed sums (must match prover).
        for claim in claims {
            claim.claimed_sum.append_to_transcript(transcript);
        }

        let alpha: F = transcript.challenge();

        // Running power of alpha: alpha^j for j = 0, 1, 2, …
        let mut alpha_pow = F::one();
        let mut combined_sum = F::zero();
        for claim in claims {
            let scaled = claim.claimed_sum.mul_pow_2(max_num_vars - claim.num_vars);
            combined_sum += alpha_pow * scaled;
            alpha_pow *= alpha;
        }

        let combined_claim = SumcheckClaim {
            num_vars: max_num_vars,
            degree: max_degree,
            claimed_sum: combined_sum,
        };

        crate::verifier::SumcheckVerifier::verify(
            &combined_claim,
            round_proofs,
            transcript,
            verifier,
        )
    }
}
