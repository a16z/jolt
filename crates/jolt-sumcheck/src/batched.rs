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

use crate::claim::SumcheckClaim;
use crate::error::SumcheckError;
use crate::handler::{ClearRoundVerifier, RoundVerifier};
use crate::proof::SumcheckProof;

/// Batched sumcheck verifier.
///
/// Recomputes the combined claim with the same scaling and batching
/// coefficients as the prover, then delegates to the single-instance
/// verifier.
pub struct BatchedSumcheckVerifier;

impl BatchedSumcheckVerifier {
    /// Verifies a batched sumcheck proof with a pluggable round verifier.
    ///
    /// Returns `(v, r)` on success, where `v` is the combined final
    /// evaluation and `r` is the full challenge vector of length
    /// `max(num_vars)`.
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if verification fails.
    #[tracing::instrument(skip_all, name = "BatchedSumcheckVerifier::verify")]
    pub fn verify_with_handler<F, T, V>(
        claims: &[SumcheckClaim<F>],
        round_proofs: &[V::RoundProof],
        transcript: &mut T,
        verifier: &V,
    ) -> Result<(F, Vec<F>), SumcheckError>
    where
        F: Field,
        T: Transcript<Challenge = F>,
        V: RoundVerifier<F>,
    {
        assert!(!claims.is_empty(), "must have at least one claim");

        let max_num_vars = claims.iter().map(|c| c.num_vars).max().unwrap();
        let max_degree = claims.iter().map(|c| c.degree).max().unwrap();

        // Fiat-Shamir: absorb claimed sums (must match prover).
        for claim in claims {
            claim.claimed_sum.append_to_transcript(transcript);
        }

        let alpha: F = transcript.challenge();

        let combined_sum: F = claims
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (j, claim)| {
                let scaled = claim.claimed_sum.mul_pow_2(max_num_vars - claim.num_vars);
                acc + pow(alpha, j) * scaled
            });

        let combined_claim = SumcheckClaim {
            num_vars: max_num_vars,
            degree: max_degree,
            claimed_sum: combined_sum,
        };

        crate::verifier::SumcheckVerifier::verify_with_handler(
            &combined_claim,
            round_proofs,
            transcript,
            verifier,
        )
    }

    /// Like [`verify_with_handler`](Self::verify_with_handler) but also
    /// returns the batching coefficient α.
    pub fn verify_with_alpha<F, T>(
        claims: &[SumcheckClaim<F>],
        proof: &SumcheckProof<F>,
        transcript: &mut T,
    ) -> Result<(F, Vec<F>, F), SumcheckError>
    where
        F: Field,
        T: Transcript<Challenge = F>,
    {
        for claim in claims {
            claim.claimed_sum.append_to_transcript(transcript);
        }
        let alpha: F = transcript.challenge();

        let max_num_vars = claims.iter().map(|c| c.num_vars).max().unwrap();
        let max_degree = claims.iter().map(|c| c.degree).max().unwrap();

        let combined_sum: F = claims
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (j, claim)| {
                let scaled = claim.claimed_sum.mul_pow_2(max_num_vars - claim.num_vars);
                acc + pow(alpha, j) * scaled
            });

        let combined_claim = SumcheckClaim {
            num_vars: max_num_vars,
            degree: max_degree,
            claimed_sum: combined_sum,
        };

        let (final_eval, challenges) = crate::verifier::SumcheckVerifier::verify_with_handler(
            &combined_claim,
            &proof.round_polynomials,
            transcript,
            &ClearRoundVerifier::new(),
        )?;

        Ok((final_eval, challenges, alpha))
    }

    /// Verifies a batched sumcheck proof with cleartext verification.
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if verification fails.
    pub fn verify<F, T>(
        claims: &[SumcheckClaim<F>],
        proof: &SumcheckProof<F>,
        transcript: &mut T,
    ) -> Result<(F, Vec<F>), SumcheckError>
    where
        F: Field,
        T: Transcript<Challenge = F>,
    {
        Self::verify_with_handler(
            claims,
            &proof.round_polynomials,
            transcript,
            &ClearRoundVerifier::new(),
        )
    }
}

/// Computes $\text{base}^{\text{exp}}$ by repeated squaring.
#[inline]
fn pow<F: Field>(base: F, exp: usize) -> F {
    if exp == 0 {
        return F::one();
    }
    let mut result = F::one();
    let mut b = base;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result *= b;
        }
        b = b.square();
        e >>= 1;
    }
    result
}
