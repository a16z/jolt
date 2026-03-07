//! Batched sumcheck: reduces multiple claims into one via random linear
//! combination.
//!
//! Supports claims with **different** `num_vars` and `degree` bounds via
//! front-loaded batching: shorter instances are active only in the last
//! `num_vars` rounds and are padded with constant dummy polynomials in
//! earlier rounds. Each claim is scaled by $2^{N - n_i}$ where $N$ is the
//! maximum `num_vars` across all claims.

use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_transcript::Transcript;

use crate::claim::SumcheckClaim;
use crate::error::SumcheckError;
use crate::handler::{ClearRoundHandler, ClearRoundVerifier, RoundHandler, RoundVerifier};
use crate::proof::SumcheckProof;
use crate::prover::SumcheckCompute;

/// Batched sumcheck prover that combines $m$ independent claims.
///
/// Given claims $C_0, \ldots, C_{m-1}$ over polynomials $g_0, \ldots, g_{m-1}$,
/// draws a batching coefficient $\alpha$ from the transcript and proves the
/// combined claim:
///
/// $$\sum_{j=0}^{m-1} \alpha^j \cdot 2^{N - n_j} \cdot C_j$$
///
/// where $N = \max_j n_j$ is the maximum number of variables. Claims with
/// fewer variables are front-padded with constant dummy rounds.
pub struct BatchedSumcheckProver;

impl BatchedSumcheckProver {
    /// Proves a batch of sumcheck claims with a pluggable round handler.
    ///
    /// The handler controls how combined round polynomials are absorbed
    /// into the transcript and what proof artifact is produced.
    ///
    /// # Panics
    ///
    /// Panics if `claims` is empty or if `claims` and `witnesses` have
    /// different lengths.
    #[tracing::instrument(skip_all, name = "BatchedSumcheckProver::prove")]
    pub fn prove_with_handler<F, T, H>(
        claims: &[SumcheckClaim<F>],
        witnesses: &mut [Box<dyn SumcheckCompute<F>>],
        transcript: &mut T,
        challenge_fn: impl Fn(T::Challenge) -> F,
        mut handler: H,
    ) -> H::Proof
    where
        F: Field,
        T: Transcript,
        H: RoundHandler<F>,
    {
        assert!(!claims.is_empty(), "must have at least one claim");
        assert_eq!(
            claims.len(),
            witnesses.len(),
            "claims and witnesses must have the same length"
        );

        let max_num_vars = claims.iter().map(|c| c.num_vars).max().unwrap();
        let max_degree = claims.iter().map(|c| c.degree).max().unwrap();

        let alpha = challenge_fn(transcript.challenge());

        let offsets: Vec<usize> = claims.iter().map(|c| max_num_vars - c.num_vars).collect();

        let mut individual_claims: Vec<F> = claims
            .iter()
            .zip(offsets.iter())
            .map(|(c, &offset)| c.claimed_sum.mul_pow_2(offset))
            .collect();

        let two_inv = (F::one() + F::one())
            .inverse()
            .expect("2 is invertible in any prime field of order > 2");

        for round in 0..max_num_vars {
            let instance_polys: Vec<UnivariatePoly<F>> = witnesses
                .iter()
                .enumerate()
                .map(|(i, witness)| {
                    let active = round >= offsets[i] && round < offsets[i] + claims[i].num_vars;
                    if active {
                        witness.round_polynomial()
                    } else {
                        UnivariatePoly::new(vec![individual_claims[i] * two_inv])
                    }
                })
                .collect();

            // Combine evaluations at points 0, 1, ..., max_degree with alpha weights.
            let num_points = max_degree + 1;
            let mut combined_evals = vec![F::zero(); num_points];
            let mut alpha_power = F::one();
            for poly in &instance_polys {
                for (t, combined) in combined_evals.iter_mut().enumerate() {
                    *combined += alpha_power * poly.evaluate(F::from_u64(t as u64));
                }
                alpha_power *= alpha;
            }

            let points: Vec<(F, F)> = combined_evals
                .into_iter()
                .enumerate()
                .map(|(t, y)| (F::from_u64(t as u64), y))
                .collect();
            let combined_poly = UnivariatePoly::interpolate(&points);

            handler.absorb_round_poly(&combined_poly, transcript);
            let challenge = challenge_fn(transcript.challenge());
            handler.on_challenge(challenge);

            for (i, poly) in instance_polys.iter().enumerate() {
                individual_claims[i] = poly.evaluate(challenge);
            }

            for (i, witness) in witnesses.iter_mut().enumerate() {
                let active = round >= offsets[i] && round < offsets[i] + claims[i].num_vars;
                if active {
                    witness.bind(challenge);
                }
            }
        }

        handler.finalize()
    }

    /// Proves a batch of sumcheck claims with cleartext round handling.
    ///
    /// Convenience wrapper using [`ClearRoundHandler`].
    ///
    /// # Panics
    ///
    /// Panics if `claims` is empty or if `claims` and `witnesses` have
    /// different lengths.
    pub fn prove<F, T>(
        claims: &[SumcheckClaim<F>],
        witnesses: &mut [Box<dyn SumcheckCompute<F>>],
        transcript: &mut T,
        challenge_fn: impl Fn(T::Challenge) -> F,
    ) -> SumcheckProof<F>
    where
        F: Field,
        T: Transcript,
    {
        let max_num_vars = claims.iter().map(|c| c.num_vars).max().unwrap_or(0);
        Self::prove_with_handler(
            claims,
            witnesses,
            transcript,
            challenge_fn,
            ClearRoundHandler::with_capacity(max_num_vars),
        )
    }
}

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
        challenge_fn: impl Fn(T::Challenge) -> F,
        verifier: &V,
    ) -> Result<(F, Vec<F>), SumcheckError>
    where
        F: Field,
        T: Transcript,
        V: RoundVerifier<F>,
    {
        assert!(!claims.is_empty(), "must have at least one claim");

        let max_num_vars = claims.iter().map(|c| c.num_vars).max().unwrap();
        let max_degree = claims.iter().map(|c| c.degree).max().unwrap();

        let alpha = challenge_fn(transcript.challenge());

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
            challenge_fn,
            verifier,
        )
    }

    /// Verifies a batched sumcheck proof with cleartext verification.
    ///
    /// Convenience wrapper using [`ClearRoundVerifier`].
    ///
    /// # Errors
    ///
    /// Returns [`SumcheckError`] if verification fails.
    pub fn verify<F, T>(
        claims: &[SumcheckClaim<F>],
        proof: &SumcheckProof<F>,
        transcript: &mut T,
        challenge_fn: impl Fn(T::Challenge) -> F,
    ) -> Result<(F, Vec<F>), SumcheckError>
    where
        F: Field,
        T: Transcript,
    {
        Self::verify_with_handler(
            claims,
            &proof.round_polynomials,
            transcript,
            challenge_fn,
            &ClearRoundVerifier,
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
