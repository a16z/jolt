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
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::claim::SumcheckClaim;
use crate::error::SumcheckError;
use crate::proof::SumcheckProof;
use crate::prover::SumcheckWitness;

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
    /// Proves a batch of sumcheck claims.
    ///
    /// Claims may have different `num_vars` and `degree` bounds. Shorter
    /// instances are front-padded with dummy rounds (constant polynomial
    /// $H(X) = \text{claim}/2$) so all instances finish at the same global
    /// round.
    ///
    /// # Panics
    ///
    /// Panics if `claims` is empty or if `claims` and `witnesses` have
    /// different lengths.
    pub fn prove<F, T>(
        claims: &[SumcheckClaim<F>],
        witnesses: &mut [Box<dyn SumcheckWitness<F>>],
        transcript: &mut T,
        challenge_fn: impl Fn(T::Challenge) -> F,
    ) -> SumcheckProof<F>
    where
        F: Field,
        T: Transcript,
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

        // Front-loaded offsets: instance i becomes active at global round
        // (max_num_vars - num_vars_i) and runs through round (max_num_vars - 1).
        let offsets: Vec<usize> = claims.iter().map(|c| max_num_vars - c.num_vars).collect();

        // Scale each claim by 2^(max_num_vars - num_vars) to account for
        // the dummy variables that are summed out trivially.
        let mut individual_claims: Vec<F> = claims
            .iter()
            .zip(offsets.iter())
            .map(|(c, &offset)| c.claimed_sum.mul_pow_2(offset))
            .collect();

        let two_inv = (F::one() + F::one())
            .inverse()
            .expect("2 is invertible in any prime field of order > 2");

        let mut round_polynomials = Vec::with_capacity(max_num_vars);

        for round in 0..max_num_vars {
            // Compute per-instance round polynomials.
            let instance_polys: Vec<UnivariatePoly<F>> = witnesses
                .iter()
                .enumerate()
                .map(|(i, witness)| {
                    let active = round >= offsets[i] && round < offsets[i] + claims[i].num_vars;
                    if active {
                        witness.round_polynomial()
                    } else {
                        // Dummy round: H(X) = claim/2 (constant).
                        // Satisfies H(0) + H(1) = claim.
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

            append_poly_to_transcript(&combined_poly, transcript);
            let challenge = challenge_fn(transcript.challenge());

            // Update per-instance running claims.
            for (i, poly) in instance_polys.iter().enumerate() {
                individual_claims[i] = poly.evaluate(challenge);
            }

            // Bind only active witnesses.
            for (i, witness) in witnesses.iter_mut().enumerate() {
                let active = round >= offsets[i] && round < offsets[i] + claims[i].num_vars;
                if active {
                    witness.bind(challenge);
                }
            }

            round_polynomials.push(combined_poly);
        }

        SumcheckProof { round_polynomials }
    }
}

/// Batched sumcheck verifier.
///
/// Recomputes the combined claim with the same scaling and batching
/// coefficients as the prover, then delegates to the single-instance
/// verifier.
pub struct BatchedSumcheckVerifier;

impl BatchedSumcheckVerifier {
    /// Verifies a batched sumcheck proof.
    ///
    /// Returns `(v, r)` on success, where `v` is the combined final
    /// evaluation and `r` is the full challenge vector of length
    /// `max(num_vars)`.
    ///
    /// To extract the challenge slice for claim `i`, take
    /// `r[offset_i..offset_i + num_vars_i]` where
    /// `offset_i = max_num_vars - num_vars_i`.
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

        crate::verifier::SumcheckVerifier::verify(&combined_claim, proof, transcript, challenge_fn)
    }
}

#[inline]
fn append_poly_to_transcript<F: Field, T: Transcript>(
    poly: &UnivariatePoly<F>,
    transcript: &mut T,
) {
    for coeff in poly.coefficients() {
        coeff.append_to_transcript(transcript);
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
