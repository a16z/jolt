//! This module provides the main engine for sum-check precompiles in Jolt ONNX.
//! It defines the `BatchedSumcheck` protocol for efficiently proving and verifying multiple sumcheck instances in parallel.
use crate::{
    field::JoltField,
    poly::unipoly::{CompressedUniPoly, UniPoly},
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{
        errors::ProofVerifyError,
        transcript::{AppendToTranscript, Transcript},
    },
};
use std::ops::Mul;

/// Trait for a sumcheck instance that can be batched with other instances.
///
/// This trait defines the interface needed to participate in the `BatchedSumcheck` protocol,
/// which reduces verifier cost and proof size by batching multiple sumcheck protocols.
pub trait BatchableSumcheckInstance<F: JoltField, ProofTranscript: Transcript> {
    /// Returns the maximum degree of the sumcheck polynomial.
    fn degree(&self) -> usize;

    /// Returns the number of rounds/variables in this sumcheck instance.
    fn num_rounds(&self) -> usize;

    /// Returns the initial claim of this sumcheck instance, i.e.
    /// input_claim = \sum_{x \in \{0, 1}^N} P(x)
    fn input_claim(&self) -> F;

    /// Computes the prover's message for a specific round of the sumcheck protocol.
    /// Returns the evaluations of the sumcheck polynomial at 0, 2, 3, ..., degree.
    /// The point evaluation at 1 can be interpolated using the previous round's claim.
    fn compute_prover_message(&self, round: usize) -> Vec<F>;

    /// Binds this sumcheck instance to the verifier's challenge from a specific round.
    /// This updates the internal state to prepare for the next round.
    fn bind(&mut self, r_j: F, round: usize);

    /// Caches polynomial opening claims needed after the sumcheck protocol completes.
    /// These openings will later be proven using either an opening proof or another sumcheck.
    fn cache_openings(&mut self);

    /// Computes the expected output claim given the verifier's challenges.
    /// This is used to verify the final result of the sumcheck protocol.
    fn expected_output_claim(&self, r: &[F]) -> F;
}

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    /// Prove a batch of sumcheck instances in parallel.
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        mut sumcheck_instances: Vec<&mut dyn BatchableSumcheckInstance<F, ProofTranscript>>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>) {
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // To see why we may need to scale by a power of two, consider a batch of
        // two sumchecks:
        //   claim_a = \sum_x P(x)             where x \in {0, 1}^M
        //   claim_b = \sum_{x, y} Q(x, y)     where x \in {0, 1}^M, y \in {0, 1}^N
        // Then the batched sumcheck is:
        //   \sum_{x, y} A * P(x) + B * Q(x, y)  where A and B are batching coefficients
        //   = A * \sum_y \sum_x P(x) + B * \sum_{x, y} Q(x, y)
        //   = A * \sum_y claim_a + B * claim_b
        //   = A * 2^N * claim_a + B * claim_b
        let mut individual_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                let num_rounds = sumcheck.num_rounds();
                sumcheck
                    .input_claim()
                    .mul_u64(1 << (max_num_rounds - num_rounds))
            })
            .collect();

        #[cfg(test)]
        let mut batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        let mut r: Vec<F> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            let remaining_rounds = max_num_rounds - round;

            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter()
                .zip(individual_claims.iter())
                .map(|(sumcheck, previous_claim)| {
                    let num_rounds = sumcheck.num_rounds();
                    if remaining_rounds > num_rounds {
                        // We haven't gotten to this sumcheck's variables yet, so
                        // the univariate polynomial is just a constant equal to
                        // the input claim, scaled by a power of 2.
                        let num_rounds = sumcheck.num_rounds();
                        let scaled_input_claim = sumcheck
                            .input_claim()
                            .mul_u64(1 << (remaining_rounds - num_rounds - 1));
                        // Constant polynomial
                        UniPoly::from_coeff(vec![scaled_input_claim])
                    } else {
                        let offset = max_num_rounds - sumcheck.num_rounds();
                        let mut univariate_poly_evals =
                            sumcheck.compute_prover_message(round - offset);
                        univariate_poly_evals.insert(1, *previous_claim - univariate_poly_evals[0]);
                        UniPoly::from_evals(&univariate_poly_evals)
                    }
                })
                .collect();

            // Linear combination of individual univariate polynomials
            let batched_univariate_poly: UniPoly<F> =
                univariate_polys.iter().zip(batching_coeffs.iter()).fold(
                    UniPoly::from_coeff(vec![]),
                    |mut batched_poly, (poly, coeff)| {
                        batched_poly += &(poly * coeff);
                        batched_poly
                    },
                );

            let compressed_poly = batched_univariate_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar();
            r.push(r_j);

            // Cache individual claims for this round
            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));

            #[cfg(test)]
            {
                // Sanity check
                let h0 = batched_univariate_poly.evaluate(&F::zero());
                let h1 = batched_univariate_poly.evaluate(&F::one());
                assert_eq!(
                    h0 + h1,
                    batched_claim,
                    "round {round}: H(0) + H(1) = {h0} + {h1} != {batched_claim}",
                );
                batched_claim = batched_univariate_poly.evaluate(&r_j);
            }

            for sumcheck in sumcheck_instances.iter_mut() {
                // If a sumcheck instance has fewer than `max_num_rounds`,
                // we wait until there are <= `sumcheck.num_rounds()` left
                // before binding its variables.
                if remaining_rounds <= sumcheck.num_rounds() {
                    let offset = max_num_rounds - sumcheck.num_rounds();
                    sumcheck.bind(r_j, round - offset);
                }
            }

            compressed_polys.push(compressed_poly);
        }

        for sumcheck in sumcheck_instances.iter_mut() {
            // Cache polynomial opening claims, to be proven using either an
            // opening proof or sumcheck (in the case of virtual polynomials).
            sumcheck.cache_openings();
        }

        (SumcheckInstanceProof::new(compressed_polys), r)
    }

    /// Verify a batch of sumcheck instances in parallel.
    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        sumcheck_instances: Vec<&dyn BatchableSumcheckInstance<F, ProofTranscript>>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let max_degree = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.degree())
            .max()
            .unwrap();
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        // To see why we may need to scale by a power of two, consider a batch of
        // two sumchecks:
        //   claim_a = \sum_x P(x)             where x \in {0, 1}^M
        //   claim_b = \sum_{x, y} Q(x, y)     where x \in {0, 1}^M, y \in {0, 1}^N
        // Then the batched sumcheck is:
        //   \sum_{x, y} A * P(x) + B * Q(x, y)  where A and B are batching coefficients
        //   = A * \sum_y \sum_x P(x) + B * \sum_{x, y} Q(x, y)
        //   = A * \sum_y claim_a + B * claim_b
        //   = A * 2^N * claim_a + B * claim_b
        let claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let num_rounds = sumcheck.num_rounds();
                sumcheck
                    .input_claim()
                    .mul_u64(1 << (max_num_rounds - num_rounds))
                    * coeff
            })
            .sum();

        let (output_claim, r_sumcheck) =
            proof.verify(claim, max_num_rounds, max_degree, transcript)?;

        let expected_output_claim = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                // If a sumcheck instance has fewer than `max_num_rounds`,
                // we wait until there are <= `sumcheck.num_rounds()` left
                // before binding its variables.
                // So, the sumcheck *actually* uses just the last `sumcheck.num_rounds()`
                // values of `r_sumcheck`.
                let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];
                sumcheck.expected_output_claim(r_slice) * coeff
            })
            .sum();

        if output_claim != expected_output_claim {
            return Err(ProofVerifyError::InternalError);
        }

        Ok(r_sumcheck)
    }
}

impl<F: JoltField> Mul<&F> for &UniPoly<F> {
    type Output = UniPoly<F>;

    fn mul(self, rhs: &F) -> UniPoly<F> {
        UniPoly::from_coeff(self.coeffs.iter().map(|c| *c * *rhs).collect::<Vec<_>>())
    }
}
