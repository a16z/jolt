#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::{AppendToTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;

use ark_serialize::*;
use rand_core::CryptoRngCore;
use std::marker::PhantomData;

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    /// Returns (proof, challenges, initial_batched_claim)
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (
        SumcheckInstanceProof<F, ProofTranscript>,
        Vec<F::Challenge>,
        F,
    ) {
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
                let input_claim = sumcheck.input_claim(opening_accumulator);
                transcript.append_scalar(&input_claim);
                input_claim.mul_pow_2(max_num_rounds - num_rounds)
            })
            .collect();

        // Compute the initial batched claim (needed for BlindFold)
        let initial_batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        #[cfg(test)]
        let mut batched_claim: F = initial_batched_claim;

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = format!("Sumcheck round {round}");
                print_current_memory_usage(label.as_str());
            }

            let remaining_rounds = max_num_rounds - round;

            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter_mut()
                .zip(individual_claims.iter())
                .map(|(sumcheck, previous_claim)| {
                    let num_rounds = sumcheck.num_rounds();
                    if remaining_rounds > num_rounds {
                        // We haven't gotten to this sumcheck's variables yet, so
                        // the univariate polynomial is just a constant equal to
                        // the input claim, scaled by a power of 2.
                        let num_rounds = sumcheck.num_rounds();
                        let scaled_input_claim = sumcheck
                            .input_claim(opening_accumulator)
                            .mul_pow_2(remaining_rounds - num_rounds - 1);
                        // Constant polynomial
                        UniPoly::from_coeff(vec![scaled_input_claim])
                    } else {
                        let offset = max_num_rounds - sumcheck.num_rounds();
                        sumcheck.compute_message(round - offset, *previous_claim)
                    }
                })
                .collect();

            // Linear combination of individual univariate polynomials
            let batched_univariate_poly: UniPoly<F> =
                univariate_polys.iter().zip(&batching_coeffs).fold(
                    UniPoly::from_coeff(vec![]),
                    |mut batched_poly, (poly, &coeff)| {
                        batched_poly += &(poly * coeff);
                        batched_poly
                    },
                );

            let compressed_poly = batched_univariate_poly.compress();

            // append the prover's message to the transcript
            compressed_poly.append_to_transcript(transcript);
            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

            // Cache individual claims for this round
            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));

            #[cfg(test)]
            {
                // Sanity check
                let h0 = batched_univariate_poly.evaluate::<F>(&F::zero());
                let h1 = batched_univariate_poly.evaluate::<F>(&F::one());
                assert_eq!(
                    h0 + h1,
                    batched_claim,
                    "round {round}: H(0) + H(1) = {h0} + {h1} != {batched_claim}"
                );
                batched_claim = batched_univariate_poly.evaluate(&r_j);
            }

            for sumcheck in sumcheck_instances.iter_mut() {
                // If a sumcheck instance has fewer than `max_num_rounds`,
                // we wait until there are <= `sumcheck.num_rounds()` left
                // before binding its variables.
                if remaining_rounds <= sumcheck.num_rounds() {
                    let offset = max_num_rounds - sumcheck.num_rounds();
                    sumcheck.ingest_challenge(r_j, round - offset);
                }
            }

            compressed_polys.push(compressed_poly);
        }

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        for sumcheck in sumcheck_instances.iter() {
            // If a sumcheck instance has fewer than `max_num_rounds`,
            // we wait until there are <= `sumcheck.num_rounds()` left
            // before binding its variables.
            // So, the sumcheck *actually* uses just the last `sumcheck.num_rounds()`
            // values of `r_sumcheck`.
            let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];

            // Cache polynomial opening claims, to be proven using either an
            // opening proof or sumcheck (in the case of virtual polynomials).
            sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
        }

        (
            SumcheckInstanceProof::new(compressed_polys),
            r_sumcheck,
            initial_batched_claim,
        )
    }

    /// Prove a batched sumcheck with Pedersen commitments (ZK mode).
    ///
    /// Instead of appending raw polynomial coefficients to the transcript,
    /// this appends Pedersen commitments. The proof still contains the
    /// coefficients for verification until BlindFold is implemented.
    ///
    /// # Security Note
    /// TODO(#ZK-SUMCHECK): The Pedersen commitments are used for Fiat-Shamir challenge
    /// derivation but are never opened/verified. The verifier appends the same commitment
    /// bytes but doesn't check they correspond to the polynomial coefficients. This allows
    /// a malicious prover to bias challenges. Fix requires either:
    /// 1. Add batch Pedersen opening proofs for all round commitments, or
    /// 2. Extend BlindFold R1CS to constrain commitment openings
    ///
    /// Returns (proof, challenges, initial_batched_claim)
    pub fn prove_zk<F: JoltField, C: JoltCurve, ProofTranscript: Transcript, R: CryptoRngCore>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
        pedersen_gens: &PedersenGenerators<C>,
        rng: &mut R,
    ) -> (
        SumcheckInstanceProof<F, ProofTranscript>,
        Vec<F::Challenge>,
        F,
    ) {
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        let mut individual_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim(opening_accumulator);
                transcript.append_scalar(&input_claim);
                input_claim.mul_pow_2(max_num_rounds - num_rounds)
            })
            .collect();

        // Compute the initial batched claim (needed for BlindFold)
        let initial_batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);
        let mut round_commitments: Vec<Vec<u8>> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = format!("Sumcheck round {round}");
                print_current_memory_usage(label.as_str());
            }

            let remaining_rounds = max_num_rounds - round;

            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter_mut()
                .zip(individual_claims.iter())
                .map(|(sumcheck, previous_claim)| {
                    let num_rounds = sumcheck.num_rounds();
                    if remaining_rounds > num_rounds {
                        let scaled_input_claim = sumcheck
                            .input_claim(opening_accumulator)
                            .mul_pow_2(remaining_rounds - num_rounds - 1);
                        UniPoly::from_coeff(vec![scaled_input_claim])
                    } else {
                        let offset = max_num_rounds - sumcheck.num_rounds();
                        sumcheck.compute_message(round - offset, *previous_claim)
                    }
                })
                .collect();

            let batched_univariate_poly: UniPoly<F> =
                univariate_polys.iter().zip(&batching_coeffs).fold(
                    UniPoly::from_coeff(vec![]),
                    |mut batched_poly, (poly, &coeff)| {
                        batched_poly += &(poly * coeff);
                        batched_poly
                    },
                );

            let compressed_poly = batched_univariate_poly.compress();

            // Generate blinding and compute Pedersen commitment
            let blinding = F::random(rng);
            let commitment =
                pedersen_gens.commit(&compressed_poly.coeffs_except_linear_term, &blinding);

            // Serialize commitment for transcript and proof
            let mut commitment_bytes = Vec::new();
            commitment
                .serialize_compressed(&mut commitment_bytes)
                .expect("Serialization should not fail");

            // Append commitment to transcript (instead of raw coefficients)
            transcript.append_message(b"UniPolyCommitment");
            transcript.append_bytes(&commitment_bytes);

            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

            // Cache individual claims for this round
            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));

            for sumcheck in sumcheck_instances.iter_mut() {
                if remaining_rounds <= sumcheck.num_rounds() {
                    let offset = max_num_rounds - sumcheck.num_rounds();
                    sumcheck.ingest_challenge(r_j, round - offset);
                }
            }

            compressed_polys.push(compressed_poly);
            round_commitments.push(commitment_bytes);
        }

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        for sumcheck in sumcheck_instances.iter() {
            let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];
            sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
        }

        (
            SumcheckInstanceProof::new_zk(compressed_polys, round_commitments),
            r_sumcheck,
            initial_batched_claim,
        )
    }

    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        sumcheck_instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript>>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
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
                let input_claim = sumcheck.input_claim(opening_accumulator);
                transcript.append_scalar(&input_claim);
                input_claim.mul_pow_2(max_num_rounds - num_rounds) * coeff
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

                // Cache polynomial opening claims, to be proven using either an
                // opening proof or sumcheck (in the case of virtual polynomials).
                sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
                let claim = sumcheck.expected_output_claim(opening_accumulator, r_slice);

                claim * coeff
            })
            .sum();

        if output_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(r_sumcheck)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct SumcheckInstanceProof<F: JoltField, ProofTranscript: Transcript> {
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
    /// Optional Pedersen commitments to round polynomials (for ZK mode).
    /// When present, these are appended to the transcript instead of raw coefficients.
    /// Each inner Vec<u8> is a serialized group element.
    pub round_commitments: Option<Vec<Vec<u8>>>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
    pub fn new(
        compressed_polys: Vec<CompressedUniPoly<F>>,
    ) -> SumcheckInstanceProof<F, ProofTranscript> {
        SumcheckInstanceProof {
            compressed_polys,
            round_commitments: None,
            _marker: PhantomData,
        }
    }

    pub fn new_zk(
        compressed_polys: Vec<CompressedUniPoly<F>>,
        round_commitments: Vec<Vec<u8>>,
    ) -> SumcheckInstanceProof<F, ProofTranscript> {
        SumcheckInstanceProof {
            compressed_polys,
            round_commitments: Some(round_commitments),
            _marker: PhantomData,
        }
    }

    /// Verify this sumcheck proof.
    /// Note: Verification does not execute the final check of sumcheck protocol: g_v(r_v) = oracle_g(r),
    /// as the oracle is not passed in. Expected that the caller will implement.
    ///
    /// Params
    /// - `claim`: Claimed evaluation
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F::Challenge>), ProofVerifyError> {
        let mut e = claim;
        let mut r: Vec<F::Challenge> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            // verify degree bound
            if self.compressed_polys[i].degree() > degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    self.compressed_polys[i].degree(),
                ));
            }

            // Append to transcript: use commitment if available, otherwise raw coefficients
            if let Some(ref commitments) = self.round_commitments {
                transcript.append_message(b"UniPolyCommitment");
                transcript.append_bytes(&commitments[i]);
            } else {
                self.compressed_polys[i].append_to_transcript(transcript);
            }

            //derive the verifier's challenge for the next round
            let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i using the hint
            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
        }

        Ok((e, r))
    }
}

/// The sumcheck proof for a univariate skip round
/// Consists of the (single) univariate polynomial sent in that round, no omission of any coefficient
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct UniSkipFirstRoundProof<F: JoltField, ProofTranscript: Transcript> {
    pub uni_poly: UniPoly<F>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> UniSkipFirstRoundProof<F, ProofTranscript> {
    pub fn new(uni_poly: UniPoly<F>) -> Self {
        Self {
            uni_poly,
            _marker: PhantomData,
        }
    }

    /// Verify only the univariate-skip first round.
    ///
    /// Params
    /// - `const N`: the first degree plus one (e.g. the size of the first evaluation domain)
    /// - `const FIRST_ROUND_POLY_NUM_COEFFS`: number of coefficients in the first-round polynomial
    /// - `degree_bound_first`: Maximum allowed degree of the first univariate polynomial
    /// - `transcript`: Fiat-Shamir transcript
    ///
    /// Returns `(r0, next_claim)` where `r0` is the verifier challenge for the first round
    /// and `next_claim` is the claimed evaluation at `r0` to be used by remaining rounds.
    pub fn verify<const N: usize, const FIRST_ROUND_POLY_NUM_COEFFS: usize>(
        &self,
        degree_bound_first: usize,
        claim: F,
        transcript: &mut ProofTranscript,
    ) -> Result<(F::Challenge, F), ProofVerifyError> {
        // Degree check for the high-degree first polynomial
        if self.uni_poly.degree() > degree_bound_first {
            return Err(ProofVerifyError::InvalidInputLength(
                degree_bound_first,
                self.uni_poly.degree(),
            ));
        }

        // Append full polynomial and derive r0
        self.uni_poly.append_to_transcript(transcript);
        let r0 = transcript.challenge_scalar_optimized::<F>();

        // Check symmetric-domain sum equals zero (initial claim), and compute next claim s1(r0)
        let (ok, next_claim) = self
            .uni_poly
            .check_sum_evals_and_set_new_claim::<N, FIRST_ROUND_POLY_NUM_COEFFS>(&claim, &r0);
        if !ok {
            return Err(ProofVerifyError::UniSkipVerificationError);
        }

        Ok((r0, next_claim))
    }
}
