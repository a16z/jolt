#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::poly::commitment::dory::{DoryContext, DoryGlobals};
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::{AppendToTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
use ark_std::log2;

use ark_serialize::*;
use std::marker::PhantomData;

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>) {
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());
        let mut trusted_advice_poly_claim: F = F::zero();
        let mut trusted_advice_poly_binded: UniPoly<F> = UniPoly::from_coeff(vec![F::zero()]);

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
                let scaling_factor = max_num_rounds - num_rounds;
                let x = input_claim.mul_pow_2(scaling_factor);
                tracing::info!("DARIVARI first individual claim: {:?}", x);

                if sumcheck.trusted_advice_dimensions().is_some() {
                    trusted_advice_poly_claim = sumcheck.input_claim(opening_accumulator);
                    tracing::info!("DARIVARI first individual claim: {:?}, scaling factor: {}", x, scaling_factor);
                }
                x
            })
            .collect();



        // #[cfg(test)]
        let mut batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

        tracing::info!("THIIISSSSSSS in the prover batched_claim: {:?}", batched_claim);

        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(max_num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(max_num_rounds);

        let row_bind_start = 6;
        let row_bind_end = 10;
        let col_bind_start = 18;
        let col_bind_end = 22;

        // Get main dimensions for trusted advice binding logic
        let _ctx = DoryGlobals::with_context(DoryContext::Main);
        let log_main_rows = log2(DoryGlobals::get_max_num_rows()) as usize;
        let _log_main_columns = log2(DoryGlobals::get_num_columns()) as usize;
        drop(_ctx);

        for round in 0..max_num_rounds {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = format!("Sumcheck round {round}");
                print_current_memory_usage(label.as_str());
            }

            let remaining_rounds = max_num_rounds - round;


            let in_row_phase = round >= row_bind_start && round < row_bind_end;
            let in_col_phase = round >= col_bind_start && round < col_bind_end;

            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter_mut()
                .zip(individual_claims.iter_mut())
                .map(|(sumcheck, previous_claim)| {
                    // Check if this is a trusted advice polynomial
                    if let Some((ta_rows, _ta_columns)) = sumcheck.trusted_advice_dimensions() {
                        // For trusted advice, we bind in two separate phases with hardcoded ranges:
                        // Phase 1 (rows): rounds [6, 10)
                        // Phase 2 (columns): rounds [18, 22)

                        let poly = if round < row_bind_start {
                            let scaling_factor = (row_bind_start - round) + (col_bind_start - row_bind_end) + (max_num_rounds - col_bind_end) - 1;
                            tracing::info!("DARIVARI in round {} before row bind start, scaling factor: {}", round, scaling_factor);
                            let scaled_claim = sumcheck
                            .input_claim(opening_accumulator)
                            .mul_pow_2(scaling_factor);
                            UniPoly::from_coeff(vec![scaled_claim])

                        } else if in_row_phase {
                            // Binding row variable
                            let ta_round = round - row_bind_start;
                            let scaling_factor = (col_bind_start - row_bind_end) + (max_num_rounds - col_bind_end);
                            tracing::info!("DARIVARI in round {} in row bind, previous claim: {} scaling factor: {}", round, trusted_advice_poly_claim, scaling_factor);
                            let x = sumcheck.compute_message(ta_round, trusted_advice_poly_claim);
                            trusted_advice_poly_binded = x.clone();
                            UniPoly::from_coeff(x.coeffs.iter().map(|coeff| coeff.mul_pow_2(scaling_factor)).collect())
                        } else if round >= row_bind_end && round < col_bind_start {
                            let scaling_factor = (col_bind_start - round) + (max_num_rounds - col_bind_end) - 1;
                            tracing::info!("DARIVARI in round {} in gap, previous claim: {} scaling factor: {}", round, trusted_advice_poly_claim, scaling_factor);
                            let scaled_input_claim = trusted_advice_poly_claim
                                .mul_pow_2(scaling_factor);

                            UniPoly::from_coeff(vec![scaled_input_claim])
                        } else if in_col_phase {
                            let ta_round = ta_rows + (round - col_bind_start);
                            let scaling_factor = (max_num_rounds - col_bind_end);

                            tracing::info!("DARIVARI in round {} in col bind, previous claim: {}, ta_round: {}, scaling factor: {}", round, trusted_advice_poly_claim, ta_round, scaling_factor);

                            let x = sumcheck.compute_message(ta_round, trusted_advice_poly_claim);
                            trusted_advice_poly_binded = x.clone();
                            UniPoly::from_coeff(x.coeffs.iter().map(|coeff| coeff.mul_pow_2(scaling_factor)).collect())
                        } else  {
                            let scaling_factor = max_num_rounds - round - 1;
                            tracing::info!("DARIVARI in round {} in final gap, previous claim: {} scaling factor: {}", round, trusted_advice_poly_claim, scaling_factor);
                            let scaled_input_claim = trusted_advice_poly_claim
                                .mul_pow_2(scaling_factor);
                            UniPoly::from_coeff(vec![scaled_input_claim])
                        };
                        tracing::info!("DARIVARI in round {} individual claim: {:?}", round, poly);

                        poly
                    } else {
                        // Standard logic for non-trusted-advice polynomials
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
            tracing::info!("DARIVARI r_j: {:?}", r_j);

            // Cache individual claims for this round
            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));

            if in_row_phase || in_col_phase {
                trusted_advice_poly_claim = trusted_advice_poly_binded.evaluate(&r_j);
            }

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
                let poly_name = sumcheck.debug_name();
                let num_rounds = sumcheck.num_rounds();
                
                // Check if this is a trusted advice polynomial
                if let Some((ta_rows, _ta_columns)) = sumcheck.trusted_advice_dimensions() {
                    // For trusted advice, binding happens in two separate phases with hardcoded ranges:
                    // Phase 1 (rows): Bind during rounds [6, 10)
                    // Phase 2 (columns): Bind during rounds [18, 22)
                    let row_bind_start = 6;
                    let row_bind_end = 10;
                    let col_bind_start = 18;
                    let col_bind_end = 22;
                    tracing::info!("row_bind_start: {}, row_bind_end: {}, col_bind_start: {}, col_bind_end: {}, log_main_rows: {}, max_num_rounds: {}", row_bind_start, row_bind_end, col_bind_start, col_bind_end, log_main_rows, max_num_rounds);

                    let should_bind_rows = round >= row_bind_start && round < row_bind_end;
                    let should_bind_cols = round >= col_bind_start && round < col_bind_end;
                    tracing::info!("should_bind_rows: {}, should_bind_cols: {}", should_bind_rows, should_bind_cols);


                    if should_bind_rows {
                        // Binding row variable
                        let ta_round = round - row_bind_start;
                        tracing::info!(
                            "DARIVARI BIND [TA rows]: poly={}, global_round={}, ta_round={}, num_rounds={}, r_j={:?}",
                            poly_name, round, ta_round, num_rounds, r_j
                        );
                        sumcheck.ingest_challenge(r_j, ta_round);
                    } else if should_bind_cols {
                        // Binding column variable
                        let ta_round = ta_rows + (round - col_bind_start);
                        tracing::info!(
                            "DARIVARI BIND [TA cols]: poly={}, global_round={}, ta_round={}, num_rounds={}, r_j={:?}",
                            poly_name, round, ta_round, num_rounds, r_j
                        );
                        sumcheck.ingest_challenge(r_j, ta_round);
                    } else {
                        tracing::info!(
                            "SKIP [TA gap]: poly={}, global_round={}, num_rounds={} (row_bind=[{},{}), col_bind=[{},{}])",
                            poly_name, round, num_rounds, row_bind_start, row_bind_end, col_bind_start, col_bind_end
                        );
                    }
                } else {
                    // Standard binding logic for non-trusted-advice polynomials
                    // If a sumcheck instance has fewer than `max_num_rounds`,
                    // we wait until there are <= `sumcheck.num_rounds()` left
                    // before binding its variables.
                    if remaining_rounds <= num_rounds {
                        let offset = max_num_rounds - num_rounds;
                        let local_round = round - offset;
                        tracing::info!(
                            "BIND [standard]: poly={}, global_round={}, local_round={}, num_rounds={}, r_j={:?}",
                            poly_name, round, local_round, num_rounds, r_j
                        );
                        sumcheck.ingest_challenge(r_j, local_round);
                    } else {
                        tracing::info!(
                            "SKIP [not started]: poly={}, global_round={}, num_rounds={}, remaining_rounds={}",
                            poly_name, round, num_rounds, remaining_rounds
                        );
                    }
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
            // Check if this is a trusted advice polynomial
            let r_slice: Vec<F::Challenge> = if let Some((ta_rows, _ta_columns)) = sumcheck.trusted_advice_dimensions() {
                // For trusted advice, we use hardcoded ranges [6, 10) and [18, 22)
                let row_bind_start = 6;
                let row_bind_end = 10;
                let col_bind_start = 18;
                let col_bind_end = 22;
                
                // Construct r_slice from the row rounds and column rounds
                let mut r_vec = Vec::with_capacity(ta_rows + (col_bind_end - col_bind_start));
                r_vec.extend_from_slice(&r_sumcheck[row_bind_start..row_bind_end]);
                r_vec.extend_from_slice(&r_sumcheck[col_bind_start..col_bind_end]);
                
                tracing::info!(
                    "PROVER cache_openings [TA]: constructing r_slice from [{}..{}) and [{}..{}), len={}",
                    row_bind_start, row_bind_end, col_bind_start, col_bind_end, r_vec.len()
                );
                r_vec
            } else {
                // If a sumcheck instance has fewer than `max_num_rounds`,
                // we wait until there are <= `sumcheck.num_rounds()` left
                // before binding its variables.
                // So, the sumcheck *actually* uses just the last `sumcheck.num_rounds()`
                // values of `r_sumcheck`.
                r_sumcheck[max_num_rounds - sumcheck.num_rounds()..].to_vec()
            };

            // Cache polynomial opening claims, to be proven using either an
            // opening proof or sumcheck (in the case of virtual polynomials).
            sumcheck.cache_openings(opening_accumulator, transcript, &r_slice);
        }
        tracing::info!("THIIISSSSSSS in the prover compressed_polys: {:?}", compressed_polys);

        (SumcheckInstanceProof::new(compressed_polys), r_sumcheck)
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
                tracing::info!("THIIISSSSSSS in the verifier input_claim: {:?}", input_claim);
                transcript.append_scalar(&input_claim);
                input_claim.mul_pow_2(max_num_rounds - num_rounds) * coeff
            })
            .sum();

        tracing::info!("THIIISSSSSSS in the verifier claim: {:?}", claim);

        let (output_claim, r_sumcheck) =
            proof.verify(claim, max_num_rounds, max_degree, transcript)?;

        tracing::info!("=== SUMCHECK VERIFY: Computing expected_output_claim ===");
        tracing::info!("  max_num_rounds={}, r_sumcheck.len()={}", max_num_rounds, r_sumcheck.len());
        tracing::info!("  num_sumcheck_instances={}", sumcheck_instances.len());
        
        let expected_output_claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .enumerate()
            .map(|(idx, (sumcheck, coeff))| {
                tracing::info!("--- Instance {} ---", idx);
                tracing::info!("  num_rounds={}, trusted_advice_dims={:?}", 
                    sumcheck.num_rounds(), 
                    sumcheck.trusted_advice_dimensions()
                );
                
                // Check if this is a trusted advice polynomial
                let r_slice: Vec<F::Challenge> = 
                if let Some((ta_rows, _ta_columns)) = sumcheck.trusted_advice_dimensions() {
                    // For trusted advice, we use hardcoded ranges [6, 10) and [18, 22)
                    let row_bind_start = 6;
                    let row_bind_end = 10;
                    let col_bind_start = 18;
                    let col_bind_end = 22;
                    
                    // Construct r_slice from the row rounds and column rounds
                    let mut r_vec = Vec::with_capacity(ta_rows + (col_bind_end - col_bind_start));
                    r_vec.extend_from_slice(&r_sumcheck[row_bind_start..row_bind_end]);
                    r_vec.extend_from_slice(&r_sumcheck[col_bind_start..col_bind_end]);
                    
                    tracing::info!(
                        "VERIFY [TA]: constructing r_slice from [{}..{}) and [{}..{}), len={}",
                        row_bind_start, row_bind_end, col_bind_start, col_bind_end, r_vec.len()
                    );
                    tracing::info!("r_vec: {:?}", r_vec);
                    r_vec
                } else {
                    // If a sumcheck instance has fewer than `max_num_rounds`,
                    // we wait until there are <= `sumcheck.num_rounds()` left
                    // before binding its variables.
                    // So, the sumcheck *actually* uses just the last `sumcheck.num_rounds()`
                    // values of `r_sumcheck`.
                    r_sumcheck[max_num_rounds - sumcheck.num_rounds()..].to_vec()
                }
                ;

                tracing::info!("  r_slice.len()={}", r_slice.len());

                // Cache polynomial opening claims, to be proven using either an
                // opening proof or sumcheck (in the case of virtual polynomials).
                sumcheck.cache_openings(opening_accumulator, transcript, &r_slice);
                let claim = sumcheck.expected_output_claim(opening_accumulator, &r_slice);

                tracing::info!("  expected_output_claim={:?}, coeff={:?}", claim, coeff);
                tracing::info!("  contribution (claim * coeff)={:?}", claim * coeff);
                
                claim * coeff
            })
            .sum();

        tracing::info!("THIIISSSSSSS output_claim: {:?}", output_claim);
        tracing::info!("THIIISSSSSSS expected_output_claim: {:?}", expected_output_claim);

        if output_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(r_sumcheck)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct SumcheckInstanceProof<F: JoltField, ProofTranscript: Transcript> {
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> SumcheckInstanceProof<F, ProofTranscript> {
    pub fn new(
        compressed_polys: Vec<CompressedUniPoly<F>>,
    ) -> SumcheckInstanceProof<F, ProofTranscript> {
        SumcheckInstanceProof {
            compressed_polys,
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

            // append the prover's message to the transcript
            self.compressed_polys[i].append_to_transcript(transcript);

            //derive the verifier's challenge for the next round
            let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i using the hint
            tracing::info!("THIIISSSSSSS in the verifier compressed_polys[i]: {:?}, r_i: {:?}", self.compressed_polys[i], r_i);
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
