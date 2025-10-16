#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::field::JoltField;
use crate::field::MaybeAllocative;
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::transcripts::{AppendToTranscript, Transcript};
use crate::utils::errors::ProofVerifyError;
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;
#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;

use ark_serialize::*;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

/// Trait for a sumcheck instance that can be batched with other instances.
///
/// This trait defines the interface needed to participate in the `BatchedSumcheck` protocol,
/// which reduces verifier cost and proof size by batching multiple sumcheck protocols.
pub trait SumcheckInstance<F: JoltField, T: Transcript>: Send + Sync + MaybeAllocative {
    /// Returns the maximum degree of the sumcheck polynomial.
    fn degree(&self) -> usize;

    /// Returns the number of rounds/variables in this sumcheck instance.
    fn num_rounds(&self) -> usize;

    /// Returns the initial claim of this sumcheck instance, i.e.
    /// input_claim = \sum_{x \in \{0, 1}^N} P(x)
    fn input_claim(&self) -> F; // TODO(moodlezoup): maybe pass this an Option<Rc<RefCell<ProverOpeningAccumulator<F>>>>

    /// Computes the prover's message for a specific round of the sumcheck protocol.
    /// Returns the evaluations of the sumcheck polynomial at 0, 2, 3, ..., degree.
    /// The point evaluation at 1 can be interpolated using the previous round's claim.
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F>;

    /// Binds this sumcheck instance to the verifier's challenge from a specific round.
    /// This updates the internal state to prepare for the next round.
    fn bind(&mut self, r_j: F::Challenge, round: usize);

    /// Computes the expected output claim given the verifier's challenges.
    /// This is used to verify the final result of the sumcheck protocol.
    fn expected_output_claim(
        &self,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F;

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F>;

    /// Caches polynomial opening claims needed after the sumcheck protocol completes.
    /// These openings will later be proven using either an opening proof or another sumcheck.
    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    );

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    );

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder);
}

pub enum SingleSumcheck {}
impl SingleSumcheck {
    /// Proves a single sumcheck instance.
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        sumcheck_instance: &mut dyn SumcheckInstance<F, ProofTranscript>,
        opening_accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>) {
        let num_rounds = sumcheck_instance.num_rounds();
        let mut r_sumcheck: Vec<F::Challenge> = Vec::with_capacity(num_rounds);
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

        let mut previous_claim = sumcheck_instance.input_claim();
        for round in 0..num_rounds {
            let mut univariate_poly_evals =
                sumcheck_instance.compute_prover_message(round, previous_claim);
            univariate_poly_evals.insert(1, previous_claim - univariate_poly_evals[0]);
            let univariate_poly = UniPoly::from_evals(&univariate_poly_evals);

            // append the prover's message to the transcript
            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

            // Cache claim for this round
            previous_claim = univariate_poly.evaluate(&r_j);

            sumcheck_instance.bind(r_j, round);
        }

        if let Some(opening_accumulator) = opening_accumulator {
            // Cache polynomial opening claims, to be proven using either an
            // opening proof or sumcheck (in the case of virtual polynomials).
            sumcheck_instance.cache_openings_prover(
                opening_accumulator,
                transcript,
                sumcheck_instance.normalize_opening_point(&r_sumcheck),
            );
        }

        (SumcheckInstanceProof::new(compressed_polys), r_sumcheck)
    }

    /// Verifies a single sumcheck instance.
    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        sumcheck_instance: &dyn SumcheckInstance<F, ProofTranscript>,
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        let (output_claim, r) = proof.verify(
            sumcheck_instance.input_claim(),
            sumcheck_instance.num_rounds(),
            sumcheck_instance.degree(),
            transcript,
        )?;

        let expected = sumcheck_instance.expected_output_claim(opening_accumulator.clone(), &r);
        if output_claim != expected {
            println!(
                "Sumcheck verify mismatch: output_claim={}, expected_output_claim={}, rounds={}, degree_bound={}",
                output_claim,
                expected,
                sumcheck_instance.num_rounds(),
                sumcheck_instance.degree()
            );
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        sumcheck_instance.cache_openings_verifier(
            opening_accumulator.unwrap(),
            transcript,
            sumcheck_instance.normalize_opening_point(&r),
        );

        Ok(r)
    }
}

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstance<F, ProofTranscript>>,
        opening_accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F>>>>,
        transcript: &mut ProofTranscript,
    ) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F::Challenge>) {
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
                    .mul_pow_2(max_num_rounds - num_rounds)
            })
            .collect();

        #[cfg(test)]
        let mut batched_claim: F = individual_claims
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(claim, coeff)| *claim * coeff)
            .sum();

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
                            .input_claim()
                            .mul_pow_2(remaining_rounds - num_rounds - 1);
                        // Constant polynomial
                        UniPoly::from_coeff(vec![scaled_input_claim])
                    } else {
                        let offset = max_num_rounds - sumcheck.num_rounds();
                        let mut univariate_poly_evals =
                            sumcheck.compute_prover_message(round - offset, *previous_claim);
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
                    sumcheck.bind(r_j, round - offset);
                }
            }

            compressed_polys.push(compressed_poly);
        }

        if let Some(opening_accumulator) = opening_accumulator {
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
                sumcheck.cache_openings_prover(
                    opening_accumulator.clone(),
                    transcript,
                    sumcheck.normalize_opening_point(r_slice),
                );
            }
        }

        (SumcheckInstanceProof::new(compressed_polys), r_sumcheck)
    }

    pub fn verify<F: JoltField, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F, ProofTranscript>,
        sumcheck_instances: Vec<&dyn SumcheckInstance<F, ProofTranscript>>,
        opening_accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
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
                sumcheck
                    .input_claim()
                    .mul_pow_2(max_num_rounds - num_rounds)
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

                if let Some(opening_accumulator) = &opening_accumulator {
                    // Cache polynomial opening claims, to be proven using either an
                    // opening proof or sumcheck (in the case of virtual polynomials).
                    sumcheck.cache_openings_verifier(
                        opening_accumulator.clone(),
                        transcript,
                        sumcheck.normalize_opening_point(r_slice),
                    );
                }
                let claim = sumcheck.expected_output_claim(opening_accumulator.clone(), r_slice);

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
            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
        }

        Ok((e, r))
    }
}

/// Trait for a single-round instance of univariate skip
/// We make a number of assumptions for the usage of this trait currently:
/// 1. There is only one univariate skip round, which happens at the beginning of a sumcheck stage
/// 2. We do not bind anything after this round. Instead during the remaining sumcheck, we
///    will stream from the trace again to initialize.
/// 3. We assume that the domain is symmetric around zero, and the prover sends the entire
///    (univariate) polynomial for this round
pub trait UniSkipFirstRoundInstance<F: JoltField, T: Transcript>:
    Send + Sync + MaybeAllocative
{
    /// The degree of the sum-check
    const DEGREE_BOUND: usize;

    /// The domain size of the sum-check. Canonically instantiated to the domain
    /// [-floor(DOMAIN_SIZE/2), ceil(DOMAIN_SIZE)/2]
    const DOMAIN_SIZE: usize;

    /// Returns the initial claim of this univariate skip round, i.e.
    /// input_claim = \sum_{-floor(S/2) <= z <= ceil(S/2)} \sum_{x \in \{0, 1}^n} P(z, x)
    /// where S = DOMAIN_SIZE
    fn input_claim(&self) -> F;

    /// Computes the full univariate polynomial to be sent in the uni-skip round.
    /// Returns a degree-bounded `UniPoly` with exactly `DEGREE_BOUND + 1` coefficients.
    fn compute_poly(&mut self) -> UniPoly<F>;

    // TODO: add flamegraph support
    // #[cfg(feature = "allocative")]
    // fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder);
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
            .check_sum_evals_and_set_new_claim::<N, FIRST_ROUND_POLY_NUM_COEFFS>(&F::zero(), &r0);
        if !ok {
            return Err(ProofVerifyError::UniSkipVerificationError);
        }

        Ok((r0, next_claim))
    }
}

/// Prove-only helper for a uni-skip first round instance.
/// Produces the proof object, the uni-skip challenge r0, and the next claim s1(r0).
pub fn prove_uniskip_round<
    F: JoltField,
    ProofTranscript: Transcript,
    I: UniSkipFirstRoundInstance<F, ProofTranscript>,
>(
    instance: &mut I,
    transcript: &mut ProofTranscript,
) -> (UniSkipFirstRoundProof<F, ProofTranscript>, F::Challenge, F) {
    let uni_poly = instance.compute_poly();
    // Append full polynomial and derive r0
    uni_poly.append_to_transcript(transcript);
    let r0: F::Challenge = transcript.challenge_scalar_optimized::<F>();
    // Evaluate next claim at r0
    let next_claim = uni_poly.evaluate::<F::Challenge>(&r0);
    (UniSkipFirstRoundProof::new(uni_poly), r0, next_claim)
}
