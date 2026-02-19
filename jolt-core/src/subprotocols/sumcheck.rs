#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::curve::JoltCurve;
use crate::field::JoltField;
#[cfg(feature = "zk")]
use crate::poly::commitment::pedersen::PedersenGenerators;
#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
use crate::poly::opening_proof::{OpeningAccumulator, ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;

use ark_serialize::*;
#[cfg(feature = "zk")]
use rand_core::CryptoRngCore;
use std::marker::PhantomData;

pub use crate::subprotocols::univariate_skip::UniSkipFirstRoundProof;

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    /// Returns (proof, challenges, initial_batched_claim)
    /// For non-ZK mode - returns ClearSumcheckProof with polynomial coefficients visible.
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (ClearSumcheckProof<F, ProofTranscript>, Vec<F::Challenge>, F) {
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        sumcheck_instances.iter().for_each(|sumcheck| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            transcript.append_scalar(b"sumcheck_claim", &input_claim);
        });
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
        let two_inv = F::from_u64(2).inverse().unwrap();

        for round in 0..max_num_rounds {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = format!("Sumcheck round {round}");
                print_current_memory_usage(label.as_str());
            }

            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter_mut()
                .zip(individual_claims.iter())
                .map(|(sumcheck, previous_claim)| {
                    let num_rounds = sumcheck.num_rounds();
                    let offset = sumcheck.round_offset(max_num_rounds);
                    let active = round >= offset && round < offset + num_rounds;
                    if active {
                        sumcheck.compute_message(round - offset, *previous_claim)
                    } else {
                        // Variable is "dummy" for this instance: polynomial is independent of it,
                        // so the round univariate is constant with H(0)=H(1)=previous_claim/2.
                        UniPoly::from_coeff(vec![*previous_claim * two_inv])
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
            transcript.append_scalars(b"sumcheck_poly", &compressed_poly.coeffs_except_linear_term);
            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

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
                let num_rounds = sumcheck.num_rounds();
                let offset = sumcheck.round_offset(max_num_rounds);
                let active = round >= offset && round < offset + num_rounds;
                if active {
                    sumcheck.ingest_challenge(r_j, round - offset);
                }
            }

            compressed_polys.push(compressed_poly);
        }

        // Allow each sumcheck instance to perform any end-of-protocol work (e.g. flushing
        // delayed bindings) after the final challenge has been ingested and before we cache
        // openings.
        for sumcheck in sumcheck_instances.iter_mut() {
            sumcheck.finalize();
        }

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        for sumcheck in sumcheck_instances.iter() {
            // Instance-local slice can start at a custom global offset.
            let offset = sumcheck.round_offset(max_num_rounds);
            let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];

            // Cache polynomial opening claims, to be proven using either an
            // opening proof or sumcheck (in the case of virtual polynomials).
            sumcheck.cache_openings(opening_accumulator, r_slice);
        }

        opening_accumulator.flush_to_transcript(transcript);

        (
            ClearSumcheckProof::new(compressed_polys),
            r_sumcheck,
            initial_batched_claim,
        )
    }

    /// Prove a batched sumcheck with Pedersen commitments (ZK mode).
    ///
    /// Instead of appending raw polynomial coefficients to the transcript,
    /// this appends Pedersen commitments. The proof contains only commitments -
    /// coefficients and blindings are stored in the accumulator for BlindFold.
    ///
    /// # Security
    /// The Pedersen commitments are verified by BlindFold's split-committed R1CS.
    /// BlindFold proves that the committed coefficients satisfy the sumcheck equations.
    ///
    /// Returns (proof, challenges, initial_batched_claim)
    #[cfg(feature = "zk")]
    pub fn prove_zk<
        F: JoltField,
        C: JoltCurve<F = F>,
        ProofTranscript: Transcript,
        R: CryptoRngCore,
    >(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        blindfold_accumulator: &mut crate::subprotocols::blindfold::BlindFoldAccumulator<F, C>,
        transcript: &mut ProofTranscript,
        pedersen_gens: &PedersenGenerators<C>,
        rng: &mut R,
    ) -> (
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
        F,
    ) {
        use crate::subprotocols::blindfold::ZkStageData;

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        // In ZK mode, don't absorb cleartext claims — polynomial commitments provide binding.
        // Batching coefficients are still unpredictable (from transcript state after commitments).
        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        let mut individual_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim(opening_accumulator);
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
        let mut round_commitments_g1: Vec<C::G1> = Vec::with_capacity(max_num_rounds);
        let mut poly_coeffs: Vec<Vec<F>> = Vec::with_capacity(max_num_rounds);
        let mut blinding_factors: Vec<F> = Vec::with_capacity(max_num_rounds);
        let mut poly_degrees: Vec<usize> = Vec::with_capacity(max_num_rounds);

        for round in 0..max_num_rounds {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = format!("Sumcheck round {round}");
                print_current_memory_usage(label.as_str());
            }

            let univariate_polys: Vec<UniPoly<F>> = sumcheck_instances
                .iter_mut()
                .zip(individual_claims.iter())
                .map(|(sumcheck, previous_claim)| {
                    let num_rounds = sumcheck.num_rounds();
                    let offset = sumcheck.round_offset(max_num_rounds);
                    let active = round >= offset && round < offset + num_rounds;
                    if active {
                        sumcheck.compute_message(round - offset, *previous_claim)
                    } else {
                        // Dummy round: polynomial is constant with H(0)=H(1)=previous_claim/2.
                        let two_inv = F::from_u64(2).inverse().unwrap();
                        UniPoly::from_coeff(vec![*previous_claim * two_inv])
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

            let blinding = F::random(rng);
            let commitment = pedersen_gens.commit(&batched_univariate_poly.coeffs, &blinding);

            transcript.append_commitment(b"sumcheck_commitment", &commitment);

            let r_j = transcript.challenge_scalar_optimized::<F>();
            r_sumcheck.push(r_j);

            individual_claims
                .iter_mut()
                .zip(univariate_polys.into_iter())
                .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));

            for sumcheck in sumcheck_instances.iter_mut() {
                let num_rounds = sumcheck.num_rounds();
                let offset = sumcheck.round_offset(max_num_rounds);
                let active = round >= offset && round < offset + num_rounds;
                if active {
                    sumcheck.ingest_challenge(r_j, round - offset);
                }
            }

            round_commitments_g1.push(commitment);
            poly_degrees.push(batched_univariate_poly.coeffs.len() - 1);
            poly_coeffs.push(batched_univariate_poly.coeffs.clone());
            blinding_factors.push(blinding);
        }

        // Allow each sumcheck instance to perform any end-of-protocol work (e.g. flushing
        // delayed bindings) after the final challenge has been ingested and before we cache
        // openings.
        for sumcheck in sumcheck_instances.iter_mut() {
            sumcheck.finalize();
        }

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        for sumcheck in sumcheck_instances.iter() {
            let num_rounds = sumcheck.num_rounds();
            let offset = sumcheck.round_offset(max_num_rounds);
            let r_slice = &r_sumcheck[offset..offset + num_rounds];
            sumcheck.cache_openings(opening_accumulator, r_slice);
        }

        let output_claim_values = opening_accumulator.take_pending_claims();
        let output_claim_ids = opening_accumulator.take_pending_claim_ids();
        let oc_committed: Vec<_> = pedersen_gens.commit_chunked(&output_claim_values, rng);
        let output_claims: Vec<(OpeningId, F)> = output_claim_ids
            .into_iter()
            .zip(output_claim_values)
            .collect();
        let (output_claims_commitments, output_claims_blindings): (Vec<_>, Vec<_>) =
            oc_committed.into_iter().unzip();
        transcript.append_commitments(b"output_claims_coms", &output_claims_commitments);

        let output_constraints: Vec<_> = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.get_params().output_claim_constraint())
            .collect();

        let constraint_challenge_values: Vec<Vec<F>> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                let num_rounds = sumcheck.num_rounds();
                let offset = sumcheck.round_offset(max_num_rounds);
                let r_slice = &r_sumcheck[offset..offset + num_rounds];
                sumcheck
                    .get_params()
                    .output_constraint_challenge_values(r_slice)
            })
            .collect();

        let input_constraints: Vec<_> = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.get_params().input_claim_constraint())
            .collect();

        let input_constraint_challenge_values: Vec<Vec<F>> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                sumcheck
                    .get_params()
                    .input_constraint_challenge_values(opening_accumulator)
            })
            .collect();

        let input_claim_scaling_exponents: Vec<usize> = sumcheck_instances
            .iter()
            .map(|sumcheck| max_num_rounds - sumcheck.num_rounds())
            .collect();

        blindfold_accumulator.push_stage_data(ZkStageData {
            initial_claim: initial_batched_claim,
            round_commitments: round_commitments_g1.clone(),
            poly_coeffs,
            blinding_factors,
            challenges: r_sumcheck.clone(),
            batching_coefficients: batching_coeffs.to_vec(),
            output_constraints,
            constraint_challenge_values,
            input_constraints,
            input_constraint_challenge_values,
            input_claim_scaling_exponents,
            output_claims,
            output_claims_blindings,
            output_claims_commitments: output_claims_commitments.clone(),
        });

        (
            SumcheckInstanceProof::new_zk(
                round_commitments_g1,
                poly_degrees,
                output_claims_commitments,
            ),
            r_sumcheck,
            initial_batched_claim,
        )
    }

    pub fn verify<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript, A: OpeningAccumulator<F>>(
        proof: &SumcheckInstanceProof<F, C, ProofTranscript>,
        sumcheck_instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript, A>>,
        opening_accumulator: &mut A,
        transcript: &mut ProofTranscript,
    ) -> Result<(Vec<F>, Vec<F::Challenge>), ProofVerifyError> {
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

        let is_zk = matches!(proof, SumcheckInstanceProof::Zk(_));
        if !is_zk {
            sumcheck_instances.iter().for_each(|sumcheck| {
                let input_claim = sumcheck.input_claim(opening_accumulator);
                transcript.append_scalar(b"sumcheck_claim", &input_claim);
            });
        }
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
                input_claim.mul_pow_2(max_num_rounds - num_rounds) * coeff
            })
            .sum();

        let (output_claim, r_sumcheck) =
            proof.verify(claim, max_num_rounds, max_degree, transcript)?;

        let expected_output_claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let offset = sumcheck.round_offset(max_num_rounds);
                let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];

                // Cache polynomial opening claims, to be proven using either an
                // opening proof or sumcheck (in the case of virtual polynomials).
                sumcheck.cache_openings(opening_accumulator, r_slice);
                let claim = sumcheck.expected_output_claim(opening_accumulator, r_slice);

                claim * coeff
            })
            .sum();

        if !is_zk {
            opening_accumulator.flush_to_transcript(transcript);
        } else if let SumcheckInstanceProof::Zk(zk_proof) = proof {
            transcript
                .append_commitments(b"output_claims_coms", &zk_proof.output_claims_commitments);
            opening_accumulator.take_pending_claims();
        }

        // In ZK mode, skip output claim verification — BlindFold proves this
        if !is_zk && output_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok((batching_coeffs, r_sumcheck))
    }

    /// Verify a standard (non-ZK) sumcheck proof without requiring a curve type.
    /// Used by opening proof reduction which doesn't need ZK mode.
    pub fn verify_standard<F: JoltField, ProofTranscript: Transcript, A: OpeningAccumulator<F>>(
        proof: &ClearSumcheckProof<F, ProofTranscript>,
        sumcheck_instances: Vec<&dyn SumcheckInstanceVerifier<F, ProofTranscript, A>>,
        opening_accumulator: &mut A,
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

        // Append input claims to transcript BEFORE deriving batching coefficients
        // (must match ordering in BatchedSumcheck::prove)
        sumcheck_instances.iter().for_each(|sumcheck| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            transcript.append_scalar(b"sumcheck_claim", &input_claim);
        });

        // Debug output for intermediate value comparison testing.
        // When enabled, prints transcript state and derived challenges to verify
        // the transpiled gnark circuit computes identical Fiat-Shamir values.
        #[cfg(feature = "debug-expected-output")]
        {
            transcript.debug_state("before_batching_coeffs");
        }
        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

        #[cfg(feature = "debug-expected-output")]
        {
            use ark_serialize::CanonicalSerialize;
            fn to_decimal<T: CanonicalSerialize>(val: &T) -> String {
                let mut bytes = Vec::new();
                val.serialize_compressed(&mut bytes).unwrap();
                num_bigint::BigUint::from_bytes_le(&bytes).to_string()
            }
            eprintln!("=== BATCHING COEFFS DEBUG ===");
            eprintln!("num_sumcheck_instances = {}", sumcheck_instances.len());
            for (i, coeff) in batching_coeffs.iter().enumerate() {
                eprintln!("batching_coeff[{}] = {}", i, to_decimal(coeff));
            }
            transcript.debug_state("after_batching_coeffs");
            eprintln!("=== END BATCHING DEBUG ===");
        }

        let claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let num_rounds = sumcheck.num_rounds();
                let input_claim = sumcheck.input_claim(opening_accumulator);
                input_claim.mul_pow_2(max_num_rounds - num_rounds) * coeff
            })
            .sum();

        let (output_claim, r_sumcheck) =
            proof.verify(claim, max_num_rounds, max_degree, transcript)?;

        let expected_output_claim = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];
                sumcheck.cache_openings(opening_accumulator, r_slice);
                let claim = sumcheck.expected_output_claim(opening_accumulator, r_slice);
                claim * coeff
            })
            .sum();

        opening_accumulator.flush_to_transcript(transcript);

        // Debug output for concrete Fr types
        #[cfg(feature = "debug-expected-output")]
        {
            use ark_serialize::CanonicalSerialize;
            fn to_decimal<T: CanonicalSerialize>(val: &T) -> String {
                let mut bytes = Vec::new();
                val.serialize_compressed(&mut bytes).unwrap();
                num_bigint::BigUint::from_bytes_le(&bytes).to_string()
            }
            eprintln!("=== SUMCHECK VERIFY DEBUG ===");
            eprintln!(
                "output_claim (from sumcheck) = {}",
                to_decimal(&output_claim)
            );
            eprintln!(
                "expected_output_claim (batched) = {}",
                to_decimal(&expected_output_claim)
            );
            eprintln!(
                "difference = {}",
                to_decimal(&(output_claim - expected_output_claim))
            );
            eprintln!("=== END SUMCHECK DEBUG ===");
        }

        if output_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(r_sumcheck)
    }
}

/// Clear (non-ZK) sumcheck proof - coefficients visible to verifier.
/// Used in non-ZK mode where the verifier evaluates polynomials directly.
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ClearSumcheckProof<F: JoltField, ProofTranscript: Transcript> {
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> ClearSumcheckProof<F, ProofTranscript> {
    pub fn new(compressed_polys: Vec<CompressedUniPoly<F>>) -> Self {
        Self {
            compressed_polys,
            _marker: PhantomData,
        }
    }

    /// Verify this standard sumcheck proof by evaluating polynomials.
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F::Challenge>), ProofVerifyError> {
        let mut e = claim;
        let mut r: Vec<F::Challenge> = Vec::new();

        if self.compressed_polys.len() != num_rounds {
            return Err(ProofVerifyError::InvalidInputLength(
                num_rounds,
                self.compressed_polys.len(),
            ));
        }
        for i in 0..self.compressed_polys.len() {
            if self.compressed_polys[i].degree() > degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    self.compressed_polys[i].degree(),
                ));
            }

            // append the prover's message to the transcript
            transcript.append_scalars(
                b"sumcheck_poly",
                &self.compressed_polys[i].coeffs_except_linear_term,
            );

            //derive the verifier's challenge for the next round
            let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            r.push(r_i);
            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
        }

        #[cfg(feature = "debug-expected-output")]
        {
            use ark_serialize::CanonicalSerialize;
            fn to_decimal<T: CanonicalSerialize>(val: &T) -> String {
                let mut bytes = Vec::new();
                val.serialize_compressed(&mut bytes).unwrap();
                num_bigint::BigUint::from_bytes_le(&bytes).to_string()
            }
            eprintln!("=== SUMCHECK CHALLENGES DEBUG ===");
            eprintln!("num_rounds = {}", num_rounds);
            for (i, r_i) in r.iter().enumerate() {
                eprintln!("sumcheck_challenge[{}] = {}", i, to_decimal(r_i));
            }
            eprintln!("=== END CHALLENGES DEBUG ===");
        }

        Ok((e, r))
    }
}

/// ZK sumcheck proof - only commitments visible, coefficients hidden in BlindFold.
/// The verifier appends commitments to transcript and derives challenges,
/// but polynomial evaluation is verified by BlindFold's R1CS constraints.
#[derive(Debug, Clone)]
pub struct ZkSumcheckProof<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript> {
    /// Pedersen commitments to round polynomials (G1 curve elements)
    pub round_commitments: Vec<C::G1>,
    /// Polynomial degrees for each round (public info needed for R1CS construction)
    pub poly_degrees: Vec<usize>,
    /// Pedersen commitments to output claims, chunked to fit generator count
    pub output_claims_commitments: Vec<C::G1>,
    _marker: PhantomData<(F, ProofTranscript)>,
}

impl<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript> CanonicalSerialize
    for ZkSumcheckProof<F, C, ProofTranscript>
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.round_commitments
            .serialize_with_mode(&mut writer, compress)?;
        self.poly_degrees
            .serialize_with_mode(&mut writer, compress)?;
        self.output_claims_commitments
            .serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.round_commitments.serialized_size(compress)
            + self.poly_degrees.serialized_size(compress)
            + self.output_claims_commitments.serialized_size(compress)
    }
}

impl<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript> ark_serialize::Valid
    for ZkSumcheckProof<F, C, ProofTranscript>
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.round_commitments.check()?;
        self.poly_degrees.check()?;
        self.output_claims_commitments.check()
    }
}

impl<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript> CanonicalDeserialize
    for ZkSumcheckProof<F, C, ProofTranscript>
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let round_commitments =
            Vec::<C::G1>::deserialize_with_mode(&mut reader, compress, validate)?;
        let poly_degrees = Vec::<usize>::deserialize_with_mode(&mut reader, compress, validate)?;
        let output_claims_commitments =
            Vec::<C::G1>::deserialize_with_mode(reader, compress, validate)?;
        Ok(Self {
            round_commitments,
            poly_degrees,
            output_claims_commitments,
            _marker: PhantomData,
        })
    }
}

impl<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript>
    ZkSumcheckProof<F, C, ProofTranscript>
{
    pub fn new(
        round_commitments: Vec<C::G1>,
        poly_degrees: Vec<usize>,
        output_claims_commitments: Vec<C::G1>,
    ) -> Self {
        Self {
            round_commitments,
            poly_degrees,
            output_claims_commitments,
            _marker: PhantomData,
        }
    }

    /// Verify ZK sumcheck by appending commitments to transcript and deriving challenges.
    /// Does NOT evaluate polynomials - that's handled by BlindFold verification.
    pub fn verify_transcript_only(
        &self,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        if self.round_commitments.len() != num_rounds {
            return Err(ProofVerifyError::InvalidInputLength(
                num_rounds,
                self.round_commitments.len(),
            ));
        }
        if self.poly_degrees.len() != num_rounds {
            return Err(ProofVerifyError::InvalidInputLength(
                num_rounds,
                self.poly_degrees.len(),
            ));
        }

        for &degree in &self.poly_degrees {
            if degree > degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(degree_bound, degree));
            }
        }

        let mut r: Vec<F::Challenge> = Vec::new();
        for commitment in &self.round_commitments {
            transcript.append_commitment(b"sumcheck_commitment", commitment);
            let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            r.push(r_i);
        }

        Ok(r)
    }
}

#[derive(Debug, Clone)]
pub enum SumcheckInstanceProof<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript> {
    /// Non-ZK: coefficients visible to verifier
    Clear(ClearSumcheckProof<F, ProofTranscript>),
    /// ZK: only commitments visible, coefficients hidden in BlindFold
    Zk(ZkSumcheckProof<F, C, ProofTranscript>),
}

impl<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript> CanonicalSerialize
    for SumcheckInstanceProof<F, C, ProofTranscript>
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        match self {
            Self::Clear(proof) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(writer, compress)
            }
            Self::Zk(proof) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                proof.serialize_with_mode(writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        1 + match self {
            Self::Clear(proof) => proof.serialized_size(compress),
            Self::Zk(proof) => proof.serialized_size(compress),
        }
    }
}

impl<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript> ark_serialize::Valid
    for SumcheckInstanceProof<F, C, ProofTranscript>
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        match self {
            Self::Clear(proof) => proof.check(),
            Self::Zk(proof) => proof.check(),
        }
    }
}

impl<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript> CanonicalDeserialize
    for SumcheckInstanceProof<F, C, ProofTranscript>
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let variant = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        match variant {
            0 => {
                let proof = ClearSumcheckProof::deserialize_with_mode(reader, compress, validate)?;
                Ok(Self::Clear(proof))
            }
            1 => {
                let proof = ZkSumcheckProof::deserialize_with_mode(reader, compress, validate)?;
                Ok(Self::Zk(proof))
            }
            _ => Err(ark_serialize::SerializationError::InvalidData),
        }
    }
}

impl<F: JoltField, C: JoltCurve<F = F>, ProofTranscript: Transcript>
    SumcheckInstanceProof<F, C, ProofTranscript>
{
    /// Create a standard (non-ZK) sumcheck proof.
    pub fn new_standard(compressed_polys: Vec<CompressedUniPoly<F>>) -> Self {
        Self::Clear(ClearSumcheckProof::new(compressed_polys))
    }

    /// Create a ZK sumcheck proof with only commitments and polynomial degrees.
    pub fn new_zk(
        round_commitments: Vec<C::G1>,
        poly_degrees: Vec<usize>,
        output_claims_commitments: Vec<C::G1>,
    ) -> Self {
        Self::Zk(ZkSumcheckProof::new(
            round_commitments,
            poly_degrees,
            output_claims_commitments,
        ))
    }

    /// Verify the sumcheck proof.
    /// For Standard: evaluates polynomials and returns (final_claim, challenges).
    /// For Zk: only derives challenges (BlindFold handles evaluation verification).
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(F, Vec<F::Challenge>), ProofVerifyError> {
        match self {
            Self::Clear(proof) => proof.verify(claim, num_rounds, degree_bound, transcript),
            Self::Zk(proof) => {
                if !cfg!(feature = "zk") {
                    return Err(ProofVerifyError::ZkFeatureRequired);
                }
                let challenges =
                    proof.verify_transcript_only(num_rounds, degree_bound, transcript)?;
                Ok((F::zero(), challenges))
            }
        }
    }

    pub fn is_zk(&self) -> bool {
        matches!(self, Self::Zk(_))
    }

    pub fn num_rounds(&self) -> usize {
        match self {
            Self::Clear(proof) => proof.compressed_polys.len(),
            Self::Zk(proof) => proof.round_commitments.len(),
        }
    }
}
