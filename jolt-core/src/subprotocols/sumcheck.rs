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

// Re-export UniSkipFirstRoundProof from univariate_skip to avoid type duplication
pub use crate::subprotocols::univariate_skip::UniSkipFirstRoundProof;

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    /// Returns (proof, challenges, initial_batched_claim)
    /// For non-ZK mode - returns StandardSumcheckProof with polynomial coefficients visible.
    pub fn prove<F: JoltField, ProofTranscript: Transcript>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> (
        StandardSumcheckProof<F, ProofTranscript>,
        Vec<F::Challenge>,
        F,
    ) {
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        // Append input claims to transcript
        sumcheck_instances.iter().for_each(|sumcheck| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            transcript.append_scalar(&input_claim);
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
            sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
        }

        (
            StandardSumcheckProof::new(compressed_polys),
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
    pub fn prove_zk<F: JoltField, C: JoltCurve, ProofTranscript: Transcript, R: CryptoRngCore>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F, ProofTranscript>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
        pedersen_gens: &PedersenGenerators<C>,
        rng: &mut R,
    ) -> (
        SumcheckInstanceProof<F, C, ProofTranscript>,
        Vec<F::Challenge>,
        F,
    ) {
        use crate::poly::opening_proof::ZkStageData;

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        // Append input claims to transcript BEFORE deriving batching coefficients
        let input_claims: Vec<F> = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.input_claim(opening_accumulator))
            .collect();
        for claim in &input_claims {
            transcript.append_scalar(claim);
        }

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
        let mut round_commitments_bytes: Vec<Vec<u8>> = Vec::with_capacity(max_num_rounds);
        let mut poly_coeffs: Vec<Vec<F>> = Vec::with_capacity(max_num_rounds);
        let mut blinding_factors: Vec<F> = Vec::with_capacity(max_num_rounds);
        let mut poly_degrees: Vec<usize> = Vec::with_capacity(max_num_rounds);

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

            // Generate blinding and compute Pedersen commitment to full coefficients
            let blinding = F::random(rng);
            let commitment = pedersen_gens.commit(&batched_univariate_poly.coeffs, &blinding);

            // Serialize commitment for transcript
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

            // Store data for BlindFold
            round_commitments_g1.push(commitment);
            round_commitments_bytes.push(commitment_bytes);
            // Polynomial degree = number of coefficients - 1
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
            let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];
            sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
        }

        // Collect output constraints and challenge values from each sumcheck instance
        let output_constraints: Vec<_> = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.get_params().output_claim_constraint())
            .collect();

        let constraint_challenge_values: Vec<Vec<F>> = sumcheck_instances
            .iter()
            .map(|sumcheck| {
                let r_slice = &r_sumcheck[max_num_rounds - sumcheck.num_rounds()..];
                sumcheck
                    .get_params()
                    .output_constraint_challenge_values(r_slice)
            })
            .collect();

        // Collect input constraints and challenge values from each sumcheck instance
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

        // Store ZK data in accumulator for BlindFold to retrieve later
        let batching_coefficients_f: Vec<F> = batching_coeffs.to_vec();
        opening_accumulator.push_zk_stage_data(ZkStageData {
            initial_claim: initial_batched_claim,
            round_commitments: round_commitments_bytes,
            poly_coeffs,
            blinding_factors,
            challenges: r_sumcheck.clone(),
            batching_coefficients: batching_coefficients_f,
            expected_evaluations: Vec::new(),
            output_constraints,
            constraint_challenge_values,
            input_constraints,
            input_constraint_challenge_values,
        });

        (
            SumcheckInstanceProof::new_zk(round_commitments_g1, poly_degrees),
            r_sumcheck,
            initial_batched_claim,
        )
    }

    pub fn verify<F: JoltField, C: JoltCurve, ProofTranscript: Transcript>(
        proof: &SumcheckInstanceProof<F, C, ProofTranscript>,
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

        // Append input claims to transcript
        sumcheck_instances.iter().for_each(|sumcheck| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            transcript.append_scalar(&input_claim);
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

        // In ZK mode (Zk variant), output_claim is F::zero() since polynomial
        // evaluation is verified by BlindFold, not by the verifier directly.
        let is_zk_mode = matches!(proof, SumcheckInstanceProof::Zk(_));

        let expected_output_claim: F = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let offset = sumcheck.round_offset(max_num_rounds);
                let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];

                // Cache polynomial opening claims, to be proven using either an
                // opening proof or sumcheck (in the case of virtual polynomials).
                sumcheck.cache_openings(opening_accumulator, transcript, r_slice);
                let claim = sumcheck.expected_output_claim(opening_accumulator, r_slice);

                claim * coeff
            })
            .sum();

        // In ZK mode, skip output claim verification - BlindFold proves this
        if !is_zk_mode && output_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(r_sumcheck)
    }

    /// Verify a standard (non-ZK) sumcheck proof without requiring a curve type.
    /// Used by opening proof reduction which doesn't need ZK mode.
    pub fn verify_standard<F: JoltField, ProofTranscript: Transcript>(
        proof: &StandardSumcheckProof<F, ProofTranscript>,
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

        // Append input claims to transcript BEFORE deriving batching coefficients
        // (must match ordering in BatchedSumcheck::prove)
        sumcheck_instances.iter().for_each(|sumcheck| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            transcript.append_scalar(&input_claim);
        });

        let batching_coeffs: Vec<F> = transcript.challenge_vector(sumcheck_instances.len());

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

/// Standard sumcheck proof - coefficients visible to verifier.
/// Used in non-ZK mode where the verifier evaluates polynomials directly.
#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct StandardSumcheckProof<F: JoltField, ProofTranscript: Transcript> {
    pub compressed_polys: Vec<CompressedUniPoly<F>>,
    _marker: PhantomData<ProofTranscript>,
}

impl<F: JoltField, ProofTranscript: Transcript> StandardSumcheckProof<F, ProofTranscript> {
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

        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            if self.compressed_polys[i].degree() > degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    self.compressed_polys[i].degree(),
                ));
            }

            self.compressed_polys[i].append_to_transcript(transcript);
            let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            r.push(r_i);
            e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
        }

        Ok((e, r))
    }
}

/// ZK sumcheck proof - only commitments visible, coefficients hidden in BlindFold.
/// The verifier appends commitments to transcript and derives challenges,
/// but polynomial evaluation is verified by BlindFold's R1CS constraints.
#[derive(Debug, Clone)]
pub struct ZkSumcheckProof<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> {
    /// Pedersen commitments to round polynomials (G1 curve elements)
    pub round_commitments: Vec<C::G1>,
    /// Polynomial degrees for each round (public info needed for R1CS construction)
    pub poly_degrees: Vec<usize>,
    _marker: PhantomData<(F, ProofTranscript)>,
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> CanonicalSerialize
    for ZkSumcheckProof<F, C, ProofTranscript>
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        self.round_commitments
            .serialize_with_mode(&mut writer, compress)?;
        self.poly_degrees.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: ark_serialize::Compress) -> usize {
        self.round_commitments.serialized_size(compress)
            + self.poly_degrees.serialized_size(compress)
    }
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> ark_serialize::Valid
    for ZkSumcheckProof<F, C, ProofTranscript>
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        self.round_commitments.check()?;
        self.poly_degrees.check()
    }
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> CanonicalDeserialize
    for ZkSumcheckProof<F, C, ProofTranscript>
{
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: ark_serialize::Compress,
        validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        let round_commitments =
            Vec::<C::G1>::deserialize_with_mode(&mut reader, compress, validate)?;
        let poly_degrees = Vec::<usize>::deserialize_with_mode(reader, compress, validate)?;
        Ok(Self {
            round_commitments,
            poly_degrees,
            _marker: PhantomData,
        })
    }
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript>
    ZkSumcheckProof<F, C, ProofTranscript>
{
    pub fn new(round_commitments: Vec<C::G1>, poly_degrees: Vec<usize>) -> Self {
        Self {
            round_commitments,
            poly_degrees,
            _marker: PhantomData,
        }
    }

    /// Verify ZK sumcheck by appending commitments to transcript and deriving challenges.
    /// Does NOT evaluate polynomials - that's handled by BlindFold verification.
    pub fn verify_transcript_only(
        &self,
        num_rounds: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F::Challenge>, ProofVerifyError> {
        if self.round_commitments.len() != num_rounds {
            return Err(ProofVerifyError::InvalidInputLength(
                num_rounds,
                self.round_commitments.len(),
            ));
        }

        let mut r: Vec<F::Challenge> = Vec::new();
        for commitment in &self.round_commitments {
            // Serialize commitment for transcript
            let mut commitment_bytes = Vec::new();
            commitment
                .serialize_compressed(&mut commitment_bytes)
                .expect("Serialization should not fail");

            transcript.append_message(b"UniPolyCommitment");
            transcript.append_bytes(&commitment_bytes);

            let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
            r.push(r_i);
        }

        Ok(r)
    }
}

/// Sumcheck proof enum - replaces old SumcheckInstanceProof throughout codebase.
#[derive(Debug, Clone)]
pub enum SumcheckInstanceProof<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> {
    /// Non-ZK: coefficients visible to verifier
    Standard(StandardSumcheckProof<F, ProofTranscript>),
    /// ZK: only commitments visible, coefficients hidden in BlindFold
    Zk(ZkSumcheckProof<F, C, ProofTranscript>),
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> CanonicalSerialize
    for SumcheckInstanceProof<F, C, ProofTranscript>
{
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        mut writer: W,
        compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        match self {
            Self::Standard(proof) => {
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
            Self::Standard(proof) => proof.serialized_size(compress),
            Self::Zk(proof) => proof.serialized_size(compress),
        }
    }
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> ark_serialize::Valid
    for SumcheckInstanceProof<F, C, ProofTranscript>
{
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        match self {
            Self::Standard(proof) => proof.check(),
            Self::Zk(proof) => proof.check(),
        }
    }
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript> CanonicalDeserialize
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
                let proof =
                    StandardSumcheckProof::deserialize_with_mode(reader, compress, validate)?;
                Ok(Self::Standard(proof))
            }
            1 => {
                let proof = ZkSumcheckProof::deserialize_with_mode(reader, compress, validate)?;
                Ok(Self::Zk(proof))
            }
            _ => Err(ark_serialize::SerializationError::InvalidData),
        }
    }
}

impl<F: JoltField, C: JoltCurve, ProofTranscript: Transcript>
    SumcheckInstanceProof<F, C, ProofTranscript>
{
    /// Create a standard (non-ZK) sumcheck proof.
    pub fn new_standard(compressed_polys: Vec<CompressedUniPoly<F>>) -> Self {
        Self::Standard(StandardSumcheckProof::new(compressed_polys))
    }

    /// Create a ZK sumcheck proof with only commitments and polynomial degrees.
    pub fn new_zk(round_commitments: Vec<C::G1>, poly_degrees: Vec<usize>) -> Self {
        Self::Zk(ZkSumcheckProof::new(round_commitments, poly_degrees))
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
            Self::Standard(proof) => proof.verify(claim, num_rounds, degree_bound, transcript),
            Self::Zk(proof) => {
                let challenges = proof.verify_transcript_only(num_rounds, transcript)?;
                // For ZK mode, we don't compute the final claim here
                // BlindFold verification ensures the R1CS constraints are satisfied
                Ok((F::zero(), challenges))
            }
        }
    }

    /// Get the number of rounds in this proof.
    pub fn num_rounds(&self) -> usize {
        match self {
            Self::Standard(proof) => proof.compressed_polys.len(),
            Self::Zk(proof) => proof.round_commitments.len(),
        }
    }
}
