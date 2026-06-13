#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use crate::curve::JoltCurve;
use crate::field::JoltField;
#[cfg(feature = "zk")]
use crate::poly::commitment::pedersen::PedersenGenerators;
#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
use crate::poly::opening_proof::{
    AbstractVerifierOpeningAccumulator, ProverOpeningAccumulator, VerifierOpeningAccumulator,
};
use crate::poly::unipoly::{CompressedUniPoly, UniPoly};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcript_msgs::{ProverFs, VerifierFs};
use crate::utils::errors::ProofVerifyError;
#[cfg(not(target_arch = "wasm32"))]
use crate::utils::profiling::print_current_memory_usage;

#[cfg(feature = "zk")]
use rand_core::CryptoRngCore;

/// Implements the standard technique for batching parallel sumchecks to reduce
/// verifier cost and proof size.
///
/// For details, refer to Jim Posen's ["Perspectives on Sumcheck Batching"](https://hackmd.io/s/HyxaupAAA).
/// We do what they describe as "front-loaded" batch sumcheck.
pub enum BatchedSumcheck {}
impl BatchedSumcheck {
    /// Returns (challenges, initial_batched_claim).
    /// For non-ZK mode - the round polynomial coefficients are written into the NARG.
    pub fn prove<F: JoltField>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut impl ProverFs<F>,
    ) -> (Vec<F::Challenge>, F) {
        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        // Input claims are shared (the verifier recomputes them from openings).
        sumcheck_instances.iter().for_each(|sumcheck| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            transcript.absorb_scalar(&input_claim);
        });
        let batching_coeffs: Vec<F> = transcript.challenge_vec(sumcheck_instances.len());

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

            // Write the prover's round polynomial into the NARG (prover-only payload).
            // The self-delimiting frame lets the verifier read back the exact (possibly
            // round-varying) number of coefficients.
            transcript.write_scalars(&compressed_poly.coeffs_except_linear_term);
            let r_j = transcript.challenge_optimized();
            r_sumcheck.push(r_j);

            individual_claims
                .iter_mut()
                .zip(univariate_polys)
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

        (r_sumcheck, initial_batched_claim)
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
    /// Returns (challenges, initial_batched_claim).
    /// The round/output-claim commitments and per-round degrees are written into the NARG.
    #[cfg(feature = "zk")]
    pub fn prove_zk<F: JoltField, C: JoltCurve<F = F>, R: CryptoRngCore>(
        mut sumcheck_instances: Vec<&mut dyn SumcheckInstanceProver<F>>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        blindfold_accumulator: &mut crate::subprotocols::blindfold::BlindFoldAccumulator<F, C>,
        transcript: &mut impl ProverFs<F>,
        pedersen_gens: &PedersenGenerators<C>,
        rng: &mut R,
    ) -> (Vec<F::Challenge>, F) {
        use crate::subprotocols::blindfold::ZkStageData;

        let max_num_rounds = sumcheck_instances
            .iter()
            .map(|sumcheck| sumcheck.num_rounds())
            .max()
            .unwrap();

        // In ZK mode, don't absorb cleartext claims — polynomial commitments provide binding.
        // Batching coefficients are still unpredictable (from transcript state after commitments).
        let batching_coeffs: Vec<F> = transcript.challenge_vec(sumcheck_instances.len());

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

            // Round commitments are prover-only payload: written into the NARG (which also
            // absorbs them) immediately before squeezing this round's challenge. The verifier
            // reads each commitment back at the SAME position in `verify_transcript_only`, so
            // the per-round interleave (write commitment → squeeze challenge) is symmetric.
            transcript.write_slice(std::slice::from_ref(&commitment));

            let r_j = transcript.challenge_optimized();
            r_sumcheck.push(r_j);

            individual_claims
                .iter_mut()
                .zip(univariate_polys)
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
            poly_degrees.push(batched_univariate_poly.coeffs.len().saturating_sub(1));
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
        // After the round loop, write the per-round polynomial degrees (public R1CS data the
        // verifier needs in stage 8), then the output-claim commitments. The verifier reads in
        // this SAME order: `poly_degrees` after its round loop (`verify_transcript_only`), then
        // `output_claims_commitments` (`BatchedSumcheck::verify`).
        transcript.write_slice(&poly_degrees);
        transcript.write_slice(&output_claims_commitments);

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

        (r_sumcheck, initial_batched_claim)
    }

    /// Returns `(batching_coeffs, r_sumcheck, zk_readback)`. `zk_readback` is `Some` only
    /// in ZK mode (the round commitments, per-round degrees, and output-claim commitments
    /// read back from the NARG, threaded to stage 8 / BlindFold); `None` otherwise.
    ///
    /// `zk_mode` is the read-frame SELECTOR (sourced from the proof's single global
    /// `zk_mode`): it picks the Clear round-polynomial path or the ZK commitment path.
    /// The NARG read order within each path is unchanged.
    pub fn verify<F: JoltField, C: JoltCurve<F = F>>(
        zk_mode: bool,
        sumcheck_instances: Vec<&dyn SumcheckInstanceVerifier<F, VerifierOpeningAccumulator<F>>>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(Vec<F>, Vec<F::Challenge>, Option<ZkSumcheckReadback<C>>), ProofVerifyError> {
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

        let is_zk = zk_mode;
        if !is_zk {
            sumcheck_instances.iter().for_each(|sumcheck| {
                let input_claim = sumcheck.input_claim(opening_accumulator);
                transcript.absorb_scalar(&input_claim);
            });
        }
        let batching_coeffs: Vec<F> = transcript.challenge_vec(sumcheck_instances.len());

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

        // SELECTOR: ZK mode reads round commitments from the NARG (BlindFold proves the
        // round polynomials); non-ZK reads the round polynomials directly. The per-path
        // NARG read order is identical to before — only the branch source changed.
        let (output_claim, r_sumcheck, zk_round_readback): (
            F,
            Vec<F::Challenge>,
            Option<(Vec<C::G1>, Vec<usize>)>,
        ) = if !is_zk {
            let (output_claim, r) =
                clear_sumcheck::verify(claim, max_num_rounds, max_degree, transcript)?;
            (output_claim, r, None)
        } else {
            #[cfg(feature = "zk")]
            {
                let (r, round_commitments, poly_degrees) =
                    zk_sumcheck::verify_transcript_only::<F, C>(
                        max_num_rounds,
                        max_degree,
                        transcript,
                    )?;
                (F::zero(), r, Some((round_commitments, poly_degrees)))
            }
            #[cfg(not(feature = "zk"))]
            {
                return Err(ProofVerifyError::ZkFeatureRequired);
            }
        };

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

        // In ZK mode, read the output-claim commitments back from the NARG (the prover wrote
        // them right after `poly_degrees`, which `verify_transcript_only` already consumed).
        // Assemble the full stage readback for stage 8.
        let zk_readback: Option<ZkSumcheckReadback<C>> = if !is_zk {
            opening_accumulator.flush_to_transcript(transcript);
            None
        } else {
            #[cfg(feature = "zk")]
            {
                let output_claims_commitments: Vec<C::G1> = transcript
                    .read_slice()
                    .map_err(|_| ProofVerifyError::SumcheckVerificationError)?;
                opening_accumulator.take_pending_claims();
                let (round_commitments, poly_degrees) =
                    zk_round_readback.ok_or(ProofVerifyError::SumcheckVerificationError)?;
                Some(ZkSumcheckReadback {
                    round_commitments,
                    poly_degrees,
                    output_claims_commitments,
                })
            }
            #[cfg(not(feature = "zk"))]
            {
                let _ = &zk_round_readback;
                return Err(ProofVerifyError::ZkFeatureRequired);
            }
        };

        // In ZK mode, skip output claim verification — BlindFold proves this
        if !is_zk && output_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok((batching_coeffs, r_sumcheck, zk_readback))
    }

    /// Verify a standard (non-ZK) sumcheck proof without requiring a curve type.
    /// Used by opening proof reduction which doesn't need ZK mode. The round
    /// polynomials are read back from the NARG.
    pub fn verify_standard<F: JoltField, A: AbstractVerifierOpeningAccumulator<F>>(
        sumcheck_instances: Vec<&dyn SumcheckInstanceVerifier<F, A>>,
        opening_accumulator: &mut A,
        transcript: &mut impl VerifierFs<F>,
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

        // Absorb input claims BEFORE deriving batching coefficients
        // (must match ordering in BatchedSumcheck::prove)
        sumcheck_instances.iter().for_each(|sumcheck| {
            let input_claim = sumcheck.input_claim(opening_accumulator);
            transcript.absorb_scalar(&input_claim);
        });

        let batching_coeffs: Vec<F> = transcript.challenge_vec(sumcheck_instances.len());

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
            clear_sumcheck::verify(claim, max_num_rounds, max_degree, transcript)?;

        let expected_output_claim = sumcheck_instances
            .iter()
            .zip(batching_coeffs.iter())
            .map(|(sumcheck, coeff)| {
                let offset = sumcheck.round_offset(max_num_rounds);
                let r_slice = &r_sumcheck[offset..offset + sumcheck.num_rounds()];
                sumcheck.cache_openings(opening_accumulator, r_slice);
                let claim = sumcheck.expected_output_claim(opening_accumulator, r_slice);
                claim * coeff
            })
            .sum();

        opening_accumulator.flush_to_transcript(transcript);

        if output_claim != expected_output_claim {
            return Err(ProofVerifyError::SumcheckVerificationError);
        }

        Ok(r_sumcheck)
    }
}

/// Clear (non-ZK) sumcheck verification namespace.
///
/// Under the NARG (Option B) the round-polynomial coefficients live in the NARG
/// byte-string — the prover writes them via `write_slice` and the verifier reads
/// them back with `read_slice`. This type carries no data; it only groups the
/// non-ZK verification logic. The Fiat-Shamir mode is selected globally by
/// `JoltProof::zk_mode`, not by a per-stage marker.
pub mod clear_sumcheck {
    use super::*;

    /// Verify this standard sumcheck by reading each round polynomial back from the
    /// NARG and evaluating it — the math is identical to the cleartext path; only the
    /// source of the coefficients changed (NARG instead of a struct field).
    pub fn verify<F: JoltField>(
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(F, Vec<F::Challenge>), ProofVerifyError> {
        let mut e = claim;
        let mut r: Vec<F::Challenge> = Vec::with_capacity(num_rounds);

        for _ in 0..num_rounds {
            // Read the prover's round polynomial back from the NARG and reconstruct it.
            let coeffs_except_linear_term: Vec<F> = transcript
                .read_scalars()
                .map_err(|_| ProofVerifyError::SumcheckVerificationError)?;
            let poly = CompressedUniPoly {
                coeffs_except_linear_term,
            };

            let poly_degree = poly.degree();
            if poly_degree == 0 || poly_degree > degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(
                    degree_bound,
                    poly_degree,
                ));
            }

            let r_i: F::Challenge = transcript.challenge_optimized();
            r.push(r_i);
            e = poly.eval_from_hint(&e, &r_i);
        }

        Ok((e, r))
    }
}

/// ZK sumcheck values read back from the NARG during verification, threaded to stage 8
/// (BlindFold) which can no longer read them from the (now data-free) proof struct.
/// Constructed only in ZK builds; the type exists in both so `BatchedSumcheck::verify`'s
/// signature is uniform (the non-ZK path always yields `None`).
#[derive(Debug, Clone)]
pub struct ZkSumcheckReadback<C: JoltCurve> {
    /// Pedersen commitments to round polynomials (G1 curve elements), one per round.
    pub round_commitments: Vec<C::G1>,
    /// Polynomial degree for each round (public info needed for R1CS construction).
    pub poly_degrees: Vec<usize>,
    /// Pedersen commitments to output claims, chunked to fit generator count.
    pub output_claims_commitments: Vec<C::G1>,
}

/// ZK sumcheck verification namespace: reads the prover-only round commitments, degrees,
/// and output-claim commitments back from the NARG (the proof struct carries no data).
#[cfg(feature = "zk")]
pub mod zk_sumcheck {
    use super::*;

    /// Verify ZK sumcheck by reading the per-round commitments back from the NARG and
    /// deriving challenges. Does NOT evaluate polynomials — that's handled by BlindFold.
    ///
    /// Reads, in the exact order the prover wrote them in `prove_zk`: per round one
    /// commitment (then squeezes that round's challenge), then the per-round `poly_degrees`
    /// frame after the loop. The output-claim commitments are read one position later, by
    /// the caller (`BatchedSumcheck::verify`), matching the prover. The read-back data is
    /// returned so stage 8 can consume it.
    pub fn verify_transcript_only<F: JoltField, C: JoltCurve<F = F>>(
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(Vec<F::Challenge>, Vec<C::G1>, Vec<usize>), ProofVerifyError> {
        let mut r: Vec<F::Challenge> = Vec::with_capacity(num_rounds);
        let mut round_commitments: Vec<C::G1> = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            let commitment: C::G1 = transcript
                .read_single()
                .map_err(|_| ProofVerifyError::SumcheckVerificationError)?;
            let r_i: F::Challenge = transcript.challenge_optimized();
            round_commitments.push(commitment);
            r.push(r_i);
        }

        // After the round loop, read the per-round polynomial degrees (matching the prover's
        // post-loop `write_slice(&poly_degrees)`).
        let poly_degrees: Vec<usize> = transcript
            .read_slice()
            .map_err(|_| ProofVerifyError::SumcheckVerificationError)?;
        if poly_degrees.len() != num_rounds {
            return Err(ProofVerifyError::InvalidInputLength(
                num_rounds,
                poly_degrees.len(),
            ));
        }
        for &degree in &poly_degrees {
            if degree > degree_bound {
                return Err(ProofVerifyError::InvalidInputLength(degree_bound, degree));
            }
        }

        Ok((r, round_commitments, poly_degrees))
    }
}
