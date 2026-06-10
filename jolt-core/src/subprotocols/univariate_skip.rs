#[cfg(feature = "zk")]
use rand_core::CryptoRngCore;

use crate::curve::JoltCurve;
use crate::field::JoltField;
#[cfg(feature = "zk")]
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::lagrange_poly::LagrangePolynomial;
#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
use crate::poly::opening_proof::{AbstractVerifierOpeningAccumulator, ProverOpeningAccumulator};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcript_msgs::{ProverFs, VerifierFs};
use crate::utils::errors::ProofVerifyError;

/// Returns the interleaved symmetric univariate-skip target indices outside the base window.
///
/// Domain is assumed to be the canonical symmetric window of size DOMAIN_SIZE with
/// base indices from start = -((DOMAIN_SIZE-1)/2) to end = start + DOMAIN_SIZE - 1.
///
/// Targets are the extended points z ∈ {−DEGREE..−1} ∪ {1..DEGREE}, interleaved as
/// [start-1, end+1, start-2, end+2, ...] until DEGREE points are produced.
#[inline]
pub const fn uniskip_targets<const DOMAIN_SIZE: usize, const DEGREE: usize>() -> [i64; DEGREE] {
    let d: i64 = DEGREE as i64;
    let ext_left: i64 = -d;
    let ext_right: i64 = d;
    let base_left: i64 = -((DOMAIN_SIZE as i64 - 1) / 2);
    let base_right: i64 = base_left + (DOMAIN_SIZE as i64) - 1;

    let mut targets: [i64; DEGREE] = [0; DEGREE];
    let mut idx = 0usize;
    let mut n = base_left - 1;
    let mut p = base_right + 1;

    while n >= ext_left && p <= ext_right && idx < DEGREE {
        targets[idx] = n;
        idx += 1;
        if idx >= DEGREE {
            break;
        }
        targets[idx] = p;
        idx += 1;
        n -= 1;
        p += 1;
    }

    while idx < DEGREE && n >= ext_left {
        targets[idx] = n;
        idx += 1;
        n -= 1;
    }

    while idx < DEGREE && p <= ext_right {
        targets[idx] = p;
        idx += 1;
        p += 1;
    }

    targets
}

/// Builds the uni-skip first-round polynomial s1 from base and extended evaluations of t1.
///
/// SPECIFIC: This helper targets the setting where s1(Y) = L(τ_high, Y) · t1(Y), with L the
/// degree-(DOMAIN_SIZE-1) Lagrange kernel over the base window and t1 a univariate of degree
/// at most 2·DEGREE (extended symmetric window size EXTENDED_SIZE = 2·DEGREE + 1).
/// Consequently, the resulting s1 has degree at most 3·DEGREE (NUM_COEFFS = 3·DEGREE + 1).
///
/// Inputs:
/// - base_evals: optional t1 evaluations on the base window (symmetric grid of size DOMAIN_SIZE).
///   When `None`, base evaluations are treated as all zeros.
/// - extended_evals: t1 evaluated on the extended symmetric grid outside the base window,
///   in the order given by `uniskip_targets::<DOMAIN_SIZE, DEGREE>()`.
/// - tau_high: the challenge used in the Lagrange kernel L(τ_high, ·) over the base window.
///
/// Returns: UniPoly s1 with exactly NUM_COEFFS coefficients.
#[inline]
pub fn build_uniskip_first_round_poly<
    F: JoltField,
    const DOMAIN_SIZE: usize,
    const DEGREE: usize,
    const EXTENDED_SIZE: usize,
    const NUM_COEFFS: usize,
>(
    base_evals: Option<&[F; DOMAIN_SIZE]>,
    extended_evals: &[F; DEGREE],
    tau_high: F::Challenge,
) -> UniPoly<F> {
    debug_assert_eq!(EXTENDED_SIZE, 2 * DEGREE + 1);
    debug_assert_eq!(NUM_COEFFS, 3 * DEGREE + 1);

    // Rebuild t1 on the full extended symmetric window
    let targets: [i64; DEGREE] = uniskip_targets::<DOMAIN_SIZE, DEGREE>();
    let mut t1_vals: [F; EXTENDED_SIZE] = [F::zero(); EXTENDED_SIZE];

    // Fill in base window evaluations when provided (otherwise treated as zeros)
    if let Some(base) = base_evals {
        let base_left: i64 = -((DOMAIN_SIZE as i64 - 1) / 2);
        for (i, &val) in base.iter().enumerate() {
            let z = base_left + (i as i64);
            let pos = (z + (DEGREE as i64)) as usize;
            t1_vals[pos] = val;
        }
    }

    // Fill in extended evaluations (outside base window)
    for (idx, &val) in extended_evals.iter().enumerate() {
        let z = targets[idx];
        let pos = (z + (DEGREE as i64)) as usize;
        t1_vals[pos] = val;
    }

    let t1_coeffs = LagrangePolynomial::<F>::interpolate_coeffs::<EXTENDED_SIZE>(&t1_vals);
    let lagrange_values = LagrangePolynomial::<F>::evals::<F::Challenge, DOMAIN_SIZE>(&tau_high);
    let lagrange_coeffs =
        LagrangePolynomial::<F>::interpolate_coeffs::<DOMAIN_SIZE>(&lagrange_values);

    let mut s1_coeffs: [F; NUM_COEFFS] = [F::zero(); NUM_COEFFS];
    for (i, &a) in lagrange_coeffs.iter().enumerate() {
        for (j, &b) in t1_coeffs.iter().enumerate() {
            s1_coeffs[i + j] += a * b;
        }
    }

    UniPoly::from_coeff(s1_coeffs.to_vec())
}

/// Prove-only helper for a uni-skip first round instance (non-ZK mode).
/// Writes the first-round polynomial into the NARG and returns the uni-skip challenge r0.
pub fn prove_uniskip_round<F: JoltField, I: SumcheckInstanceProver<F>>(
    instance: &mut I,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    transcript: &mut impl ProverFs<F>,
) {
    let input_claim = instance.input_claim(opening_accumulator);
    let uni_poly = instance.compute_message(0, input_claim);
    // Write the full first-round polynomial into the NARG and derive r0.
    transcript.write_slice(&uni_poly.coeffs);
    let r0: F::Challenge = transcript.challenge_optimized();
    instance.cache_openings(opening_accumulator, &[r0]);
    opening_accumulator.flush_to_transcript(transcript);
}

/// ZK variant: commits to coefficients instead of revealing them.
/// The polynomial coefficients are stored in the accumulator for BlindFold verification.
#[cfg(feature = "zk")]
pub fn prove_uniskip_round_zk<
    F: JoltField,
    C: JoltCurve<F = F>,
    I: SumcheckInstanceProver<F>,
    R: CryptoRngCore,
>(
    instance: &mut I,
    opening_accumulator: &mut ProverOpeningAccumulator<F>,
    blindfold_accumulator: &mut crate::subprotocols::blindfold::BlindFoldAccumulator<F, C>,
    transcript: &mut impl ProverFs<F>,
    pedersen_gens: &PedersenGenerators<C>,
    rng: &mut R,
) {
    use crate::subprotocols::blindfold::UniSkipStageData;

    let input_claim = instance.input_claim(opening_accumulator);
    let uni_poly = instance.compute_message(0, input_claim);
    let poly_degree = uni_poly.degree();

    let blinding = F::random(rng);
    let commitment = pedersen_gens.commit(&uni_poly.coeffs, &blinding);

    // The first-round commitment is prover-only payload: written into the NARG (which also
    // absorbs it) immediately before squeezing r0. The verifier reads it back at the SAME
    // position in `verify_transcript`, keeping the write-commitment → squeeze-challenge
    // interleave symmetric.
    transcript.write_slice(std::slice::from_ref(&commitment));

    let r0: F::Challenge = transcript.challenge_optimized();
    instance.cache_openings(opening_accumulator, &[r0]);

    // After the challenge, write the public polynomial degree (needed by stage 8 for R1CS
    // config), then the output-claim commitments — the verifier reads in this same order.
    transcript.write_slice(std::slice::from_ref(&poly_degree));

    let output_claim_values = opening_accumulator.take_pending_claims();
    let output_claim_ids = opening_accumulator.take_pending_claim_ids();
    let oc_committed: Vec<_> = pedersen_gens.commit_chunked(&output_claim_values, rng);
    let output_claims: Vec<(OpeningId, F)> = output_claim_ids
        .into_iter()
        .zip(output_claim_values)
        .collect();
    let output_claims_commitments: Vec<_> = oc_committed.iter().map(|(c, _)| *c).collect();
    let output_claims_blindings: Vec<_> = oc_committed.iter().map(|(_, b)| *b).collect();
    transcript.write_slice(&output_claims_commitments);

    let input_constraint = instance.get_params().input_claim_constraint();
    let input_constraint_challenge_values = instance
        .get_params()
        .input_constraint_challenge_values(opening_accumulator);
    let output_constraint = instance.get_params().output_claim_constraint();
    let output_constraint_challenge_values = instance
        .get_params()
        .output_constraint_challenge_values(&[r0]);

    blindfold_accumulator.push_uniskip_data(UniSkipStageData {
        input_claim,
        poly_coeffs: uni_poly.coeffs.clone(),
        blinding_factor: blinding,
        challenge: r0,
        poly_degree,
        commitment,
        input_constraint,
        input_constraint_challenge_values,
        output_constraint,
        output_constraint_challenge_values,
        output_claims,
        output_claims_blindings,
        output_claims_commitments: output_claims_commitments.clone(),
    });
}

/// Non-ZK univariate-skip first-round verification namespace.
///
/// Under the NARG the full first-round polynomial lives in the NARG byte-string —
/// the prover writes it via `write_slice` and the verifier reads it back with
/// `read_slice`. This type carries no data; it only groups the non-ZK first-round
/// verification logic. The Fiat-Shamir mode is selected globally by `JoltProof::zk_mode`.
///
/// ⚠️ ZK-MIGRATION NOTE (parallel to `clear_sumcheck`; see DEV-27): do NOT re-add a
/// `uni_poly` field. The first-round poly is in the NARG (Option B). The ZK path uses the
/// separate `zk_uni_skip_first_round` (Pedersen commitment + degree), NOT this path.
/// If the ZK migration ever needs the Standard first-round coeffs, read them from the NARG
/// (`read_slice`), do not restore a field.
pub mod uni_skip_first_round {
    use super::*;

    /// Verify only the univariate-skip first round by reading the polynomial back
    /// from the NARG. The checks (degree, symmetric-domain sum, evaluation) are
    /// identical to the cleartext path; only the source of the polynomial changed.
    pub fn verify<
        F: JoltField,
        const N: usize,
        const FIRST_ROUND_POLY_NUM_COEFFS: usize,
        A: AbstractVerifierOpeningAccumulator<F>,
    >(
        sumcheck_instance: &dyn SumcheckInstanceVerifier<F, A>,
        opening_accumulator: &mut A,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<F::Challenge, ProofVerifyError> {
        let degree_bound = sumcheck_instance.degree();

        // Read the full first-round polynomial back from the NARG and derive r0.
        let coeffs: Vec<F> = transcript
            .read_slice()
            .map_err(|_| ProofVerifyError::UniSkipVerificationError)?;
        // The first-round polynomial has a fixed coefficient count; reject a frame of
        // any other length before it reaches `check_sum_evals` (which indexes by
        // `FIRST_ROUND_POLY_NUM_COEFFS` and assumes exactly that many coefficients).
        if coeffs.len() != FIRST_ROUND_POLY_NUM_COEFFS {
            return Err(ProofVerifyError::InvalidInputLength(
                FIRST_ROUND_POLY_NUM_COEFFS,
                coeffs.len(),
            ));
        }
        let uni_poly = UniPoly::from_coeff(coeffs);
        if uni_poly.degree() > degree_bound {
            return Err(ProofVerifyError::InvalidInputLength(
                degree_bound,
                uni_poly.degree(),
            ));
        }
        let r0 = transcript.challenge_optimized();

        // Check symmetric-domain sum equals zero (initial claim), and compute next claim s1(r0)
        let input_claim = sumcheck_instance.input_claim(opening_accumulator);
        let input_claim_ok =
            uni_poly.check_sum_evals::<N, FIRST_ROUND_POLY_NUM_COEFFS>(input_claim);

        sumcheck_instance.cache_openings(opening_accumulator, &[r0]);
        let expected_output = uni_poly.evaluate(&r0);
        let claimed_output = sumcheck_instance.expected_output_claim(opening_accumulator, &[r0]);
        let output_claim_ok = claimed_output == expected_output;

        opening_accumulator.flush_to_transcript(transcript);

        if !input_claim_ok || !output_claim_ok {
            Err(ProofVerifyError::UniSkipVerificationError)
        } else {
            Ok(r0)
        }
    }
}

/// ZK uni-skip values read back from the NARG during verification, threaded to stage 8
/// (BlindFold) which can no longer read them from the (now data-free) proof struct.
/// Constructed only in ZK builds; the type exists in both so the uni-skip verify helpers
/// have a uniform signature (the non-ZK path always yields `None`).
#[derive(Debug, Clone)]
pub struct ZkUniSkipReadback<C: JoltCurve> {
    pub commitment: C::G1,
    pub poly_degree: usize,
    /// Pedersen commitments to output claims, chunked to fit generator count.
    pub output_claims_commitments: Vec<C::G1>,
}

/// ZK uni-skip first-round verification namespace: reads the prover-only commitment,
/// degree, and output-claim commitments back from the NARG (the proof carries no data).
#[cfg(feature = "zk")]
pub mod zk_uni_skip_first_round {
    use super::*;

    /// Verify transcript consistency only by reading the first-round values back from the
    /// NARG. The actual polynomial verification (sum check + evaluation) is done by BlindFold.
    ///
    /// Reads, in the exact order the prover wrote them in `prove_uniskip_round_zk`: the
    /// commitment (then squeezes r0), the polynomial degree, then the output-claim
    /// commitments. Returns the read-back data so stage 8 can consume it.
    pub fn verify_transcript<
        F: JoltField,
        C: JoltCurve<F = F>,
        A: AbstractVerifierOpeningAccumulator<F>,
        I: SumcheckInstanceVerifier<F, A>,
    >(
        sumcheck_instance: &I,
        opening_accumulator: &mut A,
        transcript: &mut impl VerifierFs<F>,
    ) -> Result<(F::Challenge, ZkUniSkipReadback<C>), ProofVerifyError> {
        let degree_bound = sumcheck_instance.degree();

        let commitment: C::G1 = transcript
            .read_slice()
            .map_err(|_| ProofVerifyError::UniSkipVerificationError)?
            .into_iter()
            .next()
            .ok_or(ProofVerifyError::UniSkipVerificationError)?;

        let r0: F::Challenge = transcript.challenge_optimized();
        sumcheck_instance.cache_openings(opening_accumulator, &[r0]);

        let poly_degree: usize = transcript
            .read_slice()
            .map_err(|_| ProofVerifyError::UniSkipVerificationError)?
            .into_iter()
            .next()
            .ok_or(ProofVerifyError::UniSkipVerificationError)?;
        if poly_degree > degree_bound {
            return Err(ProofVerifyError::InvalidInputLength(
                degree_bound,
                poly_degree,
            ));
        }

        let output_claims_commitments: Vec<C::G1> = transcript
            .read_slice()
            .map_err(|_| ProofVerifyError::UniSkipVerificationError)?;
        opening_accumulator.take_pending_claims();

        Ok((
            r0,
            ZkUniSkipReadback {
                commitment,
                poly_degree,
                output_claims_commitments,
            },
        ))
    }
}

