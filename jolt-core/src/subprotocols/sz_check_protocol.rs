use crate::{
    field::JoltField,
    poly::{
        commitment::{
            hyrax::{HyraxCommitment, HyraxOpeningProof},
            pedersen::PedersenGenerators,
        },
        dense_mlpoly::DensePolynomial,
        opening_proof::{ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator},
    },
    subprotocols::{
        square_and_multiply::SZCheckSumcheck,
        sumcheck::{BatchedSumcheck, SumcheckInstance, SumcheckInstanceProof},
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
    zkvm::witness::RecursionCommittedPolynomial,
};
use ark_bn254::Fq;
use ark_grumpkin::Projective as GrumpkinProjective;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
use jolt_optimizations::ExponentiationSteps;
use std::{cell::RefCell, rc::Rc};

/// Artifacts needed by the verifier for SZ check protocol
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct SZCheckArtifacts<const RATIO: usize = 1> {
    pub rho_commitments: Vec<Vec<HyraxCommitment<RATIO, GrumpkinProjective>>>,
    pub quotient_commitments: Vec<Vec<HyraxCommitment<RATIO, GrumpkinProjective>>>,
    pub base_commitments: Vec<HyraxCommitment<RATIO, GrumpkinProjective>>,
    pub num_exponentiations: usize,
    pub num_constraints_per_exponentiation: Vec<usize>,
    pub bits_per_exponentiation: Vec<Vec<bool>>,
}

/// Complete proof for SZ check protocol including sumcheck and Hyrax opening
#[derive(Clone, Debug)]
pub struct SZCheckProof<F: JoltField, ProofTranscript: Transcript, const RATIO: usize> {
    pub artifacts: SZCheckArtifacts<RATIO>,
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub r_sumcheck: Vec<F>,
    pub batched_hyrax_proof: HyraxOpeningProof<RATIO, GrumpkinProjective>,
    pub openings: crate::poly::opening_proof::Openings<F>,
}

/// Helper function to convert F to Fq when F implements Into<Fq>
fn field_to_fq<F: JoltField>(f: &F) -> Fq {
    // This is a temporary solution - in production, F should properly implement Into<Fq>
    // For now, we assume F is Fq in the sz_check context
    unsafe { std::mem::transmute_copy(f) }
}

/// Helper function to convert Vec<F> to Vec<Fq>
fn vec_field_to_fq<F: JoltField>(vec: &[F]) -> Vec<Fq> {
    vec.iter().map(field_to_fq).collect()
}

/// Helper function to commit to a polynomial
fn commit_polynomial<const RATIO: usize>(
    poly: &[Fq],
    generators: &PedersenGenerators<GrumpkinProjective>,
) -> HyraxCommitment<RATIO, GrumpkinProjective> {
    HyraxCommitment::<RATIO, GrumpkinProjective>::commit(
        &DensePolynomial::new(poly.to_vec()),
        generators,
    )
}

/// Proves the SZ check protocol, handling the complete flow from sumcheck to Hyrax
///
/// # Protocol Flow:
/// 1. Commit to all polynomials (rho, quotient, base)
/// 2. Run sumcheck protocol to reduce to opening claims
/// 3. Batch all opening claims with random challenges
/// 4. Generate Hyrax opening proof for the batched polynomial
pub fn sz_check_prove<F, ProofTranscript, const RATIO: usize>(
    exponentiation_steps_vec: Vec<ExponentiationSteps>,
    transcript: &mut ProofTranscript,
    hyrax_generators: &PedersenGenerators<GrumpkinProjective>,
) -> SZCheckProof<F, ProofTranscript, RATIO>
where
    F: JoltField + From<Fq>,
    ProofTranscript: Transcript,
{
    // Step 1: Prepare polynomials and commitments
    let num_exponentiations = exponentiation_steps_vec.len();
    let mut rho_commitments = Vec::with_capacity(num_exponentiations);
    let mut quotient_commitments = Vec::with_capacity(num_exponentiations);
    let mut base_commitments = Vec::with_capacity(num_exponentiations);

    let mut all_rho_polys = Vec::with_capacity(num_exponentiations);
    let mut all_quotient_polys = Vec::with_capacity(num_exponentiations);
    let mut all_base_polys = Vec::with_capacity(num_exponentiations);

    for steps in &exponentiation_steps_vec {
        // Commit to rho polynomials
        let rho_comms: Vec<_> = steps
            .rho_mles
            .iter()
            .map(|poly| commit_polynomial::<RATIO>(poly, hyrax_generators))
            .collect();
        rho_commitments.push(rho_comms);
        all_rho_polys.push(steps.rho_mles.clone());

        // Commit to quotient polynomials
        let quotient_comms: Vec<_> = steps
            .quotient_mles
            .iter()
            .map(|poly| commit_polynomial::<RATIO>(poly, hyrax_generators))
            .collect();
        quotient_commitments.push(quotient_comms);
        all_quotient_polys.push(steps.quotient_mles.clone());

        // Commit to base polynomial
        let base_poly = jolt_optimizations::fq12_to_multilinear_evals(&steps.base);
        let base_comm = commit_polynomial::<RATIO>(&base_poly, hyrax_generators);
        base_commitments.push(base_comm);
        all_base_polys.push(base_poly);
    }

    // Append all commitments to transcript
    for commitments in rho_commitments
        .iter()
        .chain(quotient_commitments.iter())
        .flat_map(|v| v.iter())
        .chain(base_commitments.iter())
    {
        transcript.append_serializable(commitments);
    }

    // Create artifacts for verifier
    let artifacts = SZCheckArtifacts {
        rho_commitments: rho_commitments.clone(),
        quotient_commitments: quotient_commitments.clone(),
        base_commitments: base_commitments.clone(),
        num_exponentiations,
        num_constraints_per_exponentiation: exponentiation_steps_vec
            .iter()
            .map(|steps| steps.quotient_mles.len())
            .collect(),
        bits_per_exponentiation: exponentiation_steps_vec
            .iter()
            .map(|steps| steps.bits.clone())
            .collect(),
    };

    // Step 2: Create sumcheck instances and run sumcheck protocol
    let mut sumcheck_instances: Vec<Box<dyn SumcheckInstance<F>>> = exponentiation_steps_vec
        .iter()
        .enumerate()
        .map(|(exp_idx, steps)| {
            let r: Vec<F> = transcript.challenge_vector(4);
            let gamma: F = transcript.challenge_scalar();
            Box::new(SZCheckSumcheck::new_prover(exp_idx, steps, r, gamma))
                as Box<dyn SumcheckInstance<F>>
        })
        .collect();

    let prover_accumulator = Rc::new(RefCell::new(ProverOpeningAccumulator::<F>::new()));
    let sumcheck_instances_mut: Vec<&mut dyn SumcheckInstance<F>> = sumcheck_instances
        .iter_mut()
        .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
        .collect();

    let (sumcheck_proof, r_sumcheck) = BatchedSumcheck::prove(
        sumcheck_instances_mut,
        Some(prover_accumulator.clone()),
        transcript,
    );

    // Step 3: Batch all polynomials for Hyrax opening
    let total_polys = all_rho_polys.iter().map(|v| v.len()).sum::<usize>()
        + all_quotient_polys.iter().map(|v| v.len()).sum::<usize>()
        + all_base_polys.len()
        + num_exponentiations; // for g polynomials

    let batching_challenges: Vec<F> = transcript.challenge_vector(total_polys);

    // Create batched polynomial
    let mut batched_poly = vec![F::zero(); 16]; // 2^4 = 16 for 4 variables
    let mut batched_eval = F::zero();
    let mut challenge_idx = 0;

    // Batch rho polynomials
    for (exp_idx, rho_polys) in all_rho_polys.iter().enumerate() {
        for (poly_idx, rho_poly) in rho_polys.iter().enumerate() {
            let gamma = batching_challenges[challenge_idx];
            challenge_idx += 1;

            // Get evaluation from accumulator
            let eval = prover_accumulator
                .borrow()
                .get_recursion_polynomial_opening(
                    RecursionCommittedPolynomial::SZCheckRho(exp_idx, poly_idx),
                    SumcheckId::SZCheck,
                )
                .1;
            batched_eval += gamma * eval;

            // Add to batched polynomial
            for (j, &coeff) in rho_poly.iter().enumerate() {
                batched_poly[j] += gamma * F::from(coeff);
            }
        }
    }

    // Batch quotient polynomials
    for (exp_idx, quotient_polys) in all_quotient_polys.iter().enumerate() {
        for (poly_idx, q_poly) in quotient_polys.iter().enumerate() {
            let gamma = batching_challenges[challenge_idx];
            challenge_idx += 1;

            // Get evaluation from accumulator
            let eval = prover_accumulator
                .borrow()
                .get_recursion_polynomial_opening(
                    RecursionCommittedPolynomial::SZCheckQuotient(exp_idx, poly_idx),
                    SumcheckId::SZCheck,
                )
                .1;
            batched_eval += gamma * eval;

            // Add to batched polynomial
            for (j, &coeff) in q_poly.iter().enumerate() {
                batched_poly[j] += gamma * F::from(coeff);
            }
        }
    }

    // Batch base polynomials
    for (exp_idx, base_poly) in all_base_polys.iter().enumerate() {
        let gamma = batching_challenges[challenge_idx];
        challenge_idx += 1;

        // Get evaluation from accumulator
        let eval = prover_accumulator
            .borrow()
            .get_recursion_polynomial_opening(
                RecursionCommittedPolynomial::SZCheckBase(exp_idx),
                SumcheckId::SZCheck,
            )
            .1;
        batched_eval += gamma * eval;

        // Add to batched polynomial
        for (j, &coeff) in base_poly.iter().enumerate() {
            batched_poly[j] += gamma * F::from(coeff);
        }
    }

    // Batch g polynomials
    let g_mle = jolt_optimizations::witness_gen::get_g_mle();
    for exp_idx in 0..num_exponentiations {
        let gamma = batching_challenges[challenge_idx];
        challenge_idx += 1;

        // Get evaluation from accumulator
        let eval = prover_accumulator
            .borrow()
            .get_recursion_polynomial_opening(
                RecursionCommittedPolynomial::SZCheckG(exp_idx),
                SumcheckId::SZCheck,
            )
            .1;
        batched_eval += gamma * eval;

        // Add to batched polynomial
        for (j, &coeff) in g_mle.iter().enumerate() {
            batched_poly[j] += gamma * F::from(coeff);
        }
    }

    // Step 4: Generate Hyrax opening proof
    let r_sumcheck_fq = vec_field_to_fq(&r_sumcheck);
    let batched_poly_fq = vec_field_to_fq(&batched_poly);
    let batched_dense_poly = DensePolynomial::new(batched_poly_fq);

    // Note: r_sumcheck needs to be reversed for Hyrax
    let r_sumcheck_fq_reversed: Vec<Fq> = r_sumcheck_fq.into_iter().rev().collect();

    let batched_hyrax_proof = HyraxOpeningProof::<RATIO, GrumpkinProjective>::prove(
        &batched_dense_poly,
        &r_sumcheck_fq_reversed,
        RATIO,
    );

    // Get openings from accumulator
    let openings = prover_accumulator.borrow().evaluation_openings().clone();

    SZCheckProof {
        artifacts,
        sumcheck_proof,
        r_sumcheck,
        batched_hyrax_proof,
        openings,
    }
}

/// Verifies the SZ check protocol
pub fn sz_check_verify<F, ProofTranscript, const RATIO: usize>(
    proof: &SZCheckProof<F, ProofTranscript, RATIO>,
    transcript: &mut ProofTranscript,
    hyrax_generators: &PedersenGenerators<GrumpkinProjective>,
) -> Result<(), ProofVerifyError>
where
    F: JoltField + From<Fq>,
    ProofTranscript: Transcript,
{
    let artifacts = &proof.artifacts;

    // Step 1: Reconstruct transcript state
    for commitments in artifacts
        .rho_commitments
        .iter()
        .chain(artifacts.quotient_commitments.iter())
        .flat_map(|v| v.iter())
        .chain(artifacts.base_commitments.iter())
    {
        transcript.append_serializable(commitments);
    }

    // Step 2: Create verifier sumcheck instances
    let verifier_sumcheck_instances: Vec<Box<dyn SumcheckInstance<F>>> = (0..artifacts
        .num_exponentiations)
        .map(|i| {
            let r: Vec<F> = transcript.challenge_vector(4);
            let gamma: F = transcript.challenge_scalar();
            Box::new(SZCheckSumcheck::new_verifier(
                i, // exponentiation index
                artifacts.num_constraints_per_exponentiation[i],
                artifacts.bits_per_exponentiation[i].clone(),
                r,
                gamma,
            )) as Box<dyn SumcheckInstance<F>>
        })
        .collect();

    let verifier_instances_ref: Vec<&dyn SumcheckInstance<F>> = verifier_sumcheck_instances
        .iter()
        .map(|instance| &**instance as &dyn SumcheckInstance<F>)
        .collect();

    // Step 3: Verify sumcheck with openings
    let verifier_accumulator = Rc::new(RefCell::new(VerifierOpeningAccumulator::<F>::new()));

    // Populate verifier accumulator with openings from the proof
    for (opening_id, (point, claim)) in proof.openings.iter() {
        verifier_accumulator
            .borrow_mut()
            .openings_mut()
            .insert(opening_id.clone(), (point.clone(), *claim));
    }

    let verified_r = BatchedSumcheck::verify(
        &proof.sumcheck_proof,
        verifier_instances_ref,
        Some(verifier_accumulator.clone()),
        transcript,
    )?;

    if verified_r != proof.r_sumcheck {
        return Err(ProofVerifyError::InternalError);
    }

    // Step 4: Verify Hyrax opening proof
    // Get batching challenges
    let total_polys = artifacts
        .rho_commitments
        .iter()
        .map(|v| v.len())
        .sum::<usize>()
        + artifacts
            .quotient_commitments
            .iter()
            .map(|v| v.len())
            .sum::<usize>()
        + artifacts.base_commitments.len()
        + artifacts.num_exponentiations;

    let batching_challenges: Vec<F> = transcript.challenge_vector(total_polys);

    // Compute batched commitment homomorphically
    let (L_size, _) = crate::poly::commitment::hyrax::matrix_dimensions(4, RATIO);
    let mut batched_row_commitments = vec![GrumpkinProjective::zero(); L_size];
    let mut challenge_idx = 0;

    // Batch rho commitments
    for rho_commitments_vec in &artifacts.rho_commitments {
        for commitment in rho_commitments_vec {
            let gamma = batching_challenges[challenge_idx];
            challenge_idx += 1;
            let gamma_fq = field_to_fq(&gamma);

            for (i, &com) in commitment.row_commitments.iter().enumerate() {
                batched_row_commitments[i] += com * gamma_fq;
            }
        }
    }

    // Batch quotient commitments
    for quotient_commitments_vec in &artifacts.quotient_commitments {
        for commitment in quotient_commitments_vec {
            let gamma = batching_challenges[challenge_idx];
            challenge_idx += 1;
            let gamma_fq = field_to_fq(&gamma);

            for (i, &com) in commitment.row_commitments.iter().enumerate() {
                batched_row_commitments[i] += com * gamma_fq;
            }
        }
    }

    // Batch base commitments
    for commitment in &artifacts.base_commitments {
        let gamma = batching_challenges[challenge_idx];
        challenge_idx += 1;
        let gamma_fq = field_to_fq(&gamma);

        for (i, &com) in commitment.row_commitments.iter().enumerate() {
            batched_row_commitments[i] += com * gamma_fq;
        }
    }

    // Batch g commitments
    let g_mle = jolt_optimizations::witness_gen::get_g_mle();
    let g_poly = DensePolynomial::new(g_mle);
    let g_commitment =
        HyraxCommitment::<RATIO, GrumpkinProjective>::commit(&g_poly, hyrax_generators);

    for _ in 0..artifacts.num_exponentiations {
        let gamma = batching_challenges[challenge_idx];
        challenge_idx += 1;
        let gamma_fq = field_to_fq(&gamma);

        for (i, &com) in g_commitment.row_commitments.iter().enumerate() {
            batched_row_commitments[i] += com * gamma_fq;
        }
    }

    // Create batched commitment and verify
    let batched_hyrax_commitment = HyraxCommitment {
        row_commitments: batched_row_commitments,
    };

    // Compute batched opening claim
    let mut batched_opening_fq = Fq::zero();
    for (opening_id, (_, claim)) in proof.openings.iter() {
        match opening_id {
            crate::poly::opening_proof::OpeningId::Recursion(
                RecursionCommittedPolynomial::SZCheckRho(exp_idx, poly_idx),
                _,
            ) => {
                // Calculate the challenge index for this rho polynomial
                let challenge_idx = artifacts.rho_commitments[0..*exp_idx]
                    .iter()
                    .map(|v| v.len())
                    .sum::<usize>()
                    + poly_idx;
                let gamma_fq = field_to_fq(&batching_challenges[challenge_idx]);
                batched_opening_fq += gamma_fq * field_to_fq(claim);
            }
            crate::poly::opening_proof::OpeningId::Recursion(
                RecursionCommittedPolynomial::SZCheckQuotient(exp_idx, poly_idx),
                _,
            ) => {
                // Calculate the challenge index for this quotient polynomial
                let rho_count = artifacts
                    .rho_commitments
                    .iter()
                    .map(|v| v.len())
                    .sum::<usize>();
                let challenge_idx = rho_count
                    + artifacts.quotient_commitments[0..*exp_idx]
                        .iter()
                        .map(|v| v.len())
                        .sum::<usize>()
                    + poly_idx;
                let gamma_fq = field_to_fq(&batching_challenges[challenge_idx]);
                batched_opening_fq += gamma_fq * field_to_fq(claim);
            }
            crate::poly::opening_proof::OpeningId::Recursion(
                RecursionCommittedPolynomial::SZCheckBase(exp_idx),
                _,
            ) => {
                let rho_count = artifacts
                    .rho_commitments
                    .iter()
                    .map(|v| v.len())
                    .sum::<usize>();
                let quotient_count = artifacts
                    .quotient_commitments
                    .iter()
                    .map(|v| v.len())
                    .sum::<usize>();
                let challenge_idx = rho_count + quotient_count + exp_idx;
                let gamma_fq = field_to_fq(&batching_challenges[challenge_idx]);
                batched_opening_fq += gamma_fq * field_to_fq(claim);
            }
            crate::poly::opening_proof::OpeningId::Recursion(
                RecursionCommittedPolynomial::SZCheckG(exp_idx),
                _,
            ) => {
                let rho_count = artifacts
                    .rho_commitments
                    .iter()
                    .map(|v| v.len())
                    .sum::<usize>();
                let quotient_count = artifacts
                    .quotient_commitments
                    .iter()
                    .map(|v| v.len())
                    .sum::<usize>();
                let base_count = artifacts.base_commitments.len();
                let challenge_idx = rho_count + quotient_count + base_count + exp_idx;
                let gamma_fq = field_to_fq(&batching_challenges[challenge_idx]);
                batched_opening_fq += gamma_fq * field_to_fq(claim);
            }
            _ => {}
        }
    }

    // Convert r_sumcheck to Fq and reverse for Hyrax
    let r_sumcheck_fq = vec_field_to_fq(&proof.r_sumcheck);
    let r_sumcheck_fq_reversed: Vec<Fq> = r_sumcheck_fq.into_iter().rev().collect();

    proof.batched_hyrax_proof.verify(
        hyrax_generators,
        &r_sumcheck_fq_reversed,
        &batched_opening_fq,
        &batched_hyrax_commitment,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcripts::Blake2bTranscript;
    use ark_bn254::{Fq, Fq12, Fr};
    use ark_ff::UniformRand;
    use ark_grumpkin::Projective as GrumpkinProjective;
    use ark_std::test_rng;
    use jolt_optimizations::ExponentiationSteps;

    #[test]
    fn test_single_exponentiation_sz_check() {
        const RATIO: usize = 1;
        let mut rng = test_rng();

        // Generate random exponentiation
        let base = Fq12::rand(&mut rng);
        let exponent = Fr::rand(&mut rng);
        let steps = ExponentiationSteps::new(base, exponent);
        assert!(
            steps.verify_result(),
            "ExponentiationSteps should verify correctly"
        );

        // Initialize transcripts and generators
        let mut prover_transcript = Blake2bTranscript::new(b"test_sz_check");
        let mut verifier_transcript = Blake2bTranscript::new(b"test_sz_check");
        let hyrax_generators = PedersenGenerators::<GrumpkinProjective>::new(16, b"test sz check");

        // Generate proof
        let proof =
            sz_check_prove::<Fq, _, RATIO>(vec![steps], &mut prover_transcript, &hyrax_generators);

        // Verify proof
        let verification_result =
            sz_check_verify::<Fq, _, RATIO>(&proof, &mut verifier_transcript, &hyrax_generators);

        assert!(
            verification_result.is_ok(),
            "SZ check verification should pass: {:?}",
            verification_result
        );
    }

    #[test]
    fn test_multiple_exponentiations_sz_check() {
        const RATIO: usize = 1;
        let mut rng = test_rng();

        // Generate multiple random exponentiations
        let steps_vec: Vec<_> = (0..100)
            .map(|_| {
                let base = Fq12::rand(&mut rng);
                let exponent = Fr::rand(&mut rng);
                let steps = ExponentiationSteps::new(base, exponent);
                assert!(steps.verify_result());
                steps
            })
            .collect();

        // Initialize transcripts and generators
        let mut prover_transcript = Blake2bTranscript::new(b"test_sz_check_multi");
        let mut verifier_transcript = Blake2bTranscript::new(b"test_sz_check_multi");
        let hyrax_generators = PedersenGenerators::<GrumpkinProjective>::new(16, b"test sz check");

        // Generate proof
        let proof =
            sz_check_prove::<Fq, _, RATIO>(steps_vec, &mut prover_transcript, &hyrax_generators);

        // Verify proof
        let verification_result =
            sz_check_verify::<Fq, _, RATIO>(&proof, &mut verifier_transcript, &hyrax_generators);

        assert!(
            verification_result.is_ok(),
            "SZ check verification with multiple exponentiations should pass: {:?}",
            verification_result
        );
    }

    #[test]
    fn test_edge_case_exponent_zero() {
        const RATIO: usize = 1;
        let mut rng = test_rng();

        // Test with exponent = 0
        let base = Fq12::rand(&mut rng);
        let exponent = Fr::from(0u64);
        let steps = ExponentiationSteps::new(base, exponent);
        assert!(steps.verify_result());

        // Initialize transcripts and generators
        let mut prover_transcript = Blake2bTranscript::new(b"test_sz_check_zero");
        let mut verifier_transcript = Blake2bTranscript::new(b"test_sz_check_zero");
        let hyrax_generators = PedersenGenerators::<GrumpkinProjective>::new(16, b"test sz check");

        // Generate and verify proof
        let proof =
            sz_check_prove::<Fq, _, RATIO>(vec![steps], &mut prover_transcript, &hyrax_generators);

        let verification_result =
            sz_check_verify::<Fq, _, RATIO>(&proof, &mut verifier_transcript, &hyrax_generators);

        assert!(
            verification_result.is_ok(),
            "SZ check should handle zero exponent correctly"
        );
    }

    #[test]
    fn test_edge_case_exponent_one() {
        const RATIO: usize = 1;
        let mut rng = test_rng();

        // Test with exponent = 1
        let base = Fq12::rand(&mut rng);
        let exponent = Fr::from(1u64);
        let steps = ExponentiationSteps::new(base, exponent);
        assert!(steps.verify_result());

        // Initialize transcripts and generators
        let mut prover_transcript = Blake2bTranscript::new(b"test_sz_check_one");
        let mut verifier_transcript = Blake2bTranscript::new(b"test_sz_check_one");
        let hyrax_generators = PedersenGenerators::<GrumpkinProjective>::new(16, b"test sz check");

        // Generate and verify proof
        let proof =
            sz_check_prove::<Fq, _, RATIO>(vec![steps], &mut prover_transcript, &hyrax_generators);

        let verification_result =
            sz_check_verify::<Fq, _, RATIO>(&proof, &mut verifier_transcript, &hyrax_generators);

        assert!(
            verification_result.is_ok(),
            "SZ check should handle exponent = 1 correctly"
        );
    }
}
