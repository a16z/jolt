use crate::poly::opening_proof::Openings;
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
        square_and_multiply::ExpSumcheck,
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

/// Type alias for polynomial coefficients in the BN254 base field / Grumpkin Scalar field
type PolyCoeffs = Vec<Fq>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PolynomialType {
    Rho(usize, usize),      // (exp_idx, poly_idx)
    Quotient(usize, usize), // (exp_idx, poly_idx)
    Base(usize),            // exp_idx
    G(usize),               // exp_idx
}

#[derive(Clone)]
struct ChallengeIndexer {
    rho_counts: Vec<usize>,
    quotient_counts: Vec<usize>,
    base_count: usize,
    g_count: usize,
}

impl ChallengeIndexer {
    fn new<const RATIO: usize>(artifacts: &ExpCommitments<RATIO>) -> Self {
        Self {
            rho_counts: artifacts.rho_commitments.iter().map(|v| v.len()).collect(),
            quotient_counts: artifacts
                .quotient_commitments
                .iter()
                .map(|v| v.len())
                .collect(),
            base_count: artifacts.base_commitments.len(),
            g_count: artifacts.num_exponentiations,
        }
    }

    fn get_index(&self, poly_type: PolynomialType) -> usize {
        match poly_type {
            PolynomialType::Rho(exp_idx, poly_idx) => {
                self.rho_counts[..exp_idx].iter().sum::<usize>() + poly_idx
            }
            PolynomialType::Quotient(exp_idx, poly_idx) => {
                let rho_total: usize = self.rho_counts.iter().sum();
                let quotient_offset: usize = self.quotient_counts[..exp_idx].iter().sum();
                rho_total + quotient_offset + poly_idx
            }
            PolynomialType::Base(exp_idx) => {
                let rho_total: usize = self.rho_counts.iter().sum();
                let quotient_total: usize = self.quotient_counts.iter().sum();
                rho_total + quotient_total + exp_idx
            }
            PolynomialType::G(exp_idx) => {
                let rho_total: usize = self.rho_counts.iter().sum();
                let quotient_total: usize = self.quotient_counts.iter().sum();
                rho_total + quotient_total + self.base_count + exp_idx
            }
        }
    }
}

/// Square and Multiply commitment data over grumpkin
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ExpCommitments<const RATIO: usize = 1> {
    pub rho_commitments: Vec<Vec<HyraxCommitment<RATIO, GrumpkinProjective>>>,
    pub quotient_commitments: Vec<Vec<HyraxCommitment<RATIO, GrumpkinProjective>>>,
    pub base_commitments: Vec<HyraxCommitment<RATIO, GrumpkinProjective>>,
    pub num_exponentiations: usize,
    pub num_constraints_per_exponentiation: Vec<usize>,
    pub bits_per_exponentiation: Vec<Vec<bool>>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct RecursionProof<F: JoltField, ProofTranscript: Transcript, const RATIO: usize> {
    pub commitments: ExpCommitments<RATIO>,
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub r_sumcheck: Vec<F>,
    pub hyrax_proof: HyraxOpeningProof<RATIO, GrumpkinProjective>,
    pub openings: Openings<F>,
}

#[tracing::instrument(skip_all, name = "append_commitments_to_transcript")]
fn append_commitments_to_transcript<ProofTranscript: Transcript, const RATIO: usize>(
    transcript: &mut ProofTranscript,
    rho_commitments: &[Vec<HyraxCommitment<RATIO, GrumpkinProjective>>],
    quotient_commitments: &[Vec<HyraxCommitment<RATIO, GrumpkinProjective>>],
    base_commitments: &[HyraxCommitment<RATIO, GrumpkinProjective>],
) {
    for commitments in rho_commitments
        .iter()
        .chain(quotient_commitments.iter())
        .flat_map(|v| v.iter())
        .chain(base_commitments.iter())
    {
        transcript.append_serializable(commitments);
    }
}

#[tracing::instrument(skip_all, name = "batch_polynomials_with_challenges")]
fn batch_polynomials_with_challenges<'a, F: JoltField + From<Fq> + Into<Fq>>(
    batched_poly: &mut Vec<F>,
    batched_eval: &mut F,
    polynomials: impl Iterator<Item = (PolynomialType, &'a PolyCoeffs, F, F)>, // (type, poly, eval, challenge)
) {
    for (_, poly, eval, challenge) in polynomials {
        *batched_eval += challenge * eval;
        for (idx, &coeff) in poly.iter().enumerate() {
            batched_poly[idx] += challenge * F::from(coeff);
        }
    }
}

fn get_polynomial_evaluation<F: JoltField>(
    accumulator: &Rc<RefCell<ProverOpeningAccumulator<F>>>,
    poly_type: PolynomialType,
) -> F {
    let poly_id = match poly_type {
        PolynomialType::Rho(exp_idx, poly_idx) => {
            RecursionCommittedPolynomial::RecursionRho(exp_idx, poly_idx)
        }
        PolynomialType::Quotient(exp_idx, poly_idx) => {
            RecursionCommittedPolynomial::RecursionQuotient(exp_idx, poly_idx)
        }
        PolynomialType::Base(exp_idx) => RecursionCommittedPolynomial::RecursionBase(exp_idx),
        PolynomialType::G(exp_idx) => RecursionCommittedPolynomial::RecursionG(exp_idx),
    };
    accumulator
        .borrow()
        .get_recursion_polynomial_opening(poly_id, SumcheckId::RecursionCheck)
        .1
}

#[tracing::instrument(skip_all, name = "accumulate_commitments")]
fn accumulate_commitments<'a, F: JoltField + From<Fq> + Into<Fq>, const RATIO: usize>(
    batched_row_commitments: &mut Vec<GrumpkinProjective>,
    commitments: impl Iterator<Item = &'a HyraxCommitment<RATIO, GrumpkinProjective>>,
    challenges: impl Iterator<Item = F>,
) {
    for (commitment, challenge) in commitments.zip(challenges) {
        let gamma_fq: Fq = challenge.into();
        for (i, &com) in commitment.row_commitments.iter().enumerate() {
            batched_row_commitments[i] += com * gamma_fq;
        }
    }
}

/// Proves the G_T exponetiations from dory
///
/// # Protocol:
/// 1. Commit to all polynomials (rho, quotient, base)
/// 2. Run Exp sumcheck protocol to reduce to opening claims
/// 3. Batch all opening claims with random challenges
/// 4. Generate Hyrax opening proof for the batched polynomial
#[tracing::instrument(skip_all, name = "snark_composition_prove")]
pub fn snark_composition_prove<F, ProofTranscript, const RATIO: usize>(
    exponentiation_steps_vec: Vec<ExponentiationSteps>,
    transcript: &mut ProofTranscript,
    hyrax_generators: &PedersenGenerators<GrumpkinProjective>,
) -> RecursionProof<F, ProofTranscript, RATIO>
where
    F: JoltField + From<Fq> + Into<Fq>,
    ProofTranscript: Transcript,
{
    tracing::info!(
        num_exponentiations = exponentiation_steps_vec.len(),
        "SNARK composition proving"
    );

    // Step 1: Prepare polynomials and commitments
    tracing::debug!("Preparing polynomial commitments");

    let num_exponentiations = exponentiation_steps_vec.len();
    let mut rho_commitments = Vec::with_capacity(num_exponentiations);
    let mut quotient_commitments = Vec::with_capacity(num_exponentiations);
    let mut base_commitments = Vec::with_capacity(num_exponentiations);

    let mut all_rho_polys: Vec<Vec<PolyCoeffs>> = Vec::with_capacity(num_exponentiations);
    let mut all_quotient_polys: Vec<Vec<PolyCoeffs>> = Vec::with_capacity(num_exponentiations);
    let mut all_base_polys: Vec<PolyCoeffs> = Vec::with_capacity(num_exponentiations);

    // First convert all base polynomials and store them
    for steps in &exponentiation_steps_vec {
        let base_poly = jolt_optimizations::fq12_to_multilinear_evals(&steps.base);
        all_base_polys.push(base_poly);
        all_rho_polys.push(steps.rho_mles.clone());
        all_quotient_polys.push(steps.quotient_mles.clone());
    }

    // Prepare all polynomials for batch commitment
    let mut all_polys_to_commit: Vec<&[Fq]> = Vec::new();
    let mut poly_counts = Vec::with_capacity(exponentiation_steps_vec.len());

    for i in 0..exponentiation_steps_vec.len() {
        let rho_count = all_rho_polys[i].len();
        for rho_poly in &all_rho_polys[i] {
            all_polys_to_commit.push(rho_poly);
        }

        let quotient_count = all_quotient_polys[i].len();
        for quotient_poly in &all_quotient_polys[i] {
            all_polys_to_commit.push(quotient_poly);
        }

        all_polys_to_commit.push(&all_base_polys[i]);
        poly_counts.push((rho_count, quotient_count));
    }

    // Batch commit all polynomials at once
    tracing::debug!(
        total_polynomials = all_polys_to_commit.len(),
        "Batch committing polynomials"
    );
    let all_commitments = HyraxCommitment::<RATIO, GrumpkinProjective>::batch_commit(
        &all_polys_to_commit,
        hyrax_generators,
    );

    // Reorganize commitments back into the expected structure
    let mut commitment_idx = 0;
    for (rho_count, quotient_count) in poly_counts {
        let rho_comms = all_commitments[commitment_idx..commitment_idx + rho_count].to_vec();
        rho_commitments.push(rho_comms);
        commitment_idx += rho_count;

        let quotient_comms =
            all_commitments[commitment_idx..commitment_idx + quotient_count].to_vec();
        quotient_commitments.push(quotient_comms);
        commitment_idx += quotient_count;

        base_commitments.push(all_commitments[commitment_idx].clone());
        commitment_idx += 1;
    }

    append_commitments_to_transcript(
        transcript,
        &rho_commitments,
        &quotient_commitments,
        &base_commitments,
    );

    let artifacts = ExpCommitments {
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
    tracing::debug!(
        num_instances = num_exponentiations,
        "Creating exponentiation sumcheck instances"
    );
    let mut sumcheck_instances: Vec<Box<dyn SumcheckInstance<F>>> = exponentiation_steps_vec
        .iter()
        .enumerate()
        .map(|(exp_idx, steps)| {
            let r: Vec<F> = transcript.challenge_vector(4);
            let gamma: F = transcript.challenge_scalar();
            Box::new(ExpSumcheck::new_prover(exp_idx, steps, r, gamma))
                as Box<dyn SumcheckInstance<F>>
        })
        .collect();

    let prover_accumulator = Rc::new(RefCell::new(ProverOpeningAccumulator::<F>::new()));
    let sumcheck_instances_mut: Vec<&mut dyn SumcheckInstance<F>> = sumcheck_instances
        .iter_mut()
        .map(|instance| &mut **instance as &mut dyn SumcheckInstance<F>)
        .collect();

    tracing::debug!("Running batched sumcheck protocol");
    let (sumcheck_proof, r_sumcheck) = BatchedSumcheck::prove(
        sumcheck_instances_mut,
        Some(prover_accumulator.clone()),
        transcript,
    );

    // Step 3: Batch all polynomials for Hyrax opening
    let indexer = ChallengeIndexer::new(&artifacts);
    let total_polys = all_rho_polys.iter().map(|v| v.len()).sum::<usize>()
        + all_quotient_polys.iter().map(|v| v.len()).sum::<usize>()
        + all_base_polys.len()
        + num_exponentiations;

    tracing::debug!(total_polys, "Batching polynomials for opening proof");
    let batching_challenges: Vec<F> = transcript.challenge_vector(total_polys);

    let mut batched_poly = vec![F::zero(); 16]; // 2^4 = 16 for 4 variables
    let mut batched_eval = F::zero();

    let g_mle = jolt_optimizations::witness_gen::get_g_mle();

    let poly_refs: Vec<(PolynomialType, &PolyCoeffs)> = all_rho_polys
        .iter()
        .enumerate()
        .flat_map(|(exp_idx, polys)| {
            polys.iter().enumerate().map(move |(poly_idx, poly)| {
                (PolynomialType::Rho(exp_idx, poly_idx), poly as &PolyCoeffs)
            })
        })
        .chain(
            all_quotient_polys
                .iter()
                .enumerate()
                .flat_map(|(exp_idx, polys)| {
                    polys.iter().enumerate().map(move |(poly_idx, poly)| {
                        (
                            PolynomialType::Quotient(exp_idx, poly_idx),
                            poly as &PolyCoeffs,
                        )
                    })
                }),
        )
        .chain(
            all_base_polys
                .iter()
                .enumerate()
                .map(|(exp_idx, poly)| (PolynomialType::Base(exp_idx), poly as &PolyCoeffs)),
        )
        .chain(
            (0..num_exponentiations)
                .map(|exp_idx| (PolynomialType::G(exp_idx), &g_mle as &PolyCoeffs)),
        )
        .collect();

    let all_poly_data = poly_refs.into_iter().map(|(poly_type, poly)| {
        let eval = get_polynomial_evaluation(&prover_accumulator, poly_type);
        let challenge = batching_challenges[indexer.get_index(poly_type)];
        (poly_type, poly, eval, challenge)
    });

    batch_polynomials_with_challenges(&mut batched_poly, &mut batched_eval, all_poly_data);

    // Exp sumcheck binds low to high so we reverse
    let r_sumcheck_fq: Vec<Fq> = r_sumcheck.iter().rev().map(|&x| x.into()).collect();
    let batched_poly_fq: Vec<Fq> = batched_poly.iter().map(|&x| x.into()).collect();
    let batched_dense_poly = DensePolynomial::new(batched_poly_fq);

    tracing::debug!("Generating Hyrax opening proof");
    let batched_hyrax_proof = HyraxOpeningProof::<RATIO, GrumpkinProjective>::prove(
        &batched_dense_poly,
        &r_sumcheck_fq,
        RATIO,
    );

    let openings = prover_accumulator.borrow().evaluation_openings().clone();

    tracing::info!("SNARK composition proof generated");

    RecursionProof {
        commitments: artifacts,
        sumcheck_proof,
        r_sumcheck,
        hyrax_proof: batched_hyrax_proof,
        openings,
    }
}

/// Verifies the SNARK Composition protocol
#[tracing::instrument(skip_all, name = "snark_composition_verify")]
pub fn snark_composition_verify<F, ProofTranscript, const RATIO: usize>(
    proof: &RecursionProof<F, ProofTranscript, RATIO>,
    transcript: &mut ProofTranscript,
    hyrax_generators: &PedersenGenerators<GrumpkinProjective>,
) -> Result<(), ProofVerifyError>
where
    F: JoltField + From<Fq> + Into<Fq>,
    ProofTranscript: Transcript,
{
    let artifacts = &proof.commitments;
    tracing::info!(
        num_exponentiations = artifacts.num_exponentiations,
        "SNARK composition verification"
    );

    append_commitments_to_transcript(
        transcript,
        &artifacts.rho_commitments,
        &artifacts.quotient_commitments,
        &artifacts.base_commitments,
    );

    let verifier_sumcheck_instances: Vec<Box<dyn SumcheckInstance<F>>> = (0..artifacts
        .num_exponentiations)
        .map(|i| {
            let r: Vec<F> = transcript.challenge_vector(4);
            let gamma: F = transcript.challenge_scalar();
            Box::new(ExpSumcheck::new_verifier(
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

    let verifier_accumulator = Rc::new(RefCell::new(VerifierOpeningAccumulator::<F>::new()));

    for (opening_id, (point, claim)) in proof.openings.iter() {
        verifier_accumulator
            .borrow_mut()
            .openings_mut()
            .insert(opening_id.clone(), (point.clone(), *claim));
    }

    tracing::debug!("Verifying batched sumcheck");
    let verified_r = BatchedSumcheck::verify(
        &proof.sumcheck_proof,
        verifier_instances_ref,
        Some(verifier_accumulator.clone()),
        transcript,
    )?;

    if verified_r != proof.r_sumcheck {
        tracing::error!("Sumcheck randomness mismatch");
        return Err(ProofVerifyError::InternalError);
    }

    let indexer = ChallengeIndexer::new(artifacts);
    let total_polys = indexer.rho_counts.iter().sum::<usize>()
        + indexer.quotient_counts.iter().sum::<usize>()
        + indexer.base_count
        + indexer.g_count;

    let batching_challenges: Vec<F> = transcript.challenge_vector(total_polys);

    // Compute batched commitment homomorphically
    let (L_size, _) = crate::poly::commitment::hyrax::matrix_dimensions(4, RATIO);
    let mut batched_row_commitments = vec![GrumpkinProjective::zero(); L_size];

    let mut rho_challenges = Vec::new();
    for (exp_idx, rho_vec) in artifacts.rho_commitments.iter().enumerate() {
        for (poly_idx, _) in rho_vec.iter().enumerate() {
            rho_challenges.push(
                batching_challenges[indexer.get_index(PolynomialType::Rho(exp_idx, poly_idx))],
            );
        }
    }

    let mut quotient_challenges = Vec::new();
    for (exp_idx, quot_vec) in artifacts.quotient_commitments.iter().enumerate() {
        for (poly_idx, _) in quot_vec.iter().enumerate() {
            quotient_challenges.push(
                batching_challenges[indexer.get_index(PolynomialType::Quotient(exp_idx, poly_idx))],
            );
        }
    }

    let base_challenges: Vec<F> = artifacts
        .base_commitments
        .iter()
        .enumerate()
        .map(|(exp_idx, _)| batching_challenges[indexer.get_index(PolynomialType::Base(exp_idx))])
        .collect();

    accumulate_commitments(
        &mut batched_row_commitments,
        artifacts.rho_commitments.iter().flat_map(|v| v.iter()),
        rho_challenges.into_iter(),
    );

    accumulate_commitments(
        &mut batched_row_commitments,
        artifacts.quotient_commitments.iter().flat_map(|v| v.iter()),
        quotient_challenges.into_iter(),
    );

    accumulate_commitments(
        &mut batched_row_commitments,
        artifacts.base_commitments.iter(),
        base_challenges.into_iter(),
    );

    let g_mle = jolt_optimizations::witness_gen::get_g_mle();
    let g_poly = DensePolynomial::new(g_mle);
    let g_commitment =
        HyraxCommitment::<RATIO, GrumpkinProjective>::commit(&g_poly, hyrax_generators);

    let g_challenges = (0..artifacts.num_exponentiations)
        .map(|exp_idx| batching_challenges[indexer.get_index(PolynomialType::G(exp_idx))]);

    for challenge in g_challenges {
        let gamma_fq: Fq = challenge.into();
        for (i, &com) in g_commitment.row_commitments.iter().enumerate() {
            batched_row_commitments[i] += com * gamma_fq;
        }
    }

    let batched_hyrax_commitment = HyraxCommitment {
        row_commitments: batched_row_commitments,
    };

    let mut batched_opening_fq = Fq::zero();
    for (opening_id, (_, claim)) in proof.openings.iter() {
        if let crate::poly::opening_proof::OpeningId::Recursion(poly_id, _) = opening_id {
            let poly_type = match poly_id {
                RecursionCommittedPolynomial::RecursionRho(exp_idx, poly_idx) => {
                    PolynomialType::Rho(*exp_idx, *poly_idx)
                }
                RecursionCommittedPolynomial::RecursionQuotient(exp_idx, poly_idx) => {
                    PolynomialType::Quotient(*exp_idx, *poly_idx)
                }
                RecursionCommittedPolynomial::RecursionBase(exp_idx) => {
                    PolynomialType::Base(*exp_idx)
                }
                RecursionCommittedPolynomial::RecursionG(exp_idx) => PolynomialType::G(*exp_idx),
            };

            let challenge_idx = indexer.get_index(poly_type);
            let gamma_fq: Fq = batching_challenges[challenge_idx].into();
            batched_opening_fq += gamma_fq * (*claim).into();
        }
    }

    // Reverse and cast due to LowToHigh binding
    let r_sumcheck_fq: Vec<Fq> = proof.r_sumcheck.iter().rev().map(|&x| x.into()).collect();

    proof.hyrax_proof.verify(
        hyrax_generators,
        &r_sumcheck_fq,
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
    fn test_single_exponentiation_recursion_check() {
        const RATIO: usize = 1;
        let mut rng = test_rng();

        let base = Fq12::rand(&mut rng);
        let exponent = Fr::rand(&mut rng);
        let steps = ExponentiationSteps::new(base, exponent);
        assert!(
            steps.verify_result(),
            "ExponentiationSteps should verify correctly"
        );

        let mut prover_transcript = Blake2bTranscript::new(b"test_exp_check");
        let mut verifier_transcript = Blake2bTranscript::new(b"test_exp_check");
        let hyrax_generators =
            PedersenGenerators::<GrumpkinProjective>::new(16, b"recursion check");

        let proof = snark_composition_prove::<Fq, _, RATIO>(
            vec![steps],
            &mut prover_transcript,
            &hyrax_generators,
        );

        let verification_result = snark_composition_verify::<Fq, _, RATIO>(
            &proof,
            &mut verifier_transcript,
            &hyrax_generators,
        );

        assert!(
            verification_result.is_ok(),
            "recursion check verification should pass: {:?}",
            verification_result
        );
    }

    #[test]
    fn test_multiple_exponentiations_recursion_check() {
        const RATIO: usize = 1;
        let mut rng = test_rng();

        let steps_vec: Vec<_> = (0..100)
            .map(|_| {
                let base = Fq12::rand(&mut rng);
                let exponent = Fr::rand(&mut rng);
                let steps = ExponentiationSteps::new(base, exponent);
                assert!(steps.verify_result());
                steps
            })
            .collect();

        let mut prover_transcript = Blake2bTranscript::new(b"test_exp_check_multi");
        let mut verifier_transcript = Blake2bTranscript::new(b"test_exp_check_multi");
        let hyrax_generators =
            PedersenGenerators::<GrumpkinProjective>::new(16, b"recursion check");

        let proof = snark_composition_prove::<Fq, _, RATIO>(
            steps_vec,
            &mut prover_transcript,
            &hyrax_generators,
        );

        let verification_result = snark_composition_verify::<Fq, _, RATIO>(
            &proof,
            &mut verifier_transcript,
            &hyrax_generators,
        );

        assert!(
            verification_result.is_ok(),
            "recursion check verification with multiple exponentiations should pass: {:?}",
            verification_result
        );
    }
}
