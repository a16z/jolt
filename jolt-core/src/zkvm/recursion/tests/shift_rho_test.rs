//! Tests for the EqPlusOne shift relation used by the ShiftRho sumcheck.

use ark_bn254::{Fq, Fr};
use ark_ff::{UniformRand, Zero};
use ark_std::test_rng;
use dory::backends::arkworks::ArkGT;
use serial_test::serial;

use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{DoryCommitmentScheme, DoryGlobals},
        },
        dense_mlpoly::DensePolynomial,
        eq_plus_one_poly::EqPlusOnePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::{Blake2bTranscript, Transcript},
    zkvm::recursion::stage1::packed_gt_exp::{NUM_ELEMENT_VARS, NUM_STEP_VARS},
    zkvm::recursion::stage2::shift_rho::{eq_lsb_evals, eq_plus_one_lsb_evals},
    zkvm::recursion::RecursionProver,
};

#[test]
fn test_eq_plus_one_shift_relation() {
    type F = Fr;
    type C = <F as JoltField>::Challenge;

    let mut rng = test_rng();

    let num_step_vars = 2usize;
    let num_elem_vars = 1usize;
    let step_size = 1 << num_step_vars;
    let elem_size = 1 << num_elem_vars;
    let total_size = 1 << (num_step_vars + num_elem_vars);

    // rho(s,x) table in big-endian index order over [s_bits..., x_bits...]
    let mut rho_evals = vec![F::zero(); total_size];
    for s_idx in 0..step_size {
        for x_idx in 0..elem_size {
            let idx = (s_idx << num_elem_vars) | x_idx;
            rho_evals[idx] = F::rand(&mut rng);
        }
    }

    // rho_next(s,x) = rho(s+1,x), with rho_next(max_s, x) = 0
    let mut rho_next_evals = vec![F::zero(); total_size];
    for s_idx in 0..step_size {
        for x_idx in 0..elem_size {
            let idx = (s_idx << num_elem_vars) | x_idx;
            if s_idx + 1 < step_size {
                let src_idx = ((s_idx + 1) << num_elem_vars) | x_idx;
                rho_next_evals[idx] = rho_evals[src_idx];
            }
        }
    }

    // Check the identity for a few random points.
    for _ in 0..5 {
        let r_s: Vec<C> = (0..num_step_vars)
            .map(|_| <C as UniformRand>::rand(&mut rng))
            .collect();
        let r_x: Vec<C> = (0..num_elem_vars)
            .map(|_| <C as UniformRand>::rand(&mut rng))
            .collect();

        let eq_plus_one = EqPlusOnePolynomial::<F>::evals(&r_s, None).1;
        let eq_x = EqPolynomial::<F>::evals(&r_x);

        let mut lhs = F::zero();
        for s_idx in 0..step_size {
            for x_idx in 0..elem_size {
                let idx = (s_idx << num_elem_vars) | x_idx;
                lhs += eq_plus_one[s_idx] * eq_x[x_idx] * rho_evals[idx];
            }
        }

        let r_all: Vec<C> = r_s.iter().chain(r_x.iter()).cloned().collect();
        let eq_all = EqPolynomial::<F>::evals(&r_all);
        let rhs: F = eq_all
            .iter()
            .zip(rho_next_evals.iter())
            .map(|(eq, rho_next)| *eq * *rho_next)
            .sum();

        assert_eq!(lhs, rhs);
    }
}

#[test]
#[serial]
fn test_packed_gt_rho_next_relation_from_dory() {
    use crate::poly::commitment::dory::wrappers::ArkDoryProof;

    DoryGlobals::reset();
    DoryGlobals::initialize(1 << 2, 1 << 2);

    let mut rng = test_rng();

    // Build a small Dory proof to generate real packed GT witnesses.
    let num_vars = 4;
    let poly_coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly_coefficients));

    let prover_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover(num_vars);
    let verifier_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_verifier(&prover_setup);
    let (commitment, hint) =
        <DoryCommitmentScheme as CommitmentScheme>::commit(&poly, &prover_setup);

    let mut point_transcript: Blake2bTranscript = Transcript::new(b"shift_rho_test_point");
    let point_challenges: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
        .map(|_| point_transcript.challenge_scalar_optimized::<Fr>())
        .collect();
    let evaluation = PolynomialEvaluation::evaluate(&poly, &point_challenges);

    let mut prover_transcript: Blake2bTranscript = Transcript::new(b"shift_rho_test_proof");
    let opening_proof = <DoryCommitmentScheme as CommitmentScheme>::prove(
        &prover_setup,
        &poly,
        &point_challenges,
        Some(hint),
        &mut prover_transcript,
    );

    let ark_proof = ArkDoryProof::from(opening_proof);
    let ark_commitment = ArkGT::from(commitment);

    let gamma = Fq::rand(&mut rng);
    let delta = Fq::rand(&mut rng);

    let mut witness_transcript: Blake2bTranscript = Transcript::new(b"shift_rho_test_proof");
    let prover = RecursionProver::<Fq>::new_from_dory_proof(
        &ark_proof,
        &verifier_setup,
        &mut witness_transcript,
        &point_challenges,
        &evaluation,
        &ark_commitment,
        gamma,
        delta,
    )
    .expect("Failed to build recursion prover");

    let elem_size = 1 << NUM_ELEMENT_VARS;
    let step_size = 1 << NUM_STEP_VARS;

    for (witness_idx, witness) in prover
        .constraint_system
        .packed_gt_exp_witnesses
        .iter()
        .enumerate()
    {
        for _ in 0..3 {
            let r_s: Vec<<Fq as JoltField>::Challenge> = (0..NUM_STEP_VARS)
                .map(|_| <Fq as JoltField>::Challenge::rand(&mut rng))
                .collect();
            let r_x: Vec<<Fq as JoltField>::Challenge> = (0..NUM_ELEMENT_VARS)
                .map(|_| <Fq as JoltField>::Challenge::rand(&mut rng))
                .collect();

            let eq_plus_one = eq_plus_one_lsb_evals::<Fq>(&r_s);
            let eq_s = eq_lsb_evals::<Fq>(&r_s);
            let eq_x = eq_lsb_evals::<Fq>(&r_x);

            let mut sum = Fq::zero();
            let mut sum_next = Fq::zero();
            for x in 0..elem_size {
                let base = x * step_size;
                let eq_x_val = eq_x[x];
                for s in 0..step_size {
                    let idx = base + s;
                    sum += eq_plus_one[s] * eq_x_val * witness.rho_packed[idx];
                    sum_next += eq_s[s] * eq_x_val * witness.rho_next_packed[idx];
                }
            }

            assert_eq!(
                sum, sum_next,
                "PackedGtExp rho_next shift relation failed for witness {witness_idx}"
            );
        }
    }
}
