//! Test to verify the jagged sumcheck relation with proper mapping

use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{DoryCommitmentScheme, DoryGlobals},
        },
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    },
    transcripts::{Blake2bTranscript, Transcript},
    zkvm::recursion::{
        bijection::{JaggedTransform, VarCountJaggedBijection},
        RecursionProver,
    },
};
use ark_bn254::{Fq, Fr};
use ark_ff::{UniformRand, Zero};
use ark_std::test_rng;

/// Convert index to binary representation
fn index_to_binary<F: JoltField>(idx: usize, num_vars: usize) -> Vec<F> {
    (0..num_vars)
        .map(|i| {
            if (idx >> i) & 1 == 1 {
                F::one()
            } else {
                F::zero()
            }
        })
        .collect()
}

#[test]
fn test_sumcheck_relation_with_mapping() {
    // Initialize Dory
    DoryGlobals::reset();
    DoryGlobals::initialize(1 << 2, 1 << 2);

    let mut rng = test_rng();

    // Create a test Dory proof
    let num_vars = 4;
    let poly_coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
    let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(poly_coefficients));

    let prover_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_prover(num_vars);
    let verifier_setup = <DoryCommitmentScheme as CommitmentScheme>::setup_verifier(&prover_setup);

    let (commitment, hint) =
        <DoryCommitmentScheme as CommitmentScheme>::commit(&poly, &prover_setup);

    let mut point_transcript: Blake2bTranscript = Transcript::new(b"test_point");
    let point_challenges: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
        .map(|_| point_transcript.challenge_scalar_optimized::<Fr>())
        .collect();

    let evaluation = PolynomialEvaluation::evaluate(&poly, &point_challenges);

    let mut prover_transcript: Blake2bTranscript = Transcript::new(b"dory_test_proof");
    let opening_proof = <DoryCommitmentScheme as CommitmentScheme>::prove(
        &prover_setup,
        &poly,
        &point_challenges,
        Some(hint),
        &mut prover_transcript,
    );

    let gamma = Fq::rand(&mut rng);
    let delta = Fq::rand(&mut rng);

    let mut witness_transcript: Blake2bTranscript = Transcript::new(b"dory_test_proof");

    use crate::poly::commitment::dory::wrappers::{ArkDoryProof, ArkGT};

    let ark_proof = ArkDoryProof::from(opening_proof);
    let ark_commitment = ArkGT::from(commitment);

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
    .expect("Failed to create recursion prover");

    // Get components
    let constraint_system = &prover.constraint_system;
    let (dense_poly, jagged_bijection, mapping) = constraint_system.build_dense_polynomial();

    // Create evaluation points
    let num_s_vars = constraint_system.num_s_vars();
    let num_x_vars = constraint_system.matrix.num_constraint_vars;
    let zs: Vec<Fq> = (0..num_s_vars).map(|_| Fq::rand(&mut rng)).collect();
    let zx: Vec<Fq> = (0..num_x_vars).map(|_| Fq::rand(&mut rng)).collect();

    // Compute sparse evaluation p̂(zs, zx)
    let row_size = 1 << num_x_vars;
    let mut sparse_eval = Fq::zero();
    for s_idx in 0..constraint_system.matrix.num_rows {
        for x_idx in 0..row_size {
            let flat_idx = s_idx * row_size + x_idx;
            let val = constraint_system.matrix.evaluations[flat_idx];
            if !val.is_zero() {
                let s_binary = index_to_binary::<Fq>(s_idx, num_s_vars);
                let x_binary = index_to_binary::<Fq>(x_idx, num_x_vars);
                let eq_s = EqPolynomial::mle(&s_binary, &zs);
                let eq_x = EqPolynomial::mle(&x_binary, &zx);
                sparse_eval += val * eq_s * eq_x;
            }
        }
    }

    println!("Sparse eval p̂(zs, zx) = {:?}", sparse_eval);

    // Now let's verify the sumcheck relation holds
    // The relation is: p̂(zs, zx) = ∑_{i∈{0,1}^m} q(i) · f_t(zs, zx, i)
    // where f_t(zs, zx, i) = eq(row_t(i), zs) · eq(col_t(i), zx)
    // But row_t(i) gives poly_idx, which needs to be mapped to matrix_row

    let dense_size =
        <VarCountJaggedBijection as JaggedTransform<Fq>>::dense_size(&jagged_bijection);
    let mut sumcheck_sum = Fq::zero();

    // Precompute matrix rows
    let num_polynomials = jagged_bijection.num_polynomials();
    let mut matrix_rows = Vec::with_capacity(num_polynomials);
    for poly_idx in 0..num_polynomials {
        let (constraint_idx, poly_type) = mapping.decode(poly_idx);
        let matrix_row = constraint_system
            .matrix
            .row_index(poly_type, constraint_idx);
        matrix_rows.push(matrix_row);
    }

    // Compute the sumcheck sum
    for i in 0..dense_size {
        let poly_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::row(&jagged_bijection, i);
        let eval_idx = <VarCountJaggedBijection as JaggedTransform<Fq>>::col(&jagged_bijection, i);

        // Get actual matrix row
        let matrix_row = matrix_rows[poly_idx];

        // Compute f_t(zs, zx, i) = eq(matrix_row, zs) · eq(eval_idx, zx)
        let row_binary = index_to_binary::<Fq>(matrix_row, num_s_vars);
        let col_binary = index_to_binary::<Fq>(eval_idx, num_x_vars);
        let eq_row = EqPolynomial::mle(&row_binary, &zs);
        let eq_col = EqPolynomial::mle(&col_binary, &zx);
        let f_t = eq_row * eq_col;

        // Get q(i)
        let q_i = dense_poly.Z[i];

        sumcheck_sum += q_i * f_t;
    }

    println!("Sumcheck sum = {:?}", sumcheck_sum);
    println!("Match: {}", sparse_eval == sumcheck_sum);

    // They should be equal!
    assert_eq!(
        sparse_eval, sumcheck_sum,
        "The sumcheck relation should hold: p̂(zs, zx) = ∑_i q(i) · f_t(zs, zx, i)"
    );

    // Now let's understand what happens in sumcheck
    // The sumcheck is over the polynomial h(i) = q(i) · f_t(zs, zx, i)
    // After sumcheck with challenges r_dense, we need to verify:
    // sumcheck_final_claim = ĥ(r_dense) = q̂(r_dense) · f̂_t(zs, zx, r_dense)

    let num_dense_vars = dense_poly.get_num_vars();
    let r_dense: Vec<Fq> = (0..num_dense_vars).map(|_| Fq::rand(&mut rng)).collect();

    // First, let's compute what the sumcheck polynomial h evaluates to at r_dense
    // h(i) = q(i) · f_t(zs, zx, i)
    // So ĥ(r_dense) = q̂(r_dense) · f̂_t(zs, zx, r_dense)

    // Evaluate q̂(r_dense)
    let dense_mlpoly = MultilinearPolynomial::from(dense_poly.Z.clone());
    let q_at_r: Fq = dense_mlpoly.evaluate(&r_dense);

    // Compute f̂_t(zs, zx, r_dense) using the verifier's formula
    let mut f_t_at_r = Fq::zero();
    for poly_idx in 0..jagged_bijection.num_polynomials() {
        let t_prev = jagged_bijection.cumulative_size_before(poly_idx);
        let t_curr = jagged_bijection.cumulative_size(poly_idx);

        let matrix_row = matrix_rows[poly_idx];
        let row_binary = index_to_binary::<Fq>(matrix_row, num_s_vars);
        let eq_zs_row = EqPolynomial::mle(&row_binary, &zs);

        // Sum over all valid (x_idx, dense_idx) pairs for this polynomial
        for x_idx in 0..(t_curr - t_prev) {
            let dense_idx = t_prev + x_idx;

            let x_binary = index_to_binary::<Fq>(x_idx, num_x_vars);
            let eq_zx_x = EqPolynomial::mle(&x_binary, &zx);

            let dense_binary = index_to_binary::<Fq>(dense_idx, num_dense_vars);
            let eq_r_dense = EqPolynomial::mle(&dense_binary, &r_dense);

            f_t_at_r += eq_zs_row * eq_zx_x * eq_r_dense;
        }
    }

    let sumcheck_final_claim = q_at_r * f_t_at_r;
    println!("\nAfter sumcheck:");
    println!("q̂(r_dense) = {:?}", q_at_r);
    println!("f̂_t(zs, zx, r_dense) = {:?}", f_t_at_r);
    println!(
        "Sumcheck final claim ĥ(r_dense) = {:?}",
        sumcheck_final_claim
    );

    // The sumcheck verifier would check that this final claim is consistent
    // with the univariate polynomial evaluations from the last round.
    // The key insight is that sumcheck_final_claim is NOT equal to sparse_eval,
    // but rather it's the evaluation of the summed polynomial at a random point.

    println!("\nNote: The sumcheck final claim is the evaluation of h(i) = q(i)·f_t(zs,zx,i) at r_dense,");
    println!("which is different from the original sparse polynomial evaluation p̂(zs, zx).");
}
