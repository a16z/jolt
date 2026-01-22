//! Tests for the Jagged Assist batch MLE verification (Theorem 1.5)
//!
//! These tests verify that the Jagged Assist prover and verifier correctly implement
//! the batch verification sumcheck for branching program evaluations.

use crate::{
    field::JoltField,
    poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::{Blake2bTranscript, Transcript},
    zkvm::recursion::{
        bijection::{JaggedPolynomial, JaggedTransform, VarCountJaggedBijection},
        stage3::{
            branching_program::{JaggedBranchingProgram, Point},
            jagged_assist::{JaggedAssistProver, JaggedAssistVerifier},
        },
    },
};
use ark_bn254::Fq;
use ark_ff::UniformRand;
use ark_std::{test_rng, One, Zero};

/// Create a test bijection with given polynomial sizes (must be powers of 2)
fn create_test_bijection(sizes: &[usize]) -> VarCountJaggedBijection {
    let polynomials: Vec<JaggedPolynomial> = sizes
        .iter()
        .map(|&size| {
            assert!(
                size > 0 && size.is_power_of_two(),
                "Test sizes must be powers of 2"
            );
            let num_vars = size.trailing_zeros() as usize;
            JaggedPolynomial::new(num_vars)
        })
        .collect();
    VarCountJaggedBijection::new(polynomials)
}

/// Test that the prover correctly computes claimed evaluations
#[test]
fn test_claimed_evaluations_correctness() {
    let mut rng = test_rng();
    let num_bits = 3;

    // Create random challenge points
    let r_x: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let r_dense: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();

    // Create bijection with power-of-2 sizes: [2, 2, 2, 2] -> cumulative [2, 4, 6, 8]
    let bijection = create_test_bijection(&[2, 2, 2, 2]);
    let num_polynomials = bijection.num_polynomials();

    let mut transcript: Blake2bTranscript = Transcript::new(b"test_claimed_evaluations");

    let prover = JaggedAssistProver::<Fq, Blake2bTranscript>::new(
        r_x.clone(),
        r_dense.clone(),
        &bijection,
        num_bits,
        &mut transcript,
    );

    // Verify each claimed evaluation by direct computation
    let branching_program = JaggedBranchingProgram::new(num_bits);
    let za = Point::from(r_x.clone());
    let zb = Point::from(r_dense.clone());

    for k in 0..num_polynomials {
        let t_prev = bijection.cumulative_size_before(k);
        let t_curr = bijection.cumulative_size(k);

        let zc = Point::from_usize(t_prev, num_bits);
        let zd = Point::from_usize(t_curr, num_bits);

        let expected = branching_program.eval_multilinear(&za, &zb, &zc, &zd);
        let actual = prover.claimed_evaluations[k];

        assert_eq!(
            expected, actual,
            "Claimed evaluation mismatch for polynomial {}",
            k
        );
    }
}

/// Test that prover and verifier compute the same input claim
#[test]
fn test_input_claim_consistency() {
    let mut rng = test_rng();
    let num_bits = 3;

    let r_x: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let r_dense: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let bijection = create_test_bijection(&[1, 2, 1, 4]);

    // Create prover
    let mut prover_transcript: Blake2bTranscript = Transcript::new(b"test_input_claim");
    let prover = JaggedAssistProver::<Fq, Blake2bTranscript>::new(
        r_x.clone(),
        r_dense.clone(),
        &bijection,
        num_bits,
        &mut prover_transcript,
    );

    // Create verifier with same transcript state
    let mut verifier_transcript: Blake2bTranscript = Transcript::new(b"test_input_claim");
    let verifier = JaggedAssistVerifier::<Fq, Blake2bTranscript>::new(
        prover.claimed_evaluations.clone(),
        r_x,
        r_dense,
        &bijection,
        num_bits,
        &mut verifier_transcript,
    );

    // Create dummy accumulators
    let prover_accumulator = ProverOpeningAccumulator::new(10);
    let verifier_accumulator = VerifierOpeningAccumulator::new(10);

    let prover_input = prover.input_claim(&prover_accumulator);
    let verifier_input = verifier.input_claim(&verifier_accumulator);

    assert_eq!(
        prover_input, verifier_input,
        "Prover and verifier input claims must match"
    );
}

/// Test the sumcheck polynomial degree
#[test]
fn test_sumcheck_degree() {
    let mut rng = test_rng();
    let num_bits = 2;

    let r_x: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let r_dense: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let bijection = create_test_bijection(&[1, 1, 1]);

    let mut transcript: Blake2bTranscript = Transcript::new(b"test_degree");
    let prover = JaggedAssistProver::<Fq, Blake2bTranscript>::new(
        r_x,
        r_dense,
        &bijection,
        num_bits,
        &mut transcript,
    );

    // Degree should be 2 (product of g and eq, both multilinear)
    assert_eq!(
        prover.degree(),
        2,
        "Sumcheck polynomial should have degree 2"
    );
}

/// Test that the sumcheck round messages sum correctly
#[test]
fn test_sumcheck_round_sum_property() {
    let mut rng = test_rng();
    let num_bits = 2;

    let r_x: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let r_dense: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let bijection = create_test_bijection(&[1, 1, 1]);

    let mut transcript: Blake2bTranscript = Transcript::new(b"test_round_sum");
    let mut prover = JaggedAssistProver::<Fq, Blake2bTranscript>::new(
        r_x,
        r_dense,
        &bijection,
        num_bits,
        &mut transcript,
    );

    let prover_accumulator = ProverOpeningAccumulator::new(10);
    let mut current_claim = prover.input_claim(&prover_accumulator);

    // Run through all rounds and verify h(0) + h(1) = previous_claim
    let num_rounds = prover.num_rounds();
    for round in 0..num_rounds {
        let poly = prover.compute_message(round, current_claim);

        // Verify: h(0) + h(1) = current_claim
        let h_0 = poly.evaluate(&Fq::zero());
        let h_1 = poly.evaluate(&Fq::one());
        let sum = h_0 + h_1;

        assert_eq!(
            sum, current_claim,
            "Round {}: h(0) + h(1) = {} != current_claim = {}",
            round, sum, current_claim
        );

        // Sample random challenge and update
        let challenge: Fq = Fq::rand(&mut rng);
        current_claim = poly.evaluate(&challenge);
        prover.ingest_challenge(challenge.into(), round);
    }
}

/// Test full prover-verifier interaction
#[test]
fn test_prover_verifier_consistency() {
    let mut rng = test_rng();
    let num_bits = 3;

    let r_x: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let r_dense: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let bijection = create_test_bijection(&[2, 2, 1, 2]);

    // Create prover
    let mut prover_transcript: Blake2bTranscript = Transcript::new(b"test_pv_consistency");
    let mut prover = JaggedAssistProver::<Fq, Blake2bTranscript>::new(
        r_x.clone(),
        r_dense.clone(),
        &bijection,
        num_bits,
        &mut prover_transcript,
    );

    // Create verifier
    let mut verifier_transcript: Blake2bTranscript = Transcript::new(b"test_pv_consistency");
    let verifier = JaggedAssistVerifier::<Fq, Blake2bTranscript>::new(
        prover.claimed_evaluations.clone(),
        r_x,
        r_dense,
        &bijection,
        num_bits,
        &mut verifier_transcript,
    );

    // Run sumcheck
    let prover_accumulator = ProverOpeningAccumulator::new(10);
    let verifier_accumulator = VerifierOpeningAccumulator::new(10);

    let mut current_claim = prover.input_claim(&prover_accumulator);
    let verifier_input_claim = verifier.input_claim(&verifier_accumulator);
    assert_eq!(current_claim, verifier_input_claim);

    let mut challenges: Vec<<Fq as JoltField>::Challenge> = Vec::new();
    let num_rounds = prover.num_rounds();

    for round in 0..num_rounds {
        let poly = prover.compute_message(round, current_claim);

        // Verify round sum
        let h_0 = poly.evaluate(&Fq::zero());
        let h_1 = poly.evaluate(&Fq::one());
        assert_eq!(h_0 + h_1, current_claim, "Round {} sum check failed", round);

        // Sample challenge (using deterministic rng for reproducibility)
        let challenge: Fq = Fq::rand(&mut rng);
        challenges.push(challenge.into());

        current_claim = poly.evaluate(&challenge);
        prover.ingest_challenge(challenge.into(), round);
    }

    // Verify final claim matches verifier's expected output
    let expected_output = verifier.expected_output_claim(&verifier_accumulator, &challenges);

    assert_eq!(
        current_claim, expected_output,
        "Final claim {} != expected output {}",
        current_claim, expected_output
    );
}

/// Test with edge case: single polynomial
#[test]
fn test_single_polynomial() {
    let mut rng = test_rng();
    let num_bits = 3;

    let r_x: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let r_dense: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let bijection = create_test_bijection(&[4]); // Single polynomial of size 4

    let mut prover_transcript: Blake2bTranscript = Transcript::new(b"test_single_poly");
    let mut prover = JaggedAssistProver::<Fq, Blake2bTranscript>::new(
        r_x.clone(),
        r_dense.clone(),
        &bijection,
        num_bits,
        &mut prover_transcript,
    );

    let mut verifier_transcript: Blake2bTranscript = Transcript::new(b"test_single_poly");
    let verifier = JaggedAssistVerifier::<Fq, Blake2bTranscript>::new(
        prover.claimed_evaluations.clone(),
        r_x,
        r_dense,
        &bijection,
        num_bits,
        &mut verifier_transcript,
    );

    // Run sumcheck
    let prover_accumulator = ProverOpeningAccumulator::new(10);
    let verifier_accumulator = VerifierOpeningAccumulator::new(10);

    let mut current_claim = prover.input_claim(&prover_accumulator);
    let mut challenges: Vec<<Fq as JoltField>::Challenge> = Vec::new();

    for round in 0..prover.num_rounds() {
        let poly = prover.compute_message(round, current_claim);

        let h_0 = poly.evaluate(&Fq::zero());
        let h_1 = poly.evaluate(&Fq::one());
        assert_eq!(h_0 + h_1, current_claim, "Round {} sum check failed", round);

        let challenge: Fq = Fq::rand(&mut rng);
        challenges.push(challenge.into());
        current_claim = poly.evaluate(&challenge);
        prover.ingest_challenge(challenge.into(), round);
    }

    let expected_output = verifier.expected_output_claim(&verifier_accumulator, &challenges);
    assert_eq!(current_claim, expected_output);
}

/// Test with larger number of polynomials
#[test]
fn test_many_polynomials() {
    let mut rng = test_rng();

    // With num_bits = 5, we can represent values up to 31 (2^5 - 1)
    // This allows us to have more polynomials with larger cumulative sizes
    let num_bits = 5;
    let num_polynomials = 8;

    let r_x: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let r_dense: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();

    // Create sizes that sum to 16: [2, 2, 2, 2, 2, 2, 2, 2] = 16 total
    // Cumulative sizes will be: [2, 4, 6, 8, 10, 12, 14, 16]
    // All fit within 5 bits (max value 31)
    let sizes: Vec<usize> = vec![2; num_polynomials];
    let bijection = create_test_bijection(&sizes);
    let r_dense: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();

    let mut prover_transcript: Blake2bTranscript = Transcript::new(b"test_many_polys");
    let mut prover = JaggedAssistProver::<Fq, Blake2bTranscript>::new(
        r_x.clone(),
        r_dense.clone(),
        &bijection,
        num_bits,
        &mut prover_transcript,
    );

    let mut verifier_transcript: Blake2bTranscript = Transcript::new(b"test_many_polys");
    let verifier = JaggedAssistVerifier::<Fq, Blake2bTranscript>::new(
        prover.claimed_evaluations.clone(),
        r_x,
        r_dense,
        &bijection,
        num_bits,
        &mut verifier_transcript,
    );

    // Run sumcheck
    let prover_accumulator = ProverOpeningAccumulator::new(10);
    let verifier_accumulator = VerifierOpeningAccumulator::new(10);

    let mut current_claim = prover.input_claim(&prover_accumulator);
    let mut challenges: Vec<<Fq as JoltField>::Challenge> = Vec::new();

    for round in 0..prover.num_rounds() {
        let poly = prover.compute_message(round, current_claim);

        let h_0 = poly.evaluate(&Fq::zero());
        let h_1 = poly.evaluate(&Fq::one());
        assert_eq!(h_0 + h_1, current_claim, "Round {} sum check failed", round);

        let challenge: Fq = Fq::rand(&mut rng);
        challenges.push(challenge.into());
        current_claim = poly.evaluate(&challenge);
        prover.ingest_challenge(challenge.into(), round);
    }

    let expected_output = verifier.expected_output_claim(&verifier_accumulator, &challenges);
    assert_eq!(
        current_claim, expected_output,
        "Prover-verifier mismatch with {} polynomials",
        num_polynomials
    );
}

/// Test that compute_f_jagged_with_mapping produces correct weighted sum
#[test]
fn test_compute_f_jagged_with_mapping() {
    let mut rng = test_rng();
    let num_bits = 3;

    let r_x: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let r_dense: Vec<Fq> = (0..num_bits).map(|_| Fq::rand(&mut rng)).collect();
    let bijection = create_test_bijection(&[2, 2, 1, 2]);
    let num_polynomials = bijection.num_polynomials();

    let mut transcript: Blake2bTranscript = Transcript::new(b"test_f_jagged");
    let prover = JaggedAssistProver::<Fq, Blake2bTranscript>::new(
        r_x.clone(),
        r_dense.clone(),
        &bijection,
        num_bits,
        &mut transcript,
    );

    // Create verifier
    let mut verifier_transcript: Blake2bTranscript = Transcript::new(b"test_f_jagged");
    let verifier = JaggedAssistVerifier::<Fq, Blake2bTranscript>::new(
        prover.claimed_evaluations.clone(),
        r_x,
        r_dense,
        &bijection,
        num_bits,
        &mut verifier_transcript,
    );

    // Create eq(r_s, row) values for testing (one per row)
    let num_rows = 8;
    let eq_r_s: Vec<Fq> = (0..num_rows).map(|_| Fq::rand(&mut rng)).collect();

    // Create a simple identity mapping (polynomial k -> row k)
    let matrix_rows: Vec<usize> = (0..num_polynomials).collect();

    // Compute f_jagged using verifier with mapping
    let f_jagged = verifier.compute_f_jagged_with_mapping(&eq_r_s, &matrix_rows);

    // Compute manually: Σ_k eq(r_s, matrix_rows[k]) · v_k
    let expected: Fq = matrix_rows
        .iter()
        .zip(&prover.claimed_evaluations)
        .map(|(&row, v_k)| eq_r_s.get(row).cloned().unwrap_or(Fq::zero()) * *v_k)
        .sum();

    assert_eq!(
        f_jagged, expected,
        "compute_f_jagged_with_mapping should return weighted sum of claimed evaluations"
    );
}

/// Test interleaved variable ordering consistency
#[test]
fn test_interleaved_ordering() {
    use crate::zkvm::recursion::stage3::branching_program::{get_coordinate_info, CoordType};

    let num_bits = 4;

    // Verify interleaved ordering: (a₀,b₀,c₀,d₀, a₁,b₁,c₁,d₁, ...)
    for layer in 0..num_bits {
        let base = layer * 4;

        assert_eq!(
            get_coordinate_info(base, num_bits),
            (CoordType::A, layer),
            "Variable {} should be A at layer {}",
            base,
            layer
        );
        assert_eq!(
            get_coordinate_info(base + 1, num_bits),
            (CoordType::B, layer),
            "Variable {} should be B at layer {}",
            base + 1,
            layer
        );
        assert_eq!(
            get_coordinate_info(base + 2, num_bits),
            (CoordType::C, layer),
            "Variable {} should be C at layer {}",
            base + 2,
            layer
        );
        assert_eq!(
            get_coordinate_info(base + 3, num_bits),
            (CoordType::D, layer),
            "Variable {} should be D at layer {}",
            base + 3,
            layer
        );
    }
}
