//! Full pipeline tests with hand-crafted (synthetic) witness vectors.
//!
//! Exercises prove() → verify() using manually constructed cycle witnesses
//! for NOP, ADD, LOAD, STORE instruction types. The sumcheck stages use
//! synthetic random polynomial data rather than real committed polynomials.

mod common;

use common::*;
use jolt_openings::mock::MockCommitment;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::SumcheckClaim;
use jolt_zkvm::stage::ProverStage;

#[test]
fn nop_program() {
    run_mock_e2e(&[
        nop_cycle_witness(0, 0),
        nop_cycle_witness(4, 1),
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ]);
}

#[test]
fn add_program() {
    run_mock_e2e(&[
        add_cycle_witness(0, 0, 7, 3),
        add_cycle_witness(4, 1, 10, 5),
        add_cycle_witness(8, 2, 100, 200),
        add_cycle_witness(12, 3, 1, 1),
    ]);
}

#[test]
fn load_compute_store() {
    run_mock_e2e(&[
        load_cycle_witness(0, 0, 100, 20, 42),
        add_cycle_witness(4, 1, 42, 8),
        store_cycle_witness(8, 2, 180, 20, 50),
        nop_cycle_witness(12, 3),
    ]);
}

#[test]
fn mixed_instructions() {
    run_mock_e2e(&[
        nop_cycle_witness(0, 0),
        add_cycle_witness(4, 1, 7, 3),
        load_cycle_witness(8, 2, 100, 20, 42),
        store_cycle_witness(12, 3, 180, 20, 55),
    ]);
}

#[test]
fn eight_cycle_program() {
    run_mock_e2e(&[
        nop_cycle_witness(0, 0),
        load_cycle_witness(4, 1, 0, 8, 100),
        add_cycle_witness(8, 2, 100, 50),
        store_cycle_witness(12, 3, 0, 16, 150),
        load_cycle_witness(16, 4, 0, 24, 200),
        add_cycle_witness(20, 5, 200, 150),
        store_cycle_witness(24, 6, 0, 32, 350),
        nop_cycle_witness(28, 7),
    ]);
}

#[test]
fn single_cycle() {
    run_mock_e2e(&[nop_cycle_witness(0, 0)]);
}

#[test]
fn two_cycles() {
    run_mock_e2e(&[add_cycle_witness(0, 0, 3, 4), add_cycle_witness(4, 1, 5, 6)]);
}

#[test]
fn load_add_store_nop() {
    run_mock_e2e(&[
        load_cycle_witness(0, 0, 50, 10, 99),
        add_cycle_witness(4, 1, 99, 1),
        store_cycle_witness(8, 2, 60, 0, 100),
        nop_cycle_witness(12, 3),
    ]);
}

/// Two independent claim reduction stages through the full pipeline.
#[test]
fn multi_stage_pipeline() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let config = JoltConfig { num_cycles: 4 };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

    let cycle_witnesses = vec![
        nop_cycle_witness(0, 0),
        add_cycle_witness(4, 1, 7, 3),
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ];

    let num_col_vars = key.spartan_key.num_col_vars();
    let data1 = build_reduction_data(num_col_vars, &mut rng);
    let data2 = build_reduction_data(num_col_vars, &mut rng);

    let (com_a1, ()) = MockPCS::commit(&data1.poly_a, &());
    let (com_b1, ()) = MockPCS::commit(&data1.poly_b, &());
    let (com_a2, ()) = MockPCS::commit(&data2.poly_a, &());
    let (com_b2, ()) = MockPCS::commit(&data2.poly_b, &());
    let poly_commitments: Vec<MockCommitment<Fr>> = vec![com_a1, com_b1, com_a2, com_b2];

    let mut pt = Blake2bTranscript::new(b"jolt-e2e");

    let proof = prove::<MockPCS, Blake2bTranscript>(
        &key,
        &cycle_witnesses,
        poly_commitments,
        |_r_x, r_y, _t| {
            let s1 = build_prover_stage(&data1, r_y.to_vec());
            let s2 = build_prover_stage(&data2, r_y.to_vec());
            vec![Box::new(s1), Box::new(s2)]
        },
        &mut pt,
    )
    .expect("proving should succeed");

    assert_eq!(proof.stage_proofs.len(), 2);

    let vk = jolt_verifier::JoltVerifyingKey {
        spartan_key: key.spartan_key.clone(),
        pcs_setup: (),
    };

    let mut vt = Blake2bTranscript::new(b"jolt-e2e");

    let (v_r_x, v_r_y) = jolt_verifier::verify::<MockPCS, Blake2bTranscript>(
        &proof,
        &vk,
        |_r_x, r_y, _t| {
            vec![
                build_verifier_descriptor(&data1, r_y, 0),
                build_verifier_descriptor(&data2, r_y, 2),
            ]
        },
        &mut vt,
    )
    .expect("verification should succeed");

    assert_eq!(v_r_x.len(), key.spartan_key.num_row_vars());
    assert_eq!(v_r_y.len(), key.spartan_key.num_col_vars());
}

/// Standalone claim reduction sumcheck prove/verify.
#[test]
fn claim_reduction_standalone() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let num_vars = 4;
    let n = 1usize << num_vars;

    let eq_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();
    let c0 = Fr::random(&mut rng);
    let c1 = Fr::random(&mut rng);

    let poly_a: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
    let poly_b: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

    let eq_table = EqPolynomial::new(eq_point.clone()).evaluations();
    let claimed_sum: Fr = (0..n)
        .map(|j| eq_table[j] * (c0 * poly_a[j] + c1 * poly_b[j]))
        .sum();

    let prover_stage = ClaimReductionStage::increment(
        poly_a,
        poly_b,
        eq_point.clone(),
        c0,
        c1,
        cpu(),
    );
    let mut prover_stages: Vec<Box<dyn ProverStage<Fr, Blake2bTranscript>>> =
        vec![Box::new(prover_stage)];

    let mut pt = Blake2bTranscript::new(b"cr-standalone");
    let (stage_proofs, _opening_claims) =
        jolt_zkvm::pipeline::prove_stages(&mut prover_stages, &mut pt);

    assert_eq!(stage_proofs.len(), 1);

    let desc = StageDescriptor::claim_reduction(eq_point, vec![c0, c1], claimed_sum, vec![0, 1])
        .with_reverse_challenges();

    let stage_proof = &stage_proofs[0];

    let mut vt = Blake2bTranscript::new(b"cr-standalone");

    let claims = [SumcheckClaim {
        num_vars: desc.num_vars,
        degree: desc.degree,
        claimed_sum: desc.claimed_sum,
    }];

    let (final_eval, challenges) = jolt_sumcheck::BatchedSumcheckVerifier::verify(
        &claims,
        &stage_proof.sumcheck_proof,
        &mut vt,
    )
    .expect("sumcheck verification should succeed");

    let eval_point: Vec<Fr> = challenges.iter().rev().copied().collect();
    let eq_eval = EqPolynomial::new(desc.eq_point.clone()).evaluate(&eval_point);
    let g_eval: Fr = desc
        .output_expr
        .evaluate(&stage_proof.evaluations, &desc.output_challenges);
    let expected = eq_eval * g_eval;

    assert_eq!(expected, final_eval, "eq * g should match final_eval");
    assert_eq!(stage_proof.evaluations.len(), 2);
}
