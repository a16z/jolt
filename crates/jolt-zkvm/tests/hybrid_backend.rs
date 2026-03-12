//! E2E tests using `HybridBackend<MetalBackend, CpuBackend>`.
//!
//! Validates that the hybrid GPU→CPU transition path produces correct proofs.
//! These tests exercise the real Metal backend for large buffers and the real
//! CPU backend for small buffers after the threshold transition.
//!
//! macOS-only — requires Apple Silicon with Metal support.

#![cfg(target_os = "macos")]

mod common;

use std::sync::Arc;

use common::*;
use jolt_compute::HybridBackend;
use jolt_cpu::CpuBackend;
use jolt_metal::MetalBackend;

/// Hybrid backend with a low threshold (2^10) to force the GPU→CPU transition
/// to happen during small test traces.
fn hybrid() -> Arc<HybridBackend<MetalBackend, CpuBackend>> {
    Arc::new(HybridBackend::new(MetalBackend::new(), CpuBackend, 1 << 10))
}

// ── Synthetic pipeline with hybrid backend ───────────────────────────

#[test]
fn hybrid_claim_reduction_standalone() {
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::SumcheckClaim;

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

    let data = SyntheticReduction {
        poly_a,
        poly_b,
        c0,
        c1,
    };
    let prover_stage = build_prover_stage_with_backend(&data, eq_point.clone(), hybrid());
    let mut prover_stages: Vec<Box<dyn jolt_zkvm::stage::ProverStage<Fr, Blake2bTranscript>>> =
        vec![Box::new(prover_stage)];

    let mut pt = Blake2bTranscript::new(b"hybrid-standalone");
    let (stage_proofs, _) = jolt_zkvm::pipeline::prove_stages(&mut prover_stages, &mut pt);

    assert_eq!(stage_proofs.len(), 1);

    let desc = StageDescriptor::claim_reduction(eq_point, vec![c0, c1], claimed_sum, vec![0, 1])
        .with_reverse_challenges();

    let stage_proof = &stage_proofs[0];
    let mut vt = Blake2bTranscript::new(b"hybrid-standalone");

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

#[test]
fn hybrid_nop_program() {
    run_hybrid_mock_e2e(&[
        nop_cycle_witness(0, 0),
        nop_cycle_witness(4, 1),
        nop_cycle_witness(8, 2),
        nop_cycle_witness(12, 3),
    ]);
}

#[test]
fn hybrid_add_program() {
    run_hybrid_mock_e2e(&[
        add_cycle_witness(0, 0, 7, 3),
        add_cycle_witness(4, 1, 10, 5),
        add_cycle_witness(8, 2, 100, 200),
        add_cycle_witness(12, 3, 1, 1),
    ]);
}

#[test]
fn hybrid_mixed_instructions() {
    run_hybrid_mock_e2e(&[
        nop_cycle_witness(0, 0),
        add_cycle_witness(4, 1, 7, 3),
        load_cycle_witness(8, 2, 100, 20, 42),
        store_cycle_witness(12, 3, 180, 20, 55),
    ]);
}

#[test]
fn hybrid_multi_stage_pipeline() {
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

    use jolt_openings::mock::MockCommitment;
    let (com_a1, ()) = MockPCS::commit(&data1.poly_a, &());
    let (com_b1, ()) = MockPCS::commit(&data1.poly_b, &());
    let (com_a2, ()) = MockPCS::commit(&data2.poly_a, &());
    let (com_b2, ()) = MockPCS::commit(&data2.poly_b, &());
    let poly_commitments: Vec<MockCommitment<Fr>> = vec![com_a1, com_b1, com_a2, com_b2];

    let mut pt = Blake2bTranscript::new(b"jolt-hybrid-e2e");

    let backend = hybrid();
    let proof = prove::<MockPCS, Blake2bTranscript>(
        &key,
        &cycle_witnesses,
        poly_commitments,
        |_r_x, r_y, _t| {
            let s1 = build_prover_stage_with_backend(&data1, r_y.to_vec(), Arc::clone(&backend));
            let s2 = build_prover_stage_with_backend(&data2, r_y.to_vec(), Arc::clone(&backend));
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

    let mut vt = Blake2bTranscript::new(b"jolt-hybrid-e2e");

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

// ── Host pipeline with Dory + hybrid backend ─────────────────────────

#[test]
fn hybrid_muldiv_basic() {
    let inputs = postcard::to_stdvec(&(9u32, 5u32, 3u32)).unwrap();
    prove_and_verify_guest_hybrid("muldiv-guest", &inputs);
}

#[test]
fn hybrid_fibonacci_small() {
    let inputs = postcard::to_stdvec(&5u32).unwrap();
    prove_and_verify_guest_hybrid("fibonacci-guest", &inputs);
}

// ── Helpers ──────────────────────────────────────────────────────────

fn prove_and_verify_guest_hybrid(guest_name: &str, inputs: &[u8]) {
    use jolt_dory::DoryScheme;
    use jolt_host::Program;
    use jolt_zkvm::prover::{prove, verify};

    let mut program = Program::new(guest_name);
    let (_, trace, _, _) = program.trace(inputs, &[], &[]);

    let output = prove::<DoryScheme, _>(
        &trace,
        |num_vars| {
            (
                DoryScheme::setup_prover(num_vars),
                DoryScheme::setup_verifier(num_vars),
            )
        },
        hybrid(),
    )
    .expect("prove should succeed");

    verify::<DoryScheme>(&output).expect("verify should succeed");
}

/// Run prove → verify with mock PCS and hybrid backend.
fn run_hybrid_mock_e2e(cycle_witnesses: &[Vec<Fr>]) {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let config = JoltConfig {
        num_cycles: cycle_witnesses.len(),
    };
    let key = preprocess::<Fr, MockPCS>(&config, |_| ((), ()));

    let num_col_vars = key.spartan_key.num_col_vars();
    let reduction_data = build_reduction_data(num_col_vars, &mut rng);

    let (com_a, ()) = MockPCS::commit(&reduction_data.poly_a, &());
    let (com_b, ()) = MockPCS::commit(&reduction_data.poly_b, &());
    let poly_commitments = vec![com_a, com_b];

    let mut pt = Blake2bTranscript::new(b"jolt-hybrid-e2e");
    let backend = hybrid();

    let proof = prove::<MockPCS, Blake2bTranscript>(
        &key,
        cycle_witnesses,
        poly_commitments,
        |_r_x, r_y, _t| {
            let stage = build_prover_stage_with_backend(
                &reduction_data,
                r_y.to_vec(),
                Arc::clone(&backend),
            );
            vec![Box::new(stage)]
        },
        &mut pt,
    )
    .expect("proving should succeed");

    assert_eq!(proof.stage_proofs.len(), 1);

    let vk = jolt_verifier::JoltVerifyingKey {
        spartan_key: key.spartan_key.clone(),
        pcs_setup: (),
    };

    let mut vt = Blake2bTranscript::new(b"jolt-hybrid-e2e");

    let _ = jolt_verifier::verify::<MockPCS, Blake2bTranscript>(
        &proof,
        &vk,
        |_r_x, r_y, _t| vec![build_verifier_descriptor(&reduction_data, r_y, 0)],
        &mut vt,
    )
    .expect("verification should succeed");
}
