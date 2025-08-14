#![allow(missing_docs)]
use ark_bn254::{Fq12, Fr, G1Affine};
use dory::{
    arithmetic::{Field, Group},
    curve::{
        commit_and_evaluate_batch, test_rng, ArkBn254Pairing, DummyMsm, G2AffineWrapper,
        OptimizedMsmG1, OptimizedMsmG2, StandardPolynomial,
    },
    setup::ProverSetup,
    toy_transcript::ToyTranscript,
    vmv::{
        compute_nu, compute_polynomial_commitment, create_evaluation_proof, verify_evaluation_proof,
    },
};

// Helper function to generate test environment for VMV tests
fn setup_vmv_test_environment(
    length: usize,
    max_log_n: usize,
    sigma: usize,
) -> (
    ProverSetup<ArkBn254Pairing>,
    dory::setup::VerifierSetup<ArkBn254Pairing>,
    Vec<Fr>,
    Vec<Fr>,
    usize,
) {
    let mut rng = test_rng();

    // Create setup
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, max_log_n);
    let verifier_setup = prover_setup.to_verifier_setup();

    // Calculate nu
    let nu = compute_nu(max_log_n, sigma);

    // Generate random polynomial coefficients
    let a = core::iter::repeat_with(|| Fr::random(&mut rng))
        .take(length)
        .collect::<Vec<_>>();

    // Generate random evaluation point
    let b_points = core::iter::repeat_with(|| Fr::random(&mut rng))
        .take(nu)
        .collect::<Vec<_>>();

    (prover_setup, verifier_setup, a, b_points, nu)
}

#[test]
fn test_soundness_tamper_vmv_message_c() {
    println!("=== Testing soundness: tampering with VMV message C ===");
    let mut rng = test_rng();
    let domain = b"vmv_soundness_test";

    let length: usize = 1 << 8;
    let max_log_n: usize = 9;
    let sigma: usize = 8;

    let (prover_setup, verifier_setup, a, b_points, _nu) =
        setup_vmv_test_environment(length, max_log_n, sigma);

    // Generate proof
    let transcript = ToyTranscript::new(domain);
    let mut proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &StandardPolynomial::new(&a),
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Get verification data
    let (commitment_batch, batching_factors, evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        &b_points,
        0,
        sigma,
        &prover_setup,
    );

    // Tamper with VMV message C
    if let Some(vmv_msg) = &mut proof.vmv_message {
        println!("Tampering with VMV message C...");
        vmv_msg.c = Fq12::random(&mut rng);

        // Create fresh transcript for verification
        let verify_transcript = ToyTranscript::new(domain);

        let verification_result = verify_evaluation_proof::<
            ArkBn254Pairing,
            ToyTranscript,
            OptimizedMsmG1,
            OptimizedMsmG2,
            DummyMsm<Fq12>,
        >(
            proof,
            &commitment_batch,
            &batching_factors,
            &evaluations,
            &b_points,
            sigma,
            &verifier_setup,
            verify_transcript,
        );

        assert!(
            verification_result.is_err(),
            "Verification should fail with tampered VMV C"
        );
        println!("✓ Verification correctly failed with tampered VMV C");
    }
}

#[test]
fn test_soundness_tamper_vmv_message_d2() {
    println!("=== Testing soundness: tampering with VMV message D2 ===");
    let mut rng = test_rng();
    let domain = b"vmv_soundness_test";

    let length: usize = 1 << 8;
    let max_log_n: usize = 9;
    let sigma: usize = 8;

    let (prover_setup, verifier_setup, a, b_points, _nu) =
        setup_vmv_test_environment(length, max_log_n, sigma);

    // Generate proof
    let transcript = ToyTranscript::new(domain);
    let mut proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &StandardPolynomial::new(&a),
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Get verification data
    let (commitment_batch, batching_factors, evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        &b_points,
        0,
        sigma,
        &prover_setup,
    );

    // Tamper with VMV message D2
    if let Some(vmv_msg) = &mut proof.vmv_message {
        println!("Tampering with VMV message D2...");
        vmv_msg.d2 = Fq12::random(&mut rng);

        // Create fresh transcript for verification
        let verify_transcript = ToyTranscript::new(domain);

        let verification_result = verify_evaluation_proof::<
            ArkBn254Pairing,
            ToyTranscript,
            OptimizedMsmG1,
            OptimizedMsmG2,
            DummyMsm<Fq12>,
        >(
            proof,
            &commitment_batch,
            &batching_factors,
            &evaluations,
            &b_points,
            sigma,
            &verifier_setup,
            verify_transcript,
        );

        assert!(
            verification_result.is_err(),
            "Verification should fail with tampered VMV D2"
        );
        println!("✓ Verification correctly failed with tampered VMV D2");
    }
}

#[test]
fn test_soundness_tamper_vmv_message_e1() {
    println!("=== Testing soundness: tampering with VMV message E1 ===");
    let mut rng = test_rng();
    let domain = b"vmv_soundness_test";

    let length: usize = 1 << 8;
    let max_log_n: usize = 9;
    let sigma: usize = 8;

    let (prover_setup, verifier_setup, a, b_points, _nu) =
        setup_vmv_test_environment(length, max_log_n, sigma);

    // Generate proof
    let transcript = ToyTranscript::new(domain);
    let mut proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &StandardPolynomial::new(&a),
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Get verification data
    let (commitment_batch, batching_factors, evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        &b_points,
        0,
        sigma,
        &prover_setup,
    );

    // Tamper with VMV message E1
    if let Some(vmv_msg) = &mut proof.vmv_message {
        println!("Tampering with VMV message E1...");
        vmv_msg.e1 = G1Affine::random(&mut rng);

        // Create fresh transcript for verification
        let verify_transcript = ToyTranscript::new(domain);

        let verification_result = verify_evaluation_proof::<
            ArkBn254Pairing,
            ToyTranscript,
            OptimizedMsmG1,
            OptimizedMsmG2,
            DummyMsm<Fq12>,
        >(
            proof,
            &commitment_batch,
            &batching_factors,
            &evaluations,
            &b_points,
            sigma,
            &verifier_setup,
            verify_transcript,
        );

        assert!(
            verification_result.is_err(),
            "Verification should fail with tampered VMV E1"
        );
        println!("✓ Verification correctly failed with tampered VMV E1");
    }
}

#[test]
fn test_soundness_wrong_commitment() {
    println!("=== Testing soundness: using wrong commitment ===");
    let mut rng = test_rng();
    let domain = b"vmv_soundness_test";

    let length: usize = 1 << 8;
    let max_log_n: usize = 9;
    let sigma: usize = 8;

    let (prover_setup, verifier_setup, a, b_points, _nu) =
        setup_vmv_test_environment(length, max_log_n, sigma);

    // Generate proof
    let transcript = ToyTranscript::new(domain);
    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &StandardPolynomial::new(&a),
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Get verification data but use wrong commitment
    let (mut commitment_batch, batching_factors, evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        &b_points,
        0,
        sigma,
        &prover_setup,
    );

    // Replace with random commitment
    commitment_batch[0] = Fq12::random(&mut rng);

    // Create fresh transcript for verification
    let verify_transcript = ToyTranscript::new(domain);

    let verification_result = verify_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        DummyMsm<Fq12>,
    >(
        proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        &b_points,
        sigma,
        &verifier_setup,
        verify_transcript,
    );

    assert!(
        verification_result.is_err(),
        "Verification should fail with wrong commitment"
    );
    println!("✓ Verification correctly failed with wrong commitment");
}

#[test]
fn test_soundness_wrong_evaluation() {
    println!("=== Testing soundness: claiming wrong evaluation ===");
    let mut rng = test_rng();
    let domain = b"vmv_soundness_test";

    let length: usize = 1 << 8;
    let max_log_n: usize = 9;
    let sigma: usize = 8;

    let (prover_setup, verifier_setup, a, b_points, _nu) =
        setup_vmv_test_environment(length, max_log_n, sigma);

    // Generate proof
    let transcript = ToyTranscript::new(domain);
    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &StandardPolynomial::new(&a),
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Get verification data but use wrong evaluation
    let (commitment_batch, batching_factors, mut evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        &b_points,
        0,
        sigma,
        &prover_setup,
    );

    // Replace with random evaluation
    evaluations[0] = Fr::random(&mut rng);

    let verification_result = verify_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        DummyMsm<Fq12>,
    >(
        proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        &b_points,
        sigma,
        &verifier_setup,
        ToyTranscript::new(domain),
    );

    assert!(
        verification_result.is_err(),
        "Verification should fail with wrong evaluation"
    );
    println!("✓ Verification correctly failed with wrong evaluation");
}

#[test]
fn test_soundness_wrong_evaluation_point() {
    println!("=== Testing soundness: using wrong evaluation point ===");
    let mut rng = test_rng();
    let domain = b"vmv_soundness_test";

    let length: usize = 1 << 8;
    let max_log_n: usize = 9;
    let sigma: usize = 8;

    let (prover_setup, verifier_setup, a, b_points, nu) =
        setup_vmv_test_environment(length, max_log_n, sigma);

    // Generate proof
    let transcript = ToyTranscript::new(domain);
    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &StandardPolynomial::new(&a),
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Get verification data
    let (commitment_batch, batching_factors, evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        &b_points,
        0,
        sigma,
        &prover_setup,
    );

    // Use different evaluation point for verification
    let wrong_b_points = core::iter::repeat_with(|| Fr::random(&mut rng))
        .take(nu)
        .collect::<Vec<_>>();

    let verification_result = verify_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        DummyMsm<Fq12>,
    >(
        proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        &wrong_b_points,
        sigma,
        &verifier_setup,
        ToyTranscript::new(domain),
    );

    assert!(
        verification_result.is_err(),
        "Verification should fail with wrong evaluation point"
    );
    println!("✓ Verification correctly failed with wrong evaluation point");
}

#[test]
fn test_soundness_commitment_evaluation_mismatch() {
    println!("=== Testing soundness: commitment and evaluation mismatch ===");
    let mut rng = test_rng();
    let domain = b"vmv_soundness_test";

    let length: usize = 1 << 8;
    let max_log_n: usize = 9;
    let sigma: usize = 8;

    let (prover_setup, verifier_setup, a, b_points, _nu) =
        setup_vmv_test_environment(length, max_log_n, sigma);

    // Generate a different polynomial
    let a_different = core::iter::repeat_with(|| Fr::random(&mut rng))
        .take(length)
        .collect::<Vec<_>>();

    // Generate proof for original polynomial
    let transcript = ToyTranscript::new(domain);
    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &StandardPolynomial::new(&a),
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Get commitment for different polynomial but evaluation of original
    let (commitment, _) = compute_polynomial_commitment::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        _,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a_different),
        0,
        sigma,
        &prover_setup,
    );

    let (_, _, evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        &b_points,
        0,
        sigma,
        &prover_setup,
    );

    let commitment_batch = vec![commitment];
    let batching_factors = vec![Fr::one()];

    let verification_result = verify_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        DummyMsm<Fq12>,
    >(
        proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        &b_points,
        sigma,
        &verifier_setup,
        ToyTranscript::new(domain),
    );

    assert!(
        verification_result.is_err(),
        "Verification should fail with mismatched commitment and evaluation"
    );
    println!("✓ Verification correctly failed with mismatched commitment and evaluation");
}

#[test]
fn test_soundness_wrong_batching_factors() {
    println!("=== Testing soundness: using wrong batching factors ===");
    let mut rng = test_rng();
    let domain = b"vmv_soundness_test";

    let length: usize = 1 << 8;
    let max_log_n: usize = 9;
    let sigma: usize = 8;

    let (prover_setup, verifier_setup, a, b_points, _nu) =
        setup_vmv_test_environment(length, max_log_n, sigma);

    // Generate proof
    let transcript = ToyTranscript::new(domain);
    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &StandardPolynomial::new(&a),
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Get verification data but modify batching factors
    let (commitment_batch, mut batching_factors, evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        &b_points,
        0,
        sigma,
        &prover_setup,
    );

    // Use random batching factor instead of 1
    batching_factors[0] = Fr::random(&mut rng);

    let verification_result = verify_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        DummyMsm<Fq12>,
    >(
        proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        &b_points,
        sigma,
        &verifier_setup,
        ToyTranscript::new(domain),
    );

    assert!(
        verification_result.is_err(),
        "Verification should fail with wrong batching factors"
    );
    println!("✓ Verification correctly failed with wrong batching factors");
}

#[test]
fn test_soundness_tamper_proof_structure() {
    println!("=== Testing soundness: tampering with proof structure ===");
    let mut rng = test_rng();
    let domain = b"vmv_soundness_test";

    let length: usize = 1 << 8;
    let max_log_n: usize = 9;
    let sigma: usize = 8;

    let (prover_setup, verifier_setup, a, b_points, _nu) =
        setup_vmv_test_environment(length, max_log_n, sigma);

    // Generate proof
    let transcript = ToyTranscript::new(domain);
    let mut proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &StandardPolynomial::new(&a),
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Get verification data
    let (commitment_batch, batching_factors, evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        &b_points,
        0,
        sigma,
        &prover_setup,
    );

    // Tamper with multiple parts of the proof
    if !proof.first_messages.is_empty() {
        proof.first_messages[0].d1_left = Fq12::random(&mut rng);
        proof.first_messages[0].e1_beta = G1Affine::random(&mut rng);
    }

    if !proof.second_messages.is_empty() {
        proof.second_messages[0].c_plus = Fq12::random(&mut rng);
        proof.second_messages[0].e2_plus = G2AffineWrapper::random(&mut rng);
    }

    let verification_result = verify_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        DummyMsm<Fq12>,
    >(
        proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        &b_points,
        sigma,
        &verifier_setup,
        ToyTranscript::new(domain),
    );

    assert!(
        verification_result.is_err(),
        "Verification should fail with tampered proof structure"
    );
    println!("✓ Verification correctly failed with tampered proof structure");
}

#[test]
fn test_soundness_offset_manipulation() {
    println!("=== Testing soundness: commitment offset manipulation ===");
    let domain = b"vmv_soundness_test";

    let length: usize = 1 << 8;
    let max_log_n: usize = 9;
    let sigma: usize = 8;

    let (prover_setup, verifier_setup, a, b_points, _nu) =
        setup_vmv_test_environment(length, max_log_n, sigma);

    // Generate proof with offset 0
    let transcript = ToyTranscript::new(domain);
    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &StandardPolynomial::new(&a),
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Get commitment with different offset
    let wrong_offset = 16; // Different offset
    let (commitment, _) = compute_polynomial_commitment::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        _,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        wrong_offset,
        sigma,
        &prover_setup,
    );

    // Get evaluation with correct offset
    let (_, _, evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        &b_points,
        0,
        sigma,
        &prover_setup,
    );

    let commitment_batch = vec![commitment];
    let batching_factors = vec![Fr::one()];

    let verification_result = verify_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        DummyMsm<Fq12>,
    >(
        proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        &b_points,
        sigma,
        &verifier_setup,
        ToyTranscript::new(domain),
    );

    assert!(
        verification_result.is_err(),
        "Verification should fail with wrong offset commitment"
    );
    println!("✓ Verification correctly failed with wrong offset commitment");
}

#[test]
fn test_soundness_different_polynomial_degree() {
    println!("=== Testing soundness: using different polynomial degree ===");
    let mut rng = test_rng();
    let domain = b"vmv_soundness_test";

    let length: usize = 1 << 8;
    let max_log_n: usize = 9;
    let sigma: usize = 8;

    let (prover_setup, verifier_setup, a, b_points, nu) =
        setup_vmv_test_environment(length, max_log_n, sigma);

    // Create a polynomial of different length
    let wrong_length = 1 << 7; // Half the size
    let a_wrong = core::iter::repeat_with(|| Fr::random(&mut rng))
        .take(wrong_length)
        .collect::<Vec<_>>();

    // Generate proof for wrong polynomial
    let transcript = ToyTranscript::new(domain);
    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &StandardPolynomial::new(&a_wrong),
        None,
        &b_points[..nu - 1],
        sigma - 1,
        &prover_setup,
    );

    // Get verification data for original polynomial
    let (commitment_batch, batching_factors, evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        &b_points,
        0,
        sigma,
        &prover_setup,
    );

    let verification_result = verify_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        DummyMsm<Fq12>,
    >(
        proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        &b_points,
        sigma,
        &verifier_setup,
        ToyTranscript::new(domain),
    );

    assert!(
        verification_result.is_err(),
        "Verification should fail with different polynomial degree"
    );
    println!("✓ Verification correctly failed with different polynomial degree");
}

#[test]
fn test_soundness_all_vmv_messages_tampered() {
    println!("=== Testing soundness: tampering with all VMV messages ===");
    let mut rng = test_rng();
    let domain = b"vmv_soundness_test";

    let length: usize = 1 << 8;
    let max_log_n: usize = 9;
    let sigma: usize = 8;

    let (prover_setup, verifier_setup, a, b_points, _nu) =
        setup_vmv_test_environment(length, max_log_n, sigma);

    // Generate proof
    let transcript = ToyTranscript::new(domain);
    let mut proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &StandardPolynomial::new(&a),
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Get verification data
    let (commitment_batch, batching_factors, evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        &b_points,
        0,
        sigma,
        &prover_setup,
    );

    // Tamper with all VMV messages
    if let Some(vmv_msg) = &mut proof.vmv_message {
        println!("Tampering with all VMV messages...");
        vmv_msg.c = Fq12::random(&mut rng);
        vmv_msg.d2 = Fq12::random(&mut rng);
        vmv_msg.e1 = G1Affine::random(&mut rng);

        // Create fresh transcript for verification
        let verify_transcript = ToyTranscript::new(domain);

        let verification_result = verify_evaluation_proof::<
            ArkBn254Pairing,
            ToyTranscript,
            OptimizedMsmG1,
            OptimizedMsmG2,
            DummyMsm<Fq12>,
        >(
            proof,
            &commitment_batch,
            &batching_factors,
            &evaluations,
            &b_points,
            sigma,
            &verifier_setup,
            verify_transcript,
        );

        assert!(
            verification_result.is_err(),
            "Verification should fail with all VMV messages tampered"
        );
        println!("✓ Verification correctly failed with all VMV messages tampered");
    }
}

#[test]
fn test_soundness_relationship_attack() {
    println!("=== Testing soundness: VMV relationship attack ===");
    let mut rng = test_rng();
    let domain = b"vmv_soundness_test";

    let length: usize = 1 << 8;
    let max_log_n: usize = 9;
    let sigma: usize = 8;

    let (prover_setup, verifier_setup, a, b_points, _nu) =
        setup_vmv_test_environment(length, max_log_n, sigma);

    // Generate two different polynomials
    let a2 = core::iter::repeat_with(|| Fr::random(&mut rng))
        .take(length)
        .collect::<Vec<_>>();

    // Generate proof for first polynomial
    let transcript = ToyTranscript::new(domain);
    let mut proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &StandardPolynomial::new(&a),
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Get VMV message from a different polynomial
    let transcript2 = ToyTranscript::new(domain);
    let proof2 = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript2,
        &StandardPolynomial::new(&a2),
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    // Mix VMV messages from different proofs
    if let (Some(vmv_msg1), Some(vmv_msg2)) = (&mut proof.vmv_message, &proof2.vmv_message) {
        println!("Mixing VMV messages from different proofs...");
        vmv_msg1.c = vmv_msg2.c.clone();
        // Keep d2 and e1 from original proof
    }

    // Get verification data for first polynomial
    let (commitment_batch, batching_factors, evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &StandardPolynomial::new(&a),
        &b_points,
        0,
        sigma,
        &prover_setup,
    );

    let verification_result = verify_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        DummyMsm<Fq12>,
    >(
        proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        &b_points,
        sigma,
        &verifier_setup,
        ToyTranscript::new(domain),
    );

    assert!(
        verification_result.is_err(),
        "Verification should fail with mixed VMV messages"
    );
    println!("✓ Verification correctly failed with mixed VMV messages");
}
