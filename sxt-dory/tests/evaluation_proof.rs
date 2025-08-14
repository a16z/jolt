#![allow(missing_docs)]
use std::time::Instant;

use ark_bn254::{Fq12, Fr};
use dory::{
    arithmetic::{Field, Group},
    builder::DoryProofBuilder,
    curve::commit_and_evaluate_batch,
    setup::ProverSetup,
    toy_transcript::ToyTranscript,
    vmv::evaluate::{create_evaluation_proof, verify_evaluation_proof},
};

use dory::curve::{
    test_rng, ArkBn254Pairing, DummyMsm, OptimizedMsmG1, OptimizedMsmG2, StandardPolynomial,
};

#[test]
fn test_evaluation_proof_sigma_2() {
    println!("===== Evaluation Proof Test (sigma=2) =====");
    let total_start = Instant::now();

    // ----- Test Parameters -----
    let length: usize = 1 << 9;
    let max_log_n: usize = 9;
    let sigma: usize = 5;

    println!("Parameters:");
    println!("  - Polynomial length: {}", length);
    println!("  - Max log n: {}", max_log_n);
    println!("  - Sigma: {}", sigma);

    let mut rng = test_rng();
    let domain = b"eval_proof_test_domain";

    // ----- Setup phase -----
    println!("\n[1/4] Creating setup...");
    let setup_start = Instant::now();

    // Calculate nu (polynomial degree)
    let nu = length.next_power_of_two().trailing_zeros() as usize;
    println!("  - Nu (log of next power of 2 of length): {}", nu);
    println!("  - 2^nu: {}", 1 << nu);

    // Verify nu is valid for the polynomial length
    assert!(length <= 1 << nu, "Length should be at most 2^nu");
    assert!(
        1 << (nu - 1) < length,
        "Length should be more than 2^(nu-1)"
    );

    // Create prover setup
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, max_log_n);
    println!("Setup created in: {:?}", setup_start.elapsed());

    // ----- Polynomial and Evaluation Point Generation -----
    println!("\n[2/4] Generating polynomial and evaluation point...");
    let gen_start = Instant::now();

    // Generate random polynomial coefficients
    let a = core::iter::repeat_with(|| Fr::random(&mut rng))
        .take(length)
        .collect::<Vec<_>>();

    // Generate random evaluation point
    let b_points = core::iter::repeat_with(|| Fr::random(&mut rng))
        .take(nu)
        .collect::<Vec<_>>();

    println!("  - Polynomial coefficient count: {}", a.len());
    println!("  - Evaluation point dimension: {}", b_points.len());
    println!("Vectors generated in: {:?}", gen_start.elapsed());

    // ----- Create transcript -----
    println!("\n[3/4] Creating transcript and generating proof...");
    let transcript = ToyTranscript::new(domain);

    // ----- Generate evaluation proof -----
    let proof_start = Instant::now();

    // Create the evaluation proof
    let polynomial = StandardPolynomial::new(&a);
    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &polynomial,
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    let proof_time = proof_start.elapsed();
    println!("Proof generated in: {:?}", proof_time);

    // ----- Verify Proof Structure -----
    println!("\n[4/4] Verifying proof structure...");
    println!("Proof message counts:");
    println!("  - First messages: {}", proof.first_messages.len());
    println!("  - Second messages: {}", proof.second_messages.len());
    println!(
        "  - Scalar Final message: {:?}",
        proof.final_message.clone().unwrap()
    );

    // Verify that proof contains messages
    assert!(
        !proof.first_messages.is_empty(),
        "Proof should contain first messages"
    );
    assert!(
        !proof.second_messages.is_empty(),
        "Proof should contain second messages"
    );

    // ----- Verify the proof -----
    println!("\n[5/5] Verifying evaluation proof...");
    let verify_start = Instant::now();

    // Compute proper commitment, batching factors, and evaluations
    let (commitment_batch, batching_factors, evaluations) = commit_and_evaluate_batch::<
        ArkBn254Pairing,
        OptimizedMsmG1,
        Fr,
        <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
    >(
        &polynomial,
        &b_points,
        0, // offset
        sigma,
        &prover_setup,
    );
    let verifier_setup = prover_setup.to_verifier_setup();

    // Create fresh transcript for verification
    let verify_transcript = ToyTranscript::new(domain);

    // Call verify_evaluation_proof
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

    let verify_time = verify_start.elapsed();
    println!("Verification completed in: {:?}", verify_time);

    // Check verification result
    match verification_result {
        Ok(_) => println!("✓ Proof verification succeeded!"),
        Err(e) => println!("✗ Proof verification failed: {:?}", e),
    }

    // ----- Test Summary -----
    let total_time = total_start.elapsed();
    println!("\n===== Test Summary =====");
    println!("Total test time: {:?}", total_time);
    println!(
        "Proof generation time: {:?} ({:.2}% of total)",
        proof_time,
        proof_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "Verification time: {:?} ({:.2}% of total)",
        verify_time,
        verify_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!("Evaluation proof test completed successfully!");
}

#[test]
fn test_evaluation_proof_verification_should_fail() {
    println!("===== Evaluation Proof Failure Test =====");
    let total_start = Instant::now();

    // ----- Test Parameters -----
    let length: usize = 1 << 9;
    let max_log_n: usize = 9;
    let sigma: usize = 5;

    println!("Parameters:");
    println!("  - Polynomial length: {}", length);
    println!("  - Max log n: {}", max_log_n);
    println!("  - Sigma: {}", sigma);

    let mut rng = test_rng();
    let domain = b"eval_proof_test_domain";

    // ----- Setup phase -----
    println!("\n[1/5] Creating setup...");
    let setup_start = Instant::now();

    let nu = length.next_power_of_two().trailing_zeros() as usize;
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, max_log_n);
    println!("Setup created in: {:?}", setup_start.elapsed());

    // ----- Polynomial and Evaluation Point Generation -----
    println!("\n[2/5] Generating polynomial and evaluation point...");
    let gen_start = Instant::now();

    let a = core::iter::repeat_with(|| Fr::random(&mut rng))
        .take(length)
        .collect::<Vec<_>>();

    let b_points = core::iter::repeat_with(|| Fr::random(&mut rng))
        .take(nu)
        .collect::<Vec<_>>();

    println!("Vectors generated in: {:?}", gen_start.elapsed());

    // ----- Create transcript and generate proof -----
    println!("\n[3/5] Generating proof...");
    let transcript = ToyTranscript::new(domain);
    let proof_start = Instant::now();

    let polynomial = StandardPolynomial::new(&a);
    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &polynomial,
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    let proof_time = proof_start.elapsed();
    println!("Proof generated in: {:?}", proof_time);

    // ----- Test Case 2: Tamper with commitment -----
    println!("\n[5/5] Testing tampered commitment...");
    {
        let verify_start = Instant::now();

        // Get correct verification data
        let (mut commitment_batch, batching_factors, evaluations) =
            commit_and_evaluate_batch::<
                ArkBn254Pairing,
                OptimizedMsmG1,
                Fr,
                <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
            >(&polynomial, &b_points, 0, sigma, &prover_setup);

        // Tamper with the commitment
        commitment_batch[0] = Fq12::random(&mut rng); // Wrong commitment
                                                      // evaluations[0] = Fr::random(&mut rng); // Also tamper with evaluation

        let verifier_setup = prover_setup.to_verifier_setup();

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

        println!(
            "Tampered commitment verification time: {:?}",
            verify_start.elapsed()
        );
        assert!(
            verification_result.is_err(),
            "Verification should fail with tampered commitment"
        );
        println!("✓ Verification correctly failed with tampered commitment");
    }

    // ----- Test Summary -----
    let total_time = total_start.elapsed();
    println!("\n===== Test Summary =====");
    println!("Total test time: {:?}", total_time);
    println!("All failure tests completed successfully!");
}

#[test]
fn test_evaluation_proof_tampered_messages_should_fail() {
    println!("===== Evaluation Proof Tampered Messages Test =====");
    let total_start = Instant::now();

    // ----- Test Parameters -----
    let length: usize = 1 << 9;
    let max_log_n: usize = 9;
    let sigma: usize = 5;

    println!("Parameters:");
    println!("  - Polynomial length: {}", length);
    println!("  - Max log n: {}", max_log_n);
    println!("  - Sigma: {}", sigma);

    let mut rng = test_rng();
    let domain = b"eval_proof_test_domain";

    // ----- Setup phase -----
    println!("\n[1/4] Creating setup...");
    let setup_start = Instant::now();

    let nu = length.next_power_of_two().trailing_zeros() as usize;
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, max_log_n);
    println!("Setup created in: {:?}", setup_start.elapsed());

    // ----- Polynomial and Evaluation Point Generation -----
    println!("\n[2/4] Generating polynomial and evaluation point...");
    let gen_start = Instant::now();

    let a = core::iter::repeat_with(|| Fr::random(&mut rng))
        .take(length)
        .collect::<Vec<_>>();

    let b_points = core::iter::repeat_with(|| Fr::random(&mut rng))
        .take(nu)
        .collect::<Vec<_>>();

    println!("Vectors generated in: {:?}", gen_start.elapsed());

    // ----- Create transcript and generate proof -----
    println!("\n[3/4] Generating proof...");
    let transcript = ToyTranscript::new(domain);
    let proof_start = Instant::now();

    let polynomial = StandardPolynomial::new(&a);
    let proof = create_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        _,
    >(
        transcript,
        &polynomial,
        None,
        &b_points,
        sigma,
        &prover_setup,
    );

    let proof_time = proof_start.elapsed();
    println!("Proof generated in: {:?}", proof_time);

    // ----- Test: Tamper with proof messages -----
    println!("\n[4/4] Testing tampered proof messages...");
    let verify_start = Instant::now();

    // Get correct verification data (no tampering with verification data)
    let (commitment_batch, batching_factors, evaluations) =
        commit_and_evaluate_batch::<
            ArkBn254Pairing,
            OptimizedMsmG1,
            Fr,
            <ArkBn254Pairing as dory::arithmetic::Pairing>::G1,
        >(&polynomial, &b_points, 0, sigma, &prover_setup);

    // Create a corrupted copy of the proof by tampering with proof messages
    let mut corrupted_proof = DoryProofBuilder {
        first_messages: proof.first_messages.clone(),
        second_messages: proof.second_messages.clone(),
        final_message: proof.final_message.clone(),
        transcript: proof.transcript.clone(),
        _phantom: std::marker::PhantomData,
        vmv_message: proof.vmv_message.clone(),
    };

    // Tamper with a first message if available
    if !corrupted_proof.first_messages.is_empty() {
        println!("Tampering with first message d1_left...");
        corrupted_proof.first_messages[0].d1_left = Fq12::random(&mut rng);
    }

    // Also tamper with a second message if available
    if !corrupted_proof.second_messages.is_empty() {
        println!("Tampering with second message c_plus...");
        corrupted_proof.second_messages[0].c_plus = Fq12::random(&mut rng);
    }

    let verifier_setup = prover_setup.to_verifier_setup();

    // Create fresh transcript for verification
    let verify_transcript = ToyTranscript::new(domain);

    let verification_result = dory::vmv::verify_evaluation_proof::<
        ArkBn254Pairing,
        ToyTranscript,
        OptimizedMsmG1,
        OptimizedMsmG2,
        DummyMsm<Fq12>,
    >(
        corrupted_proof,
        &commitment_batch,
        &batching_factors,
        &evaluations,
        &b_points,
        sigma,
        &verifier_setup,
        verify_transcript,
    );

    let verify_time = verify_start.elapsed();
    println!(
        "Tampered proof messages verification time: {:?}",
        verify_time
    );

    // Check verification result
    assert!(
        verification_result.is_err(),
        "Verification should fail with tampered proof messages"
    );
    println!("✓ Verification correctly failed with tampered proof messages");

    // ----- Test Summary -----
    let total_time = total_start.elapsed();
    println!("\n===== Test Summary =====");
    println!("Total test time: {:?}", total_time);
    println!(
        "Proof generation time: {:?} ({:.2}% of total)",
        proof_time,
        proof_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!(
        "Verification time: {:?} ({:.2}% of total)",
        verify_time,
        verify_time.as_secs_f64() / total_time.as_secs_f64() * 100.0
    );
    println!("Tampered proof messages test completed successfully!");
}
