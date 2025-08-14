#![allow(missing_docs)]
use std::time::Instant;

use ark_bn254::{Fq12, Fr, G1Affine};
use dory::{
    arithmetic::{Field, Group, MultiScalarMul, Pairing},
    builder::{DoryProofBuilder, DoryVerifyBuilder},
    inner_product::{inner_product_prove, inner_product_verify},
    setup::ProverSetup,
    state::{DoryProverState, DoryVerifierState},
    toy_transcript::ToyTranscript,
};

use dory::curve::{test_rng, ArkBn254Pairing, G2AffineWrapper, OptimizedMsmG1, OptimizedMsmG2};

#[test]
fn test_inner_product_prove_verify() {
    println!("Starting inner product test...");
    let total_start = Instant::now();

    // Create deterministic RNG for testing
    let mut rng = test_rng();

    // Test parameters
    let domain = b"test_domain";
    let log_n = 10; // This means vectors of length 2^10 = 1024
    let vector_size = 1 << log_n;
    println!("Vector size: {}", vector_size);

    // ----- Setup phase -----
    let setup_start = Instant::now();

    // Create setup
    println!("Creating setup...");
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, 2 * log_n);
    let verifier_setup = prover_setup.to_verifier_setup();
    println!("Setup created in: {:?}", setup_start.elapsed());

    // ----- Vector generation phase -----
    let vector_gen_start = Instant::now();

    println!("Generating random vectors...");
    // Generate random vectors for prover state
    let v1: Vec<G1Affine> = (0..vector_size)
        .map(|_| G1Affine::random(&mut rng))
        .collect();
    let v2: Vec<G2AffineWrapper> = (0..vector_size)
        .map(|_| G2AffineWrapper::random(&mut rng))
        .collect();
    let s1: Vec<Fr> = (0..vector_size).map(|_| Fr::random(&mut rng)).collect();
    let s2: Vec<Fr> = (0..vector_size).map(|_| Fr::random(&mut rng)).collect();
    println!("Vectors generated in: {:?}", vector_gen_start.elapsed());

    // ----- Initial state calculation phase -----
    let init_state_start = Instant::now();

    println!("Creating initial states...");
    // Create initial state
    let prover_state = DoryProverState::new(v1.clone(), v2.clone(), s1.clone(), s2.clone(), log_n);

    // Create initial value for C (inner product of v1 and v2)
    println!("Computing C (inner product)...");
    let c_start = Instant::now();
    let c = ArkBn254Pairing::multi_pair(&v1, &v2);
    println!("C computed in: {:?}", c_start.elapsed());

    // Create the initial values for D1 and D2
    println!("Computing D1 and D2...");
    let d_start = Instant::now();
    let d_1 = ArkBn254Pairing::multi_pair(&v1, &prover_setup.g2_vec()[..1 << log_n]);
    let d_2 = ArkBn254Pairing::multi_pair(&prover_setup.g1_vec()[..1 << log_n], &v2);
    println!("D1 and D2 computed in: {:?}", d_start.elapsed());

    // Create the initial values for E1 and E2
    println!("Computing E1 and E2...");
    let e_start = Instant::now();
    let e_1 = OptimizedMsmG1::msm(&prover_setup.g1_vec()[..1 << log_n], &s2);
    let e_2 = OptimizedMsmG2::msm(&prover_setup.g2_vec()[..1 << log_n], &s1);
    println!("E1 and E2 computed in: {:?}", e_start.elapsed());

    // Create verifier state
    let verifier_state = DoryVerifierState::new_with_s(c, d_1, d_2, e_1, e_2, s1, s2, log_n);
    println!(
        "Initial states created in: {:?}",
        init_state_start.elapsed()
    );

    // ----- Proof generation phase -----
    println!("Generating proof...");
    let proof_start = Instant::now();

    // Create proof builder
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);

    // Generate proof
    let proof = inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
        builder,
        prover_state,
        &prover_setup,
        log_n,
    );
    println!("Proof generated in: {:?}", proof_start.elapsed());

    // create a verifier
    let verify_builder =
        DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
            proof,
            ToyTranscript::new(domain),
        );

    // ----- Verification phase -----
    println!("Verifying proof...");
    let verify_start = Instant::now();

    // Verify proof
    let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);
    println!("Proof verified in: {:?}", verify_start.elapsed());

    // Check the verification result
    assert!(result.is_ok(), "Proof verification failed");

    println!("Total test time: {:?}", total_start.elapsed());
    println!("Test completed successfully!");
}

#[test]
fn test_inner_product_verify_should_fail() {
    println!("Starting failing verification test...");
    let total_start = Instant::now();

    // Create deterministic RNG for testing
    let mut rng = test_rng();

    // Test parameters
    let domain = b"test_domain";
    let log_n = 9; // Use a smaller size for faster testing
    let vector_size = 1 << log_n;
    println!("Vector size: {}", vector_size);

    // ----- Setup phase -----
    println!("Creating setup...");
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, 2 * log_n);
    let verifier_setup = prover_setup.to_verifier_setup();

    // ----- Vector generation phase -----
    println!("Generating random vectors...");
    // Generate random vectors for prover state
    let v1: Vec<G1Affine> = (0..vector_size)
        .map(|_| G1Affine::random(&mut rng))
        .collect();
    let v2: Vec<G2AffineWrapper> = (0..vector_size)
        .map(|_| G2AffineWrapper::random(&mut rng))
        .collect();
    let s1: Vec<Fr> = (0..vector_size).map(|_| Fr::random(&mut rng)).collect();
    let s2: Vec<Fr> = (0..vector_size).map(|_| Fr::random(&mut rng)).collect();

    // ----- Initial state calculation phase -----
    println!("Creating initial states...");
    // Create initial state
    let prover_state = DoryProverState::new(v1.clone(), v2.clone(), s1.clone(), s2.clone(), log_n);

    // Create initial value for C (inner product of v1 and v2)
    let c = ArkBn254Pairing::multi_pair(&v1, &v2);

    // Create the initial values for D1 and D2
    let d_1 = ArkBn254Pairing::multi_pair(&v1, &prover_setup.g2_vec()[..1 << log_n]);
    let d_2 = ArkBn254Pairing::multi_pair(&prover_setup.g1_vec()[..1 << log_n], &v2);

    // Create the initial values for E1 and E2
    let e_1 = OptimizedMsmG1::msm(&prover_setup.g1_vec()[..1 << log_n], &s2);
    let e_2 = OptimizedMsmG2::msm(&prover_setup.g2_vec()[..1 << log_n], &s1);

    // Create verifier state
    let verifier_state = DoryVerifierState::new_with_s(c, d_1, d_2, e_1, e_2, s1, s2, log_n);

    // ----- Proof generation phase -----
    println!("Generating proof...");

    // Create proof builder
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);

    // Generate proof
    let proof_builder = inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
        builder,
        prover_state,
        &prover_setup,
        log_n,
    );

    // ----- Tamper with the proof -----
    println!("\n=== Testing tampered proofs ===");

    // Test Case 1: Tamper with a first message
    {
        println!("\n--- Test 1: Tampering with first reduce message ---");
        let mut corrupt_proof = DoryProofBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, _> {
            first_messages: proof_builder.first_messages.clone(),
            second_messages: proof_builder.second_messages.clone(),
            final_message: proof_builder.final_message.clone(),
            transcript: proof_builder.transcript.clone(),
            _phantom: std::marker::PhantomData,
            vmv_message: None,
        };

        if !corrupt_proof.first_messages.is_empty() {
            // Corrupt d1_left in the first message
            println!("Corrupting d1_left in first message...");
            let corrupt_d1_left = Fq12::random(&mut rng);
            corrupt_proof.first_messages[0].d1_left = corrupt_d1_left;

            let verify_transcript = ToyTranscript::new(domain);

            // create a verifier
            let verify_builder = DoryVerifyBuilder::<
                G1Affine,
                G2AffineWrapper,
                Fq12,
                Fr,
                ToyTranscript,
            >::new_from_proof(corrupt_proof, verify_transcript);

            // Test verification
            println!("Verifying corrupted proof...");
            let result =
                inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

            println!("Verification result: {:?}", result);
            assert!(
                result.is_err(),
                "Corrupted first message should cause verification to fail"
            );
            if let Err(round) = result {
                println!("Verification correctly failed at round: {}", round);
            }
        }
    }

    println!("Total test time: {:?}", total_start.elapsed());
    println!("All tests completed successfully!");
}
