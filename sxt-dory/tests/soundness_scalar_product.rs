#![allow(missing_docs)]
use ark_bn254::{Fq12, Fr, G1Affine};
use ark_ff::UniformRand;
use dory::{
    arithmetic::{Field, Group, MultiScalarMul, Pairing},
    builder::{DoryProofBuilder, DoryVerifyBuilder},
    inner_product::{inner_product_prove, inner_product_verify},
    messages::ScalarProductMessage,
    setup::ProverSetup,
    state::{DoryProverState, DoryVerifierState},
    toy_transcript::ToyTranscript,
};

use dory::curve::{test_rng, ArkBn254Pairing, G2AffineWrapper, OptimizedMsmG1, OptimizedMsmG2};

// Helper function to generate test environment
fn setup_scalar_product_test_environment(
    log_n: usize,
) -> (
    ProverSetup<ArkBn254Pairing>,
    dory::setup::VerifierSetup<ArkBn254Pairing>,
    DoryProverState<ArkBn254Pairing>,
    DoryVerifierState<ArkBn254Pairing>,
) {
    let mut rng = test_rng();
    let vector_size = 1 << log_n;

    // Setup - max_log_n should be 2 * log_n because g1_vec/g2_vec have length sqrt(n)
    let prover_setup = ProverSetup::<ArkBn254Pairing>::new(&mut rng, 2 * log_n);
    let verifier_setup = prover_setup.to_verifier_setup();

    // Generate vectors
    let v1: Vec<G1Affine> = (0..vector_size)
        .map(|_| G1Affine::random(&mut rng))
        .collect();
    let v2: Vec<G2AffineWrapper> = (0..vector_size)
        .map(|_| G2AffineWrapper::random(&mut rng))
        .collect();
    let s1: Vec<Fr> = (0..vector_size).map(|_| Fr::random(&mut rng)).collect();
    let s2: Vec<Fr> = (0..vector_size).map(|_| Fr::random(&mut rng)).collect();

    // Create states
    let prover_state = DoryProverState::new(v1.clone(), v2.clone(), s1.clone(), s2.clone(), log_n);
    let c = ArkBn254Pairing::multi_pair(&v1, &v2);
    let d_1 = ArkBn254Pairing::multi_pair(&v1, &prover_setup.g2_vec()[..1 << log_n]);
    let d_2 = ArkBn254Pairing::multi_pair(&prover_setup.g1_vec()[..1 << log_n], &v2);
    let e_1 = OptimizedMsmG1::msm(&prover_setup.g1_vec()[..1 << log_n], &s2);
    let e_2 = OptimizedMsmG2::msm(&prover_setup.g2_vec()[..1 << log_n], &s1);
    let verifier_state = DoryVerifierState::new_with_s(c, d_1, d_2, e_1, e_2, s1, s2, log_n);

    (prover_setup, verifier_setup, prover_state, verifier_state)
}

#[test]
fn test_soundness_scalar_product_wrong_e1() {
    println!("=== Testing soundness: scalar product with wrong E1 ===");
    let mut rng = test_rng();
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    // Tamper with final scalar product message E1
    if let Some(final_msg) = &mut proof_builder.final_message {
        println!("Tampering with scalar product E1...");
        final_msg.e1 = G1Affine::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(result.is_err(), "Verification should fail with wrong E1");
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_scalar_product_wrong_e2() {
    println!("=== Testing soundness: scalar product with wrong E2 ===");
    let mut rng = test_rng();
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    // Tamper with final scalar product message E2
    if let Some(final_msg) = &mut proof_builder.final_message {
        println!("Tampering with scalar product E2...");
        final_msg.e2 = G2AffineWrapper::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(result.is_err(), "Verification should fail with wrong E2");
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_scalar_product_both_wrong() {
    println!("=== Testing soundness: scalar product with both E1 and E2 wrong ===");
    let mut rng = test_rng();
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    // Tamper with both E1 and E2
    if let Some(final_msg) = &mut proof_builder.final_message {
        println!("Tampering with both scalar product E1 and E2...");
        final_msg.e1 = G1Affine::random(&mut rng);
        final_msg.e2 = G2AffineWrapper::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with both E1 and E2 wrong"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_scalar_product_scaled_values() {
    println!("=== Testing soundness: scalar product with scaled E1 and E2 ===");
    let mut rng = test_rng();
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    // Scale both E1 and E2 by some factor
    if let Some(final_msg) = &mut proof_builder.final_message {
        println!("Scaling scalar product E1 and E2...");
        let scale = Fr::random(&mut rng);
        let scale_inv = scale.inv().unwrap();

        // Scale E1 by scale and E2 by scale_inv to try to maintain the pairing
        final_msg.e1 = final_msg.e1.scale(&scale);
        final_msg.e2 = final_msg.e2.scale(&scale_inv);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with scaled E1 and E2"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_scalar_product_relationship_attack() {
    println!("=== Testing soundness: scalar product relationship attack ===");
    let domain = b"scalar_product_test";
    let log_n = 8;

    // Generate two different test environments
    let (prover_setup1, _, prover_state1, _) = setup_scalar_product_test_environment(log_n);
    let (prover_setup2, verifier_setup, prover_state2, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof for first state
    let builder1 = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let proof1 = inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
        builder1,
        prover_state1,
        &prover_setup1,
        log_n,
    );

    // Generate proof for second state
    let builder2 = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof2 = inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
        builder2,
        prover_state2,
        &prover_setup2,
        log_n,
    );

    // Mix scalar product messages from different proofs
    if let (Some(final_msg1), Some(final_msg2)) = (&proof1.final_message, &mut proof2.final_message)
    {
        println!("Mixing scalar product messages from different proofs...");
        // Take E1 from proof1 but keep E2 from proof2
        final_msg2.e1 = final_msg1.e1.clone();

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof2,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with mixed scalar product messages"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_scalar_product_missing_message() {
    println!("=== Testing soundness: missing scalar product message ===");
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    // Remove the final scalar product message
    println!("Removing scalar product message...");
    proof_builder.final_message = None;

    let verify_builder =
        DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
            proof_builder,
            ToyTranscript::new(domain),
        );
    let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

    assert!(
        result.is_err(),
        "Verification should fail with missing scalar product message"
    );
    if let Err(round) = result {
        println!("✓ Verification correctly failed at round: {}", round);
    }
}

#[test]
fn test_soundness_scalar_product_pairing_check() {
    println!("=== Testing soundness: scalar product pairing equation check ===");
    let mut rng = test_rng();
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    // Create E1 and E2 that don't satisfy the pairing equation
    if let Some(final_msg) = &mut proof_builder.final_message {
        println!("Creating E1 and E2 that violate pairing equation...");

        // Use random elements that are unlikely to satisfy the verification equation
        let random_e1 = G1Affine::random(&mut rng);
        let random_e2 = G2AffineWrapper::random(&mut rng);

        final_msg.e1 = random_e1;
        final_msg.e2 = random_e2;

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail when pairing equation is not satisfied"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_scalar_product_after_valid_rounds() {
    println!("=== Testing soundness: tampering scalar product after valid rounds ===");
    let mut rng = test_rng();
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate a valid proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    // Ensure all rounds are valid but tamper only with final scalar product
    if let Some(final_msg) = &mut proof_builder.final_message {
        println!("Tampering only the final scalar product message...");

        // Make a small change to E1
        let tampered_e1 = final_msg.e1.add(&G1Affine::rand(&mut rng));
        final_msg.e1 = tampered_e1;

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail even with small tampering in scalar product"
        );
        if let Err(round) = result {
            println!(
                "✓ Verification correctly failed at round: {} (should be final round)",
                round
            );
            assert_eq!(
                round, log_n,
                "Should fail at the final scalar product verification"
            );
        }
    }
}

#[test]
fn test_soundness_scalar_product_identity_elements() {
    println!("=== Testing soundness: scalar product with identity elements ===");
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    // Set E1 and E2 to identity elements
    if let Some(final_msg) = &mut proof_builder.final_message {
        println!("Setting scalar product E1 and E2 to identity elements...");
        final_msg.e1 = G1Affine::identity();
        final_msg.e2 = G2AffineWrapper::identity();

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with identity elements"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_scalar_product_consistency_check() {
    println!("=== Testing soundness: scalar product consistency with inner product state ===");
    let mut rng = test_rng();
    let domain = b"scalar_product_test";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_scalar_product_test_environment(log_n);

    // Generate proof
    let builder = DoryProofBuilder::<
        G1Affine,
        G2AffineWrapper,
        Fq12,
        Fr,
        ToyTranscript,
    >::new_with_toy_transcript(domain);
    let mut proof_builder =
        inner_product_prove::<_, _, _, _, _, _, _, OptimizedMsmG1, OptimizedMsmG2>(
            builder,
            prover_state,
            &prover_setup,
            log_n,
        );

    // Create a new scalar product message that's inconsistent with the folded state
    if let Some(final_msg) = &mut proof_builder.final_message {
        println!("Creating inconsistent scalar product message...");

        // Generate completely new v1 and v2
        let new_v1 = G1Affine::random(&mut rng);
        let new_v2 = G2AffineWrapper::random(&mut rng);

        // Create scalar product message from different vectors
        let new_msg = ScalarProductMessage {
            e1: new_v1,
            e2: new_v2,
        };

        *final_msg = new_msg;

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with inconsistent scalar product message"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}
