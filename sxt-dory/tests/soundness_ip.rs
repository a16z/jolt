#![allow(missing_docs)]
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

// Helper function to generate test vectors and states
fn setup_test_environment(
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
fn test_soundness_tamper_d1_left() {
    println!("=== Testing soundness: tampering with d1_left ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with d1_left
    if !proof_builder.first_messages.is_empty() {
        proof_builder.first_messages[0].d1_left = Fq12::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted d1_left"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_d1_right() {
    println!("=== Testing soundness: tampering with d1_right ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with d1_right
    if !proof_builder.first_messages.is_empty() {
        proof_builder.first_messages[0].d1_right = Fq12::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted d1_right"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_d2_left() {
    println!("=== Testing soundness: tampering with d2_left ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with d2_left
    if !proof_builder.first_messages.is_empty() {
        proof_builder.first_messages[0].d2_left = Fq12::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted d2_left"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_d2_right() {
    println!("=== Testing soundness: tampering with d2_right ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with d2_right
    if !proof_builder.first_messages.is_empty() {
        proof_builder.first_messages[0].d2_right = Fq12::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted d2_right"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_e1_beta() {
    println!("=== Testing soundness: tampering with e1_beta ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with e1_beta
    if !proof_builder.first_messages.is_empty() {
        proof_builder.first_messages[0].e1_beta = G1Affine::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted e1_beta"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_e2_beta() {
    println!("=== Testing soundness: tampering with e2_beta ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with e2_beta
    if !proof_builder.first_messages.is_empty() {
        proof_builder.first_messages[0].e2_beta = G2AffineWrapper::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted e2_beta"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_c_plus() {
    println!("=== Testing soundness: tampering with c_plus ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with c_plus
    if !proof_builder.second_messages.is_empty() {
        proof_builder.second_messages[0].c_plus = Fq12::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted c_plus"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_c_minus() {
    println!("=== Testing soundness: tampering with c_minus ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with c_minus
    if !proof_builder.second_messages.is_empty() {
        proof_builder.second_messages[0].c_minus = Fq12::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted c_minus"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_e1_plus() {
    println!("=== Testing soundness: tampering with e1_plus ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with e1_plus
    if !proof_builder.second_messages.is_empty() {
        proof_builder.second_messages[0].e1_plus = G1Affine::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted e1_plus"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_e1_minus() {
    println!("=== Testing soundness: tampering with e1_minus ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with e1_minus
    if !proof_builder.second_messages.is_empty() {
        proof_builder.second_messages[0].e1_minus = G1Affine::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted e1_minus"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_e2_plus() {
    println!("=== Testing soundness: tampering with e2_plus ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with e2_plus
    if !proof_builder.second_messages.is_empty() {
        proof_builder.second_messages[0].e2_plus = G2AffineWrapper::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted e2_plus"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_e2_minus() {
    println!("=== Testing soundness: tampering with e2_minus ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with e2_minus
    if !proof_builder.second_messages.is_empty() {
        proof_builder.second_messages[0].e2_minus = G2AffineWrapper::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted e2_minus"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_final_e1() {
    println!("=== Testing soundness: tampering with final e1 ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with final e1
    if let Some(final_msg) = &mut proof_builder.final_message {
        final_msg.e1 = G1Affine::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted final e1"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_final_e2() {
    println!("=== Testing soundness: tampering with final e2 ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with final e2
    if let Some(final_msg) = &mut proof_builder.final_message {
        final_msg.e2 = G2AffineWrapper::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with corrupted final e2"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_swap_d1_values() {
    println!("=== Testing soundness: swapping d1_left and d1_right ===");
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Swap d1_left and d1_right
    if !proof_builder.first_messages.is_empty() {
        let temp = proof_builder.first_messages[0].d1_left.clone();
        proof_builder.first_messages[0].d1_left = proof_builder.first_messages[0].d1_right.clone();
        proof_builder.first_messages[0].d1_right = temp;

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with swapped d1 values"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_swap_c_values() {
    println!("=== Testing soundness: swapping c_plus and c_minus ===");
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Swap c_plus and c_minus
    if !proof_builder.second_messages.is_empty() {
        let temp = proof_builder.second_messages[0].c_plus.clone();
        proof_builder.second_messages[0].c_plus = proof_builder.second_messages[0].c_minus.clone();
        proof_builder.second_messages[0].c_minus = temp;

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with swapped c values"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_scale_d1_values() {
    println!("=== Testing soundness: scaling d1 values ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Scale d1 values by a random factor
    if !proof_builder.first_messages.is_empty() {
        let scale = Fr::random(&mut rng);
        proof_builder.first_messages[0].d1_left =
            proof_builder.first_messages[0].d1_left.scale(&scale);
        proof_builder.first_messages[0].d1_right =
            proof_builder.first_messages[0].d1_right.scale(&scale);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with scaled values"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_multi_round_tampering() {
    println!("=== Testing soundness: tampering across multiple rounds ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with messages in different rounds
    if proof_builder.first_messages.len() >= 2 {
        // Tamper with first round
        proof_builder.first_messages[0].d1_left = Fq12::random(&mut rng);
        // Tamper with second round
        proof_builder.first_messages[1].e1_beta = G1Affine::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with multi-round tampering"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_tamper_last_round() {
    println!("=== Testing soundness: tampering with last round ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Tamper with the last round
    let last_round = proof_builder.first_messages.len() - 1;
    if !proof_builder.first_messages.is_empty() {
        proof_builder.first_messages[last_round].d2_right = Fq12::random(&mut rng);

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail with last round tampering"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}

#[test]
fn test_soundness_maintain_sum_attack() {
    println!("=== Testing soundness: maintaining sum but with wrong values ===");
    let mut rng = test_rng();
    let domain = b"test_domain";
    let log_n = 8;

    let (prover_setup, verifier_setup, prover_state, verifier_state) =
        setup_test_environment(log_n);

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

    // Maintain sum but use wrong individual values
    if !proof_builder.first_messages.is_empty() {
        // Get the sum of d1_left and d1_right
        let d1_sum = proof_builder.first_messages[0]
            .d1_left
            .add(&proof_builder.first_messages[0].d1_right);

        // Create new values that maintain the sum but are individually wrong
        let random_val = Fq12::random(&mut rng);
        proof_builder.first_messages[0].d1_left = random_val.clone();
        proof_builder.first_messages[0].d1_right = d1_sum - random_val;

        let verify_builder =
            DoryVerifyBuilder::<G1Affine, G2AffineWrapper, Fq12, Fr, ToyTranscript>::new_from_proof(
                proof_builder,
                ToyTranscript::new(domain),
            );
        let result = inner_product_verify(verify_builder, verifier_state, &verifier_setup, log_n);

        assert!(
            result.is_err(),
            "Verification should fail even with maintained sum"
        );
        if let Err(round) = result {
            println!("✓ Verification correctly failed at round: {}", round);
        }
    }
}
