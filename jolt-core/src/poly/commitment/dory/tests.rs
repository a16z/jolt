#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::super::*;
    use crate::field::JoltField;
    use crate::poly::commitment::commitment_scheme::CommitmentScheme;
    use crate::poly::commitment::dory::{bind_opening_inputs, DoryContext};
    use crate::poly::dense_mlpoly::DensePolynomial;
    use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
    use crate::transcripts::{Blake2bTranscript, Transcript};
    use ark_ff::biginteger::S128;
    use ark_std::rand::{thread_rng, Rng};
    use ark_std::{UniformRand, Zero};
    use serial_test::serial;
    type Fr = ark_bn254::Fr;

    fn test_commitment_scheme_with_poly(
        poly: MultilinearPolynomial<Fr>,
        poly_type_name: &str,
        prover_setup: &ArkworksProverSetup,
        verifier_setup: &ArkworksVerifierSetup,
    ) {
        let num_vars = poly.get_num_vars();

        let mut rng = thread_rng();
        let opening_point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        let (commitment, row_commitments) = DoryCommitmentScheme::commit(&poly, prover_setup);

        let evaluation = <MultilinearPolynomial<Fr> as PolynomialEvaluation<Fr>>::evaluate(
            &poly,
            &opening_point,
        );

        let mut prove_transcript = Blake2bTranscript::new(b"dory_test");
        bind_opening_inputs::<Fr, _>(&mut prove_transcript, &opening_point, &evaluation);
        let (proof, _y_blinding) = DoryCommitmentScheme::prove(
            prover_setup,
            &poly,
            &opening_point,
            Some(row_commitments),
            &mut prove_transcript,
        );

        let mut verify_transcript = Blake2bTranscript::new(b"dory_test");
        bind_opening_inputs::<Fr, _>(&mut verify_transcript, &opening_point, &evaluation);
        let verification_result = DoryCommitmentScheme::verify(
            &proof,
            verifier_setup,
            &mut verify_transcript,
            &opening_point,
            &evaluation,
            &commitment,
        );

        assert!(
            verification_result.is_ok(),
            "Dory verification failed for {poly_type_name}: {verification_result:?}"
        );
    }

    fn setup_dory_for_test(num_vars: usize) -> (ArkworksProverSetup, ArkworksVerifierSetup) {
        // Reset globals to ensure clean state
        DoryGlobals::reset();

        let num_coeffs = 1 << num_vars;
        // Dense polynomial: K = 1, T = num_coeffs
        let _guard = DoryGlobals::initialize_context(1, num_coeffs, DoryContext::Main, None);

        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        (prover_setup, verifier_setup)
    }

    #[test]
    #[serial]
    fn test_dory_large_scalars() {
        let num_vars = 10;
        let num_coeffs = 1 << num_vars;
        let (prover_setup, verifier_setup) = setup_dory_for_test(num_vars);

        let mut rng = thread_rng();
        let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs));

        test_commitment_scheme_with_poly(poly, "LargeScalars", &prover_setup, &verifier_setup);
    }

    #[test]
    #[serial]
    fn test_dory_bool_scalars() {
        let num_vars = 10;
        let num_coeffs = 1 << num_vars;
        let (prover_setup, verifier_setup) = setup_dory_for_test(num_vars);

        let mut rng = thread_rng();
        let coeffs: Vec<bool> = (0..num_coeffs).map(|_| rng.gen::<bool>()).collect();
        let poly: MultilinearPolynomial<Fr> = coeffs.into();

        test_commitment_scheme_with_poly(poly, "BoolScalars", &prover_setup, &verifier_setup);
    }

    #[test]
    #[serial]
    fn test_dory_u8_scalars() {
        let num_vars = 10;
        let num_coeffs = 1 << num_vars;
        let (prover_setup, verifier_setup) = setup_dory_for_test(num_vars);

        let mut rng = thread_rng();
        let coeffs: Vec<u8> = (0..num_coeffs).map(|_| rng.gen::<u8>()).collect();
        let poly: MultilinearPolynomial<Fr> = coeffs.into();

        test_commitment_scheme_with_poly(poly, "U8Scalars", &prover_setup, &verifier_setup);
    }

    #[test]
    #[serial]
    fn test_dory_u16_scalars() {
        let num_vars = 10;
        let num_coeffs = 1 << num_vars;
        let (prover_setup, verifier_setup) = setup_dory_for_test(num_vars);

        let mut rng = thread_rng();
        let coeffs: Vec<u16> = (0..num_coeffs).map(|_| rng.gen::<u16>()).collect();
        let poly: MultilinearPolynomial<Fr> = coeffs.into();

        test_commitment_scheme_with_poly(poly, "U16Scalars", &prover_setup, &verifier_setup);
    }

    #[test]
    #[serial]
    fn test_dory_u32_scalars() {
        let num_vars = 10;
        let num_coeffs = 1 << num_vars;
        let (prover_setup, verifier_setup) = setup_dory_for_test(num_vars);

        let mut rng = thread_rng();
        let coeffs: Vec<u32> = (0..num_coeffs).map(|_| rng.gen::<u32>()).collect();
        let poly: MultilinearPolynomial<Fr> = coeffs.into();

        test_commitment_scheme_with_poly(poly, "U32Scalars", &prover_setup, &verifier_setup);
    }

    #[test]
    #[serial]
    fn test_dory_u64_scalars() {
        let num_vars = 10;
        let num_coeffs = 1 << num_vars;
        let (prover_setup, verifier_setup) = setup_dory_for_test(num_vars);

        let mut rng = thread_rng();
        let coeffs: Vec<u64> = (0..num_coeffs).map(|_| rng.gen::<u64>()).collect();
        let poly: MultilinearPolynomial<Fr> = coeffs.into();

        test_commitment_scheme_with_poly(poly, "U64Scalars", &prover_setup, &verifier_setup);
    }

    #[test]
    #[serial]
    fn test_dory_u128_scalars() {
        let num_vars = 10;
        let num_coeffs = 1 << num_vars;
        let (prover_setup, verifier_setup) = setup_dory_for_test(num_vars);

        let mut rng = thread_rng();
        let coeffs: Vec<u128> = (0..num_coeffs).map(|_| rng.gen::<u128>()).collect();
        let poly: MultilinearPolynomial<Fr> = coeffs.into();

        test_commitment_scheme_with_poly(poly, "U128Scalars", &prover_setup, &verifier_setup);
    }

    #[test]
    #[serial]
    fn test_dory_i64_scalars() {
        let num_vars = 10;
        let num_coeffs = 1 << num_vars;
        let (prover_setup, verifier_setup) = setup_dory_for_test(num_vars);

        let mut rng = thread_rng();
        let coeffs: Vec<i64> = (0..num_coeffs).map(|_| rng.gen::<i64>()).collect();
        let poly: MultilinearPolynomial<Fr> = coeffs.into();

        test_commitment_scheme_with_poly(poly, "I64Scalars", &prover_setup, &verifier_setup);
    }

    #[test]
    #[serial]
    fn test_dory_i128_scalars() {
        let num_vars = 10;
        let num_coeffs = 1 << num_vars;
        let (prover_setup, verifier_setup) = setup_dory_for_test(num_vars);

        let mut rng = thread_rng();
        // MSM only supports i128 values in the range [-u64::MAX, u64::MAX]
        let coeffs: Vec<i128> = (0..num_coeffs)
            .map(|_| {
                let val = rng.gen::<i64>() as i128;
                if rng.gen::<bool>() {
                    val
                } else {
                    -val
                }
            })
            .collect();
        let poly: MultilinearPolynomial<Fr> = coeffs.into();

        test_commitment_scheme_with_poly(poly, "I128Scalars", &prover_setup, &verifier_setup);
    }

    #[test]
    #[serial]
    fn test_dory_s128_scalars() {
        let num_vars = 10;
        let num_coeffs = 1 << num_vars;
        let (prover_setup, verifier_setup) = setup_dory_for_test(num_vars);

        let mut rng = thread_rng();
        // S128 is a signed 128-bit integer type from arkworks
        // MSM only supports i128 values in the range [-u64::MAX, u64::MAX]
        let coeffs: Vec<S128> = (0..num_coeffs)
            .map(|_| {
                let val = rng.gen::<i64>() as i128;
                S128::from(if rng.gen::<bool>() { val } else { -val })
            })
            .collect();
        let poly: MultilinearPolynomial<Fr> = coeffs.into();

        test_commitment_scheme_with_poly(poly, "S128Scalars", &prover_setup, &verifier_setup);
    }

    #[test]
    #[serial]
    fn test_dory_soundness() {
        use ark_std::UniformRand;

        // Reset globals to ensure clean state
        DoryGlobals::reset();

        let num_vars = 10;
        let num_coeffs = 1 << num_vars;

        // Dense polynomial: K = 1, T = num_coeffs
        let _guard = DoryGlobals::initialize_context(1, num_coeffs, DoryContext::Main, None);

        let mut rng = thread_rng();
        let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs.clone()));

        let opening_point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        let (commitment, row_commitments) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        let mut prove_transcript = Blake2bTranscript::new(DoryCommitmentScheme::protocol_name());

        let correct_evaluation = poly.evaluate(&opening_point);

        bind_opening_inputs::<Fr, _>(&mut prove_transcript, &opening_point, &correct_evaluation);
        let (proof, _y_blinding) = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &opening_point,
            Some(row_commitments),
            &mut prove_transcript,
        );

        // Test 1: Tamper with the evaluation
        {
            let tampered_evaluation = Fr::rand(&mut rng);

            let mut verify_transcript =
                Blake2bTranscript::new(DoryCommitmentScheme::protocol_name());
            bind_opening_inputs::<Fr, _>(
                &mut verify_transcript,
                &opening_point,
                &tampered_evaluation,
            );
            let result = DoryCommitmentScheme::verify(
                &proof,
                &verifier_setup,
                &mut verify_transcript,
                &opening_point,
                &tampered_evaluation,
                &commitment,
            );

            assert!(
                result.is_err(),
                "Verification should fail with tampered evaluation"
            );
        }

        // Test 1b: Tamper with the committed evaluation in ZK proofs
        #[cfg(all(feature = "prover", feature = "zk"))]
        {
            let mut tampered_proof = proof.clone();
            if let Some(ref mut y_com) = tampered_proof.y_com {
                *y_com = *y_com + verifier_setup.g1_0;
            } else if let Some(ref mut e2) = tampered_proof.e2 {
                *e2 = *e2 + verifier_setup.g2_0;
            } else {
                panic!("ZK proof missing committed evaluation fields");
            }

            let mut verify_transcript =
                Blake2bTranscript::new(DoryCommitmentScheme::protocol_name());
            bind_opening_inputs::<Fr, _>(
                &mut verify_transcript,
                &opening_point,
                &correct_evaluation,
            );
            let result = DoryCommitmentScheme::verify(
                &tampered_proof,
                &verifier_setup,
                &mut verify_transcript,
                &opening_point,
                &correct_evaluation,
                &commitment,
            );

            assert!(
                result.is_err(),
                "Verification should fail with tampered committed evaluation"
            );
        }

        // Test 2: Tamper with the opening point
        {
            let tampered_opening_point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
                .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
                .collect();

            let mut verify_transcript =
                Blake2bTranscript::new(DoryCommitmentScheme::protocol_name());
            bind_opening_inputs::<Fr, _>(
                &mut verify_transcript,
                &tampered_opening_point,
                &correct_evaluation,
            );
            let result = DoryCommitmentScheme::verify(
                &proof,
                &verifier_setup,
                &mut verify_transcript,
                &tampered_opening_point,
                &correct_evaluation,
                &commitment,
            );

            assert!(
                result.is_err(),
                "Verification should fail with tampered opening point"
            );
        }

        // Test 3: Use wrong commitment
        {
            // Create a different polynomial and its commitment
            let wrong_coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
            let wrong_poly =
                MultilinearPolynomial::LargeScalars(DensePolynomial::new(wrong_coeffs));
            let (wrong_commitment, _) = DoryCommitmentScheme::commit(&wrong_poly, &prover_setup);

            let mut verify_transcript =
                Blake2bTranscript::new(DoryCommitmentScheme::protocol_name());
            bind_opening_inputs::<Fr, _>(
                &mut verify_transcript,
                &opening_point,
                &correct_evaluation,
            );
            let result = DoryCommitmentScheme::verify(
                &proof,
                &verifier_setup,
                &mut verify_transcript,
                &opening_point,
                &correct_evaluation,
                &wrong_commitment,
            );

            assert!(
                result.is_err(),
                "Verification should fail with wrong commitment"
            );
        }

        // Test 4: Use wrong domain in transcript
        {
            let mut verify_transcript = Blake2bTranscript::new(b"wrong_domain");
            bind_opening_inputs::<Fr, _>(
                &mut verify_transcript,
                &opening_point,
                &correct_evaluation,
            );
            let result = DoryCommitmentScheme::verify(
                &proof,
                &verifier_setup,
                &mut verify_transcript,
                &opening_point,
                &correct_evaluation,
                &commitment,
            );

            assert!(
                result.is_err(),
                "Verification should fail with wrong transcript domain"
            );
        }

        // Test 5: Verify that correct proof still passes
        {
            let mut verify_transcript =
                Blake2bTranscript::new(DoryCommitmentScheme::protocol_name());
            bind_opening_inputs::<Fr, _>(
                &mut verify_transcript,
                &opening_point,
                &correct_evaluation,
            );
            let result = DoryCommitmentScheme::verify(
                &proof,
                &verifier_setup,
                &mut verify_transcript,
                &opening_point,
                &correct_evaluation,
                &commitment,
            );

            assert!(
                result.is_ok(),
                "Verification should succeed with correct proof"
            );
        }
    }

    #[test]
    #[serial]
    fn test_dory_one_hot() {
        use crate::poly::one_hot_polynomial::OneHotPolynomial;

        DoryGlobals::reset();

        // Use K=32, T=32 to ensure the test exercises both row and column variables
        // in the Dory matrix (log2(32*32) = 10 variables, split as sigma=5, nu=5)
        let K = 32;
        let T = 32;

        let _guard = DoryGlobals::initialize_context(K, T, DoryContext::Main, None);

        let mut rng = thread_rng();
        let nonzero_indices: Vec<Option<u8>> = (0..T)
            .map(|_| {
                if rng.gen::<bool>() {
                    Some(rng.gen::<u8>() % K as u8)
                } else {
                    None
                }
            })
            .collect();

        let one_hot_poly = OneHotPolynomial::from_indices(nonzero_indices, K);
        let num_vars = one_hot_poly.get_num_vars();
        let poly = MultilinearPolynomial::OneHot(one_hot_poly);

        let opening_point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        let (commitment, row_commitments) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        let evaluation = <MultilinearPolynomial<Fr> as PolynomialEvaluation<Fr>>::evaluate(
            &poly,
            &opening_point,
        );

        let mut prove_transcript = Blake2bTranscript::new(b"dory_test");
        bind_opening_inputs::<Fr, _>(&mut prove_transcript, &opening_point, &evaluation);
        let (proof, _y_blinding) = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &opening_point,
            Some(row_commitments),
            &mut prove_transcript,
        );

        let mut verify_transcript = Blake2bTranscript::new(b"dory_test");
        bind_opening_inputs::<Fr, _>(&mut verify_transcript, &opening_point, &evaluation);
        let verification_result = DoryCommitmentScheme::verify(
            &proof,
            &verifier_setup,
            &mut verify_transcript,
            &opening_point,
            &evaluation,
            &commitment,
        );

        assert!(
            verification_result.is_ok(),
            "Dory verification failed for OneHot: {verification_result:?}"
        );
    }

    #[test]
    #[serial]
    fn test_dory_homomorphic_combination() {
        DoryGlobals::reset();

        let num_vars = 10;
        let num_coeffs = 1 << num_vars;
        let num_polys = 5;

        let _guard = DoryGlobals::initialize_context(1, num_coeffs, DoryContext::Main, None);

        let mut rng = thread_rng();

        // Step 1: Generate 5 random polynomials
        let polys: Vec<MultilinearPolynomial<Fr>> = (0..num_polys)
            .map(|_| {
                let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
                MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs))
            })
            .collect();

        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        // Step 2: Commit to each polynomial
        let commitments_and_hints: Vec<_> = polys
            .iter()
            .map(|poly| DoryCommitmentScheme::commit(poly, &prover_setup))
            .collect();

        let commitments: Vec<_> = commitments_and_hints.iter().map(|(c, _)| *c).collect();
        let hints: Vec<_> = commitments_and_hints.into_iter().map(|(_, h)| h).collect();

        // Step 3: Generate 5 random coefficients
        let coeffs: Vec<Fr> = (0..num_polys).map(|_| Fr::rand(&mut rng)).collect();

        // Step 4: Homomorphically combine commitments and hints
        let combined_commitment = DoryCommitmentScheme::combine_commitments(&commitments, &coeffs);
        let combined_hint = DoryCommitmentScheme::combine_hints(hints, &coeffs);

        // Step 5: Generate evaluation point first
        let opening_point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        // Step 6: Compute expected evaluation as linear combination: eval = coeff[0]*P0(r) + ... + coeff[4]*P4(r)
        let mut evaluation = Fr::zero();
        for (poly, coeff) in polys.iter().zip(coeffs.iter()) {
            let poly_eval = poly.evaluate(&opening_point);
            evaluation += *coeff * poly_eval;
        }

        // Step 7: Compute combined polynomial: P = coeff[0]*P0 + coeff[1]*P1 + ... + coeff[4]*P4
        let poly_refs: Vec<&MultilinearPolynomial<Fr>> = polys.iter().collect();
        let combined_poly = DensePolynomial::linear_combination(&poly_refs, &coeffs);
        let combined_poly = MultilinearPolynomial::from(combined_poly.Z);

        // Step 8: Create evaluation proof using combined commitment and hint
        let mut prove_transcript = Blake2bTranscript::new(b"dory_homomorphic_test");
        bind_opening_inputs::<Fr, _>(&mut prove_transcript, &opening_point, &evaluation);
        let (proof, _y_blinding) = DoryCommitmentScheme::prove(
            &prover_setup,
            &combined_poly,
            &opening_point,
            Some(combined_hint),
            &mut prove_transcript,
        );

        // Step 9: Verify the proof
        let mut verify_transcript = Blake2bTranscript::new(b"dory_homomorphic_test");
        bind_opening_inputs::<Fr, _>(&mut verify_transcript, &opening_point, &evaluation);
        let result = DoryCommitmentScheme::verify(
            &proof,
            &verifier_setup,
            &mut verify_transcript,
            &opening_point,
            &evaluation,
            &combined_commitment,
        );

        assert!(
            result.is_ok(),
            "Verification should succeed for homomorphically combined commitment: {result:?}"
        );
    }

    #[test]
    #[serial]
    fn test_dory_batch_commit_e2e() {
        DoryGlobals::reset();

        let num_vars = 10;
        let num_coeffs = 1 << num_vars;
        let num_polys = 5;

        let _guard = DoryGlobals::initialize_context(1, num_coeffs, DoryContext::Main, None);

        let mut rng = thread_rng();

        // Step 1: Generate 5 random polynomials
        let polys: Vec<MultilinearPolynomial<Fr>> = (0..num_polys)
            .map(|_| {
                let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();
                MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs))
            })
            .collect();

        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        // Step 2: Use batch_commit
        let commitments_and_hints = DoryCommitmentScheme::batch_commit(&polys, &prover_setup);

        let commitments: Vec<_> = commitments_and_hints.iter().map(|(c, _)| *c).collect();
        let hints: Vec<_> = commitments_and_hints.into_iter().map(|(_, h)| h).collect();

        // Step 3: Generate random coefficients (like gamma powers in opening_proof.rs)
        let coeffs: Vec<Fr> = (0..num_polys).map(|_| Fr::rand(&mut rng)).collect();

        // Step 4: Homomorphically combine commitments and hints
        let combined_commitment = DoryCommitmentScheme::combine_commitments(&commitments, &coeffs);
        let combined_hint = DoryCommitmentScheme::combine_hints(hints, &coeffs);

        // Step 5: Generate evaluation point
        let opening_point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        // Step 6: Compute expected evaluation as linear combination
        let mut evaluation = Fr::zero();
        for (poly, coeff) in polys.iter().zip(coeffs.iter()) {
            let poly_eval = poly.evaluate(&opening_point);
            evaluation += *coeff * poly_eval;
        }

        // Step 7: Create combined polynomial
        let poly_refs: Vec<&MultilinearPolynomial<Fr>> = polys.iter().collect();
        let combined_poly = DensePolynomial::linear_combination(&poly_refs, &coeffs);
        let combined_poly = MultilinearPolynomial::from(combined_poly.Z);

        // Step 8: Verify that directly committing to the combined polynomial gives the same result
        // as homomorphically combining the individual commitments
        let (direct_commitment, direct_hint) =
            DoryCommitmentScheme::commit(&combined_poly, &prover_setup);

        // The commitments should match
        assert_eq!(
            combined_commitment, direct_commitment,
            "Homomorphically combined commitment should match direct commitment to RLC"
        );

        // Step 9: Create evaluation proof using combined hint
        let mut prove_transcript = Blake2bTranscript::new(b"dory_batch_commit_e2e_test");
        bind_opening_inputs::<Fr, _>(&mut prove_transcript, &opening_point, &evaluation);
        let (proof, _y_blinding) = DoryCommitmentScheme::prove(
            &prover_setup,
            &combined_poly,
            &opening_point,
            Some(combined_hint),
            &mut prove_transcript,
        );

        // Step 10: Verify the proof
        let mut verify_transcript = Blake2bTranscript::new(b"dory_batch_commit_e2e_test");
        bind_opening_inputs::<Fr, _>(&mut verify_transcript, &opening_point, &evaluation);
        let result = DoryCommitmentScheme::verify(
            &proof,
            &verifier_setup,
            &mut verify_transcript,
            &opening_point,
            &evaluation,
            &combined_commitment,
        );

        assert!(
            result.is_ok(),
            "Verification should succeed with batch_commit flow: {result:?}"
        );

        // Step 11: Also verify that proving with the direct hint works
        let mut prove_transcript2 = Blake2bTranscript::new(b"dory_batch_commit_e2e_test");
        bind_opening_inputs::<Fr, _>(&mut prove_transcript2, &opening_point, &evaluation);
        let (proof2, _y_blinding2) = DoryCommitmentScheme::prove(
            &prover_setup,
            &combined_poly,
            &opening_point,
            Some(direct_hint),
            &mut prove_transcript2,
        );

        let mut verify_transcript2 = Blake2bTranscript::new(b"dory_batch_commit_e2e_test");
        bind_opening_inputs::<Fr, _>(&mut verify_transcript2, &opening_point, &evaluation);
        let result2 = DoryCommitmentScheme::verify(
            &proof2,
            &verifier_setup,
            &mut verify_transcript2,
            &opening_point,
            &evaluation,
            &direct_commitment,
        );

        assert!(
            result2.is_ok(),
            "Verification should also succeed with direct commitment: {result2:?}"
        );
    }

    #[test]
    fn test_dory_layout_address_cycle_conversions() {
        let K = 4; // 4 addresses
        let T = 8; // 8 cycles

        // Test CycleMajor layout: index = address * T + cycle
        let cycle_major = DoryLayout::CycleMajor;

        // Address 0: indices 0-7, Address 1: indices 8-15, etc.
        assert_eq!(cycle_major.address_cycle_to_index(0, 0, K, T), 0); // addr 0, cycle 0
        assert_eq!(cycle_major.address_cycle_to_index(0, 1, K, T), 1); // addr 0, cycle 1
        assert_eq!(cycle_major.address_cycle_to_index(0, 7, K, T), 7); // addr 0, cycle 7
        assert_eq!(cycle_major.address_cycle_to_index(1, 0, K, T), 8); // addr 1, cycle 0
        assert_eq!(cycle_major.address_cycle_to_index(1, 1, K, T), 9); // addr 1, cycle 1
        assert_eq!(cycle_major.address_cycle_to_index(3, 7, K, T), 31); // addr 3, cycle 7

        // Test reverse: index_to_address_cycle
        assert_eq!(cycle_major.index_to_address_cycle(0, K, T), (0, 0));
        assert_eq!(cycle_major.index_to_address_cycle(1, K, T), (0, 1));
        assert_eq!(cycle_major.index_to_address_cycle(8, K, T), (1, 0));
        assert_eq!(cycle_major.index_to_address_cycle(31, K, T), (3, 7));

        // Test AddressMajor layout: index = cycle * K + address
        let addr_major = DoryLayout::AddressMajor;

        // Cycle 0: indices 0-3, Cycle 1: indices 4-7, etc.
        assert_eq!(addr_major.address_cycle_to_index(0, 0, K, T), 0); // addr 0, cycle 0
        assert_eq!(addr_major.address_cycle_to_index(1, 0, K, T), 1); // addr 1, cycle 0
        assert_eq!(addr_major.address_cycle_to_index(3, 0, K, T), 3); // addr 3, cycle 0
        assert_eq!(addr_major.address_cycle_to_index(0, 1, K, T), 4); // addr 0, cycle 1
        assert_eq!(addr_major.address_cycle_to_index(1, 1, K, T), 5); // addr 1, cycle 1
        assert_eq!(addr_major.address_cycle_to_index(3, 7, K, T), 31); // addr 3, cycle 7

        // Test reverse: index_to_address_cycle
        assert_eq!(addr_major.index_to_address_cycle(0, K, T), (0, 0));
        assert_eq!(addr_major.index_to_address_cycle(1, K, T), (1, 0));
        assert_eq!(addr_major.index_to_address_cycle(4, K, T), (0, 1));
        assert_eq!(addr_major.index_to_address_cycle(31, K, T), (3, 7));

        // Verify round-trip for both layouts
        for addr in 0..K {
            for cycle in 0..T {
                let idx = cycle_major.address_cycle_to_index(addr, cycle, K, T);
                assert_eq!(cycle_major.index_to_address_cycle(idx, K, T), (addr, cycle));

                let idx = addr_major.address_cycle_to_index(addr, cycle, K, T);
                assert_eq!(addr_major.index_to_address_cycle(idx, K, T), (addr, cycle));
            }
        }
    }

    #[test]
    #[serial]
    fn test_dory_layout_global_state() {
        DoryGlobals::reset();

        // Default should be CycleMajor
        assert_eq!(DoryGlobals::get_layout(), DoryLayout::CycleMajor);

        // Set to AddressMajor
        DoryGlobals::set_layout(DoryLayout::AddressMajor);
        assert_eq!(DoryGlobals::get_layout(), DoryLayout::AddressMajor);

        // Set back to CycleMajor
        DoryGlobals::set_layout(DoryLayout::CycleMajor);
        assert_eq!(DoryGlobals::get_layout(), DoryLayout::CycleMajor);
    }

    /// Dense polynomials are treated as k=1, so `AddressMajor` and `CycleMajor`
    /// degenerate to same computation for Dory commitments
    /// Hence, we expect them to produce the same commitment.
    #[test]
    #[serial]
    fn test_dory_layout_dense_polynomials_same_commitment() {
        DoryGlobals::reset();

        let num_vars = 10;
        let num_coeffs = 1 << num_vars;

        let _ = DoryGlobals::initialize_context(1, num_coeffs, DoryContext::Main, None);

        let mut rng = thread_rng();
        let coeffs: Vec<Fr> = (0..num_coeffs).map(|_| Fr::rand(&mut rng)).collect();

        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);

        DoryGlobals::set_layout(DoryLayout::CycleMajor);
        let poly1 = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs.clone()));
        let (commitment_cycle_major, _) = DoryCommitmentScheme::commit(&poly1, &prover_setup);

        DoryGlobals::set_layout(DoryLayout::AddressMajor);
        let poly2 = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coeffs));
        let (commitment_addr_major, _) = DoryCommitmentScheme::commit(&poly2, &prover_setup);

        assert_eq!(
            commitment_cycle_major, commitment_addr_major,
            "Dense polynomials should produce the same commitment with any layout"
        );
        DoryGlobals::set_layout(DoryLayout::CycleMajor);
    }

    #[test]
    fn test_dory_layout_enum_methods() {
        let K = 8; // addresses
        let T = 16; // cycles

        let cycle_major = DoryLayout::CycleMajor;
        let addr_major = DoryLayout::AddressMajor;

        let addr = 3;
        let cycle = 7;

        let idx_cycle = cycle_major.address_cycle_to_index(addr, cycle, K, T);
        let idx_addr = addr_major.address_cycle_to_index(addr, cycle, K, T);

        // CycleMajor: index = addr * T + cycle = 3 * 16 + 7 = 55
        assert_eq!(idx_cycle, 55);

        // AddressMajor: index = cycle * K + addr = 7 * 8 + 3 = 59
        assert_eq!(idx_addr, 59);

        assert_eq!(
            cycle_major.index_to_address_cycle(idx_cycle, K, T),
            (addr, cycle)
        );
        assert_eq!(
            addr_major.index_to_address_cycle(idx_addr, K, T),
            (addr, cycle)
        );
    }

    /// Test that AddressMajor one-hot polynomial proof/verify works correctly.
    #[test]
    #[serial]
    fn test_dory_one_hot_address_major() {
        use crate::poly::one_hot_polynomial::OneHotPolynomial;

        DoryGlobals::reset();

        let K = 32;
        let T = 32;

        let _guard = DoryGlobals::initialize_context(
            K,
            T,
            DoryContext::Main,
            Some(DoryLayout::AddressMajor),
        );

        let mut rng = thread_rng();
        let nonzero_indices: Vec<Option<u8>> = (0..T)
            .map(|_| {
                if rng.gen::<bool>() {
                    Some(rng.gen::<u8>() % K as u8)
                } else {
                    None
                }
            })
            .collect();

        let one_hot_poly = OneHotPolynomial::from_indices(nonzero_indices, K);
        let num_vars = one_hot_poly.get_num_vars();
        let poly = MultilinearPolynomial::OneHot(one_hot_poly);

        let opening_point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        let (commitment, row_commitments) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        let evaluation = <MultilinearPolynomial<Fr> as PolynomialEvaluation<Fr>>::evaluate(
            &poly,
            &opening_point,
        );

        let mut prove_transcript = Blake2bTranscript::new(b"dory_test");
        bind_opening_inputs::<Fr, _>(&mut prove_transcript, &opening_point, &evaluation);
        let (proof, _y_binding) = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &opening_point,
            Some(row_commitments),
            &mut prove_transcript,
        );

        let mut verify_transcript = Blake2bTranscript::new(b"dory_test");
        bind_opening_inputs::<Fr, _>(&mut verify_transcript, &opening_point, &evaluation);
        let verification_result = DoryCommitmentScheme::verify(
            &proof,
            &verifier_setup,
            &mut verify_transcript,
            &opening_point,
            &evaluation,
            &commitment,
        );

        assert!(
            verification_result.is_ok(),
            "Dory verification failed for AddressMajor OneHot: {verification_result:?}"
        );
    }

    /// Test VMP correctness for AddressMajor layout with RLC polynomial (dense + one-hot).
    #[test]
    #[serial]
    fn test_vmp_address_major_rlc() {
        use crate::poly::one_hot_polynomial::OneHotPolynomial;
        use crate::poly::rlc_polynomial::RLCPolynomial;

        DoryGlobals::reset();

        let K = 16usize;
        let T = 64usize;

        let _guard = DoryGlobals::initialize_context(
            K,
            T,
            DoryContext::Main,
            Some(DoryLayout::AddressMajor),
        );

        let num_columns = DoryGlobals::get_num_columns();
        let num_rows = DoryGlobals::get_max_num_rows();

        let mut rng = thread_rng();

        let dense_coeffs: Vec<Fr> = (0..T).map(|_| Fr::rand(&mut rng)).collect();

        let nonzero_indices: Vec<Option<u8>> = (0..T)
            .map(|_| {
                if rng.gen::<bool>() {
                    Some(rng.gen::<u8>() % K as u8)
                } else {
                    None
                }
            })
            .collect();
        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices.clone(), K);

        let dense_rlc_coeff: Fr = Fr::rand(&mut rng);
        let one_hot_rlc_coeff: Fr = Fr::rand(&mut rng);

        let rlc_dense: Vec<Fr> = dense_coeffs.iter().map(|c| *c * dense_rlc_coeff).collect();
        let rlc_poly = RLCPolynomial {
            dense_rlc: rlc_dense.clone(),
            one_hot_rlc: vec![(
                one_hot_rlc_coeff,
                std::sync::Arc::new(MultilinearPolynomial::OneHot(one_hot_poly.clone())),
            )],
            streaming_context: None,
        };

        let left_vec: Vec<Fr> = (0..num_rows).map(|_| Fr::rand(&mut rng)).collect();

        let vmp_result = rlc_poly.vector_matrix_product(&left_vec);

        let mut expected = vec![Fr::zero(); num_columns];
        let cycles_per_row = DoryGlobals::address_major_cycles_per_row();

        // Dense contribution for AddressMajor layout:
        // Dense coefficients occupy evenly-spaced columns (every K-th column).
        // Coefficient i maps to: row = i / cycles_per_row, col = (i % cycles_per_row) * K
        for (i, &coeff) in rlc_dense.iter().enumerate() {
            let row = i / cycles_per_row;
            let col = (i % cycles_per_row) * K;
            if row < num_rows && col < num_columns {
                expected[col] += left_vec[row] * coeff;
            }
        }

        // One-hot contribution: uses AddressMajor layout
        for (cycle, k_opt) in nonzero_indices.iter().enumerate() {
            if let Some(k) = k_opt {
                let k = *k as usize;
                // AddressMajor: global_index = cycle * K + address
                let global_index = DoryLayout::AddressMajor.address_cycle_to_index(k, cycle, K, T);
                let row = global_index / num_columns;
                let col = global_index % num_columns;
                if row < num_rows && col < num_columns {
                    expected[col] += left_vec[row] * one_hot_rlc_coeff;
                }
            }
        }

        // Compare results
        for (col, (actual, exp)) in vmp_result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                *actual, *exp,
                "VMP mismatch at column {col}: actual={actual:?}, expected={exp:?}"
            );
        }
    }
}
