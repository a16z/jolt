#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::super::*;
    use crate::field::JoltField;
    use crate::poly::commitment::commitment_scheme::CommitmentScheme;
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
        let proof = DoryCommitmentScheme::prove(
            prover_setup,
            &poly,
            &opening_point,
            Some(row_commitments),
            &mut prove_transcript,
        );

        let mut verify_transcript = Blake2bTranscript::new(b"dory_test");
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
        let _guard = DoryGlobals::initialize(1, num_coeffs);

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
        let _guard = DoryGlobals::initialize(1, num_coeffs);

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

        let proof = DoryCommitmentScheme::prove(
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

        // Test 2: Tamper with the opening point
        {
            let tampered_opening_point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
                .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
                .collect();

            let mut verify_transcript =
                Blake2bTranscript::new(DoryCommitmentScheme::protocol_name());
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

        let K = 8;
        let T = 8;

        let _guard = DoryGlobals::initialize(K, T);

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
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &opening_point,
            Some(row_commitments),
            &mut prove_transcript,
        );

        let mut verify_transcript = Blake2bTranscript::new(b"dory_test");
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

        let num_vars = 8;
        let num_coeffs = 1 << num_vars;
        let num_polys = 5;

        let _guard = DoryGlobals::initialize(1, num_coeffs);

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
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &combined_poly,
            &opening_point,
            Some(combined_hint),
            &mut prove_transcript,
        );

        // Step 9: Verify the proof
        let mut verify_transcript = Blake2bTranscript::new(b"dory_homomorphic_test");
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

        let num_vars = 8;
        let num_coeffs = 1 << num_vars;
        let num_polys = 5;

        let _guard = DoryGlobals::initialize(1, num_coeffs);

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
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &combined_poly,
            &opening_point,
            Some(combined_hint),
            &mut prove_transcript,
        );

        // Step 10: Verify the proof
        let mut verify_transcript = Blake2bTranscript::new(b"dory_batch_commit_e2e_test");
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
        let proof2 = DoryCommitmentScheme::prove(
            &prover_setup,
            &combined_poly,
            &opening_point,
            Some(direct_hint),
            &mut prove_transcript2,
        );

        let mut verify_transcript2 = Blake2bTranscript::new(b"dory_batch_commit_e2e_test");
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
}
