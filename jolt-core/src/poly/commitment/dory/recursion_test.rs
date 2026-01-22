#[cfg(test)]
mod recursion_tests {
    use super::super::recursion::{JoltWitness, JoltWitnessGenerator};
    use super::super::{DoryCommitmentScheme, DoryGlobals, BN254};
    use crate::poly::commitment::dory::gt_exp_witness::Base4ExponentiationSteps;
    use crate::{
        field::JoltField,
        poly::{
            commitment::commitment_scheme::{CommitmentScheme, RecursionExt},
            dense_mlpoly::DensePolynomial,
            multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
        },
        transcripts::Transcript,
    };
    use ark_bn254::{Fq12, Fr};
    use ark_ff::{Field, One, PrimeField, UniformRand, Zero};
    use dory::{backends::arkworks::ArkGT, recursion::WitnessGenerator};
    use rand::thread_rng;
    use serial_test::serial;

    #[test]
    fn test_witness_generation_for_gt_exp() {
        let mut rng = thread_rng();

        let base = ArkGT(Fq12::rand(&mut rng));
        let scalar_fr = Fr::rand(&mut rng);
        let scalar = super::super::wrappers::jolt_to_ark(&scalar_fr);
        let result = ArkGT(base.0.pow(scalar_fr.into_bigint()));

        let witness =
            <JoltWitnessGenerator as WitnessGenerator<JoltWitness, BN254>>::generate_gt_exp(
                &base, &scalar, &result,
            );

        assert_eq!(witness.base, base.0, "Base should match");
        assert_eq!(witness.exponent, scalar_fr, "Exponent should match");
        assert_eq!(witness.result, result.0, "Result should match");

        let exp_steps = Base4ExponentiationSteps::new(base.0, scalar_fr);
        assert!(
            exp_steps.verify_result(),
            "ExponentiationSteps should verify correctly"
        );
        assert_eq!(
            exp_steps.result, result.0,
            "Results should match between witness and ExponentiationSteps"
        );

        let expected_steps = (witness.bits.len() + 1) / 2;
        assert_eq!(
            witness.quotient_mles.len(),
            expected_steps,
            "Should have one quotient per base-4 digit"
        );
        assert_eq!(
            witness.rho_mles.len(),
            expected_steps + 1,
            "Should have rho_0, ..., rho_t"
        );
    }

    #[test]
    fn test_special_cases() {
        let mut rng = thread_rng();
        let base = ArkGT(Fq12::rand(&mut rng));

        {
            let scalar = super::super::wrappers::jolt_to_ark(&Fr::zero());
            let result = ArkGT(Fq12::one());

            let witness =
                <JoltWitnessGenerator as WitnessGenerator<JoltWitness, BN254>>::generate_gt_exp(
                    &base, &scalar, &result,
                );

            assert_eq!(witness.result, Fq12::one());
            assert_eq!(witness.bits.len(), 0, "Zero exponent should have no bits");
            assert_eq!(witness.quotient_mles.len(), 0);
            assert_eq!(witness.rho_mles.len(), 1);
        }

        {
            let scalar = super::super::wrappers::jolt_to_ark(&Fr::one());
            let result = base;

            let witness =
                <JoltWitnessGenerator as WitnessGenerator<JoltWitness, BN254>>::generate_gt_exp(
                    &base, &scalar, &result,
                );

            assert_eq!(witness.result, base.0);
            assert_eq!(witness.bits.len(), 1);
            assert!(witness.bits[0], "Single bit should be 1");
        }

        {
            let identity_base = ArkGT(Fq12::one());
            let scalar_fr = Fr::rand(&mut rng);
            let scalar = super::super::wrappers::jolt_to_ark(&scalar_fr);
            let result = ArkGT(Fq12::one());

            let witness =
                <JoltWitnessGenerator as WitnessGenerator<JoltWitness, BN254>>::generate_gt_exp(
                    &identity_base,
                    &scalar,
                    &result,
                );

            assert_eq!(witness.result, Fq12::one());
        }
    }

    #[test]
    #[serial]
    fn test_verify_recursive_witness_generation() {
        // Reset DoryGlobals before initializing
        DoryGlobals::reset();
        let K = 1 << 2; // 2^2 = 4
        let T = 1 << 2; // 2^2 = 4
        DoryGlobals::initialize(K, T);

        // Setup
        let num_vars = 4;
        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        // Create polynomial
        let mut rng = thread_rng();
        let size = 1 << num_vars; // 2^4 = 16
        let coefficients: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));

        // Commit
        let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        // Create evaluation point
        let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        // Generate proof using DoryCommitmentScheme
        let mut prover_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &point,
            Some(hint),
            &mut prover_transcript,
        );

        // Evaluate polynomial
        let evaluation = PolynomialEvaluation::evaluate(&poly, &point);

        // Use extension trait for witness generation
        let mut witness_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let (witnesses, hints) = DoryCommitmentScheme::witness_gen(
            &proof,
            &verifier_setup,
            &mut witness_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Witness generation should succeed");

        // Now we have both witnesses (for proving) and hints (for verification)
        println!(
            "Successfully generated hints for {} rounds",
            hints.num_rounds
        );
        println!("Collected {} GT exp witnesses", witnesses.gt_exp.len());

        // Now verify with hint
        let mut hint_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        DoryCommitmentScheme::verify_with_hint(
            &proof,
            &verifier_setup,
            &mut hint_transcript,
            &point,
            &evaluation,
            &commitment,
            &hints,
        )
        .expect("Verify with hint should succeed");
    }

    #[test]
    #[serial]
    fn test_verify_recursive_matches_normal_verify() {
        // This test ensures that verify_recursive produces the same result
        // as normal verification when witness generation is enabled

        // Reset DoryGlobals
        DoryGlobals::reset();
        let K = 1 << 2;
        let T = 1 << 2;
        DoryGlobals::initialize(K, T);

        // Setup
        let num_vars = 4;
        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        // Create polynomial
        let mut rng = thread_rng();
        let size = 1 << num_vars;
        let coefficients: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));

        // Commit
        let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        // Create evaluation point
        let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        // Generate proof
        let mut prover_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &point,
            Some(hint),
            &mut prover_transcript,
        );

        let evaluation = PolynomialEvaluation::evaluate(&poly, &point);

        // First verify normally using DoryCommitmentScheme
        let mut normal_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let normal_result = DoryCommitmentScheme::verify(
            &proof,
            &verifier_setup,
            &mut normal_transcript,
            &point,
            &evaluation,
            &commitment,
        );

        assert!(normal_result.is_ok(), "Normal verification should succeed");

        // Now generate witnesses and hints using extension trait
        let mut witness_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let (witnesses, hints) = DoryCommitmentScheme::witness_gen(
            &proof,
            &verifier_setup,
            &mut witness_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Witness generation should succeed");

        // Both witnesses and hints successfully generated
        assert!(hints.num_rounds > 0, "Should have generated hints");
        assert!(
            !witnesses.gt_exp.is_empty(),
            "Should have collected witnesses"
        );

        // Verify with hint should also succeed
        let mut hint_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let hint_result = DoryCommitmentScheme::verify_with_hint(
            &proof,
            &verifier_setup,
            &mut hint_transcript,
            &point,
            &evaluation,
            &commitment,
            &hints,
        );

        assert!(hint_result.is_ok(), "Verify with hint should also succeed");
    }
}
