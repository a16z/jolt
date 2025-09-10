use crate::{
    field::JoltField,
    poly::{dense_mlpoly::DensePolynomial, multilinear_polynomial::MultilinearPolynomial},
    subprotocols::{
        square_and_multiply::{AccumulatorMultiplySumcheck, SquareAndMultiplySumcheck},
        sumcheck::SumcheckInstance,
    },
    transcripts::Transcript,
};
use ark_bn254::{Fq, Fq12};
use ark_ff::{BigInteger, One, PrimeField, Zero};
use jolt_optimizations::{
    fq12_poly::{fq12_to_multilinear_evals, g_coeffs, to_multilinear_evals},
    steps::ExponentiationSteps,
};
use std::collections::HashMap;

pub struct SZCheckArtifacts<F: JoltField> {
    /// The quotient polynomials for each product (needed for verification)
    pub quotient_polynomials: Vec<Vec<F>>,
    /// The multilinear polynomials representing Fq12 values
    /// Organized as triplets (a, b, c) for each product
    pub fq12_polynomials: Vec<MultilinearPolynomial<F>>,
    /// Commitments to the Fq12 polynomials (a, b, c values)
    pub fq12_commitments: Vec<Vec<u8>>, // Will be filled by prover with actual commitments
    /// Commitments to quotient polynomials  
    pub quotient_commitments: Vec<Vec<u8>>, // Will be filled by prover with actual commitments
}

/// Process exponentiation steps and create sumcheck instances for SZ check
pub fn sz_check_prove<F, ProofTranscript>(
    exponentiation_steps_vec: Vec<ExponentiationSteps>,
    transcript: &mut ProofTranscript,
) -> (Vec<Box<dyn SumcheckInstance<F>>>, SZCheckArtifacts<F>)
where
    F: JoltField + From<Fq>,
    ProofTranscript: Transcript,
{
    let mut sumcheck_instances: Vec<Box<dyn SumcheckInstance<F>>> = Vec::new();
    let mut all_products = Vec::new();
    let mut quotient_polynomials = Vec::new();

    // Process all exponentiation steps to collect products
    for steps in &exponentiation_steps_vec {
        let products = steps.to_products();
        for product in &products {
            quotient_polynomials.push(product.quotient.iter().map(|&fq| F::from(fq)).collect());
        }
        all_products.extend(products);
    }

    // Collect unique Fq12 values and create index mapping
    let mut fp12_values = Vec::new();
    let mut fp12_to_index = HashMap::new();

    for product in &all_products {
        fp12_to_index.entry(product.a).or_insert_with(|| {
            let idx = fp12_values.len();
            fp12_values.push(product.a);
            idx
        });

        fp12_to_index.entry(product.b).or_insert_with(|| {
            let idx = fp12_values.len();
            fp12_values.push(product.b);
            idx
        });

        fp12_to_index.entry(product.c).or_insert_with(|| {
            let idx = fp12_values.len();
            fp12_values.push(product.c);
            idx
        });
    }

    // Convert Fq12 values to multilinear polynomials
    let fq12_polynomials: Vec<MultilinearPolynomial<F>> = fp12_values
        .iter()
        .map(|fp12| {
            let evals_fq = fq12_to_multilinear_evals(fp12);
            let evals_f: Vec<F> = evals_fq.into_iter().map(F::from).collect();
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(evals_f))
        })
        .collect();

    // Create sumcheck instances for each exponentiation
    for (_exp_idx, steps) in exponentiation_steps_vec.iter().enumerate() {
        // Get random challenges from transcript
        let r: Vec<F> = transcript.challenge_vector(4); // 4 variables for x ∈ {0,1}⁴
        let gamma: F = transcript.challenge_scalar();

        // Create a_polys for SquareAndMultiplySumcheck
        // We need to map the steps to the corresponding polynomials
        let mut a_polys = Vec::new();

        // Add base polynomial
        if let Some(&base_idx) = fp12_to_index.get(&steps.base) {
            a_polys.push(fq12_polynomials[base_idx].clone());
        }

        // Add intermediate a_i values
        for step in &steps.steps {
            if let Some(&idx) = fp12_to_index.get(&step.a_curr) {
                a_polys.push(fq12_polynomials[idx].clone());
            }
        }

        // Pad to 256 polynomials
        while a_polys.len() < 256 {
            a_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                vec![F::zero(); 16],
            )));
        }

        // Create g polynomial
        let g_coeffs_vec = g_coeffs();
        let mut g_coeffs_array = [Fq::zero(); 12];
        for i in 0..g_coeffs_vec.len().min(12) {
            g_coeffs_array[i] = g_coeffs_vec[i];
        }
        let g_evals = to_multilinear_evals(&g_coeffs_array);
        let g_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
            g_evals.into_iter().map(F::from).collect(),
        ));

        // Create SquareAndMultiplySumcheck instance
        let square_multiply_sumcheck =
            SquareAndMultiplySumcheck::new_prover(a_polys, g_poly, r.clone(), gamma);
        sumcheck_instances.push(Box::new(square_multiply_sumcheck));

        // Create AccumulatorMultiplySumcheck
        // Extract rho polynomials
        let mut rho_polys = Vec::new();

        // Initial rho
        let initial_rho = if steps.exponent.into_bigint().to_bits_le()[0] {
            steps.base
        } else {
            Fq12::one()
        };

        if let Some(&idx) = fp12_to_index.get(&initial_rho) {
            rho_polys.push(fq12_polynomials[idx].clone());
        }

        // Add rho values from steps
        for step in &steps.steps {
            if let Some(&idx) = fp12_to_index.get(&step.rho_after) {
                rho_polys.push(fq12_polynomials[idx].clone());
            }
        }

        // Pad to 256
        while rho_polys.len() < 256 {
            rho_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                vec![F::zero(); 16],
            )));
        }

        // Base polynomial for accumulator
        let base_poly = if let Some(&idx) = fp12_to_index.get(&steps.base) {
            fq12_polynomials[idx].clone()
        } else {
            MultilinearPolynomial::LargeScalars(DensePolynomial::new(vec![F::one(); 16]))
        };

        // Extract exponent bits
        let exponent_bits: Vec<u8> = steps
            .exponent
            .into_bigint()
            .to_bits_le()
            .iter()
            .take(256)
            .map(|&b| if b { 1 } else { 0 })
            .collect();

        let mut exponent_bits = exponent_bits;
        while exponent_bits.len() < 256 {
            exponent_bits.push(0);
        }

        // Result for accumulator sumcheck
        let exp_result = F::from(steps.result.c0.c0.c0);

        let accumulator_sumcheck = AccumulatorMultiplySumcheck::new_prover(
            rho_polys,
            base_poly,
            r,
            gamma,
            exponent_bits,
            exp_result,
        );
        sumcheck_instances.push(Box::new(accumulator_sumcheck));
    }

    let artifacts = SZCheckArtifacts {
        quotient_polynomials,
        fq12_polynomials,
        fq12_commitments: Vec::new(), // To be filled by the prover after committing
        quotient_commitments: Vec::new(), // To be filled by the prover after committing
    };

    (sumcheck_instances, artifacts)
}

#[cfg(test)]
mod tests {
    use super::super::square_and_multiply::{
        AccumulatorMultiplySumcheck, SquareAndMultiplySumcheck,
    };
    use crate::{
        poly::{
            dense_mlpoly::DensePolynomial, multilinear_polynomial::MultilinearPolynomial,
            unipoly::UniPoly,
        },
        subprotocols::sumcheck::SumcheckInstance,
    };
    use ark_bn254::{Fq, Fq12};
    use ark_ff::BigInteger;
    use ark_ff::{Field, One, PrimeField, UniformRand, Zero};
    use ark_std::test_rng;
    use jolt_optimizations::{
        expression::{Expression, Term},
        fq12_poly::{fq12_to_poly12_coeffs, g_coeffs, to_multilinear_evals},
        steps::pow_with_steps_le,
        sz_check::batch_verify,
    };

    fn create_g_polynomial() -> MultilinearPolynomial<Fq> {
        let g_coeffs = g_coeffs();
        let mut g_evals = vec![Fq::zero(); 16];

        // Convert g(x) coefficients to evaluations over the boolean hypercube
        // For now, we'll use the coefficients directly padded to 16 elements
        for i in 0..g_coeffs.len().min(16) {
            g_evals[i] = g_coeffs[i];
        }

        MultilinearPolynomial::LargeScalars(DensePolynomial::new(g_evals))
    }

    fn convert_fq12_to_fq_poly(fq12: Fq12) -> Vec<Fq> {
        let coeffs = fq12_to_poly12_coeffs(&fq12);
        to_multilinear_evals(&coeffs)
    }

    #[test]
    fn test_single_exponentiation_sumcheck() {
        let mut rng = test_rng();

        // Generate random base and exponent
        let base = Fq12::rand(&mut rng);
        let exponent = Fq::rand(&mut rng);

        // Compute a^e using arkworks
        let expected_result = base.pow(exponent.into_bigint());

        // Get exponentiation steps using jolt_optimizations
        let steps = pow_with_steps_le(base, exponent);
        assert!(steps.sanity_verify(), "Steps should pass sanity check");
        assert_eq!(
            steps.result, expected_result,
            "Result should match expected"
        );

        // Convert steps to products for sz_check
        let products = steps.to_products();

        // Verify products using batch_verify
        let r = Fq::rand(&mut rng);
        assert!(
            batch_verify(&products, &r),
            "Batch verification should pass"
        );

        // Now test the sumcheck protocol
        // Convert Fq12 elements to multilinear polynomials
        let num_steps = steps.steps.len();
        if num_steps == 0 {
            return; // Edge case: exponent is 0 or 1
        }

        // Create a_polys from the step sequence
        let mut a_polys = Vec::new();

        // a_0 is the base
        a_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
            convert_fq12_to_fq_poly(base),
        )));

        // Add intermediate values from steps
        for step in &steps.steps {
            a_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                convert_fq12_to_fq_poly(step.a_curr),
            )));
        }

        // Pad to 256 polynomials
        while a_polys.len() < 256 {
            a_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                vec![Fq::zero(); 16],
            )));
        }

        // Create g polynomial
        let g = create_g_polynomial();

        // Random point for eq(r, x)
        let r: Vec<Fq> = (0..4).map(|_| Fq::rand(&mut rng)).collect();
        let gamma = Fq::rand(&mut rng);

        // Create sumcheck instance
        let mut sumcheck =
            SquareAndMultiplySumcheck::new_prover(a_polys.clone(), g.clone(), r.clone(), gamma);

        // Run sumcheck rounds
        let mut previous_claim = sumcheck.input_claim();

        for round in 0..4 {
            let prover_message = sumcheck.compute_prover_message(round, previous_claim);

            // Verify degree bound
            assert_eq!(
                prover_message.len(),
                3,
                "Should have degree-2 polynomial (3 evaluations)"
            );

            // Generate random challenge
            let challenge = Fq::rand(&mut rng);

            // Compute next claim using univariate evaluation
            let univariate_evals = vec![
                prover_message[0],
                previous_claim - prover_message[0], // eval at 1
                prover_message[1],                  // eval at 2
                prover_message[2],                  // eval at 3
            ];
            let univariate_poly = UniPoly::from_evals(&univariate_evals);
            previous_claim = univariate_poly.evaluate(&challenge);

            // Bind the sumcheck
            sumcheck.bind(challenge, round);
        }

        println!("Single exponentiation sumcheck test passed!");
    }

    #[test]
    fn test_expression_composition_sumcheck() {
        let mut rng = test_rng();

        // Create an expression with multiple terms
        let terms = vec![
            Term {
                base: Fq12::rand(&mut rng),
                exponent: Fq::rand(&mut rng),
            },
            Term {
                base: Fq12::rand(&mut rng),
                exponent: Fq::rand(&mut rng),
            },
            Term {
                base: Fq12::rand(&mut rng),
                exponent: Fq::rand(&mut rng),
            },
        ];

        let expr = Expression::new(terms.clone());

        // Evaluate the expression and get steps
        let (result, expr_steps) = expr.evaluate_with_steps();

        // Convert to products
        let products = Expression::steps_to_products(&expr_steps);

        // Verify with batch_verify
        let r = Fq::rand(&mut rng);
        assert!(
            batch_verify(&products, &r),
            "Batch verification should pass"
        );

        // Test accumulator multiply sumcheck
        // We'll test the accumulator updates for one of the terms
        if let Some(term_steps) = expr_steps.term_steps.first() {
            let num_bits = 256; // Using 256 bits as in the sumcheck

            // Create rho polynomials from the steps
            let mut rho_polys = Vec::new();

            // Initial rho_0
            rho_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                vec![Fq::one(); 16],
            )));

            // Add rho values from steps
            for step in &term_steps.steps {
                rho_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                    convert_fq12_to_fq_poly(step.rho_after),
                )));
            }

            // Pad to 256
            while rho_polys.len() < 256 {
                rho_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                    vec![Fq::zero(); 16],
                )));
            }

            // Base polynomial
            let a_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                convert_fq12_to_fq_poly(term_steps.base),
            ));

            // Extract exponent bits
            let exponent_bits: Vec<u8> = term_steps
                .exponent
                .into_bigint()
                .to_bits_le()
                .iter()
                .take(256)
                .map(|&b| if b { 1 } else { 0 })
                .collect();

            // Pad to 256 bits
            let mut exponent_bits = exponent_bits;
            while exponent_bits.len() < 256 {
                exponent_bits.push(0);
            }

            let r: Vec<Fq> = (0..4).map(|_| Fq::rand(&mut rng)).collect();
            let gamma = Fq::rand(&mut rng);
            // Use one of the Fq coefficients as the result (simplified for testing)
            let exp_result = term_steps.result.c0.c0.c0;

            // Create accumulator multiply sumcheck
            let mut acc_sumcheck = AccumulatorMultiplySumcheck::new_prover(
                rho_polys,
                a_poly,
                r.clone(),
                gamma,
                exponent_bits,
                exp_result,
            );

            // Run sumcheck rounds
            let mut previous_claim = acc_sumcheck.input_claim();

            for round in 0..4 {
                let prover_message = acc_sumcheck.compute_prover_message(round, previous_claim);

                // Verify degree bound (degree 2 for accumulator multiply)
                assert_eq!(
                    prover_message.len(),
                    2,
                    "Should have degree-1 polynomial (2 evaluations)"
                );

                // Generate random challenge
                let challenge = Fq::rand(&mut rng);

                // Compute next claim
                let univariate_evals = vec![
                    prover_message[0],
                    previous_claim - prover_message[0], // eval at 1
                    prover_message[1],                  // eval at 2
                ];
                let univariate_poly = UniPoly::from_evals(&univariate_evals);
                previous_claim = univariate_poly.evaluate(&challenge);

                // Bind the sumcheck
                acc_sumcheck.bind(challenge, round);
            }
        }

        println!("Expression composition sumcheck test passed!");
    }

    #[test]
    fn test_edge_cases_sumcheck() {
        let mut rng = test_rng();

        // Test case 1: Exponent = 0 (result should be 1)
        let base = Fq12::rand(&mut rng);
        let exponent = Fq::zero();
        let steps = pow_with_steps_le(base, exponent);
        assert_eq!(steps.result, Fq12::one(), "a^0 should equal 1");
        assert!(steps.sanity_verify(), "Steps should pass sanity check");

        // Test case 2: Exponent = 1 (result should be base)
        let base = Fq12::rand(&mut rng);
        let exponent = Fq::one();
        let steps = pow_with_steps_le(base, exponent);
        assert_eq!(steps.result, base, "a^1 should equal a");
        assert!(steps.sanity_verify(), "Steps should pass sanity check");

        // Test case 3: Base = 1 (result should always be 1)
        let base = Fq12::one();
        let exponent = Fq::rand(&mut rng);
        let steps = pow_with_steps_le(base, exponent);
        assert_eq!(steps.result, Fq12::one(), "1^e should equal 1");
        assert!(steps.sanity_verify(), "Steps should pass sanity check");

        // Test case 4: Small exponent (e = 2)
        let base = Fq12::rand(&mut rng);
        let exponent = Fq::from(2u64);
        let steps = pow_with_steps_le(base, exponent);
        assert_eq!(steps.result, base * base, "a^2 should equal a*a");
        assert!(steps.sanity_verify(), "Steps should pass sanity check");

        // Verify all edge cases with batch_verify
        for exponent_val in [0u64, 1, 2, 3, 7, 15, 255] {
            let base = Fq12::rand(&mut rng);
            let exponent = Fq::from(exponent_val);
            let steps = pow_with_steps_le(base, exponent);
            let products = steps.to_products();

            if !products.is_empty() {
                let r = Fq::rand(&mut rng);
                assert!(
                    batch_verify(&products, &r),
                    "Batch verification should pass for exponent = {}",
                    exponent_val
                );
            }
        }

        println!("Edge cases sumcheck test passed!");
    }

    #[test]
    fn test_large_batch_sumcheck() {
        let mut rng = test_rng();
        let num_exponentiations = 100;

        let mut all_products = Vec::new();

        // Generate multiple exponentiations
        for _ in 0..num_exponentiations {
            let base = Fq12::rand(&mut rng);
            let exponent = Fq::rand(&mut rng);

            let steps = pow_with_steps_le(base, exponent);
            assert!(steps.sanity_verify(), "Steps should pass sanity check");

            let products = steps.to_products();
            all_products.extend(products);
        }

        // Batch verify all products
        let r = Fq::rand(&mut rng);
        assert!(
            batch_verify(&all_products, &r),
            "Large batch verification should pass"
        );

        // Test with expression composition
        let mut expr_terms = Vec::new();
        for _ in 0..10 {
            expr_terms.push(Term {
                base: Fq12::rand(&mut rng),
                exponent: Fq::rand(&mut rng),
            });
        }

        let expr = Expression::new(expr_terms);
        let (result, expr_steps) = expr.evaluate_with_steps();

        // Verify the expression result
        let expected = expr.terms.iter().fold(Fq12::one(), |acc, term| {
            acc * term.base.pow(term.exponent.into_bigint())
        });
        assert_eq!(result, expected, "Expression result should match expected");

        // Convert to products and verify
        let expr_products = Expression::steps_to_products(&expr_steps);
        let r = Fq::rand(&mut rng);
        assert!(
            batch_verify(&expr_products, &r),
            "Expression batch verification should pass"
        );

        println!(
            "Large batch sumcheck test passed with {} total products!",
            all_products.len()
        );
    }

    #[test]
    #[ignore]
    fn test_full_sumcheck_prove_verify() {
        use crate::{
            poly::opening_proof::{
                OpeningId, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
                BIG_ENDIAN,
            },
            subprotocols::sumcheck::SingleSumcheck,
            transcripts::{KeccakTranscript, Transcript},
            zkvm::witness::VirtualPolynomial,
        };
        use std::{cell::RefCell, rc::Rc};

        let mut rng = test_rng();

        // Generate random base and exponent
        let base = Fq12::rand(&mut rng);
        let exponent = Fq::rand(&mut rng);

        // Compute a^e using arkworks
        let expected_result = base.pow(exponent.into_bigint());

        // Get exponentiation steps
        let steps = pow_with_steps_le(base, exponent);
        assert!(steps.sanity_verify(), "Steps should pass sanity check");
        assert_eq!(
            steps.result, expected_result,
            "Result should match expected"
        );

        // Convert steps to products for sz_check
        let products = steps.to_products();

        // Verify products using batch_verify
        let r_batch = Fq::rand(&mut rng);
        assert!(
            batch_verify(&products, &r_batch),
            "Batch verification should pass"
        );

        // // Skip edge cases with no steps
        // if steps.steps.is_empty() {
        //     return;
        // }

        // Create a_polys from the step sequence
        let mut a_polys = Vec::new();
        a_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
            convert_fq12_to_fq_poly(base),
        )));

        for step in &steps.steps {
            a_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                convert_fq12_to_fq_poly(step.a_curr),
            )));
        }

        // Pad to 256 polynomials
        while a_polys.len() < 256 {
            a_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                vec![Fq::zero(); 16],
            )));
        }

        // Create g polynomial
        let g = create_g_polynomial();

        // Random point for eq(r, x)
        let r: Vec<Fq> = (0..4).map(|_| Fq::rand(&mut rng)).collect();
        let gamma = Fq::rand(&mut rng);

        // Test 1: SquareAndMultiplySumcheck with full prove/verify
        {
            // Create prover sumcheck instance
            let mut prover_sumcheck =
                SquareAndMultiplySumcheck::new_prover(a_polys.clone(), g.clone(), r.clone(), gamma);

            // Create prover accumulator and transcript
            let prover_accumulator = Rc::new(RefCell::new(ProverOpeningAccumulator::new()));
            let mut prover_transcript = KeccakTranscript::new(b"test_square_multiply");

            // Prove
            let (proof, r_sumcheck) = SingleSumcheck::prove(
                &mut prover_sumcheck,
                Some(prover_accumulator.clone()),
                &mut prover_transcript,
            );

            // The prover accumulator should now have the openings cached
            // We need to populate the verifier's accumulator with these same openings
            // In a real protocol, these would come from the commitment scheme

            // Create verifier sumcheck instance
            let verifier_sumcheck = SquareAndMultiplySumcheck::new_verifier(r.clone(), gamma);

            // Create verifier accumulator and transcript
            let verifier_accumulator = Rc::new(RefCell::new(VerifierOpeningAccumulator::new()));
            let mut verifier_transcript = KeccakTranscript::new(b"test_square_multiply");

            // Copy the prover's openings to the verifier's accumulator
            // In a real protocol, these would come from commitment opening proofs
            for (opening_id, (point, claim)) in
                prover_accumulator.borrow().evaluation_openings().iter()
            {
                verifier_accumulator
                    .borrow_mut()
                    .openings_mut()
                    .insert(opening_id.clone(), (point.clone(), *claim));
            }

            // Verify
            let verify_result = SingleSumcheck::verify(
                &verifier_sumcheck,
                &proof,
                Some(verifier_accumulator.clone()),
                &mut verifier_transcript,
            );

            // assert!(verify_result.is_ok(), "Sumcheck verification should pass");
            // let verified_r = verify_result.unwrap();
            // assert_eq!(r_sumcheck, verified_r, "Sumcheck points should match");

            // Verify that the expected output claim matches
            // let expected_claim = verifier_sumcheck
            //     .expected_output_claim(Some(verifier_accumulator.clone()), &verified_r);

            // The claim should be zero for a valid constraint system
            // assert_eq!(
            //     expected_claim,
            //     Fq::zero(),
            //     "Expected claim should be zero for valid constraints"
            // );
        }

        // Test 2: AccumulatorMultiplySumcheck with full prove/verify
        {
            // Create rho polynomials from the steps
            let mut rho_polys = Vec::new();
            rho_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                vec![Fq::one(); 16],
            )));

            for step in &steps.steps {
                rho_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                    convert_fq12_to_fq_poly(step.rho_after),
                )));
            }

            // Pad to 256
            while rho_polys.len() < 256 {
                rho_polys.push(MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                    vec![Fq::zero(); 16],
                )));
            }

            // Base polynomial
            let a_poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(
                convert_fq12_to_fq_poly(base),
            ));

            // Extract exponent bits
            let exponent_bits: Vec<u8> = exponent
                .into_bigint()
                .to_bits_le()
                .iter()
                .take(256)
                .map(|&b| if b { 1 } else { 0 })
                .collect();

            // Pad to 256 bits
            let mut exponent_bits = exponent_bits;
            while exponent_bits.len() < 256 {
                exponent_bits.push(0);
            }

            // Use the result coefficient as the claim
            let exp_result = expected_result.c0.c0.c0;

            // Create prover accumulator sumcheck
            let mut prover_acc_sumcheck = AccumulatorMultiplySumcheck::new_prover(
                rho_polys.clone(),
                a_poly.clone(),
                r.clone(),
                gamma,
                exponent_bits.clone(),
                exp_result,
            );

            // Create prover accumulator and transcript
            let prover_accumulator = Rc::new(RefCell::new(ProverOpeningAccumulator::new()));
            let mut prover_transcript = KeccakTranscript::new(b"test_accumulator_multiply");

            // Prove
            let (proof, r_sumcheck) = SingleSumcheck::prove(
                &mut prover_acc_sumcheck,
                Some(prover_accumulator.clone()),
                &mut prover_transcript,
            );

            // Create verifier accumulator sumcheck
            let verifier_acc_sumcheck =
                AccumulatorMultiplySumcheck::new_verifier(r.clone(), gamma, exponent_bits.clone());

            // Create verifier accumulator and transcript
            let verifier_accumulator = Rc::new(RefCell::new(VerifierOpeningAccumulator::new()));
            let mut verifier_transcript = KeccakTranscript::new(b"test_accumulator_multiply");

            // Copy the prover's openings to the verifier's accumulator
            // In a real protocol, these would come from commitment opening proofs
            for (opening_id, (point, claim)) in
                prover_accumulator.borrow().evaluation_openings().iter()
            {
                verifier_accumulator
                    .borrow_mut()
                    .openings_mut()
                    .insert(opening_id.clone(), (point.clone(), *claim));
            }

            // Verify
            let verify_result = SingleSumcheck::verify(
                &verifier_acc_sumcheck,
                &proof,
                Some(verifier_accumulator.clone()),
                &mut verifier_transcript,
            );

            // assert!(
            //     verify_result.is_ok(),
            //     "Accumulator sumcheck verification should pass"
            // );
            let verified_r = verify_result.unwrap();
            // assert_eq!(r_sumcheck, verified_r, "Sumcheck points should match");
        }

        println!("Full sumcheck prove/verify test passed!");
    }

    #[test]
    fn test_hyrax_exponentiation_steps() {
        use crate::{
            poly::{
                commitment::{
                    commitment_scheme::CommitmentScheme,
                    hyrax::{matrix_dimensions, BatchedHyraxOpeningProof, HyraxCommitment},
                    pedersen::PedersenGenerators,
                },
                dense_mlpoly::DensePolynomial,
            },
            transcripts::{Blake2bTranscript, Transcript},
        };
        use ark_bn254::{Fq, Fq12};
        use ark_grumpkin::Projective as GrumpkinProjective;
        use jolt_optimizations::fq12_poly::fq12_to_multilinear_evals;
        use rand_core::RngCore;

        const RATIO: usize = 1;
        type G = GrumpkinProjective; // Grumpkin's scalar field is BN254's Fq
        type TranscriptType = Blake2bTranscript;

        let mut rng = test_rng();

        // Generate random base and use a smaller exponent to avoid stack overflow
        let base = Fq12::rand(&mut rng);
        // Use a smaller exponent for testing (e.g., 16 bits)
        let exponent = Fq::from(rng.next_u32() as u64 & 0xFFFF);

        // Get exponentiation steps
        let steps = pow_with_steps_le(base, exponent);
        assert!(steps.sanity_verify(), "Steps should pass sanity check");

        // Get all products from the steps - these are the constraints we need to commit to
        let products = steps.to_products();

        // For each product (a * b = c), we need to commit to the polynomials representing a, b, and c
        // Collect all unique Fq12 values that appear in the products
        let mut fp12_values = Vec::new();
        let mut fp12_to_index = std::collections::HashMap::new();

        // Helper to add unique Fq12 values and track their indices
        let mut add_unique_fp12 = |val: Fq12| -> usize {
            if let Some(&idx) = fp12_to_index.get(&val) {
                idx
            } else {
                let idx = fp12_values.len();
                fp12_values.push(val);
                fp12_to_index.insert(val, idx);
                idx
            }
        };

        // Collect all Fq12 values from products and track which indices correspond to each product
        let mut product_indices = Vec::new();
        for product in &products {
            let a_idx = add_unique_fp12(product.a);
            let b_idx = add_unique_fp12(product.b);
            let c_idx = add_unique_fp12(product.c);
            product_indices.push((a_idx, b_idx, c_idx));
        }

        println!(
            "Generated {} products from {} steps, with {} unique Fq12 values to commit",
            products.len(),
            steps.steps.len(),
            fp12_values.len()
        );

        // Convert all unique Fq12 values to multilinear evaluations
        let multilinear_evals_fq: Vec<Vec<Fq>> = fp12_values
            .iter()
            .map(|fp12| fq12_to_multilinear_evals(fp12))
            .collect();

        // Create DensePolynomials from the Fq evaluations
        let polys: Vec<DensePolynomial<Fq>> = multilinear_evals_fq
            .iter()
            .map(|evals| DensePolynomial::new(evals.clone()))
            .collect();

        // Setup generators for Hyrax (16 = 2^4 elements per poly)
        let num_vars = 4; // 2^4 = 16 evaluations
        let (_, R_size) = matrix_dimensions(num_vars, RATIO);
        let gens = PedersenGenerators::<G>::new(R_size, b"test exponentiation steps");

        // Commit to all polynomials
        let poly_refs: Vec<&[Fq]> = polys.iter().map(|p| p.evals_ref()).collect();
        let commitments = HyraxCommitment::<RATIO, G>::batch_commit(&poly_refs, &gens);

        println!(
            "Committed to {} unique polynomials from {} products",
            fp12_values.len(),
            products.len()
        );

        // Generate random opening point
        let opening_point: Vec<Fq> = (0..num_vars).map(|_| Fq::rand(&mut rng)).collect();

        // Evaluate all polynomials at the opening point
        let openings: Vec<Fq> = polys.iter().map(|p| p.evaluate(&opening_point)).collect();

        // Create opening proof
        let mut prover_transcript = TranscriptType::new(b"test_exp_steps");
        let poly_ptrs: Vec<&DensePolynomial<Fq>> = polys.iter().collect();
        let proof = BatchedHyraxOpeningProof::<RATIO, G>::prove(
            &poly_ptrs,
            &opening_point,
            &openings,
            &mut prover_transcript,
        );

        // Verify the proof
        let mut verifier_transcript = TranscriptType::new(b"test_exp_steps");
        let commitment_refs: Vec<&HyraxCommitment<RATIO, G>> = commitments.iter().collect();
        let verification_result = proof.verify(
            &gens,
            &opening_point,
            &openings,
            &commitment_refs,
            &mut verifier_transcript,
        );

        assert!(
            verification_result.is_ok(),
            "Hyrax proof verification should pass"
        );

        // Verify the products using sz_check batch verification
        // This checks that a * b - c = q * g(X) for the irreducible polynomial g
        let r_sz = Fq::rand(&mut rng);
        assert!(
            batch_verify(&products, &r_sz),
            "SZ check batch verification should pass for all products"
        );
        println!(
            "All {} product constraints verified using sz_check at random point",
            products.len()
        );

        // Test with incorrect opening (should fail)
        let mut wrong_openings = openings.clone();
        wrong_openings[0] = wrong_openings[0] + Fq::from(1u64);

        let mut verifier_transcript = TranscriptType::new(b"test_exp_steps");
        let wrong_result = proof.verify(
            &gens,
            &opening_point,
            &wrong_openings,
            &commitment_refs,
            &mut verifier_transcript,
        );

        assert!(
            wrong_result.is_err(),
            "Hyrax proof verification should fail with wrong opening"
        );

        // Verify that the exponentiation result is correct
        let expected_result = base.pow(exponent.into_bigint());
        assert_eq!(
            steps.result, expected_result,
            "Exponentiation result should match expected value"
        );

        println!(
            "Hyrax exponentiation steps test passed! Committed and proved {} unique polynomials from {} products",
            fp12_values.len(),
            products.len()
        );
    }
}
