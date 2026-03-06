//! Spartan-based SNARK for R1CS constraint systems.
//!
//! This crate implements the Spartan proving system, which reduces
//! R1CS satisfiability to a sumcheck argument and polynomial evaluation
//! queries. It is generic over the polynomial commitment scheme and
//! scalar field, making it usable with any backend (Dory, HyperKZG, etc.).
//!
//! # Protocol overview
//!
//! Given an R1CS instance $(A, B, C)$ and a satisfying witness $z$ with
//! $Az \circ Bz = Cz$, the prover:
//!
//! 1. Commits to the witness polynomial $\tilde{z}$.
//! 2. Runs an outer sumcheck proving
//!    $\sum_{x \in \{0,1\}^n} \widetilde{eq}(x, \tau) \cdot (\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - \widetilde{Cz}(x)) = 0$
//!    for a random $\tau$ sampled via Fiat-Shamir.
//! 3. Provides opening proofs for the witness polynomial.
//!
//! # Crate structure
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`r1cs`] | R1CS trait and sparse implementation |
//! | [`key`] | Spartan key with precomputed matrix MLEs |
//! | [`prover`] | Proof generation |
//! | [`verifier`] | Proof verification |
//! | [`proof`] | Proof data structure |
//! | [`ir_r1cs`] | Bridge from `jolt-ir` R1CS emission to Spartan |
//! | [`uni_skip`] | Univariate skip optimization strategy |
//! | [`error`] | Error types |

pub mod error;
pub mod ir_r1cs;
pub mod key;
pub mod proof;
pub mod prover;
pub mod r1cs;
pub mod uni_skip;
pub mod verifier;

pub use error::SpartanError;
pub use ir_r1cs::build_witness;
pub use key::SpartanKey;
pub use proof::SpartanProof;
pub use prover::SpartanProver;
pub use r1cs::{SimpleR1CS, R1CS};
pub use uni_skip::FirstRoundStrategy;
pub use verifier::SpartanVerifier;

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Field;
    use jolt_field::Fr;
    use jolt_openings::mock::MockCommitmentScheme;
    use jolt_transcript::{Blake2bTranscript, Transcript};

    type MockPCS = MockCommitmentScheme<Fr>;

    /// Builds the `x * x = y` circuit with 1 constraint and 3 variables:
    /// z = [1, x, y].
    fn x_squared_circuit() -> SimpleR1CS<Fr> {
        SimpleR1CS::new(
            1,
            3,
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 2, Fr::from_u64(1))],
        )
    }

    fn prove_helper(
        r1cs: &SimpleR1CS<Fr>,
        key: &SpartanKey<Fr>,
        witness: &[Fr],
        label: &'static [u8],
    ) -> Result<(SpartanProof<Fr, MockPCS>, Blake2bTranscript), SpartanError> {
        let mut transcript = Blake2bTranscript::new(label);
        let proof = SpartanProver::prove::<MockPCS, _>(
            r1cs,
            key,
            witness,
            &(),
            &mut transcript,
            FirstRoundStrategy::Standard,
        )?;
        Ok((proof, transcript))
    }

    #[test]
    fn prove_and_verify_x_squared() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

        let (proof, _) =
            prove_helper(&r1cs, &key, &witness, b"spartan-test").expect("proving should succeed");

        let mut verifier_transcript = Blake2bTranscript::new(b"spartan-test");
        SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut verifier_transcript)
            .expect("verification should succeed");
    }

    #[test]
    fn reject_unsatisfied_witness() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(10)];

        let result = prove_helper(&r1cs, &key, &witness, b"spartan-test");
        assert!(
            matches!(result, Err(SpartanError::ConstraintViolation(0))),
            "expected constraint violation at index 0"
        );
    }

    #[test]
    fn prove_and_verify_multiple_constraints() {
        // x * x = y, y * x = z
        let r1cs = SimpleR1CS::new(
            2,
            4,
            vec![(0, 1, Fr::from_u64(1)), (1, 2, Fr::from_u64(1))],
            vec![(0, 1, Fr::from_u64(1)), (1, 1, Fr::from_u64(1))],
            vec![(0, 2, Fr::from_u64(1)), (1, 3, Fr::from_u64(1))],
        );
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(4),
            Fr::from_u64(8),
        ];

        let (proof, _) =
            prove_helper(&r1cs, &key, &witness, b"spartan-multi").expect("proving should succeed");

        let mut verifier_transcript = Blake2bTranscript::new(b"spartan-multi");
        SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut verifier_transcript)
            .expect("verification should succeed");
    }

    #[test]
    fn prove_and_verify_chain_multiplication() {
        // x^2 = y, x^3 = z, x^4 = w, x^5 = v
        // Constraint 0: A[0,1]*x * B[0,1]*x = C[0,2]*y
        // Constraint 1: A[1,2]*y * B[1,1]*x = C[1,3]*z
        // Constraint 2: A[2,3]*z * B[2,1]*x = C[2,4]*w
        // Constraint 3: A[3,4]*w * B[3,1]*x = C[3,5]*v
        let one = Fr::from_u64(1);
        let r1cs = SimpleR1CS::new(
            4,
            6,
            vec![(0, 1, one), (1, 2, one), (2, 3, one), (3, 4, one)],
            vec![(0, 1, one), (1, 1, one), (2, 1, one), (3, 1, one)],
            vec![(0, 2, one), (1, 3, one), (2, 4, one), (3, 5, one)],
        );
        let key = SpartanKey::from_r1cs(&r1cs);
        // x=3: 1, 3, 9, 27, 81, 243
        let witness = [
            Fr::from_u64(1),
            Fr::from_u64(3),
            Fr::from_u64(9),
            Fr::from_u64(27),
            Fr::from_u64(81),
            Fr::from_u64(243),
        ];

        let (proof, _) =
            prove_helper(&r1cs, &key, &witness, b"spartan-chain").expect("proving should succeed");

        let mut vt = Blake2bTranscript::new(b"spartan-chain");
        SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt)
            .expect("verification should succeed");
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn witness_length_mismatch_panics() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        // Only 2 elements instead of 3
        let witness = [Fr::from_u64(1), Fr::from_u64(3)];
        let _ = prove_helper(&r1cs, &key, &witness, b"spartan-mismatch");
    }

    #[test]
    fn tampered_az_eval_rejected() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

        let (mut proof, _) =
            prove_helper(&r1cs, &key, &witness, b"spartan-tamper").expect("proving should succeed");

        proof.az_eval += Fr::from_u64(1);

        let mut vt = Blake2bTranscript::new(b"spartan-tamper");
        let result = SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt);
        assert!(result.is_err(), "tampered az_eval should be rejected");
    }

    #[test]
    fn tampered_bz_eval_rejected() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

        let (mut proof, _) =
            prove_helper(&r1cs, &key, &witness, b"spartan-tamper").expect("proving should succeed");

        proof.bz_eval += Fr::from_u64(1);

        let mut vt = Blake2bTranscript::new(b"spartan-tamper");
        let result = SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt);
        assert!(result.is_err(), "tampered bz_eval should be rejected");
    }

    #[test]
    fn tampered_cz_eval_rejected() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

        let (mut proof, _) =
            prove_helper(&r1cs, &key, &witness, b"spartan-tamper").expect("proving should succeed");

        proof.cz_eval += Fr::from_u64(1);

        let mut vt = Blake2bTranscript::new(b"spartan-tamper");
        let result = SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt);
        assert!(result.is_err(), "tampered cz_eval should be rejected");
    }

    #[test]
    fn tampered_witness_eval_rejected() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

        let (mut proof, _) =
            prove_helper(&r1cs, &key, &witness, b"spartan-tamper").expect("proving should succeed");

        proof.witness_eval += Fr::from_u64(1);

        let mut vt = Blake2bTranscript::new(b"spartan-tamper");
        let result = SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt);
        assert!(
            matches!(result, Err(SpartanError::InnerEvaluationMismatch)),
            "tampered witness_eval should fail inner eval check, got: {result:?}"
        );
    }

    #[test]
    fn tampered_inner_sumcheck_rejected() {
        // Use a larger circuit so the inner sumcheck has multiple rounds
        let one = Fr::from_u64(1);
        let r1cs = SimpleR1CS::new(
            4,
            6,
            vec![(0, 1, one), (1, 2, one), (2, 3, one), (3, 4, one)],
            vec![(0, 1, one), (1, 1, one), (2, 1, one), (3, 1, one)],
            vec![(0, 2, one), (1, 3, one), (2, 4, one), (3, 5, one)],
        );
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [
            Fr::from_u64(1),
            Fr::from_u64(3),
            Fr::from_u64(9),
            Fr::from_u64(27),
            Fr::from_u64(81),
            Fr::from_u64(243),
        ];

        let (mut proof, _) = prove_helper(&r1cs, &key, &witness, b"spartan-inner-tamper")
            .expect("proving should succeed");

        // Replace the first inner sumcheck round polynomial with a shifted copy
        let orig = &proof.inner_sumcheck_proof.round_polynomials[0];
        let tampered_coeffs: Vec<Fr> = orig
            .coefficients()
            .iter()
            .enumerate()
            .map(|(i, &c)| if i == 0 { c + Fr::from_u64(1) } else { c })
            .collect();
        proof.inner_sumcheck_proof.round_polynomials[0] =
            jolt_poly::UnivariatePoly::new(tampered_coeffs);

        let mut vt = Blake2bTranscript::new(b"spartan-inner-tamper");
        let result = SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt);
        assert!(
            result.is_err(),
            "tampered inner sumcheck should be rejected"
        );
    }

    #[test]
    fn tampered_outer_sumcheck_rejected() {
        // Need a multi-constraint circuit so the outer sumcheck has rounds
        let one = Fr::from_u64(1);
        let r1cs = SimpleR1CS::new(
            2,
            4,
            vec![(0, 1, one), (1, 2, one)],
            vec![(0, 1, one), (1, 1, one)],
            vec![(0, 2, one), (1, 3, one)],
        );
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(4),
            Fr::from_u64(8),
        ];

        let (mut proof, _) = prove_helper(&r1cs, &key, &witness, b"spartan-outer-tamper")
            .expect("proving should succeed");

        // Replace the first outer sumcheck round polynomial with a shifted copy
        let orig = &proof.outer_sumcheck_proof.round_polynomials[0];
        let tampered_coeffs: Vec<Fr> = orig
            .coefficients()
            .iter()
            .enumerate()
            .map(|(i, &c)| if i == 0 { c + Fr::from_u64(1) } else { c })
            .collect();
        proof.outer_sumcheck_proof.round_polynomials[0] =
            jolt_poly::UnivariatePoly::new(tampered_coeffs);

        let mut vt = Blake2bTranscript::new(b"spartan-outer-tamper");
        let result = SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt);
        assert!(
            result.is_err(),
            "tampered outer sumcheck should be rejected"
        );
    }

    mod uniskip_tests {
        use super::*;

        fn prove_uniskip(
            r1cs: &SimpleR1CS<Fr>,
            key: &SpartanKey<Fr>,
            witness: &[Fr],
            label: &'static [u8],
        ) -> Result<(SpartanProof<Fr, MockPCS>, Blake2bTranscript), SpartanError> {
            let mut transcript = Blake2bTranscript::new(label);
            let proof = SpartanProver::prove::<MockPCS, _>(
                r1cs,
                key,
                witness,
                &(),
                &mut transcript,
                FirstRoundStrategy::UnivariateSkip,
            )?;
            Ok((proof, transcript))
        }

        #[test]
        fn uniskip_prove_verify_x_squared() {
            let r1cs = x_squared_circuit();
            let key = SpartanKey::from_r1cs(&r1cs);
            let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

            let (proof, _) = prove_uniskip(&r1cs, &key, &witness, b"spartan-uniskip")
                .expect("proving should succeed");

            let mut vt = Blake2bTranscript::new(b"spartan-uniskip");
            SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt)
                .expect("verification should succeed");
        }

        #[test]
        fn uniskip_prove_verify_multiple_constraints() {
            let one = Fr::from_u64(1);
            let r1cs = SimpleR1CS::new(
                2,
                4,
                vec![(0, 1, one), (1, 2, one)],
                vec![(0, 1, one), (1, 1, one)],
                vec![(0, 2, one), (1, 3, one)],
            );
            let key = SpartanKey::from_r1cs(&r1cs);
            let witness = [
                Fr::from_u64(1),
                Fr::from_u64(2),
                Fr::from_u64(4),
                Fr::from_u64(8),
            ];

            let (proof, _) = prove_uniskip(&r1cs, &key, &witness, b"spartan-uniskip-multi")
                .expect("proving should succeed");

            let mut vt = Blake2bTranscript::new(b"spartan-uniskip-multi");
            SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt)
                .expect("verification should succeed");
        }

        #[test]
        fn uniskip_prove_verify_chain_multiplication() {
            let one = Fr::from_u64(1);
            let r1cs = SimpleR1CS::new(
                4,
                6,
                vec![(0, 1, one), (1, 2, one), (2, 3, one), (3, 4, one)],
                vec![(0, 1, one), (1, 1, one), (2, 1, one), (3, 1, one)],
                vec![(0, 2, one), (1, 3, one), (2, 4, one), (3, 5, one)],
            );
            let key = SpartanKey::from_r1cs(&r1cs);
            let witness = [
                Fr::from_u64(1),
                Fr::from_u64(3),
                Fr::from_u64(9),
                Fr::from_u64(27),
                Fr::from_u64(81),
                Fr::from_u64(243),
            ];

            let (proof, _) = prove_uniskip(&r1cs, &key, &witness, b"spartan-uniskip-chain")
                .expect("proving should succeed");

            let mut vt = Blake2bTranscript::new(b"spartan-uniskip-chain");
            SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt)
                .expect("verification should succeed");
        }

        #[test]
        fn uniskip_produces_same_proof_as_standard() {
            let one = Fr::from_u64(1);
            let r1cs = SimpleR1CS::new(
                2,
                4,
                vec![(0, 1, one), (1, 2, one)],
                vec![(0, 1, one), (1, 1, one)],
                vec![(0, 2, one), (1, 3, one)],
            );
            let key = SpartanKey::from_r1cs(&r1cs);
            let witness = [
                Fr::from_u64(1),
                Fr::from_u64(2),
                Fr::from_u64(4),
                Fr::from_u64(8),
            ];

            let (standard_proof, _) = prove_helper(&r1cs, &key, &witness, b"spartan-cmp")
                .expect("standard proving should succeed");
            let (uniskip_proof, _) = prove_uniskip(&r1cs, &key, &witness, b"spartan-cmp")
                .expect("uniskip proving should succeed");

            // The first round polynomial should be identical
            assert_eq!(
                standard_proof.outer_sumcheck_proof.round_polynomials[0].coefficients(),
                uniskip_proof.outer_sumcheck_proof.round_polynomials[0].coefficients(),
                "first round polynomial should match between Standard and UnivariateSkip"
            );

            // Full proofs should be identical (same transcript → same challenges)
            assert_eq!(standard_proof.az_eval, uniskip_proof.az_eval);
            assert_eq!(standard_proof.bz_eval, uniskip_proof.bz_eval);
            assert_eq!(standard_proof.cz_eval, uniskip_proof.cz_eval);
            assert_eq!(standard_proof.witness_eval, uniskip_proof.witness_eval);
        }
    }

    mod ir_tests {
        use super::*;
        use jolt_ir::{ExprBuilder, R1csVar};

        /// Full pipeline: ExprBuilder → SoP → emit_r1cs → build_witness →
        /// SpartanKey → prove → verify.
        fn ir_prove_verify(
            emission: &jolt_ir::R1csEmission<Fr>,
            witness: &[Fr],
            label: &'static [u8],
        ) {
            let key = SpartanKey::from_r1cs(emission);

            let mut prover_transcript = Blake2bTranscript::new(label);
            let proof = SpartanProver::prove::<MockPCS, _>(
                emission,
                &key,
                witness,
                &(),
                &mut prover_transcript,
                FirstRoundStrategy::Standard,
            )
            .expect("proving should succeed");

            let mut verifier_transcript = Blake2bTranscript::new(label);
            SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut verifier_transcript)
                .expect("verification should succeed");
        }

        #[test]
        fn ir_prove_verify_simple_mul() {
            // x * y
            let b = ExprBuilder::new();
            let x = b.opening(0);
            let y = b.opening(1);
            let expr = b.build(x * y);
            let sop = expr.to_sum_of_products();

            let opening_vars = [R1csVar(1), R1csVar(2)];
            let mut next_var = 3;
            let emission = sop.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

            let opening_vals = [Fr::from_u64(7), Fr::from_u64(11)];
            let witness = build_witness(&emission, &opening_vars, &opening_vals);

            ir_prove_verify(&emission, &witness, b"ir-simple-mul");
        }

        #[test]
        fn ir_prove_verify_booleanity() {
            // gamma * (h^2 - h) with baked challenge
            let b = ExprBuilder::new();
            let h = b.opening(0);
            let gamma = b.challenge(0);
            let expr = b.build(gamma * (h * h - h));
            let sop = expr.to_sum_of_products();

            let opening_vars = [R1csVar(1)];
            let challenge_vals = [Fr::from_u64(5)];
            let mut next_var = 2;
            let emission = sop.emit_r1cs::<Fr>(&opening_vars, &challenge_vals, &mut next_var);

            let opening_vals = [Fr::from_u64(3)];
            let witness = build_witness(&emission, &opening_vars, &opening_vals);

            // Sanity: output should be 5 * (9 - 3) = 30
            let output = witness[emission.output_var.index()];
            assert_eq!(output, Fr::from_u64(30));

            ir_prove_verify(&emission, &witness, b"ir-booleanity");
        }

        #[test]
        fn ir_prove_verify_distribution() {
            // (a + b) * (c - d) → 4 SoP terms: a*c, -a*d, b*c, -b*d
            let b = ExprBuilder::new();
            let a = b.opening(0);
            let bv = b.opening(1);
            let c = b.opening(2);
            let d = b.opening(3);
            let expr = b.build((a + bv) * (c - d));
            let sop = expr.to_sum_of_products();

            let opening_vars = [R1csVar(1), R1csVar(2), R1csVar(3), R1csVar(4)];
            let mut next_var = 5;
            let emission = sop.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

            // (2+3) * (7-1) = 5 * 6 = 30
            let opening_vals = [
                Fr::from_u64(2),
                Fr::from_u64(3),
                Fr::from_u64(7),
                Fr::from_u64(1),
            ];
            let witness = build_witness(&emission, &opening_vars, &opening_vals);

            let output = witness[emission.output_var.index()];
            assert_eq!(output, Fr::from_u64(30));

            ir_prove_verify(&emission, &witness, b"ir-distribution");
        }

        #[test]
        fn ir_prove_verify_weighted_sum() {
            // alpha*a + beta*b (challenges baked as constants)
            let b = ExprBuilder::new();
            let a = b.opening(0);
            let bv = b.opening(1);
            let alpha = b.challenge(0);
            let beta = b.challenge(1);
            let expr = b.build(alpha * a + beta * bv);
            let sop = expr.to_sum_of_products();

            let opening_vars = [R1csVar(1), R1csVar(2)];
            let challenge_vals = [Fr::from_u64(3), Fr::from_u64(7)];
            let mut next_var = 3;
            let emission = sop.emit_r1cs::<Fr>(&opening_vars, &challenge_vals, &mut next_var);

            // 3*10 + 7*20 = 30 + 140 = 170
            let opening_vals = [Fr::from_u64(10), Fr::from_u64(20)];
            let witness = build_witness(&emission, &opening_vars, &opening_vals);

            let output = witness[emission.output_var.index()];
            assert_eq!(output, Fr::from_u64(170));

            ir_prove_verify(&emission, &witness, b"ir-weighted-sum");
        }

        #[test]
        fn ir_reject_bad_witness() {
            let b = ExprBuilder::new();
            let x = b.opening(0);
            let y = b.opening(1);
            let expr = b.build(x * y);
            let sop = expr.to_sum_of_products();

            let opening_vars = [R1csVar(1), R1csVar(2)];
            let mut next_var = 3;
            let emission = sop.emit_r1cs::<Fr>(&opening_vars, &[], &mut next_var);

            // Build a correct witness then tamper with it
            let mut witness = build_witness(
                &emission,
                &opening_vars,
                &[Fr::from_u64(7), Fr::from_u64(11)],
            );
            // Corrupt the output variable
            let output_idx = emission.output_var.index();
            witness[output_idx] += Fr::from_u64(1);

            let key = SpartanKey::from_r1cs(&emission);
            let mut transcript = Blake2bTranscript::new(b"ir-bad-witness");
            let result = SpartanProver::prove::<MockPCS, _>(
                &emission,
                &key,
                &witness,
                &(),
                &mut transcript,
                FirstRoundStrategy::Standard,
            );
            assert!(
                matches!(result, Err(SpartanError::ConstraintViolation(_))),
                "tampered witness should be rejected"
            );
        }
    }
}
