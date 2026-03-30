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
//! | [`uniform_key`] | Uniform Spartan key for repeated-constraint R1CS |
//! | [`uniform_prover`] | Prover for uniform R1CS (per-cycle witnesses) |
//! | [`uniform_verifier`] | Verifier for uniform Spartan proofs |
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
pub mod uniform_key;
pub mod uniform_prover;
pub mod uniform_verifier;
pub mod verifier;

pub use error::SpartanError;
pub use ir_r1cs::build_witness;
pub use key::SpartanKey;
pub use proof::SpartanProof;
pub use prover::SpartanProver;
pub use r1cs::{SimpleR1CS, R1CS};
pub use uni_skip::{uniskip_round_poly, FirstRoundStrategy};
pub use uniform_key::UniformSpartanKey;
pub use uniform_prover::{UniformSpartanProof, UniformSpartanProver};
pub use uniform_verifier::UniformSpartanVerifier;
pub use verifier::SpartanVerifier;

// Re-export relaxed types (still PCS-aware, used by BlindFold)
pub use proof::RelaxedSpartanProof;

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Field;
    use jolt_field::Fr;
    use jolt_openings::mock::MockCommitmentScheme;
    use jolt_openings::CommitmentScheme;
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

    /// Pads witness, commits via MockPCS, and appends commitment to transcript.
    fn commit_witness(witness: &[Fr], padded_len: usize, transcript: &mut Blake2bTranscript) {
        let mut padded = vec![Fr::from_u64(0); padded_len];
        let copy_len = witness.len().min(padded_len);
        padded[..copy_len].copy_from_slice(&witness[..copy_len]);
        let (commitment, ()) = MockPCS::commit(&padded, &());
        transcript.append_bytes(format!("{commitment:?}").as_bytes());
    }

    fn prove_helper(
        r1cs: &SimpleR1CS<Fr>,
        key: &SpartanKey<Fr>,
        witness: &[Fr],
        label: &'static [u8],
    ) -> Result<(SpartanProof<Fr>, Blake2bTranscript), SpartanError> {
        let mut transcript = Blake2bTranscript::new(label);
        commit_witness(witness, key.num_variables_padded, &mut transcript);
        let proof = SpartanProver::prove(
            r1cs,
            key,
            witness,
            &mut transcript,
            FirstRoundStrategy::Standard,
        )?;
        Ok((proof, transcript))
    }

    fn verify_helper(
        key: &SpartanKey<Fr>,
        proof: &SpartanProof<Fr>,
        witness: &[Fr],
        label: &'static [u8],
    ) -> Result<(), SpartanError> {
        let mut vt = Blake2bTranscript::new(label);
        commit_witness(witness, key.num_variables_padded, &mut vt);
        SpartanVerifier::verify(key, proof, &mut vt)
    }

    #[test]
    fn prove_and_verify_x_squared() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

        let (proof, _) =
            prove_helper(&r1cs, &key, &witness, b"spartan-test").expect("proving should succeed");

        verify_helper(&key, &proof, &witness, b"spartan-test")
            .expect("verification should succeed");
    }

    #[test]
    fn verify_with_challenges_matches_prover() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

        let mut pt = Blake2bTranscript::new(b"spartan-challenges");
        commit_witness(&witness, key.num_variables_padded, &mut pt);
        let (proof, prover_r_x, prover_r_y) = SpartanProver::prove_with_challenges(
            &r1cs,
            &key,
            &witness,
            &mut pt,
            FirstRoundStrategy::Standard,
        )
        .expect("proving should succeed");

        let mut vt = Blake2bTranscript::new(b"spartan-challenges");
        commit_witness(&witness, key.num_variables_padded, &mut vt);
        let (verifier_r_x, verifier_r_y) =
            SpartanVerifier::verify_with_challenges(&key, &proof, &mut vt)
                .expect("verification should succeed");

        assert_eq!(prover_r_x, verifier_r_x, "r_x must match");
        assert_eq!(prover_r_y, verifier_r_y, "r_y must match");
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

        verify_helper(&key, &proof, &witness, b"spartan-multi")
            .expect("verification should succeed");
    }

    #[test]
    fn prove_and_verify_chain_multiplication() {
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

        let (proof, _) =
            prove_helper(&r1cs, &key, &witness, b"spartan-chain").expect("proving should succeed");

        verify_helper(&key, &proof, &witness, b"spartan-chain")
            .expect("verification should succeed");
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn witness_length_mismatch_panics() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
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

        let result = verify_helper(&key, &proof, &witness, b"spartan-tamper");
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

        let result = verify_helper(&key, &proof, &witness, b"spartan-tamper");
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

        let result = verify_helper(&key, &proof, &witness, b"spartan-tamper");
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

        let result = verify_helper(&key, &proof, &witness, b"spartan-tamper");
        assert!(
            matches!(result, Err(SpartanError::InnerEvaluationMismatch)),
            "tampered witness_eval should fail inner eval check, got: {result:?}"
        );
    }

    #[test]
    fn tampered_inner_sumcheck_rejected() {
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

        let orig = &proof.inner_sumcheck_proof.round_polynomials[0];
        let tampered_coeffs: Vec<Fr> = orig
            .coefficients()
            .iter()
            .enumerate()
            .map(|(i, &c)| if i == 0 { c + Fr::from_u64(1) } else { c })
            .collect();
        proof.inner_sumcheck_proof.round_polynomials[0] =
            jolt_poly::UnivariatePoly::new(tampered_coeffs);

        let result = verify_helper(&key, &proof, &witness, b"spartan-inner-tamper");
        assert!(
            result.is_err(),
            "tampered inner sumcheck should be rejected"
        );
    }

    #[test]
    fn tampered_outer_sumcheck_rejected() {
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

        let orig = &proof.outer_sumcheck_proof.round_polynomials[0];
        let tampered_coeffs: Vec<Fr> = orig
            .coefficients()
            .iter()
            .enumerate()
            .map(|(i, &c)| if i == 0 { c + Fr::from_u64(1) } else { c })
            .collect();
        proof.outer_sumcheck_proof.round_polynomials[0] =
            jolt_poly::UnivariatePoly::new(tampered_coeffs);

        let result = verify_helper(&key, &proof, &witness, b"spartan-outer-tamper");
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
        ) -> Result<(SpartanProof<Fr>, Blake2bTranscript), SpartanError> {
            let mut transcript = Blake2bTranscript::new(label);
            commit_witness(witness, key.num_variables_padded, &mut transcript);
            let proof = SpartanProver::prove(
                r1cs,
                key,
                witness,
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

            verify_helper(&key, &proof, &witness, b"spartan-uniskip")
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

            verify_helper(&key, &proof, &witness, b"spartan-uniskip-multi")
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

            verify_helper(&key, &proof, &witness, b"spartan-uniskip-chain")
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

            assert_eq!(
                standard_proof.outer_sumcheck_proof.round_polynomials[0].coefficients(),
                uniskip_proof.outer_sumcheck_proof.round_polynomials[0].coefficients(),
                "first round polynomial should match between Standard and UnivariateSkip"
            );

            assert_eq!(standard_proof.az_eval, uniskip_proof.az_eval);
            assert_eq!(standard_proof.bz_eval, uniskip_proof.bz_eval);
            assert_eq!(standard_proof.cz_eval, uniskip_proof.cz_eval);
            assert_eq!(standard_proof.witness_eval, uniskip_proof.witness_eval);
        }
    }

    mod ir_tests {
        use super::*;
        use jolt_ir::{ExprBuilder, R1csVar};

        fn ir_prove_verify(
            emission: &jolt_ir::R1csEmission<Fr>,
            witness: &[Fr],
            label: &'static [u8],
        ) {
            let key = SpartanKey::from_r1cs(emission);

            let mut prover_transcript = Blake2bTranscript::new(label);
            commit_witness(witness, key.num_variables_padded, &mut prover_transcript);
            let proof = SpartanProver::prove(
                emission,
                &key,
                witness,
                &mut prover_transcript,
                FirstRoundStrategy::Standard,
            )
            .expect("proving should succeed");

            verify_helper(&key, &proof, witness, label).expect("verification should succeed");
        }

        #[test]
        fn ir_prove_verify_simple_mul() {
            // x * y
            let b = ExprBuilder::new();
            let x = b.opening(0);
            let y = b.opening(1);
            let expr = b.build(x * y);
            let sop = expr.to_composition_formula();

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
            let sop = expr.to_composition_formula();

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
            let sop = expr.to_composition_formula();

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
            let sop = expr.to_composition_formula();

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
            let sop = expr.to_composition_formula();

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
            commit_witness(&witness, key.num_variables_padded, &mut transcript);
            let result = SpartanProver::prove(
                &emission,
                &key,
                &witness,
                &mut transcript,
                FirstRoundStrategy::Standard,
            );
            assert!(
                matches!(result, Err(SpartanError::ConstraintViolation(_))),
                "tampered witness should be rejected"
            );
        }
    }

    mod relaxed_tests {
        use super::*;
        use jolt_openings::CommitmentScheme;
        use num_traits::{One, Zero};

        /// Pads a slice to the next power-of-two length with zeros.
        fn pad(data: &[Fr], padded_len: usize) -> Vec<Fr> {
            let mut v = vec![Fr::zero(); padded_len];
            v[..data.len()].copy_from_slice(data);
            v
        }

        /// u=1, E=0 — equivalent to standard R1CS. prove_relaxed should succeed
        /// and verify_relaxed should accept.
        #[test]
        fn prove_relaxed_standard_instance() {
            let r1cs = x_squared_circuit();
            let key = SpartanKey::from_r1cs(&r1cs);
            let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];
            let u = Fr::one();
            let error = vec![Fr::zero()];

            // Commitments must be over padded data to match proving internals
            let w_padded = pad(&witness, key.num_variables_padded);
            let e_padded = pad(&error, key.num_constraints_padded);
            let (w_commitment, ()) = MockPCS::commit(&w_padded, &());
            let (e_commitment, ()) = MockPCS::commit(&e_padded, &());

            let mut transcript = Blake2bTranscript::new(b"relaxed-standard");
            let proof = SpartanProver::prove_relaxed::<MockPCS, _>(
                &r1cs,
                &key,
                u,
                &witness,
                &error,
                &w_commitment,
                &e_commitment,
                &(),
                &mut transcript,
            )
            .expect("proving should succeed");

            let mut vt = Blake2bTranscript::new(b"relaxed-standard");
            SpartanVerifier::verify_relaxed::<MockPCS, _>(
                &key,
                u,
                &w_commitment,
                &e_commitment,
                &proof,
                &(),
                &mut vt,
            )
            .expect("verification should succeed");
        }

        /// prove_relaxed on a manually constructed relaxed instance (non-trivial u and E).
        #[test]
        fn prove_relaxed_manual_instance() {
            let r1cs = x_squared_circuit();
            let key = SpartanKey::from_r1cs(&r1cs);

            // z = [1, x, y], constraint: x*x = y
            // Pick u=2, witness=[1, 3, 9], so Az*Bz=3*3=9, u*Cz=2*9=18
            // E = Az*Bz - u*Cz = 9 - 18 = -9
            let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];
            let u = Fr::from_u64(2);
            let error = vec![Fr::from_u64(9) - Fr::from_u64(18)]; // -9

            let w_padded = pad(&witness, key.num_variables_padded);
            let e_padded = pad(&error, key.num_constraints_padded);
            let (w_commitment, ()) = MockPCS::commit(&w_padded, &());
            let (e_commitment, ()) = MockPCS::commit(&e_padded, &());

            let mut transcript = Blake2bTranscript::new(b"relaxed-manual");
            let proof = SpartanProver::prove_relaxed::<MockPCS, _>(
                &r1cs,
                &key,
                u,
                &witness,
                &error,
                &w_commitment,
                &e_commitment,
                &(),
                &mut transcript,
            )
            .expect("proving should succeed");

            let mut vt = Blake2bTranscript::new(b"relaxed-manual");
            SpartanVerifier::verify_relaxed::<MockPCS, _>(
                &key,
                u,
                &w_commitment,
                &e_commitment,
                &proof,
                &(),
                &mut vt,
            )
            .expect("verification should succeed");
        }

        /// Tampered error vector should be rejected by the prover.
        #[test]
        fn relaxed_reject_bad_error() {
            let r1cs = x_squared_circuit();
            let key = SpartanKey::from_r1cs(&r1cs);
            let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];
            let u = Fr::one();
            let error = vec![Fr::from_u64(1)]; // Wrong: should be 0

            let w_padded = pad(&witness, key.num_variables_padded);
            let e_padded = pad(&error, key.num_constraints_padded);
            let (w_commitment, ()) = MockPCS::commit(&w_padded, &());
            let (e_commitment, ()) = MockPCS::commit(&e_padded, &());

            let mut transcript = Blake2bTranscript::new(b"relaxed-bad-error");
            let result = SpartanProver::prove_relaxed::<MockPCS, _>(
                &r1cs,
                &key,
                u,
                &witness,
                &error,
                &w_commitment,
                &e_commitment,
                &(),
                &mut transcript,
            );

            assert!(
                matches!(result, Err(SpartanError::RelaxedConstraintViolation(0))),
                "tampered error should cause constraint violation at index 0"
            );
        }

        /// Tampered e_eval in proof should be rejected by verifier.
        #[test]
        fn relaxed_tampered_e_eval_rejected() {
            let r1cs = x_squared_circuit();
            let key = SpartanKey::from_r1cs(&r1cs);
            let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];
            let u = Fr::one();
            let error = vec![Fr::zero()];

            let w_padded = pad(&witness, key.num_variables_padded);
            let e_padded = pad(&error, key.num_constraints_padded);
            let (w_commitment, ()) = MockPCS::commit(&w_padded, &());
            let (e_commitment, ()) = MockPCS::commit(&e_padded, &());

            let mut transcript = Blake2bTranscript::new(b"relaxed-tampered-e");
            let mut proof = SpartanProver::prove_relaxed::<MockPCS, _>(
                &r1cs,
                &key,
                u,
                &witness,
                &error,
                &w_commitment,
                &e_commitment,
                &(),
                &mut transcript,
            )
            .expect("proving should succeed");

            // Tamper e_eval
            proof.e_eval += Fr::from_u64(1);

            let mut vt = Blake2bTranscript::new(b"relaxed-tampered-e");
            let result = SpartanVerifier::verify_relaxed::<MockPCS, _>(
                &key,
                u,
                &w_commitment,
                &e_commitment,
                &proof,
                &(),
                &mut vt,
            );
            assert!(result.is_err(), "tampered e_eval should be rejected");
        }
    }

    mod uniform_tests {
        use super::*;
        use num_traits::One;

        fn test_key(num_cycles: usize) -> UniformSpartanKey<Fr> {
            let one = Fr::from_u64(1);
            UniformSpartanKey::new(
                num_cycles,
                2,
                4,
                vec![vec![(1, one)], vec![(2, one)]],
                vec![vec![(1, one)], vec![(1, one)]],
                vec![vec![(2, one)], vec![(3, one)]],
            )
        }

        fn make_cycle_witness(x: u64) -> Vec<Fr> {
            vec![
                Fr::one(),
                Fr::from_u64(x),
                Fr::from_u64(x * x),
                Fr::from_u64(x * x * x),
            ]
        }

        /// Flattens per-cycle witnesses into a single interleaved vector.
        fn flatten_witnesses(key: &UniformSpartanKey<Fr>, cycle_witnesses: &[Vec<Fr>]) -> Vec<Fr> {
            let total_cols_padded = key.total_cols().next_power_of_two();
            let mut flat = vec![Fr::from_u64(0); total_cols_padded];
            for (c, w) in cycle_witnesses.iter().enumerate() {
                let base = c * key.num_vars_padded;
                for (v, &val) in w.iter().enumerate().take(key.num_vars) {
                    flat[base + v] = val;
                }
            }
            flat
        }

        fn uniform_commit(flat: &[Fr], transcript: &mut Blake2bTranscript) {
            let (commitment, ()) = MockPCS::commit(flat, &());
            transcript.append_bytes(format!("{commitment:?}").as_bytes());
        }

        #[test]
        fn uniform_prove_verify_single_cycle() {
            let key = test_key(1);
            let witnesses = vec![make_cycle_witness(3)];
            let flat = flatten_witnesses(&key, &witnesses);

            let mut transcript = Blake2bTranscript::new(b"uniform-single");
            uniform_commit(&flat, &mut transcript);
            let proof = UniformSpartanProver::prove_dense(&key, &flat, &mut transcript)
                .expect("proving should succeed");

            let mut vt = Blake2bTranscript::new(b"uniform-single");
            uniform_commit(&flat, &mut vt);
            UniformSpartanVerifier::verify(&key, &proof, &mut vt)
                .expect("verification should succeed");
        }

        #[test]
        fn uniform_prove_verify_two_cycles() {
            let key = test_key(2);
            let witnesses = vec![make_cycle_witness(3), make_cycle_witness(5)];
            let flat = flatten_witnesses(&key, &witnesses);

            let mut transcript = Blake2bTranscript::new(b"uniform-two");
            uniform_commit(&flat, &mut transcript);
            let proof = UniformSpartanProver::prove_dense(&key, &flat, &mut transcript)
                .expect("proving should succeed");

            let mut vt = Blake2bTranscript::new(b"uniform-two");
            uniform_commit(&flat, &mut vt);
            UniformSpartanVerifier::verify(&key, &proof, &mut vt)
                .expect("verification should succeed");
        }

        #[test]
        fn uniform_prove_verify_four_cycles() {
            let key = test_key(4);
            let witnesses = vec![
                make_cycle_witness(2),
                make_cycle_witness(3),
                make_cycle_witness(5),
                make_cycle_witness(7),
            ];
            let flat = flatten_witnesses(&key, &witnesses);

            let mut transcript = Blake2bTranscript::new(b"uniform-four");
            uniform_commit(&flat, &mut transcript);
            let proof = UniformSpartanProver::prove_dense(&key, &flat, &mut transcript)
                .expect("proving should succeed");

            let mut vt = Blake2bTranscript::new(b"uniform-four");
            uniform_commit(&flat, &mut vt);
            UniformSpartanVerifier::verify(&key, &proof, &mut vt)
                .expect("verification should succeed");
        }

        #[test]
        fn uniform_reject_bad_witness() {
            let key = test_key(2);
            let mut witnesses = vec![make_cycle_witness(3), make_cycle_witness(5)];
            witnesses[1][2] = Fr::from_u64(26); // y should be 25
            let flat = flatten_witnesses(&key, &witnesses);

            let mut transcript = Blake2bTranscript::new(b"uniform-bad");
            uniform_commit(&flat, &mut transcript);
            let result = UniformSpartanProver::prove_dense(&key, &flat, &mut transcript);
            assert!(
                matches!(result, Err(SpartanError::ConstraintViolation(_))),
                "corrupted witness should be rejected"
            );
        }

        #[test]
        fn uniform_tampered_az_eval_rejected() {
            let key = test_key(2);
            let witnesses = vec![make_cycle_witness(3), make_cycle_witness(7)];
            let flat = flatten_witnesses(&key, &witnesses);

            let mut transcript = Blake2bTranscript::new(b"uniform-tamper");
            uniform_commit(&flat, &mut transcript);
            let mut proof = UniformSpartanProver::prove_dense(&key, &flat, &mut transcript)
                .expect("proving should succeed");

            proof.az_eval += Fr::from_u64(1);

            let mut vt = Blake2bTranscript::new(b"uniform-tamper");
            uniform_commit(&flat, &mut vt);
            let result = UniformSpartanVerifier::verify(&key, &proof, &mut vt);
            assert!(result.is_err(), "tampered az_eval should be rejected");
        }

        #[test]
        fn uniform_tampered_witness_eval_rejected() {
            let key = test_key(2);
            let witnesses = vec![make_cycle_witness(3), make_cycle_witness(7)];
            let flat = flatten_witnesses(&key, &witnesses);

            let mut transcript = Blake2bTranscript::new(b"uniform-tamper-w");
            uniform_commit(&flat, &mut transcript);
            let mut proof = UniformSpartanProver::prove_dense(&key, &flat, &mut transcript)
                .expect("proving should succeed");

            proof.witness_eval += Fr::from_u64(1);

            let mut vt = Blake2bTranscript::new(b"uniform-tamper-w");
            uniform_commit(&flat, &mut vt);
            let result = UniformSpartanVerifier::verify(&key, &proof, &mut vt);
            assert!(
                matches!(result, Err(SpartanError::InnerEvaluationMismatch)),
                "tampered witness_eval should fail inner check, got: {result:?}"
            );
        }

        #[test]
        fn uniform_single_constraint_key() {
            let one = Fr::from_u64(1);
            let key = UniformSpartanKey::new(
                2,
                1,
                3,
                vec![vec![(1, one)]],
                vec![vec![(1, one)]],
                vec![vec![(2, one)]],
            );

            let witnesses = vec![
                vec![Fr::one(), Fr::from_u64(4), Fr::from_u64(16)],
                vec![Fr::one(), Fr::from_u64(6), Fr::from_u64(36)],
            ];
            let flat = flatten_witnesses(&key, &witnesses);

            let mut transcript = Blake2bTranscript::new(b"uniform-single-constr");
            uniform_commit(&flat, &mut transcript);
            let proof = UniformSpartanProver::prove_dense(&key, &flat, &mut transcript)
                .expect("proving should succeed");

            let mut vt = Blake2bTranscript::new(b"uniform-single-constr");
            uniform_commit(&flat, &mut vt);
            UniformSpartanVerifier::verify(&key, &proof, &mut vt)
                .expect("verification should succeed");
        }

        #[test]
        fn prove_dense_with_challenges_returns_valid_vectors() {
            let key = test_key(4);
            let witnesses = vec![
                make_cycle_witness(2),
                make_cycle_witness(3),
                make_cycle_witness(5),
                make_cycle_witness(7),
            ];
            let flat = flatten_witnesses(&key, &witnesses);

            let mut pt = Blake2bTranscript::new(b"uniform-challenges");
            uniform_commit(&flat, &mut pt);
            let (proof, r_x, r_y) =
                UniformSpartanProver::prove_dense_with_challenges(&key, &flat, &mut pt)
                    .expect("proving should succeed");

            assert_eq!(r_x.len(), key.num_row_vars());
            assert_eq!(r_y.len(), key.num_col_vars());

            let mut vt = Blake2bTranscript::new(b"uniform-challenges");
            uniform_commit(&flat, &mut vt);
            UniformSpartanVerifier::verify(&key, &proof, &mut vt)
                .expect("verification should succeed");

            assert!(!r_x.is_empty());
            assert!(!r_y.is_empty());
        }
    }
}
