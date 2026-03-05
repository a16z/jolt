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
//! | [`uni_skip`] | Univariate skip optimization strategy |
//! | [`error`] | Error types |

pub mod error;
pub mod key;
pub mod proof;
pub mod prover;
pub mod r1cs;
pub mod uni_skip;
pub mod verifier;

pub use error::SpartanError;
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
        let proof = SpartanProver::prove::<MockPCS, _>(r1cs, key, witness, &(), &mut transcript)?;
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
        SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &mut verifier_transcript)
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
        SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &mut verifier_transcript)
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
        SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &mut vt)
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
    fn tampered_proof_rejected() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

        let (mut proof, _) =
            prove_helper(&r1cs, &key, &witness, b"spartan-tamper").expect("proving should succeed");

        proof.az_eval += Fr::from_u64(1);

        let mut verifier_transcript = Blake2bTranscript::new(b"spartan-tamper");
        let result = SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &mut verifier_transcript);
        assert!(result.is_err(), "tampered proof should be rejected");
    }
}
