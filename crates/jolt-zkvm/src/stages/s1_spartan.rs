//! Stage 1: Spartan R1CS proof.
//!
//! Wraps [`SpartanProver::prove_with_challenges`] to produce a Spartan proof
//! and extract the outer/inner sumcheck challenge vectors needed by downstream
//! stages.
//!
//! This stage does **not** implement [`ProverStage`](crate::stage::ProverStage)
//! because Spartan runs both outer and inner sumchecks internally, sharing
//! Az/Bz/Cz intermediates. Splitting into build/extract would require
//! decomposing the Spartan prover.

use std::marker::PhantomData;

use jolt_field::Field;
use jolt_openings::{CommitmentScheme, ProverClaim};
use jolt_spartan::{
    FirstRoundStrategy, SpartanError, SpartanKey, SpartanProof, SpartanProver, SpartanVerifier,
    R1CS,
};
use jolt_transcript::Transcript;

/// Thin wrapper around Spartan proving that surfaces challenge vectors.
pub struct SpartanStage<PCS: CommitmentScheme> {
    _marker: PhantomData<PCS>,
}

/// Output of [`SpartanStage::prove`].
pub struct SpartanResult<F: Field, PCS: CommitmentScheme> {
    /// The Spartan proof (outer + inner sumcheck + opening).
    pub proof: SpartanProof<F, PCS>,
    /// Witness polynomial opening claim at the inner sumcheck challenge point.
    pub witness_opening_claim: ProverClaim<F>,
    /// Outer sumcheck challenge vector, needed by downstream claim reductions.
    pub r_x: Vec<F>,
    /// Inner sumcheck challenge vector (witness evaluation point).
    pub r_y: Vec<F>,
}

impl<PCS: CommitmentScheme> SpartanStage<PCS> {
    /// Runs the full Spartan protocol and returns the proof alongside
    /// the challenge vectors and witness opening claim.
    #[tracing::instrument(skip_all, name = "SpartanStage::prove")]
    pub fn prove<T: Transcript<Challenge = u128>>(
        r1cs: &impl R1CS<PCS::Field>,
        key: &SpartanKey<PCS::Field>,
        witness: &[PCS::Field],
        witness_evals: &[PCS::Field],
        pcs_setup: &PCS::ProverSetup,
        transcript: &mut T,
        strategy: FirstRoundStrategy,
    ) -> Result<SpartanResult<PCS::Field, PCS>, SpartanError> {
        let (proof, r_x, r_y) = SpartanProver::prove_with_challenges::<PCS, T>(
            r1cs, key, witness, pcs_setup, transcript, strategy,
        )?;

        let witness_opening_claim = ProverClaim {
            evaluations: witness_evals.to_vec(),
            point: r_y.clone(),
            eval: proof.witness_eval,
        };

        Ok(SpartanResult {
            proof,
            witness_opening_claim,
            r_x,
            r_y,
        })
    }

    /// Verifies a Spartan proof.
    ///
    /// Delegates directly to [`SpartanVerifier::verify`].
    #[tracing::instrument(skip_all, name = "SpartanStage::verify")]
    pub fn verify<T: Transcript<Challenge = u128>>(
        key: &SpartanKey<PCS::Field>,
        proof: &SpartanProof<PCS::Field, PCS>,
        verifier_setup: &PCS::VerifierSetup,
        transcript: &mut T,
    ) -> Result<(), SpartanError> {
        SpartanVerifier::verify::<PCS, T>(key, proof, verifier_setup, transcript)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_openings::mock::MockCommitmentScheme;
    use jolt_spartan::SimpleR1CS;
    use jolt_transcript::{Blake2bTranscript, Transcript};

    type MockPCS = MockCommitmentScheme<Fr>;

    fn x_squared_circuit() -> SimpleR1CS<Fr> {
        SimpleR1CS::new(
            1,
            3,
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 1, Fr::from_u64(1))],
            vec![(0, 2, Fr::from_u64(1))],
        )
    }

    #[test]
    fn prove_and_verify_round_trip() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

        let mut pt = Blake2bTranscript::new(b"s1-test");
        let result = SpartanStage::<MockPCS>::prove(
            &r1cs,
            &key,
            &witness,
            &witness,
            &(),
            &mut pt,
            FirstRoundStrategy::Standard,
        )
        .expect("proving should succeed");

        let mut vt = Blake2bTranscript::new(b"s1-test");
        SpartanStage::<MockPCS>::verify(&key, &result.proof, &(), &mut vt)
            .expect("verification should succeed");
    }

    #[test]
    fn witness_claim_eval_matches_proof() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

        let mut pt = Blake2bTranscript::new(b"s1-claim");
        let result = SpartanStage::<MockPCS>::prove(
            &r1cs,
            &key,
            &witness,
            &witness,
            &(),
            &mut pt,
            FirstRoundStrategy::Standard,
        )
        .expect("proving should succeed");

        assert_eq!(result.witness_opening_claim.eval, result.proof.witness_eval);
        assert_eq!(result.witness_opening_claim.point, result.r_y);
    }

    #[test]
    fn challenge_vectors_have_correct_length() {
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

        let mut pt = Blake2bTranscript::new(b"s1-dims");
        let result = SpartanStage::<MockPCS>::prove(
            &r1cs,
            &key,
            &witness,
            &witness,
            &(),
            &mut pt,
            FirstRoundStrategy::Standard,
        )
        .expect("proving should succeed");

        assert_eq!(result.r_x.len(), key.num_sumcheck_vars());
        assert_eq!(result.r_y.len(), key.num_witness_vars());
    }

    #[test]
    fn bad_witness_rejected() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(10)];

        let mut pt = Blake2bTranscript::new(b"s1-bad");
        let result = SpartanStage::<MockPCS>::prove(
            &r1cs,
            &key,
            &witness,
            &witness,
            &(),
            &mut pt,
            FirstRoundStrategy::Standard,
        );
        assert!(matches!(result, Err(SpartanError::ConstraintViolation(0))));
    }

    #[test]
    fn uniskip_matches_standard() {
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

        let mut pt_std = Blake2bTranscript::new(b"s1-cmp");
        let std_result = SpartanStage::<MockPCS>::prove(
            &r1cs,
            &key,
            &witness,
            &witness,
            &(),
            &mut pt_std,
            FirstRoundStrategy::Standard,
        )
        .expect("standard should succeed");

        let mut pt_uni = Blake2bTranscript::new(b"s1-cmp");
        let uni_result = SpartanStage::<MockPCS>::prove(
            &r1cs,
            &key,
            &witness,
            &witness,
            &(),
            &mut pt_uni,
            FirstRoundStrategy::UnivariateSkip,
        )
        .expect("uniskip should succeed");

        assert_eq!(std_result.proof.witness_eval, uni_result.proof.witness_eval);
        assert_eq!(std_result.r_x, uni_result.r_x);
        assert_eq!(std_result.r_y, uni_result.r_y);
    }
}
