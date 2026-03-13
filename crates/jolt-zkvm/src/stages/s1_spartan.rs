//! Stage 1: Spartan R1CS proof (PIOP only — no PCS).
//!
//! Wraps the Spartan prover/verifier to produce a Spartan proof and extract
//! the outer/inner sumcheck challenge vectors needed by downstream stages.
//!
//! The caller is responsible for committing the witness polynomial and
//! appending the commitment to the Fiat-Shamir transcript **before** calling
//! `prove`. The witness opening proof is handled by S8 (batch opening phase).

use jolt_field::Field;
use jolt_openings::ProverClaim;
use jolt_spartan::{
    FirstRoundStrategy, SpartanError, SpartanKey, SpartanProof, SpartanProver, SpartanVerifier,
    UniformSpartanKey, UniformSpartanProof, UniformSpartanProver, UniformSpartanVerifier, R1CS,
};
use jolt_transcript::Transcript;

/// Thin wrapper around standard Spartan proving that surfaces challenge vectors.
pub struct SpartanStage;

/// Output of [`SpartanStage::prove`].
pub struct SpartanResult<F: Field> {
    /// The Spartan proof (outer + inner sumcheck, PIOP only).
    pub proof: SpartanProof<F>,
    /// Witness polynomial opening claim at the inner sumcheck challenge point.
    pub witness_opening_claim: ProverClaim<F>,
    /// Outer sumcheck challenge vector, needed by downstream claim reductions.
    pub r_x: Vec<F>,
    /// Inner sumcheck challenge vector (witness evaluation point).
    pub r_y: Vec<F>,
}

impl SpartanStage {
    /// Runs the full Spartan PIOP and returns the proof alongside
    /// the challenge vectors and witness opening claim.
    ///
    /// The caller must commit the witness and append the commitment to the
    /// transcript before calling this.
    #[tracing::instrument(skip_all, name = "SpartanStage::prove")]
    pub fn prove<F, T>(
        r1cs: &impl R1CS<F>,
        key: &SpartanKey<F>,
        witness: &[F],
        witness_evals: &[F],
        transcript: &mut T,
        strategy: FirstRoundStrategy,
    ) -> Result<SpartanResult<F>, SpartanError>
    where
        F: Field,
        T: Transcript<Challenge = F>,
    {
        let (proof, r_x, r_y) =
            SpartanProver::prove_with_challenges(r1cs, key, witness, transcript, strategy)?;

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

    /// Verifies a Spartan proof (PIOP only).
    ///
    /// The caller must append the witness commitment to the transcript before
    /// calling this, and verify the witness opening proof afterward.
    #[tracing::instrument(skip_all, name = "SpartanStage::verify")]
    pub fn verify<F, T>(
        key: &SpartanKey<F>,
        proof: &SpartanProof<F>,
        transcript: &mut T,
    ) -> Result<(Vec<F>, Vec<F>), SpartanError>
    where
        F: Field,
        T: Transcript<Challenge = F>,
    {
        SpartanVerifier::verify_with_challenges(key, proof, transcript)
    }
}

/// Output of [`UniformSpartanStage::prove`].
pub struct UniformSpartanResult<F: Field> {
    /// The uniform Spartan proof (PIOP only).
    pub proof: UniformSpartanProof<F>,
    /// Witness polynomial opening claim at the inner sumcheck challenge point.
    pub witness_opening_claim: ProverClaim<F>,
    /// Outer sumcheck challenge vector (`log2(total_rows)` elements).
    ///
    /// Decomposes as `(r_cycle, r_constraint)` — the cycle and within-cycle
    /// dimensions of the row index. Downstream stages use `r_cycle` as the
    /// eq point for claim reductions.
    pub r_x: Vec<F>,
    /// Inner sumcheck challenge vector (`log2(total_cols)` elements).
    ///
    /// Decomposes as `(r_cycle, r_var)` — the cycle and within-cycle
    /// dimensions of the column index. This is the evaluation point for the
    /// interleaved witness polynomial.
    pub r_y: Vec<F>,
}

/// Thin wrapper around the uniform Spartan prover (PIOP only).
///
/// Uses [`UniformSpartanKey`] (per-cycle sparse constraints) instead of the
/// dense [`SpartanKey`], enabling O(K) key storage regardless of cycle count.
pub struct UniformSpartanStage;

impl UniformSpartanStage {
    /// Runs the uniform Spartan PIOP (dense mode) and returns the proof
    /// alongside challenge vectors and witness opening claim.
    ///
    /// The caller must commit the witness and append the commitment to the
    /// transcript before calling this.
    ///
    /// # Arguments
    ///
    /// * `key` — Uniform Spartan key with per-cycle sparse constraints.
    /// * `flat_witness` — Flat interleaved witness vector of length
    ///   `total_cols_padded`. Layout: `flat_witness[c * num_vars_padded + v]`.
    /// * `witness_evals` — Evaluations for the opening claim (same flat vector).
    /// * `transcript` — Fiat-Shamir transcript (with witness commitment already
    ///   appended).
    #[tracing::instrument(skip_all, name = "UniformSpartanStage::prove")]
    pub fn prove<F, T>(
        key: &UniformSpartanKey<F>,
        flat_witness: &[F],
        witness_evals: &[F],
        transcript: &mut T,
    ) -> Result<UniformSpartanResult<F>, SpartanError>
    where
        F: Field,
        T: Transcript<Challenge = F>,
    {
        let (proof, r_x, r_y) =
            UniformSpartanProver::prove_dense_with_challenges(key, flat_witness, transcript)?;

        let witness_opening_claim = ProverClaim {
            evaluations: witness_evals.to_vec(),
            point: r_y.clone(),
            eval: proof.witness_eval,
        };

        Ok(UniformSpartanResult {
            proof,
            witness_opening_claim,
            r_x,
            r_y,
        })
    }

    /// Verifies a uniform Spartan proof (PIOP only).
    ///
    /// Returns `(r_x, r_y)` challenge vectors for downstream stages.
    /// The caller must append the witness commitment to the transcript before
    /// calling this, and verify the witness opening proof afterward.
    #[tracing::instrument(skip_all, name = "UniformSpartanStage::verify")]
    pub fn verify<F, T>(
        key: &UniformSpartanKey<F>,
        proof: &UniformSpartanProof<F>,
        transcript: &mut T,
    ) -> Result<(Vec<F>, Vec<F>), SpartanError>
    where
        F: Field,
        T: Transcript<Challenge = F>,
    {
        UniformSpartanVerifier::verify_with_challenges(key, proof, transcript)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use jolt_openings::mock::MockCommitmentScheme;
    use jolt_openings::CommitmentScheme;
    use jolt_spartan::SimpleR1CS;
    use jolt_transcript::{Blake2bTranscript, Transcript};

    type MockPCS = MockCommitmentScheme<Fr>;

    fn commit_witness(witness: &[Fr], padded_len: usize, transcript: &mut Blake2bTranscript) {
        let mut padded = vec![Fr::from_u64(0); padded_len];
        let copy_len = witness.len().min(padded_len);
        padded[..copy_len].copy_from_slice(&witness[..copy_len]);
        let (commitment, ()) = MockPCS::commit(&padded, &());
        transcript.append_bytes(format!("{commitment:?}").as_bytes());
    }

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
        commit_witness(&witness, key.num_variables_padded, &mut pt);
        let result = SpartanStage::prove(
            &r1cs,
            &key,
            &witness,
            &witness,
            &mut pt,
            FirstRoundStrategy::Standard,
        )
        .expect("proving should succeed");

        let mut vt = Blake2bTranscript::new(b"s1-test");
        commit_witness(&witness, key.num_variables_padded, &mut vt);
        let _ = SpartanStage::verify(&key, &result.proof, &mut vt)
            .expect("verification should succeed");
    }

    #[test]
    fn witness_claim_eval_matches_proof() {
        let r1cs = x_squared_circuit();
        let key = SpartanKey::from_r1cs(&r1cs);
        let witness = [Fr::from_u64(1), Fr::from_u64(3), Fr::from_u64(9)];

        let mut pt = Blake2bTranscript::new(b"s1-claim");
        commit_witness(&witness, key.num_variables_padded, &mut pt);
        let result = SpartanStage::prove(
            &r1cs,
            &key,
            &witness,
            &witness,
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
        commit_witness(&witness, key.num_variables_padded, &mut pt);
        let result = SpartanStage::prove(
            &r1cs,
            &key,
            &witness,
            &witness,
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
        commit_witness(&witness, key.num_variables_padded, &mut pt);
        let result = SpartanStage::prove(
            &r1cs,
            &key,
            &witness,
            &witness,
            &mut pt,
            FirstRoundStrategy::Standard,
        );
        assert!(matches!(result, Err(SpartanError::ConstraintViolation(0))));
    }

    mod uniform {
        use super::*;
        use jolt_spartan::UniformSpartanKey;
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

        fn make_flat_witness(key: &UniformSpartanKey<Fr>, cycle_witnesses: &[Vec<Fr>]) -> Vec<Fr> {
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
        fn uniform_prove_and_verify_round_trip() {
            let key = test_key(4);
            let witnesses = vec![
                make_cycle_witness(2),
                make_cycle_witness(3),
                make_cycle_witness(5),
                make_cycle_witness(7),
            ];
            let flat = make_flat_witness(&key, &witnesses);

            let mut pt = Blake2bTranscript::new(b"uniform-s1");
            uniform_commit(&flat, &mut pt);
            let result = UniformSpartanStage::prove(&key, &flat, &flat, &mut pt)
                .expect("proving should succeed");

            assert_eq!(result.r_x.len(), key.num_row_vars());
            assert_eq!(result.r_y.len(), key.num_col_vars());
            assert_eq!(result.witness_opening_claim.eval, result.proof.witness_eval);
            assert_eq!(result.witness_opening_claim.point, result.r_y);

            let mut vt = Blake2bTranscript::new(b"uniform-s1");
            uniform_commit(&flat, &mut vt);
            let _ = UniformSpartanStage::verify(&key, &result.proof, &mut vt)
                .expect("verification should succeed");
        }

        #[test]
        fn uniform_bad_witness_rejected() {
            let key = test_key(2);
            let mut witnesses = vec![make_cycle_witness(3), make_cycle_witness(5)];
            witnesses[1][2] = Fr::from_u64(26); // y should be 25

            let flat = make_flat_witness(&key, &witnesses);

            let mut pt = Blake2bTranscript::new(b"uniform-s1-bad");
            uniform_commit(&flat, &mut pt);
            let result = UniformSpartanStage::prove(&key, &flat, &flat, &mut pt);
            assert!(matches!(result, Err(SpartanError::ConstraintViolation(_))));
        }
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
        commit_witness(&witness, key.num_variables_padded, &mut pt_std);
        let std_result = SpartanStage::prove(
            &r1cs,
            &key,
            &witness,
            &witness,
            &mut pt_std,
            FirstRoundStrategy::Standard,
        )
        .expect("standard should succeed");

        let mut pt_uni = Blake2bTranscript::new(b"s1-cmp");
        commit_witness(&witness, key.num_variables_padded, &mut pt_uni);
        let uni_result = SpartanStage::prove(
            &r1cs,
            &key,
            &witness,
            &witness,
            &mut pt_uni,
            FirstRoundStrategy::UnivariateSkip,
        )
        .expect("uniskip should succeed");

        assert_eq!(std_result.proof.witness_eval, uni_result.proof.witness_eval);
        assert_eq!(std_result.r_x, uni_result.r_x);
        assert_eq!(std_result.r_y, uni_result.r_y);
    }
}
