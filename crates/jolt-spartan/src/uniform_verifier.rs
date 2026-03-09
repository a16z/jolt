//! Verifier for uniform (repeated-constraint) Spartan proofs.
//!
//! Uses the [`UniformSpartanKey`] sparse representation to evaluate matrix
//! MLEs, avoiding the dense MLE storage required by the standard verifier.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{SumcheckClaim, SumcheckVerifier};
use jolt_transcript::Transcript;
use num_traits::Zero;

use crate::error::SpartanError;
use crate::uniform_key::UniformSpartanKey;
use crate::uniform_prover::UniformSpartanProof;

/// Stateless uniform Spartan verifier.
pub struct UniformSpartanVerifier;

impl UniformSpartanVerifier {
    /// Verifies a uniform Spartan proof.
    ///
    /// # Protocol
    ///
    /// 1. Absorb witness commitment, sample $\tau$.
    /// 2. Verify outer sumcheck → obtain $r_x$.
    /// 3. Check: $\widetilde{eq}(r_x, \tau) \cdot (Az \cdot Bz - Cz) = v$.
    /// 4. Absorb evaluation claims, sample $\rho_A, \rho_B, \rho_C$.
    /// 5. Verify inner sumcheck → obtain $r_y$.
    /// 6. Evaluate matrix MLEs via the sparse key at $(r_x, r_y)$.
    /// 7. Check: $M(r_x, r_y) \cdot z(r_y) = v_{\text{inner}}$.
    /// 8. Verify witness opening proof at $r_y$.
    #[tracing::instrument(skip_all, name = "UniformSpartanVerifier::verify")]
    pub fn verify<PCS, T>(
        key: &UniformSpartanKey<PCS::Field>,
        proof: &UniformSpartanProof<PCS::Field, PCS>,
        verifier_setup: &PCS::VerifierSetup,
        transcript: &mut T,
    ) -> Result<(), SpartanError>
    where
        PCS: CommitmentScheme,
        T: Transcript<Challenge = u128>,
    {
        Self::verify_with_challenges::<PCS, T>(key, proof, verifier_setup, transcript)
            .map(|_| ())
    }

    /// Verifies a uniform Spartan proof and returns the challenge vectors.
    ///
    /// Same as [`verify`](Self::verify) but returns `(r_x, r_y)` — the outer
    /// and inner sumcheck challenge points. Downstream stages need these to
    /// construct eq-weighted sumcheck claims.
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "UniformSpartanVerifier::verify_with_challenges")]
    pub fn verify_with_challenges<PCS, T>(
        key: &UniformSpartanKey<PCS::Field>,
        proof: &UniformSpartanProof<PCS::Field, PCS>,
        verifier_setup: &PCS::VerifierSetup,
        transcript: &mut T,
    ) -> Result<(Vec<PCS::Field>, Vec<PCS::Field>), SpartanError>
    where
        PCS: CommitmentScheme,
        T: Transcript<Challenge = u128>,
    {
        let total_rows_padded = key.total_rows().next_power_of_two();
        let total_cols_padded = key.total_cols().next_power_of_two();
        let num_row_vars = total_rows_padded.trailing_zeros() as usize;
        let num_col_vars = total_cols_padded.trailing_zeros() as usize;

        transcript.append_bytes(format!("{:?}", proof.witness_commitment).as_bytes());

        let tau: Vec<PCS::Field> = (0..num_row_vars)
            .map(|_| PCS::Field::from_u128(transcript.challenge()))
            .collect();

        let outer_claim = SumcheckClaim {
            num_vars: num_row_vars,
            degree: 3,
            claimed_sum: PCS::Field::zero(),
        };

        let (outer_final_eval, r_x) = SumcheckVerifier::verify(
            &outer_claim,
            &proof.outer_sumcheck_proof,
            transcript,
            |c: u128| PCS::Field::from_u128(c),
        )?;

        let eq_eval = EqPolynomial::new(tau).evaluate(&r_x);
        let expected = eq_eval * (proof.az_eval * proof.bz_eval - proof.cz_eval);
        if expected != outer_final_eval {
            return Err(SpartanError::OuterEvaluationMismatch);
        }

        transcript.append(&proof.az_eval);
        transcript.append(&proof.bz_eval);
        transcript.append(&proof.cz_eval);

        let rho_a = PCS::Field::from_u128(transcript.challenge());
        let rho_b = PCS::Field::from_u128(transcript.challenge());
        let rho_c = PCS::Field::from_u128(transcript.challenge());

        let inner_claim = SumcheckClaim {
            num_vars: num_col_vars,
            degree: 2,
            claimed_sum: rho_a * proof.az_eval + rho_b * proof.bz_eval + rho_c * proof.cz_eval,
        };

        let (inner_final_eval, r_y) = SumcheckVerifier::verify(
            &inner_claim,
            &proof.inner_sumcheck_proof,
            transcript,
            |c: u128| PCS::Field::from_u128(c),
        )?;

        let (a_eval, b_eval, c_eval) = key.evaluate_matrix_mles(&r_x, &r_y);
        let combined_matrix_eval = rho_a * a_eval + rho_b * b_eval + rho_c * c_eval;

        if combined_matrix_eval * proof.witness_eval != inner_final_eval {
            return Err(SpartanError::InnerEvaluationMismatch);
        }

        PCS::verify(
            &proof.witness_commitment,
            &r_y,
            proof.witness_eval,
            &proof.witness_opening_proof,
            verifier_setup,
            transcript,
        )?;

        Ok((r_x, r_y))
    }
}
