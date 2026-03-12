//! Verifier for uniform (repeated-constraint) Spartan proofs (pure PIOP).
//!
//! Uses the [`UniformSpartanKey`] sparse representation to evaluate matrix
//! MLEs, avoiding the dense MLE storage required by the standard verifier.
//!
//! The caller is responsible for appending the witness commitment to the
//! transcript before calling verify, and for checking the witness opening
//! proof after verify returns.

use jolt_field::WithChallenge;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{SumcheckClaim, SumcheckVerifier};
use jolt_transcript::Transcript;

use crate::error::SpartanError;
use crate::uniform_key::UniformSpartanKey;
use crate::uniform_prover::UniformSpartanProof;

/// Stateless uniform Spartan verifier.
pub struct UniformSpartanVerifier;

impl UniformSpartanVerifier {
    /// Verifies a uniform Spartan proof (PIOP only — no PCS checks).
    ///
    /// The caller must:
    /// 1. Append the witness commitment to the transcript before calling this.
    /// 2. Verify the witness opening proof at `r_y` after this returns.
    ///
    /// # Protocol
    ///
    /// 1. Sample $\tau$ (commitment already absorbed by caller).
    /// 2. Verify outer sumcheck → obtain $r_x$.
    /// 3. Check: $\widetilde{eq}(r_x, \tau) \cdot (Az \cdot Bz - Cz) = v$.
    /// 4. Absorb evaluation claims, sample $\rho_A, \rho_B, \rho_C$.
    /// 5. Verify inner sumcheck → obtain $r_y$.
    /// 6. Evaluate matrix MLEs via the sparse key at $(r_x, r_y)$.
    /// 7. Check: $M(r_x, r_y) \cdot z(r_y) = v_{\text{inner}}$.
    #[tracing::instrument(skip_all, name = "UniformSpartanVerifier::verify")]
    pub fn verify<F, T>(
        key: &UniformSpartanKey<F>,
        proof: &UniformSpartanProof<F>,
        transcript: &mut T,
    ) -> Result<(), SpartanError>
    where
        F: WithChallenge,
        F::Challenge: From<T::Challenge>,
        T: Transcript,
    {
        Self::verify_with_challenges(key, proof, transcript).map(|_| ())
    }

    /// Verifies a uniform Spartan proof and returns the challenge vectors.
    ///
    /// Same as [`verify`](Self::verify) but returns `(r_x, r_y)` — the outer
    /// and inner sumcheck challenge points. Downstream stages need `r_x` for
    /// eq-weighted sumcheck claims, and `r_y` is the witness evaluation point
    /// that the caller uses for PCS verification.
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "UniformSpartanVerifier::verify_with_challenges")]
    pub fn verify_with_challenges<F, T>(
        key: &UniformSpartanKey<F>,
        proof: &UniformSpartanProof<F>,
        transcript: &mut T,
    ) -> Result<(Vec<F>, Vec<F>), SpartanError>
    where
        F: WithChallenge,
        F::Challenge: From<T::Challenge>,
        T: Transcript,
    {
        let total_rows_padded = key.total_rows().next_power_of_two();
        let total_cols_padded = key.total_cols().next_power_of_two();
        let num_row_vars = total_rows_padded.trailing_zeros() as usize;
        let num_col_vars = total_cols_padded.trailing_zeros() as usize;

        let tau: Vec<F> = (0..num_row_vars)
            .map(|_| F::Challenge::from(transcript.challenge()).into())
            .collect();

        let outer_claim = SumcheckClaim {
            num_vars: num_row_vars,
            degree: 3,
            claimed_sum: F::zero(),
        };

        let (outer_final_eval, r_x) = SumcheckVerifier::verify(
            &outer_claim,
            &proof.outer_sumcheck_proof,
            transcript,
        )?;

        let eq_eval = EqPolynomial::new(tau).evaluate(&r_x);
        let expected = eq_eval * (proof.az_eval * proof.bz_eval - proof.cz_eval);
        if expected != outer_final_eval {
            return Err(SpartanError::OuterEvaluationMismatch);
        }

        transcript.append(&proof.az_eval);
        transcript.append(&proof.bz_eval);
        transcript.append(&proof.cz_eval);

        let rho_a = F::Challenge::from(transcript.challenge()).into();
        let rho_b = F::Challenge::from(transcript.challenge()).into();
        let rho_c = F::Challenge::from(transcript.challenge()).into();

        let inner_claim = SumcheckClaim {
            num_vars: num_col_vars,
            degree: 2,
            claimed_sum: rho_a * proof.az_eval + rho_b * proof.bz_eval + rho_c * proof.cz_eval,
        };

        let (inner_final_eval, r_y) = SumcheckVerifier::verify(
            &inner_claim,
            &proof.inner_sumcheck_proof,
            transcript,
        )?;

        let (a_eval, b_eval, c_eval) = key.evaluate_matrix_mles(&r_x, &r_y);
        let combined_matrix_eval = rho_a * a_eval + rho_b * b_eval + rho_c * c_eval;

        if combined_matrix_eval * proof.witness_eval != inner_final_eval {
            return Err(SpartanError::InnerEvaluationMismatch);
        }

        Ok((r_x, r_y))
    }
}
