//! Spartan verifier: checks a proof of R1CS satisfiability (pure PIOP).
//!
//! The verifier:
//! 1. Replays the Fiat-Shamir transcript to derive the same challenges as the prover.
//! 2. Verifies the outer sumcheck proof.
//! 3. Checks the final outer sumcheck evaluation against the MLE evaluations.
//! 4. Verifies the inner sumcheck proof.
//! 5. Checks the inner sumcheck evaluation against the matrix MLEs and witness evaluation.
//!
//! The caller appends the witness commitment to the transcript before calling
//! verify, and checks the witness opening proof after verify returns.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{SumcheckClaim, SumcheckVerifier};
use jolt_transcript::Transcript;
use num_traits::Zero;

use crate::error::SpartanError;
use crate::key::SpartanKey;
use crate::proof::{RelaxedSpartanProof, SpartanProof};

/// Stateless Spartan verifier.
///
/// Checks a [`SpartanProof`] against a [`SpartanKey`] by replaying the
/// Fiat-Shamir transcript and verifying each sub-protocol. Does NOT
/// check the witness opening proof — the caller handles PCS verification.
pub struct SpartanVerifier;

impl SpartanVerifier {
    /// Verifies a Spartan proof (PIOP only — no PCS checks).
    ///
    /// The caller must:
    /// 1. Append the witness commitment to the transcript before calling this.
    /// 2. Verify the witness opening proof at `r_y` after this returns.
    ///
    /// # Protocol
    ///
    /// 1. Sample $\tau$ (commitment already absorbed by caller).
    /// 2. Verify the outer sumcheck.
    /// 3. At the outer sumcheck challenge point $r_x$, check:
    ///    $$\widetilde{eq}(r_x, \tau) \cdot (\widetilde{Az}(r_x) \cdot \widetilde{Bz}(r_x) - \widetilde{Cz}(r_x)) = v$$
    /// 4. Absorb evaluation claims, sample $\rho_A, \rho_B, \rho_C$.
    /// 5. Verify inner sumcheck with claim $\rho_A \cdot az + \rho_B \cdot bz + \rho_C \cdot cz$.
    /// 6. At inner challenge $r_y$, check the final evaluation against
    ///    matrix MLEs and witness evaluation.
    #[tracing::instrument(skip_all, name = "SpartanVerifier::verify")]
    pub fn verify<F, T>(
        key: &SpartanKey<F>,
        proof: &SpartanProof<F>,
        transcript: &mut T,
    ) -> Result<(), SpartanError>
    where
        F: Field,
        T: Transcript<Challenge = u128>,
    {
        Self::verify_with_challenges(key, proof, transcript).map(|_| ())
    }

    /// Verifies a Spartan proof and returns the challenge vectors.
    ///
    /// Same as [`verify`](Self::verify) but returns `(r_x, r_y)`.
    /// The caller needs `r_y` to verify the witness opening proof via PCS.
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "SpartanVerifier::verify_with_challenges")]
    pub fn verify_with_challenges<F, T>(
        key: &SpartanKey<F>,
        proof: &SpartanProof<F>,
        transcript: &mut T,
    ) -> Result<(Vec<F>, Vec<F>), SpartanError>
    where
        F: Field,
        T: Transcript<Challenge = u128>,
    {
        let num_sc_vars = key.num_sumcheck_vars();
        let tau: Vec<F> = (0..num_sc_vars)
            .map(|_| F::from_u128(transcript.challenge()))
            .collect();

        let outer_claim = SumcheckClaim {
            num_vars: num_sc_vars,
            degree: 3,
            claimed_sum: F::zero(),
        };

        let (outer_final_eval, r_x) = SumcheckVerifier::verify(
            &outer_claim,
            &proof.outer_sumcheck_proof,
            transcript,
            |c: u128| F::from_u128(c),
        )?;

        let eq_eval = EqPolynomial::new(tau).evaluate(&r_x);
        let expected = eq_eval * (proof.az_eval * proof.bz_eval - proof.cz_eval);
        if expected != outer_final_eval {
            return Err(SpartanError::OuterEvaluationMismatch);
        }

        transcript.append(&proof.az_eval);
        transcript.append(&proof.bz_eval);
        transcript.append(&proof.cz_eval);

        let rho_a = F::from_u128(transcript.challenge());
        let rho_b = F::from_u128(transcript.challenge());
        let rho_c = F::from_u128(transcript.challenge());

        let num_witness_vars = key.num_witness_vars();
        let inner_claim = SumcheckClaim {
            num_vars: num_witness_vars,
            degree: 2,
            claimed_sum: rho_a * proof.az_eval + rho_b * proof.bz_eval + rho_c * proof.cz_eval,
        };

        let (inner_final_eval, r_y) = SumcheckVerifier::verify(
            &inner_claim,
            &proof.inner_sumcheck_proof,
            transcript,
            |c: u128| F::from_u128(c),
        )?;

        let (a_eval, b_eval, c_eval) = key.evaluate_matrix_mles(&r_x, &r_y);
        let combined_matrix_eval = rho_a * a_eval + rho_b * b_eval + rho_c * c_eval;

        if combined_matrix_eval * proof.witness_eval != inner_final_eval {
            return Err(SpartanError::InnerEvaluationMismatch);
        }

        Ok((r_x, r_y))
    }

    /// Verifies a relaxed Spartan proof for $Az \circ Bz = u \cdot Cz + E$.
    ///
    /// The verifier receives the relaxed scalar $u$, witness/error commitments,
    /// and the proof. The outer sumcheck check becomes:
    /// $$\widetilde{eq}(r_x, \tau) \cdot (\widetilde{Az}(r_x) \cdot \widetilde{Bz}(r_x) - u \cdot \widetilde{Cz}(r_x) - \widetilde{E}(r_x)) = v$$
    #[tracing::instrument(skip_all, name = "SpartanVerifier::verify_relaxed")]
    pub fn verify_relaxed<PCS, T>(
        key: &SpartanKey<PCS::Field>,
        u: PCS::Field,
        w_commitment: &PCS::Output,
        e_commitment: &PCS::Output,
        proof: &RelaxedSpartanProof<PCS::Field, PCS>,
        verifier_setup: &PCS::VerifierSetup,
        transcript: &mut T,
    ) -> Result<(), SpartanError>
    where
        PCS: CommitmentScheme,
        T: Transcript<Challenge = u128>,
    {
        transcript.append_bytes(format!("{w_commitment:?}").as_bytes());
        transcript.append_bytes(format!("{e_commitment:?}").as_bytes());

        let num_sc_vars = key.num_sumcheck_vars();
        let tau: Vec<PCS::Field> = (0..num_sc_vars)
            .map(|_| PCS::Field::from_u128(transcript.challenge()))
            .collect();

        let outer_claim = SumcheckClaim {
            num_vars: num_sc_vars,
            degree: 3,
            claimed_sum: PCS::Field::zero(),
        };

        let (outer_final_eval, r_x) = SumcheckVerifier::verify(
            &outer_claim,
            &proof.outer_sumcheck_proof,
            transcript,
            |c: u128| PCS::Field::from_u128(c),
        )?;

        // Check: eq(r_x, tau) * (Az(r_x)*Bz(r_x) - u*Cz(r_x) - E(r_x)) == outer_final_eval
        let eq_eval = EqPolynomial::new(tau).evaluate(&r_x);
        let expected = eq_eval * (proof.az_eval * proof.bz_eval - u * proof.cz_eval - proof.e_eval);
        if expected != outer_final_eval {
            return Err(SpartanError::OuterEvaluationMismatch);
        }

        transcript.append(&proof.az_eval);
        transcript.append(&proof.bz_eval);
        transcript.append(&proof.cz_eval);
        transcript.append(&proof.e_eval);

        let rho_a = PCS::Field::from_u128(transcript.challenge());
        let rho_b = PCS::Field::from_u128(transcript.challenge());
        let rho_c = PCS::Field::from_u128(transcript.challenge());

        let num_witness_vars = key.num_witness_vars();
        let inner_claim = SumcheckClaim {
            num_vars: num_witness_vars,
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
            w_commitment,
            &r_y,
            proof.witness_eval,
            &proof.witness_opening_proof,
            verifier_setup,
            transcript,
        )?;

        PCS::verify(
            e_commitment,
            &r_x,
            proof.e_eval,
            &proof.error_opening_proof,
            verifier_setup,
            transcript,
        )?;

        Ok(())
    }
}
