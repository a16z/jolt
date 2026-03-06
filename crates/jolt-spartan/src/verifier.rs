//! Spartan verifier: checks a proof of R1CS satisfiability.
//!
//! The verifier:
//! 1. Replays the Fiat-Shamir transcript to derive the same challenges as the prover.
//! 2. Verifies the outer sumcheck proof.
//! 3. Checks the final sumcheck evaluation against the MLE evaluations.
//! 4. Verifies the witness polynomial opening proof.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::EqPolynomial;
use jolt_sumcheck::{SumcheckClaim, SumcheckVerifier};
use jolt_transcript::Transcript;
use num_traits::Zero;

use crate::error::SpartanError;
use crate::key::SpartanKey;
use crate::proof::SpartanProof;

/// Stateless Spartan verifier.
///
/// Checks a [`SpartanProof`] against a [`SpartanKey`] by replaying the
/// Fiat-Shamir transcript and verifying each sub-protocol.
pub struct SpartanVerifier;

impl SpartanVerifier {
    /// Verifies a Spartan proof.
    ///
    /// # Protocol
    ///
    /// 1. Absorb the witness commitment into the transcript.
    /// 2. Sample $\tau$ and verify the outer sumcheck.
    /// 3. At the sumcheck challenge point $r_x$, check:
    ///    $$\widetilde{eq}(r_x, \tau) \cdot (\widetilde{Az}(r_x) \cdot \widetilde{Bz}(r_x) - \widetilde{Cz}(r_x)) = v$$
    ///    where $v$ is the final sumcheck evaluation.
    /// 4. Sample a witness evaluation point $r_y$ and verify the opening proof.
    ///
    /// # Errors
    ///
    /// Returns [`SpartanError`] if any verification step fails.
    pub fn verify<PCS, T>(
        key: &SpartanKey<PCS::Field>,
        proof: &SpartanProof<PCS::Field, PCS>,
        verifier_setup: &PCS::VerifierSetup,
        transcript: &mut T,
    ) -> Result<(), SpartanError>
    where
        PCS: CommitmentScheme,
        T: Transcript<Challenge = u128>,
    {
        // Absorb commitment (must match prover's transcript)
        transcript.append_bytes(format!("{:?}", proof.witness_commitment).as_bytes());

        // Sample tau
        let num_sc_vars = key.num_sumcheck_vars();
        let tau: Vec<PCS::Field> = (0..num_sc_vars)
            .map(|_| PCS::Field::from_u128(transcript.challenge()))
            .collect();

        // Verify the outer sumcheck
        let claim = SumcheckClaim {
            num_vars: num_sc_vars,
            degree: 3,
            claimed_sum: PCS::Field::zero(),
        };

        let (final_eval, challenges) =
            SumcheckVerifier::verify(&claim, &proof.sumcheck_proof, transcript, |c: u128| {
                PCS::Field::from_u128(c)
            })?;

        // Check the final evaluation:
        // eq(r_x, tau) * (Az(r_x) * Bz(r_x) - Cz(r_x)) == final_eval
        let eq_eval = EqPolynomial::new(tau).evaluate(&challenges);
        let expected = eq_eval * (proof.az_eval * proof.bz_eval - proof.cz_eval);
        if expected != final_eval {
            return Err(SpartanError::EvaluationMismatch);
        }

        // Sample a witness evaluation point and verify the opening proof
        let witness_point: Vec<PCS::Field> = (0..key.num_witness_vars())
            .map(|_| PCS::Field::from_u128(transcript.challenge()))
            .collect();

        PCS::verify(
            &proof.witness_commitment,
            &witness_point,
            proof.witness_eval,
            &proof.witness_opening_proof,
            verifier_setup,
            transcript,
        )?;

        Ok(())
    }
}
