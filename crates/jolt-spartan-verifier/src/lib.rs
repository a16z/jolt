//! Clear Spartan for a fixed sparse R1CS, with direct matrix verification.
//!
//! Keys are application-authenticated. This baseline evaluates sparse matrices
//! directly: it does not claim succinct verification in the matrix size or ZK.

// In the jolt-verifier runtime closure: stricter panic and unsafe discipline
// than the workspace lints (specs/verifier-closure-lints.md).
#![forbid(unsafe_code)]
#![deny(
    clippy::indexing_slicing,
    clippy::get_unwrap,
    clippy::string_slice,
    clippy::fallible_impl_from,
    clippy::mem_forget,
    clippy::exit,
    clippy::panic_in_result_fn,
    clippy::let_underscore_must_use,
    clippy::host_endian_bytes,
    clippy::wildcard_enum_match_arm
)]

mod key;

use jolt_field::{CanonicalDecode, JoltField};
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_poly::EqPolynomial;
use jolt_r1cs::ConstraintMatrixEvalError;
use jolt_sumcheck::{SumcheckClaim, SumcheckError, SumcheckVerifier};
use jolt_transcript::{Sponge, TranscriptError, VerifierTranscript};
use thiserror::Error;

pub use key::SpartanKey;

pub const OUTER_DEGREE: usize = 3;
pub const INNER_DEGREE: usize = 2;

/// Outer sumcheck summand `eq(tau, x) * (Az(x) * Bz(x) - Cz(x))`.
#[inline]
pub fn outer_relation<F: JoltField>(eq: F, a: F, b: F, c: F) -> F {
    eq * (a * b - c)
}

/// Inner sumcheck summand `L(y) * w(y)` for the weighted matrix row `L`.
#[inline]
pub fn inner_relation<F: JoltField>(linear: F, witness: F) -> F {
    linear * witness
}

#[derive(Debug, Error)]
pub enum SpartanError<F: JoltField> {
    #[error("invalid R1CS shape or public/private partition")]
    InvalidShape,
    #[error("public input count does not match the key")]
    PublicInputs,
    #[error("private witness length does not match the key")]
    WitnessLength,
    #[error("R1CS witness is unsatisfied at row {0}")]
    Unsatisfied(usize),
    #[error("Spartan outer final claim failed")]
    OuterClaim,
    #[error("Spartan inner final claim failed")]
    InnerClaim,
    #[error("internal sumcheck proof shape differs from its checked dimensions")]
    InternalShape,
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError<F>),
    #[error(transparent)]
    Matrix(#[from] ConstraintMatrixEvalError),
    #[error(transparent)]
    Opening(#[from] OpeningsError),
    #[error(transparent)]
    Transcript(#[from] TranscriptError),
}

impl<F: JoltField + CanonicalDecode> SpartanKey<F> {
    /// Verify against this authenticated relation and the application's PCS key,
    /// reading the proof from `transcript`.
    ///
    /// The application's key policy must fix PCS setup and transcript choices,
    /// and the caller finishes the transcript.
    pub fn verify<PCS: CommitmentScheme<Field = F>, H: Sponge>(
        &self,
        public_inputs: &[F],
        pcs_setup: &PCS::VerifierSetup,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<(), SpartanError<F>> {
        self.bind_statement(public_inputs, transcript)?;
        let witness_commitment = PCS::receive_commitment(pcs_setup, transcript)?;
        let tau = self.draw_tau(transcript);
        let outer = SumcheckVerifier::verify_compressed(
            &SumcheckClaim {
                num_vars: self.row_vars(),
                degree: OUTER_DEGREE,
                claimed_sum: F::zero(),
            },
            transcript,
        )?;
        let mut outer_evaluations = [F::zero(); 3];
        let row_weights = EqPolynomial::new(outer.point.as_slice().to_vec()).evaluations();
        let (weights, claim) = self.begin_inner(
            &row_weights,
            public_inputs,
            &mut outer_evaluations,
            transcript,
        )?;
        self.check_outer(&tau, outer.point.as_slice(), outer.value, outer_evaluations)?;
        let inner = SumcheckVerifier::verify_compressed(
            &SumcheckClaim {
                num_vars: self.witness_vars(),
                degree: INNER_DEGREE,
                claimed_sum: claim,
            },
            transcript,
        )?;
        let column_weights = EqPolynomial::new(inner.point.as_slice().to_vec()).evaluations();
        let column_weights = column_weights
            .get(..self.witness_len())
            .ok_or(SpartanError::InternalShape)?;
        let linear = self.matrices().linear_form_bilinear_eval(
            &row_weights,
            column_weights,
            self.public_columns(),
            self.witness_len(),
            weights,
        )?;
        let mut witness_evaluation = F::zero();
        Self::exchange_witness_evaluation(&mut witness_evaluation, transcript)?;
        if inner.value != inner_relation(linear, witness_evaluation) {
            return Err(SpartanError::InnerClaim);
        }
        PCS::verify(
            &witness_commitment,
            inner.point.as_slice(),
            witness_evaluation,
            pcs_setup,
            transcript,
        )?;
        Ok(())
    }
}
