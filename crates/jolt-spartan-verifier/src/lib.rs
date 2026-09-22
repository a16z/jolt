//! Clear Spartan for a fixed sparse R1CS, with direct matrix verification.
//!
//! Keys are application-authenticated. This baseline evaluates sparse matrices
//! directly: it does not claim succinct verification in the matrix size or ZK.

#![forbid(unsafe_code)]
#![deny(clippy::indexing_slicing, clippy::panic_in_result_fn)]

mod key;

use jolt_field::JoltField;
use jolt_openings::{CommitmentScheme, OpeningsError};
use jolt_poly::EqPolynomial;
use jolt_r1cs::ConstraintMatrixEvalError;
use jolt_sumcheck::{
    BooleanHypercube, CompressedSumcheckProof, SumcheckClaim, SumcheckError,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{AppendToTranscript, Transcript};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use key::SpartanKey;

pub const OUTER_DEGREE: usize = 3;
pub const INNER_DEGREE: usize = 2;

/// Wire proof: only clear sumcheck rounds and one ordinary PCS opening.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, C: Serialize, O: Serialize",
    deserialize = "C: Deserialize<'de>, O: Deserialize<'de>"
))]
pub struct SpartanProof<F: JoltField, C, O> {
    pub witness_commitment: C,
    pub outer: CompressedSumcheckProof<F>,
    pub outer_evaluations: [F; 3],
    pub inner: CompressedSumcheckProof<F>,
    pub witness_evaluation: F,
    pub opening: O,
}

#[derive(Debug, Error)]
pub enum SpartanError<F: JoltField> {
    #[error("Spartan round interpolation requires field characteristic greater than three")]
    UnsupportedField,
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
}

impl<F: JoltField + AppendToTranscript> SpartanKey<F> {
    /// Verify against this authenticated relation and the application's PCS key.
    ///
    /// The application's key policy must fix PCS setup and transcript choices.
    /// Container decoding limits and authentication remain application-owned.
    pub fn verify<PCS: CommitmentScheme<Field = F>>(
        &self,
        public_inputs: &[F],
        proof: &SpartanProof<F, PCS::Output, PCS::Proof>,
        pcs_setup: &PCS::VerifierSetup,
        transcript: &mut impl Transcript<Challenge = F>,
    ) -> Result<(), SpartanError<F>>
    where
        PCS::Output: AppendToTranscript,
    {
        let tau = self.begin(public_inputs, &proof.witness_commitment, transcript)?;
        let outer = proof.outer.verify(
            &SumcheckClaim {
                num_vars: self.row_vars(),
                degree: OUTER_DEGREE,
                claimed_sum: F::zero(),
            },
            BooleanHypercube,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            transcript,
        )?;
        self.check_outer(
            &tau,
            outer.point.as_slice(),
            outer.value,
            proof.outer_evaluations,
        )?;
        let row_weights = EqPolynomial::new(outer.point.as_slice().to_vec()).evaluations();
        let (weights, claim) = self.begin_inner(
            &row_weights,
            public_inputs,
            proof.outer_evaluations,
            transcript,
        )?;
        let inner = proof.inner.verify(
            &SumcheckClaim {
                num_vars: self.witness_vars(),
                degree: INNER_DEGREE,
                claimed_sum: claim,
            },
            BooleanHypercube,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
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
        if inner.value != linear * proof.witness_evaluation {
            return Err(SpartanError::InnerClaim);
        }
        Self::append_witness_evaluation(proof.witness_evaluation, transcript);
        PCS::verify(
            &proof.witness_commitment,
            inner.point.as_slice(),
            proof.witness_evaluation,
            &proof.opening,
            pcs_setup,
            transcript,
        )?;
        Ok(())
    }
}
