//! BlindFold's proof and the order its messages enter the transcript. The
//! prover, the verifier and the tests absorb them through these helpers;
//! only the sumcheck rounds between them belong to `jolt-sumcheck`.
//!
//! Every count below is fixed by [`BlindFoldDimensions`], so the verifier
//! checks each length before absorbing anything. The committed relaxed
//! instance needs no message of its own beyond its auxiliary rows: its pad
//! rows are the pad commitment absorbed before the first challenge, its
//! evaluation commitment comes from the PCS opening, and its remaining rows
//! are fixed padding.

use jolt_crypto::VectorCommitmentOpening;
use jolt_field::Field;
use jolt_sumcheck::CompressedSumcheckProof;
use jolt_transcript::{Encoding, Transcript};
use serde::{Deserialize, Serialize};

use super::{BlindFoldDimensions, RelaxedError};

/// Degree bound of the outer (folded R1CS) sumcheck.
pub(crate) const OUTER_SUMCHECK_DEGREE: usize = 3;
/// Degree bound of the inner (witness) sumcheck.
pub(crate) const INNER_SUMCHECK_DEGREE: usize = 2;

/// A BlindFold proof over committed pads, its messages in transcript order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, Com: Serialize",
    deserialize = "F: Deserialize<'de>, Com: Deserialize<'de>"
))]
pub struct BlindFoldProof<F: Field, Com> {
    /// Absorbed before the folding challenge.
    pub folding: FoldingCommitments<F, Com>,
    /// Absorbed after the folding challenge, before the outer challenges.
    pub folded_evaluation: FoldedEvaluation<F>,
    pub outer_sumcheck: CompressedSumcheckProof<F>,
    /// Absorbed after the outer sumcheck, before the inner weights.
    pub outer_claims: OuterClaims<F>,
    pub inner_sumcheck: CompressedSumcheckProof<F>,
    /// The folded witness opened at the inner sumcheck's point; absorbed
    /// last.
    pub witness_opening: VectorCommitmentOpening<F>,
}

/// The committed instance's auxiliary rows and the random instance and
/// cross term the prover folds it with.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, Com: Serialize",
    deserialize = "F: Deserialize<'de>, Com: Deserialize<'de>"
))]
pub struct FoldingCommitments<F, Com> {
    pub auxiliary_rows: Vec<Com>,
    pub random_u: F,
    pub random_pad_rows: Vec<Com>,
    pub random_auxiliary_rows: Vec<Com>,
    pub random_error_rows: Vec<Com>,
    pub random_eval: Com,
    pub cross_term_error_rows: Vec<Com>,
}

impl<F: Encoding, Com: Encoding> FoldingCommitments<F, Com> {
    /// Checks every count against `dimensions`.
    pub(crate) fn check_shape(&self, dimensions: &BlindFoldDimensions) -> Result<(), RelaxedError> {
        let auxiliary_rows = dimensions.witness_rows.auxiliary.len();
        check_len(
            "auxiliary row commitments",
            auxiliary_rows,
            self.auxiliary_rows.len(),
        )?;
        check_len(
            "random pad row commitments",
            dimensions.witness_rows.pads.len(),
            self.random_pad_rows.len(),
        )?;
        check_len(
            "random auxiliary row commitments",
            auxiliary_rows,
            self.random_auxiliary_rows.len(),
        )?;
        check_len(
            "random error row commitments",
            dimensions.error.row_count,
            self.random_error_rows.len(),
        )?;
        check_len(
            "cross-term error row commitments",
            dimensions.error.row_count,
            self.cross_term_error_rows.len(),
        )
    }

    pub(crate) fn append<T: Transcript + ?Sized>(&self, transcript: &mut T) {
        transcript.append_values(&self.auxiliary_rows);
        transcript.append(&self.random_u);
        transcript.append_values(&self.random_pad_rows);
        transcript.append_values(&self.random_auxiliary_rows);
        transcript.append_values(&self.random_error_rows);
        transcript.append(&self.random_eval);
        transcript.append_values(&self.cross_term_error_rows);
    }
}

/// The folded final evaluation, its blind, and the folded witness opened at
/// their two dedicated coordinates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct FoldedEvaluation<F> {
    pub output: F,
    pub blinding: F,
    pub output_opening: VectorCommitmentOpening<F>,
    pub blinding_opening: VectorCommitmentOpening<F>,
}

impl<F: Encoding> FoldedEvaluation<F> {
    pub(crate) fn check_shape(&self, witness_row_len: usize) -> Result<(), RelaxedError> {
        check_opening_len(
            "folded eval output opening",
            witness_row_len,
            &self.output_opening,
        )?;
        check_opening_len(
            "folded eval blinding opening",
            witness_row_len,
            &self.blinding_opening,
        )
    }

    pub(crate) fn append<T: Transcript + ?Sized>(&self, transcript: &mut T) {
        transcript.append(&self.output);
        transcript.append(&self.blinding);
        append_opening(transcript, &self.output_opening);
        append_opening(transcript, &self.blinding_opening);
    }
}

/// `Az`, `Bz`, `Cz` at the outer point and the folded error rows opened there.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub struct OuterClaims<F> {
    pub abc: [F; 3],
    pub error_opening: VectorCommitmentOpening<F>,
}

impl<F: Encoding> OuterClaims<F> {
    pub(crate) fn append<T: Transcript + ?Sized>(&self, transcript: &mut T) {
        transcript.append_values(&self.abc);
        append_opening(transcript, &self.error_opening);
    }
}

/// Absorbs an opening: its length (the prover chose the vector), the vector,
/// then the blinding.
pub(crate) fn append_opening<F, T>(transcript: &mut T, opening: &VectorCommitmentOpening<F>)
where
    F: Encoding,
    T: Transcript + ?Sized,
{
    transcript.append(&(opening.combined_vector.len() as u64));
    transcript.append_values(&opening.combined_vector);
    transcript.append(&opening.combined_blinding);
}

/// Checks an opening's vector against the committed row length.
pub(crate) fn check_opening_len<F>(
    name: &'static str,
    row_len: usize,
    opening: &VectorCommitmentOpening<F>,
) -> Result<(), RelaxedError> {
    check_len(name, row_len, opening.combined_vector.len())
}

fn check_len(name: &'static str, expected: usize, actual: usize) -> Result<(), RelaxedError> {
    if expected == actual {
        Ok(())
    } else {
        Err(RelaxedError::LengthMismatch {
            name,
            expected,
            actual,
        })
    }
}
