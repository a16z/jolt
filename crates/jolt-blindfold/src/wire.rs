//! BlindFold's prover messages and their order in the transcript.
//!
//! Every count below is fixed by [`BlindFoldDimensions`] and the protocol's
//! final-opening layout, so no length travels with a message. The committed
//! relaxed instance needs no message of its own beyond its auxiliary rows: its
//! round, output-claim, and evaluation commitments are already in the
//! transcript from the stages that produced them, and its remaining rows are
//! fixed padding.

use jolt_crypto::VectorCommitmentOpening;
use jolt_field::{CanonicalDecode, JoltField};
use jolt_transcript::{ProverTranscript, Sponge, TranscriptError, VerifierTranscript};

use crate::BlindFoldDimensions;

/// The commitments the prover sends before the folding challenge.
pub(crate) struct FoldingCommitments<F, Com> {
    pub(crate) auxiliary_rows: Vec<Com>,
    pub(crate) random_u: F,
    pub(crate) random_rounds: Vec<Com>,
    pub(crate) random_output_claim_rows: Vec<Com>,
    pub(crate) random_auxiliary_rows: Vec<Com>,
    pub(crate) random_error_rows: Vec<Com>,
    pub(crate) random_evals: Vec<Com>,
    pub(crate) cross_term_error_rows: Vec<Com>,
}

impl<F: JoltField, Com: CanonicalDecode> FoldingCommitments<F, Com> {
    pub(crate) fn send<H: Sponge>(&self, transcript: &mut ProverTranscript<H>) {
        transcript.send_all(&self.auxiliary_rows);
        transcript.send(&self.random_u);
        transcript.send_all(&self.random_rounds);
        transcript.send_all(&self.random_output_claim_rows);
        transcript.send_all(&self.random_auxiliary_rows);
        transcript.send_all(&self.random_error_rows);
        transcript.send_all(&self.random_evals);
        transcript.send_all(&self.cross_term_error_rows);
    }

    pub(crate) fn receive<H: Sponge>(
        dimensions: &BlindFoldDimensions,
        eval_count: usize,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<Self, TranscriptError> {
        Ok(Self {
            auxiliary_rows: transcript.receive_n(dimensions.auxiliary_rows)?,
            random_u: transcript.receive()?,
            random_rounds: transcript.receive_n(dimensions.coefficient_rows)?,
            random_output_claim_rows: transcript.receive_n(dimensions.output_claim_rows)?,
            random_auxiliary_rows: transcript.receive_n(dimensions.auxiliary_rows)?,
            random_error_rows: transcript.receive_n(dimensions.error.row_count)?,
            random_evals: transcript.receive_n(eval_count)?,
            cross_term_error_rows: transcript.receive_n(dimensions.error.row_count)?,
        })
    }
}

/// Sends a committed-row opening: the combined row, then its blinding.
pub(crate) fn send_opening<F: JoltField, H: Sponge>(
    opening: &VectorCommitmentOpening<F>,
    transcript: &mut ProverTranscript<H>,
) {
    transcript.send_all(&opening.combined_vector);
    transcript.send(&opening.combined_blinding);
}

/// Receives a committed-row opening of a `row_len`-entry row.
pub(crate) fn receive_opening<F: JoltField, H: Sponge>(
    row_len: usize,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<VectorCommitmentOpening<F>, TranscriptError> {
    Ok(VectorCommitmentOpening {
        combined_vector: transcript.receive_n(row_len)?,
        combined_blinding: transcript.receive()?,
    })
}
