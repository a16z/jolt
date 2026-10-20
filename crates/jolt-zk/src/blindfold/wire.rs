//! BlindFold's prover messages and their order in the transcript. The prover,
//! the verifier, and the tests all send and read them through these types;
//! only the sumcheck rounds between them belong to `jolt-sumcheck`.
//!
//! Every count below is fixed by [`BlindFoldDimensions`], so no length travels
//! with a message. The committed relaxed instance needs no message of its own
//! beyond its auxiliary rows: its pad rows are the pad commitment sent before
//! the first challenge, its evaluation commitment comes from the PCS opening,
//! and its remaining rows are fixed padding.

use jolt_algebra::{CanonicalDecode, JoltField};
use jolt_crypto::VectorCommitmentOpening;
use jolt_transcript::{ProverTranscript, Sponge, TranscriptError, VerifierTranscript};

use super::BlindFoldDimensions;

/// The commitments the prover sends before the folding challenge.
pub(crate) struct FoldingCommitments<F, Com> {
    pub(crate) auxiliary_rows: Vec<Com>,
    pub(crate) random_u: F,
    pub(crate) random_pad_rows: Vec<Com>,
    pub(crate) random_auxiliary_rows: Vec<Com>,
    pub(crate) random_error_rows: Vec<Com>,
    pub(crate) random_eval: Com,
    pub(crate) cross_term_error_rows: Vec<Com>,
}

impl<F: JoltField, Com: CanonicalDecode> FoldingCommitments<F, Com> {
    pub(crate) fn send<H: Sponge>(&self, transcript: &mut ProverTranscript<H>) {
        transcript.send_all(&self.auxiliary_rows);
        transcript.send(&self.random_u);
        transcript.send_all(&self.random_pad_rows);
        transcript.send_all(&self.random_auxiliary_rows);
        transcript.send_all(&self.random_error_rows);
        transcript.send(&self.random_eval);
        transcript.send_all(&self.cross_term_error_rows);
    }

    pub(crate) fn receive<H: Sponge>(
        dimensions: &BlindFoldDimensions,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<Self, TranscriptError> {
        let auxiliary_rows = dimensions.witness_rows.auxiliary.len();
        Ok(Self {
            auxiliary_rows: transcript.receive_n(auxiliary_rows)?,
            random_u: transcript.receive()?,
            random_pad_rows: transcript.receive_n(dimensions.witness_rows.pads.len())?,
            random_auxiliary_rows: transcript.receive_n(auxiliary_rows)?,
            random_error_rows: transcript.receive_n(dimensions.error.row_count)?,
            random_eval: transcript.receive()?,
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

/// The folded residual evaluation and its blind, sent after the folding
/// challenge.
pub(crate) struct FoldedEvaluation<F> {
    pub(crate) output: F,
    pub(crate) blinding: F,
}

impl<F: JoltField> FoldedEvaluation<F> {
    pub(crate) fn send<H: Sponge>(&self, transcript: &mut ProverTranscript<H>) {
        transcript.send(&self.output);
        transcript.send(&self.blinding);
    }

    pub(crate) fn receive<H: Sponge>(
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<Self, TranscriptError> {
        Ok(Self {
            output: transcript.receive()?,
            blinding: transcript.receive()?,
        })
    }
}

/// The outer folded-R1CS sumcheck's terminal claims: `Az`, `Bz`, and `Cz` at
/// the outer point, then the opening of the folded error rows there.
pub(crate) struct OuterClaims<F> {
    pub(crate) abc: [F; 3],
    pub(crate) error_opening: VectorCommitmentOpening<F>,
}

impl<F: JoltField> OuterClaims<F> {
    pub(crate) fn send<H: Sponge>(&self, transcript: &mut ProverTranscript<H>) {
        transcript.send_all(&self.abc);
        send_opening(&self.error_opening, transcript);
    }

    pub(crate) fn receive<H: Sponge>(
        error_row_len: usize,
        transcript: &mut VerifierTranscript<'_, H>,
    ) -> Result<Self, TranscriptError> {
        Ok(Self {
            abc: [
                transcript.receive()?,
                transcript.receive()?,
                transcript.receive()?,
            ],
            error_opening: receive_opening(error_row_len, transcript)?,
        })
    }
}
