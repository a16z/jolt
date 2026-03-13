//! Transcript binding helpers for opening claims.
//!
//! These functions standardize how opening claim data is absorbed into
//! Fiat-Shamir transcripts. Standard mode appends the cleartext evaluation;
//! ZK mode appends a hiding commitment to the evaluation instead.

use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, Transcript};

/// Appends an opening claim (point + cleartext eval) to the transcript.
///
/// Used in standard (non-ZK) mode where the evaluation value is public.
pub fn bind_opening_claim<F: Field>(transcript: &mut impl Transcript, point: &[F], eval: &F) {
    for coord in point {
        coord.append_to_transcript(transcript);
    }
    eval.append_to_transcript(transcript);
}

/// Appends an opening claim (point + eval commitment) to the transcript.
///
/// Used in ZK mode where the evaluation is hidden behind a commitment.
/// Generic over any `AppendToTranscript` type — works with group elements,
/// lattice commitments, hash digests, etc.
pub fn bind_opening_claim_zk<F: Field, C: AppendToTranscript>(
    transcript: &mut impl Transcript,
    point: &[F],
    eval_commitment: &C,
) {
    for coord in point {
        coord.append_to_transcript(transcript);
    }
    eval_commitment.append_to_transcript(transcript);
}
