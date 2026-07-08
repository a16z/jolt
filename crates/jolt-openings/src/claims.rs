//! Stateless claim types for PCS operations.

use jolt_field::Field;
use jolt_poly::{Point, HIGH_TO_LOW};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationClaim<F> {
    pub point: Point<HIGH_TO_LOW, F>,
    pub value: F,
}

impl<F> EvaluationClaim<F> {
    pub fn new(point: impl Into<Point<HIGH_TO_LOW, F>>, value: F) -> Self {
        Self {
            point: point.into(),
            value,
        }
    }
}

impl<F> AppendToTranscript for EvaluationClaim<F>
where
    F: Field,
{
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&LabelWithCount(b"opening_point", self.point.len() as u64));
        for coordinate in self.point.as_slice() {
            coordinate.append_to_transcript(transcript);
        }
        transcript.append(&Label(b"opening_eval"));
        self.value.append_to_transcript(transcript);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZkEvaluationClaim<'a, F, C> {
    pub point: &'a [F],
    pub hiding_commitment: &'a C,
}

impl<'a, F, C> ZkEvaluationClaim<'a, F, C> {
    pub fn new(point: &'a [F], hiding_commitment: &'a C) -> Self {
        Self {
            point,
            hiding_commitment,
        }
    }
}

impl<F, C> AppendToTranscript for ZkEvaluationClaim<'_, F, C>
where
    F: Field,
    C: AppendToTranscript,
{
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&LabelWithCount(
            b"zk_opening_point",
            self.point.len() as u64,
        ));
        for coordinate in self.point {
            coordinate.append_to_transcript(transcript);
        }
        transcript.append(&Label(b"zk_eval_commitment"));
        self.hiding_commitment.append_to_transcript(transcript);
    }
}

/// Verifier-side opening claim: commitment, point, and claimed value.
#[derive(Clone, Debug)]
pub struct VerifierOpeningClaim<F: Field, C> {
    pub commitment: C,
    pub evaluation: EvaluationClaim<F>,
}

pub(crate) struct VerifierRlcClaims<'a, F: Field, C>(pub &'a [VerifierOpeningClaim<F, C>]);

impl<F, C> AppendToTranscript for VerifierRlcClaims<'_, F, C>
where
    F: Field,
{
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&LabelWithCount(b"rlc_claims", self.0.len() as u64));
        for claim in self.0 {
            claim.evaluation.value.append_to_transcript(transcript);
        }
    }
}
