use serde::{Deserialize, Serialize};

use crate::{Point, HIGH_TO_LOW};

#[cfg(feature = "transcript")]
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};

/// An evaluation of a multilinear polynomial at a point.
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

#[cfg(feature = "transcript")]
impl<F> AppendToTranscript for EvaluationClaim<F>
where
    F: AppendToTranscript,
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
