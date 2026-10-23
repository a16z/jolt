use serde::{Deserialize, Serialize};

use crate::{Point, HIGH_TO_LOW};

#[cfg(feature = "transcript")]
use jolt_transcript::{Encoding, Label, LabelWithCount};

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

/// A [`LabelWithCount`] word carrying the point's length, the coordinates,
/// a [`Label`] word and the value: self-delimiting when `F`'s encoding is
/// fixed-width.
#[cfg(feature = "transcript")]
impl<F: Encoding> Encoding for EvaluationClaim<F> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut out = Vec::new();
        out.extend_from_slice(
            LabelWithCount(b"opening_point", self.point.len() as u64)
                .encode()
                .as_ref(),
        );
        for coordinate in self.point.as_slice() {
            out.extend_from_slice(coordinate.encode().as_ref());
        }
        out.extend_from_slice(Label(b"opening_eval").encode().as_ref());
        out.extend_from_slice(self.value.encode().as_ref());
        out
    }
}
