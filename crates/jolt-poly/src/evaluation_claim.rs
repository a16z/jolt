use serde::{Deserialize, Serialize};

use crate::{Point, HIGH_TO_LOW};

#[cfg(feature = "transcript")]
use jolt_transcript::Encoding;

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

/// The coordinates, then the value. Positional: the caller binds the point's
/// length before the claim is absorbed.
#[cfg(feature = "transcript")]
impl<F: Encoding> Encoding for EvaluationClaim<F> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut out = Vec::new();
        for coordinate in self.point.as_slice() {
            out.extend_from_slice(coordinate.encode().as_ref());
        }
        out.extend_from_slice(self.value.encode().as_ref());
        out
    }
}
