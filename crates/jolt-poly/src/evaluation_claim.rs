use serde::{Deserialize, Serialize};

use crate::{Point, HIGH_TO_LOW};

#[cfg(feature = "transcript")]
use jolt_transcript::Encoding;

/// A claim that a polynomial takes `value` at `point`.
///
/// # Security
/// With the `transcript` feature, a claim is absorbed as its coordinates and
/// then its value, with no count. Claims whose points differ in length can
/// then absorb as the same bytes, so the encoding binds a claim only when the
/// point's length is already fixed. A caller that absorbs a claim built from
/// proof data must first check `point.len()` against the length its protocol
/// fixes from bound data, or absorb that length as a `u64` word itself.
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
