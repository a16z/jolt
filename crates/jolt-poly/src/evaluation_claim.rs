use serde::{Deserialize, Serialize};

use crate::{Point, HIGH_TO_LOW};

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
