use crate::{AdditiveAccumulator, AdditiveGroup};

/// Associates an additive redundant accumulator with an element type.
pub trait WithAccumulator: AdditiveGroup {
    /// Accumulator type.
    type Accumulator: AdditiveAccumulator<Element = Self>;
}
