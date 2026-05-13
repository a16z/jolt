use crate::{AdditiveGroup, FromPrimitiveInt, RingAccumulator, RingCore};

/// Associates an additive redundant accumulator with an element type.
pub trait WithAccumulator: AdditiveGroup + RingCore + FromPrimitiveInt {
    /// Accumulator type.
    type Accumulator: RingAccumulator<Element = Self>;
}
