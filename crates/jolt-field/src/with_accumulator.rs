use crate::{AdditiveGroup, FromPrimitiveInt, RingAccumulator, RingCore};

/// Associates an additive redundant accumulator with an element type.
///
/// The accumulator is a full [`RingAccumulator`], so generic kernels bounded
/// on `Field` can use deferred-reduction fused multiply-adds
/// (`fmadd(a, b)`), not just plain additive accumulation. Every implementor
/// is a ring with primitive-integer embeddings, which the ring-accumulator
/// contract requires.
pub trait WithAccumulator: AdditiveGroup + RingCore + FromPrimitiveInt {
    /// Accumulator type.
    type Accumulator: RingAccumulator<Element = Self>;
}
