use num_traits::Zero;
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

/// Minimal additive group operations shared by fields, rings, and accumulators.
pub trait AdditiveGroup:
    Sized
    + Clone
    + Copy
    + Send
    + Sync
    + Zero
    + Add<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + AddAssign<Self>
    + Sub<Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + SubAssign<Self>
    + Neg<Output = Self>
{
}
