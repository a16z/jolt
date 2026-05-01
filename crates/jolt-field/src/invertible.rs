use crate::RingCore;

/// Ring-level inversion capability with explicit zero handling.
pub trait Invertible: RingCore {
    /// Multiplicative inverse, or `None` for the zero element.
    fn inverse(&self) -> Option<Self>;

    /// Multiplicative inverse with zero mapped to zero.
    #[inline]
    fn inv_or_zero(self) -> Self {
        self.inverse().unwrap_or_else(Self::zero)
    }
}
