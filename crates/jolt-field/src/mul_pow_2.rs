use crate::{FromPrimitiveInt, RingCore};

/// Multiplication by powers of two.
pub trait MulPow2: RingCore + FromPrimitiveInt {
    /// Multiplies this ring element by the integer `2^pow`.
    #[inline]
    fn mul_pow_2(&self, pow: usize) -> Self {
        assert!(pow <= 255, "pow > 255");
        let mut res = *self;
        let mut p = pow;
        while p >= 64 {
            res *= Self::from_u64(1 << 63);
            p -= 63;
        }
        res * Self::from_u64(1 << p)
    }
}
