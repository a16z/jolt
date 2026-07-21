use crate::{FieldCore, FromPrimitiveInt};
use num_traits::Zero;

/// Canonical integer representation for a prime-field element.
pub trait CanonicalField: FieldCore + FromPrimitiveInt {
    /// Returns the unique representative in `[0, p)`.
    fn to_canonical_u128(self) -> u128;

    /// Returns the bit width of the field modulus.
    fn modulus_bits() -> u32;

    /// Constructs an element when `val` is a canonical representative.
    fn from_canonical_u128_checked(val: u128) -> Option<Self>;

    /// Constructs an element by reducing `val` modulo the field modulus.
    fn from_canonical_u128_reduced(val: u128) -> Self;
}

/// Field types with a cheap division-by-two operation.
pub trait HalvingField: FieldCore {
    /// Divides this element by two.
    fn half(self) -> Self;

    /// Returns the multiplicative inverse of two.
    #[inline]
    fn two_inv() -> Self {
        Self::one().half()
    }
}

/// Balanced signed-digit lookup support for small power-of-two bases.
pub trait BalancedDigitLookup: FromPrimitiveInt + Zero + Copy {
    /// Builds the balanced digit table for `1 <= log_basis <= 6`.
    fn digit_lut(log_basis: u32) -> [Self; 64] {
        debug_assert!(log_basis > 0 && log_basis <= 6);
        let basis = 1usize << log_basis;
        let half_basis = (basis >> 1) as i64;
        std::array::from_fn(|i| {
            if i < basis {
                Self::from_i64(i as i64 - half_basis)
            } else {
                Self::zero()
            }
        })
    }
}

/// Metadata for a pseudo-Mersenne modulus `2^k - c`.
pub trait PseudoMersenneField: CanonicalField {
    /// Exponent `k` in `2^k - c`.
    const MODULUS_BITS: u32;

    /// Offset `c` in `2^k - c`.
    const MODULUS_OFFSET: u128;
}

/// Field with a precomputed primitive root of a supported smooth subgroup.
pub trait SmoothFftField: CanonicalField + PseudoMersenneField {
    /// Order of the supported smooth multiplicative subgroup.
    const SMOOTH_SUBGROUP_ORDER: usize;

    /// Canonical representation of its primitive root.
    const SMOOTH_OMEGA: u128;
}
