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

/// Builds the balanced signed-digit table for `1 <= log_basis <= 6`.
pub fn balanced_digit_lut<F: FromPrimitiveInt + Zero + Copy>(log_basis: u32) -> [F; 64] {
    debug_assert!(log_basis > 0 && log_basis <= 6);
    let basis = 1usize << log_basis;
    let half_basis = (basis >> 1) as i64;
    std::array::from_fn(|i| {
        if i < basis {
            F::from_i64(i as i64 - half_basis)
        } else {
            F::zero()
        }
    })
}

/// Metadata for a pseudo-Mersenne modulus `2^k - c`.
pub trait PseudoMersenneField: CanonicalField {
    /// Exponent `k` in `2^k - c`.
    const MODULUS_BITS: u32;

    /// Offset `c` in `2^k - c`.
    const MODULUS_OFFSET: u128;
}
