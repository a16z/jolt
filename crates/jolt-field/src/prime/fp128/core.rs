use super::*;

/// 128-bit prime field element for primes of the form `p = 2^128 - c`.
///
/// Stored as `[u64; 2]` (lo, hi) for 8-byte alignment and direct limb access.
///
/// The offset `c = 2^128 - p` and all derived constants are computed at
/// compile time from the const-generic `P`. Instantiating `Fp128` with a
/// modulus that is not of this form is a compile-time error.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Fp128<const P: u128>(pub(crate) [u64; 2]);

impl<const P: u128> PartialEq for Fp128<P> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<const P: u128> Eq for Fp128<P> {}

impl<const P: u128> Fp128<P> {
    /// Offset `c = 2^128 − p`.  Validated at compile time.
    pub const C: u128 = {
        let c = 0u128.wrapping_sub(P);
        assert!(P != 0, "modulus must be nonzero");
        assert!(P & 1 == 1, "modulus must be odd");
        assert!(
            c < (1u128 << 32),
            "C must be < 2^32 (asm fold-2 uses single mul)"
        );
        assert!(
            c * (c + 1) < P,
            "C(C+1) < P required for fused canonicalize"
        );
        c
    };
    /// Low 64 bits of `C` (always equals `C` since `C < 2^32`).
    pub const C_LO: u64 = Self::C as u64;

    /// Create from a canonical representative in `[0, p)`.
    #[inline]
    pub fn from_canonical_u128(x: u128) -> Self {
        debug_assert!(x < P);
        Self(from_u128(x))
    }

    /// Additive identity.
    #[inline]
    pub fn zero() -> Self {
        Self(pack(0, 0))
    }

    /// Multiplicative identity.
    #[inline]
    pub fn one() -> Self {
        Self(pack(1, 0))
    }

    /// Check whether this element is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0 == [0, 0]
    }

    /// Multiplicative inverse, or `None` for zero.
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        <Self as Invertible>::inverse(self)
    }

    /// Construct from a `u64` reduced modulo the field modulus.
    #[inline]
    pub fn from_u64(val: u64) -> Self {
        Self(from_u128(val as u128))
    }

    /// Construct from an `i64` reduced modulo the field modulus.
    #[inline]
    pub fn from_i64(val: i64) -> Self {
        Self::from_i64_const(val)
    }

    /// Construct from an `i8` reduced modulo the field modulus.
    #[inline]
    pub fn from_i8(val: i8) -> Self {
        Self::from_i64(val as i64)
    }

    /// Return the canonical representative in `[0, p)`.
    #[inline]
    pub fn to_canonical_u128(self) -> u128 {
        to_u128(self.0)
    }

    /// Const-evaluable `from_i64`. Embeds a small signed integer into `Fp`.
    pub const fn from_i64_const(val: i64) -> Self {
        if val >= 0 {
            Self(from_u128(val as u128))
        } else {
            Self(Self::sub_raw_portable(
                pack(0, 0),
                from_u128(val.unsigned_abs() as u128),
            ))
        }
    }

    /// Const-evaluable lookup table for balanced digits in `[-b/2, b/2)`
    /// where `b = 2^log_basis`. Requires `log_basis <= 6`.
    ///
    /// # Panics
    ///
    /// Panics if `log_basis` is outside `1..=6`.
    pub const fn digit_lut(log_basis: u32) -> [Self; 64] {
        assert!(log_basis > 0 && log_basis <= 6);
        let b = 1u32 << log_basis;
        let half_b = (b / 2) as i64;
        let mut lut = [Self(pack(0, 0)); 64];
        let mut i = 0u32;
        while i < b {
            lut[i as usize] = Self::from_i64_const(i as i64 - half_b);
            i += 1;
        }
        lut
    }
}
