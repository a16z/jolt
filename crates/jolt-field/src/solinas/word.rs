//! Single-word pseudo-Mersenne prime fields: one Solinas fold algebra
//! stamped at `u32` ([`Fp32`]) and `u64` ([`Fp64`]) storage.
//!
//! The fold point `k` and offset `c = 2^k − p` are computed at compile time
//! from the const-generic modulus; the `C(C+1) < P` precondition for the
//! fused two-fold-plus-canonicalize reduction is const-asserted in exactly
//! one place. Per-width differences enter only through the `mul`/`random`
//! macro arguments (the `u64` width has a fold-entirely-in-`u64` product
//! path for sub-word primes, with a BMI2 variant on x86-64).

use crate::PseudoMersenne;
use crate::{CanonicalBytes, CanonicalEncoding, Field, NaiveAccumulator, Ring, WithAccumulator};
use rand_core::RngCore;

/// Trial-division primality check, cheap enough for CTFE at u32 scale.
/// (64-bit moduli skip the check: 2^32 const-eval iterations is not viable.)
const fn is_small_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut d = 3u64;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

macro_rules! define_solinas_prime {
    (
        $(#[$doc:meta])* $name:ident,
        word: $word:ident,
        from_canonical: $from_canon:ident,
        to_canonical: $to_canon:ident,
        double: $double:ty,
        mul_wide_raw: $mul_wide_raw:ident($raw:ty),
        mul($ma:ident, $mb:ident): $mul:expr,
        random($rng:ident): $random:expr $(,)?
    ) => {
        $(#[$doc])*
        #[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
        #[repr(transparent)]
        pub struct $name<const P: $word>(pub(crate) $word);

        impl<const P: $word> $name<P> {
            /// Fold point: smallest `k` such that `P <= 2^k`.
            pub(crate) const BITS: u32 = <$word>::BITS - P.leading_zeros();

            /// Offset `c = 2^k − P`. Instantiating with a modulus that
            /// violates the Solinas preconditions is a compile-time error.
            pub const C: $word = {
                let c = if Self::BITS == <$word>::BITS {
                    (0 as $word).wrapping_sub(P)
                } else {
                    ((1 as $word) << Self::BITS) - P
                };
                assert!(P != 0, "modulus must be nonzero");
                assert!(P & 1 == 1, "modulus must be odd");
                assert!(
                    <$word>::BITS > 32 || is_small_prime(P as u64),
                    "modulus must be prime"
                );
                assert!(
                    (c as u128) * (c as u128 + 1) < P as u128,
                    "C(C+1) < P required for fused canonicalize"
                );
                c
            };

            /// Mask for the low `BITS` bits of a double word.
            const MASK: $double = if Self::BITS == <$word>::BITS {
                <$word>::MAX as $double
            } else {
                ((1 as $double) << Self::BITS) - 1
            };

            const MASK128: u128 = Self::MASK as u128;

            /// Conditional subtract of a folded value down to `[0, P)`.
            #[inline(always)]
            fn canonicalize_folded(v: $double) -> $word {
                if Self::BITS < <$word>::BITS {
                    let x = v as $word;
                    x.min(x.wrapping_sub(P))
                } else {
                    let reduced = v.wrapping_sub(P as $double);
                    let borrow = reduced >> (<$double>::BITS - 1);
                    reduced.wrapping_add(borrow.wrapping_neg() & (P as $double)) as $word
                }
            }

            /// Loop-fold Solinas reduction of an arbitrary double word.
            #[inline(always)]
            fn reduce_double(x: $double) -> $word {
                let mut v = x;
                while v >> Self::BITS != 0 {
                    v = (v & Self::MASK) + (Self::C as $double) * (v >> Self::BITS);
                }
                Self::canonicalize_folded(v)
            }

            /// Loop-fold Solinas reduction of an arbitrary `u128`.
            #[inline(always)]
            fn reduce_u128(x: u128) -> $word {
                let mut v = x;
                while v >> Self::BITS != 0 {
                    v = (v & Self::MASK128) + (Self::C as u128) * (v >> Self::BITS);
                }
                Self::canonicalize_folded(v as $double)
            }

            /// Two-fold Solinas reduction for products `< 2^{2·BITS}`.
            #[inline(always)]
            fn reduce_product(x: $double) -> $word {
                let c = Self::C as $double;
                let f1 = (x & Self::MASK) + c * (x >> Self::BITS);
                let f2 = (f1 & Self::MASK) + c * (f1 >> Self::BITS);
                Self::canonicalize_folded(f2)
            }

            #[inline(always)]
            fn add_raw(a: $word, b: $word) -> $word {
                if Self::BITS < <$word>::BITS {
                    let s = a.wrapping_add(b);
                    s.min(s.wrapping_sub(P))
                } else {
                    // Full-word: fold the carry with 2^k ≡ C, then subtract.
                    let (s, overflow) = a.overflowing_add(b);
                    let folded = s.wrapping_add((overflow as $word).wrapping_neg() & Self::C);
                    folded.min(folded.wrapping_sub(P))
                }
            }

            #[inline(always)]
            fn sub_raw(a: $word, b: $word) -> $word {
                let (d, underflow) = a.overflowing_sub(b);
                // If subtraction borrowed, subtracting -P modulo the word
                // adds P. At full width, -P is the small Solinas offset C.
                d.wrapping_sub(
                    (underflow as $word).wrapping_neg() & P.wrapping_neg()
                )
            }

            #[inline(always)]
            fn mul_raw(a: $word, b: $word) -> $word {
                let ($ma, $mb) = (a, b);
                $mul
            }

            fn pow(self, mut exp: u64) -> Self {
                let mut base = self;
                let mut acc = <Self as num_traits::One>::one();
                while exp > 0 {
                    if (exp & 1) == 1 {
                        acc *= base;
                    }
                    base = Self(Self::mul_raw(base.0, base.0));
                    exp >>= 1;
                }
                acc
            }

            /// Create from a canonical representative in `[0, P)`.
            #[inline]
            pub fn $from_canon(x: $word) -> Self {
                debug_assert!(x < P);
                Self(x)
            }

            /// Return the canonical representative in `[0, P)`.
            #[inline]
            pub fn $to_canon(self) -> $word {
                self.0
            }

            /// Extract the canonical value.
            #[inline(always)]
            pub fn to_limbs(self) -> $word {
                self.0
            }

            /// Widening multiply to a double word, **no reduction**.
            #[inline(always)]
            pub fn mul_wide(self, other: Self) -> $double {
                (self.0 as $double) * (other.0 as $double)
            }

            /// Widening multiply by a raw word operand, **no reduction**.
            #[inline(always)]
            pub fn $mul_wide_raw(self, other: $raw) -> $double {
                (self.0 as $double) * (other as $double)
            }

            /// Reduce a double word via Solinas folding to a canonical element.
            #[inline(always)]
            pub fn solinas_reduce(x: $double) -> Self {
                Self(Self::reduce_double(x))
            }
        }

        $crate::impl_ring_ops!(impl[const P: $word] $name<P> {
            add(a, b): $name(Self::add_raw(a.0, b.0)),
            sub(a, b): $name(Self::sub_raw(a.0, b.0)),
            mul(a, b): $name(Self::mul_raw(a.0, b.0)),
            neg(a): $name(Self::sub_raw(0, a.0)),
            zero: $name(0),
            one: $name((P > 1) as $word),
        });

        impl<const P: $word> std::fmt::Display for $name<P> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl<const P: $word> Ring for $name<P> {
            #[inline(always)]
            fn from_u64(v: u64) -> Self {
                Self(Self::reduce_double(v as $double))
            }

            #[inline(always)]
            fn from_i64(v: i64) -> Self {
                if v >= 0 {
                    Self::from_u64(v as u64)
                } else {
                    -Self::from_u64(v.unsigned_abs())
                }
            }

            #[inline(always)]
            fn from_u128(v: u128) -> Self {
                Self(Self::reduce_u128(v))
            }

            #[inline(always)]
            fn from_i128(v: i128) -> Self {
                if v >= 0 {
                    Self::from_u128(v as u128)
                } else {
                    -Self::from_u128(v.unsigned_abs())
                }
            }

            #[inline(always)]
            fn square(&self) -> Self {
                Self(Self::mul_raw(self.0, self.0))
            }
        }

        impl<const P: $word> Field for $name<P> {
            #[inline(always)]
            fn inverse(&self) -> Option<Self> {
                let inv = self.inv_or_zero();
                if num_traits::Zero::is_zero(self) {
                    None
                } else {
                    Some(inv)
                }
            }

            /// Fermat inversion with branchless zero-masking.
            #[inline(always)]
            fn inv_or_zero(self) -> Self {
                let candidate = self.pow((P as u64).wrapping_sub(2));
                let nz = ((self.0 | self.0.wrapping_neg()) >> (<$word>::BITS - 1)) & 1;
                let mask = (0 as $word).wrapping_sub(nz);
                Self(candidate.0 & mask)
            }

            #[inline(always)]
            fn random<R: RngCore>($rng: &mut R) -> Self {
                $random
            }

            #[inline]
            fn half(self) -> Self {
                let x = self.0 as $double;
                Self(((x + (x & 1) * P as $double) >> 1) as $word)
            }

            #[inline]
            fn two_inv() -> Self {
                <Self as num_traits::One>::one().half()
            }
        }

        impl<const P: $word> CanonicalBytes for $name<P> {
            const NUM_BYTES: usize = (<$word>::BITS / 8) as usize;

            #[inline(always)]
            fn to_bytes_le(&self, out: &mut [u8]) {
                assert_eq!(out.len(), Self::NUM_BYTES);
                out.copy_from_slice(&self.0.to_le_bytes());
            }
        }

        impl<const P: $word> CanonicalEncoding for $name<P> {
            const MODULUS_BITS: u32 = Self::BITS;

            #[inline(always)]
            fn from_bytes_le_reduced(bytes: &[u8]) -> Self {
                if bytes.len() <= 16 {
                    let mut padded = [0u8; 16];
                    padded[..bytes.len()].copy_from_slice(bytes);
                    return Self::from_u128(u128::from_le_bytes(padded));
                }
                $crate::solinas::reduce_le_bytes_mod_order(bytes)
            }

            #[inline]
            fn from_bytes_le_checked(bytes: &[u8]) -> Option<Self> {
                let arr: [u8; (<$word>::BITS / 8) as usize] = bytes.try_into().ok()?;
                Self::from_u128_checked(<$word>::from_le_bytes(arr) as u128)
            }

            #[inline]
            fn to_u128_checked(&self) -> Option<u128> {
                Some(self.0 as u128)
            }

            #[inline]
            fn from_u128_checked(v: u128) -> Option<Self> {
                (v < P as u128).then(|| Self(v as $word))
            }

            #[inline]
            fn from_u128_reduced(v: u128) -> Self {
                Self(Self::reduce_u128(v))
            }

            #[inline]
            fn canonical_u32_slice(values: &[Self]) -> Option<&[u32]> {
                canonical_u32_slice!($word, values)
            }

            #[inline]
            fn canonical_u64_slice(values: &[Self]) -> Option<&[u64]> {
                canonical_u64_slice!($word, values)
            }

            #[inline]
            fn num_bits(&self) -> u32 {
                <$word>::BITS - self.0.leading_zeros()
            }

            #[inline]
            fn from_scalar_challenge_bytes(bytes: &[u8]) -> Self {
                Self::from_bytes_le_reduced(bytes)
            }
        }

        $crate::impl_serde_bytes!(impl[const P: $word] $name<P>, (<$word>::BITS / 8) as usize);

        impl<const P: $word> WithAccumulator for $name<P> {
            type Accumulator = NaiveAccumulator<Self>;
            type SmallScalarAccumulator = NaiveAccumulator<Self>;
            type SignedProductAccumulator = NaiveAccumulator<Self>;
        }

    };
}

macro_rules! canonical_u32_slice {
    (u32, $values:ident) => {{
        // SAFETY: `Fp32` is transparent over one `u32`, and every constructor
        // and arithmetic operation maintains a canonical representative.
        Some(unsafe { std::slice::from_raw_parts($values.as_ptr().cast(), $values.len()) })
    }};
    ($word:ident, $values:ident) => {{
        let _ = $values;
        None
    }};
}

macro_rules! canonical_u64_slice {
    (u64, $values:ident) => {{
        // SAFETY: `Fp64` is transparent over one `u64`, and every constructor
        // and arithmetic operation maintains a canonical representative.
        Some(unsafe { std::slice::from_raw_parts($values.as_ptr().cast(), $values.len()) })
    }};
    ($word:ident, $values:ident) => {{
        let _ = $values;
        None
    }};
}

define_solinas_prime!(
    /// Prime field element for primes `p = 2^k − c` stored as `u32`.
    Fp32,
    word: u32,
    from_canonical: from_canonical_u32,
    to_canonical: to_canonical_u32,
    double: u64,
    mul_wide_raw: mul_wide_u32(u32),
    mul(a, b): Self::reduce_product((a as u64) * (b as u64)),
    random(rng): Self(super::sample_uniform_below(rng, P as u128, Self::BITS) as u32),
);

impl<const P: u32> PseudoMersenne for Fp32<P> {
    const OFFSET: u128 = Self::C as u128;

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn ext4_mul(a: [Self; 4], b: [Self; 4]) -> [Self; 4] {
        if Self::BITS < 32 {
            return crate::schedules::ext4_mul_coeffs(a, b);
        }
        super::unreduced::fp_ext4_mul_to_accum_fp32(a, b)
            .0
            .map(Self::from_u128_reduced)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn ext4_square(a: [Self; 4]) -> [Self; 4] {
        if Self::BITS < 32 {
            return crate::schedules::ext4_square_coeffs(a);
        }
        super::unreduced::fp_ext4_square_to_accum_fp32(a)
            .0
            .map(Self::from_u128_reduced)
    }
}

define_solinas_prime!(
    /// Prime field element for primes `p = 2^k − c` stored as `u64`.
    Fp64,
    word: u64,
    from_canonical: from_canonical_u64,
    to_canonical: to_canonical_u64,
    double: u128,
    mul_wide_raw: mul_wide_u64(u64),
    mul(a, b): {
        let (lo, hi) = mul64_wide(a, b);
        Self::reduce_product_wide(lo, hi)
    },
    random(rng): Self(super::sample_uniform_below(rng, P as u128, Self::BITS) as u64),
);

impl<const P: u64> PseudoMersenne for Fp64<P> {
    const OFFSET: u128 = Self::C as u128;
}

impl<const P: u64> Fp64<P> {
    /// Mask for the low `BITS` bits in a word.
    pub(crate) const MASK64: u64 = if Self::BITS < 64 {
        (1u64 << Self::BITS) - 1
    } else {
        u64::MAX
    };

    /// Whether both product folds stay in a single word.
    pub(crate) const FOLD_IN_U64: bool =
        Self::BITS < 64 && (Self::C as u128) < (1u128 << (64 - Self::BITS));

    /// Reduces a product supplied as exact low and high words.
    #[inline(always)]
    pub(crate) fn reduce_product_wide(lo: u64, hi: u64) -> u64 {
        if Self::FOLD_IN_U64 {
            let high = (lo >> Self::BITS) | (hi << (64 - Self::BITS));
            let f1 = (lo & Self::MASK64) + mul_c_narrow(Self::C, high);
            let f2 = (f1 & Self::MASK64) + mul_c_narrow(Self::C, f1 >> Self::BITS);
            let reduced = f2.wrapping_sub(P);
            reduced.wrapping_add((reduced >> 63).wrapping_neg() & P)
        } else if Self::BITS < 64 {
            Self::reduce_sub_word_wide(lo, hi, 0)
        } else {
            Self::reduce_product((lo as u128) | ((hi as u128) << 64))
        }
    }

    /// Two-fold sub-word reduction. `high_overflow` is the portion of
    /// `x >> BITS` above one word, which can be nonzero for three products.
    #[inline(always)]
    pub(super) fn reduce_sub_word_wide(lo: u64, hi: u64, high_overflow: u64) -> u64 {
        let high = (lo >> Self::BITS) | (hi << (64 - Self::BITS));
        let c_high = (Self::C as u128) * (high as u128)
            + (((Self::C as u128) * (high_overflow as u128)) << 64);
        let (fold1_lo, carry) = (lo & Self::MASK64).overflowing_add(c_high as u64);
        let fold1_hi = ((c_high >> 64) as u64) + u64::from(carry);
        let fold1_high = (fold1_lo >> Self::BITS) | (fold1_hi << (64 - Self::BITS));
        let fold2 = (fold1_lo & Self::MASK64) + mul_c_narrow(Self::C, fold1_high);
        let reduced = fold2.wrapping_sub(P);
        reduced.wrapping_add(u64::from(fold2 < P).wrapping_neg() & P)
    }
}

/// `a * b` widening to 128 bits; returns `(lo, hi)`. Shared with the
/// two-limb field (`fp128.rs`).
#[inline(always)]
pub(super) fn mul64_wide(a: u64, b: u64) -> (u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        let mut hi = 0;
        // SAFETY: the BMI2 intrinsic is gated by its required target feature.
        let lo = unsafe { std::arch::x86_64::_mulx_u64(a, b, &mut hi) };
        (lo, hi)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let prod = (a as u128) * (b as u128);
        (prod as u64, (prod >> 64) as u64)
    }
}

/// `c * x` split into u32-wide halves so LLVM emits `umull` on aarch64
/// instead of promoting to `u128` (valid because `C < sqrt(P) < 2^32`);
/// x86-64 keeps the single fast 64-bit multiply.
#[inline(always)]
fn mul_c_narrow(c: u64, x: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        c.wrapping_mul(x)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let (c, x_lo, x_hi) = (c as u32 as u64, x as u32 as u64, x >> 32);
        (c * x_lo).wrapping_add((c * x_hi) << 32)
    }
}
