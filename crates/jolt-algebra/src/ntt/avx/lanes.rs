//! Width-generic lane arithmetic for the x86 transforms.
//!
//! The transform passes are written once over [`Lanes`] and instantiated for
//! `i32` lanes in `__m256i` and `__m512i` and for `i16` lanes in [`I16x16`].
//! Every method is `#[inline(always)]`, so the intrinsics are inlined into the
//! target-feature caller.
//!
//! Between forward passes, butterfly sums are stored in `[0, 2p)` and products
//! in `(-p, p)`; between inverse passes, every value is in `[0, 2p)`. The
//! helpers on [`Modulus`] move values between these ranges, each with one add
//! or subtract and one unsigned minimum. The transforms require `p` below
//! `2^30` for `i32` and below `2^14` for `i16`, which `NttPrime::new` checks
//! but the public `NttPrime` fields do not enforce. Under that bound `4p` fits
//! the unsigned lane, `2p` fits the signed lane, and no intermediate below
//! wraps.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::ntt::prime::PrimeWidth;

/// A vector of lanes with the operations the transforms need.
pub(super) trait Lanes: Copy {
    /// Lane element.
    type Elem: PrimeWidth;
    /// A constant multiplier prepared for this lane width.
    type Multiplier: Multiply<Self>;
    /// Number of lanes.
    const LANES: usize;

    /// Load `LANES` values from `ptr`.
    unsafe fn load(ptr: *const Self::Elem) -> Self;
    /// Store `LANES` values to `ptr`.
    unsafe fn store(self, ptr: *mut Self::Elem);
    /// Load `LANES` signed bytes from `ptr`, sign-extended.
    unsafe fn load_i8(ptr: *const i8) -> Self;
    /// Load `LANES` signed 16-bit values from `ptr`, sign-extended.
    unsafe fn load_i16(ptr: *const i16) -> Self;
    /// Broadcast one value.
    unsafe fn splat(value: Self::Elem) -> Self;
    /// Lane-wise wrapping addition.
    unsafe fn add(self, rhs: Self) -> Self;
    /// Lane-wise wrapping subtraction.
    unsafe fn sub(self, rhs: Self) -> Self;
    /// Lane-wise unsigned minimum.
    unsafe fn min_unsigned(self, rhs: Self) -> Self;
}

/// Constant multipliers `w` with their Montgomery quotients `w * p^{-1} mod R`.
pub(super) trait Multiply<V: Lanes>: Copy {
    /// Prepare `LANES` multipliers from their values and quotients.
    unsafe fn new(value: V, quotient: V) -> Self;
    /// Montgomery product `x * w / R mod p`, bit-identical to
    /// `NttPrime::mont_mul_raw`. The result lies in `(-p, p)` for every `x`
    /// because `|w| < p`.
    unsafe fn mul(self, x: V, m: Modulus<V>) -> V;
}

/// The prime and its double, broadcast.
#[derive(Clone, Copy)]
pub(super) struct Modulus<V> {
    pub(super) p: V,
    two_p: V,
}

impl<V: Lanes> Modulus<V> {
    #[inline(always)]
    pub(super) unsafe fn splat(p: V::Elem) -> Self {
        // SAFETY: the caller's target features cover `V`.
        unsafe {
            Self {
                p: V::splat(p),
                two_p: V::splat(p.wrapping_add(p)),
            }
        }
    }

    /// `[0, 4p) -> [0, 2p)`.
    #[inline(always)]
    pub(super) unsafe fn reduce_4p(self, x: V) -> V {
        // SAFETY: the caller's target features cover `V`.
        unsafe { x.min_unsigned(x.sub(self.two_p)) }
    }

    /// `(-2p, 2p) -> [0, 2p)`.
    #[inline(always)]
    pub(super) unsafe fn fold(self, x: V) -> V {
        // SAFETY: the caller's target features cover `V`.
        unsafe { x.min_unsigned(x.add(self.two_p)) }
    }

    /// `[0, 2p) -> [0, p)`.
    #[inline(always)]
    pub(super) unsafe fn canonical(self, x: V) -> V {
        // SAFETY: the caller's target features cover `V`.
        unsafe { x.min_unsigned(x.sub(self.p)) }
    }

    /// `(-p, p) -> [0, p)`.
    #[inline(always)]
    pub(super) unsafe fn canonical_signed(self, x: V) -> V {
        // SAFETY: the caller's target features cover `V`.
        unsafe { x.min_unsigned(x.add(self.p)) }
    }

    /// The per-lane sum offsets `(-2p, 2p)`: a butterfly sum of two `[0, 2p)`
    /// values reduces with `-2p` ([`Self::reduce_4p`]) and a sum of two
    /// `(-p, p)` values with `2p` ([`Self::fold`]).
    #[inline(always)]
    pub(super) unsafe fn sum_offsets(self) -> (V, V) {
        // SAFETY: the caller's target features cover `V`.
        unsafe { (self.p.sub(self.p).sub(self.two_p), self.two_p) }
    }

    /// Reduce a butterfly sum to `[0, 2p)` with the per-lane offsets from
    /// [`Self::sum_offsets`].
    #[inline(always)]
    pub(super) unsafe fn reduce_sum(self, x: V, offset: V) -> V {
        // SAFETY: the caller's target features cover `V`.
        unsafe { x.min_unsigned(x.add(offset)) }
    }
}

/// The 64-bit products `i32` lanes use for Montgomery multiplication.
pub(super) trait I32Lanes: Lanes<Elem = i32> {
    /// Signed 64-bit products of the even lanes.
    unsafe fn mul_even_i64(self, rhs: Self) -> Self;
    /// Unsigned 64-bit products of the even lanes.
    unsafe fn mul_even_u64(self, rhs: Self) -> Self;
    /// Copy every odd lane into the even lane below it.
    unsafe fn odd_to_even(self) -> Self;
    /// Gather the high halves of the 64-bit lanes of `even` and `odd` into
    /// the even and odd `i32` lanes of one vector.
    unsafe fn merge_high_halves(even: Self, odd: Self) -> Self;
}

impl Lanes for __m256i {
    type Elem = i32;
    type Multiplier = I32Multiplier<Self>;
    const LANES: usize = 8;

    #[inline(always)]
    unsafe fn load(ptr: *const i32) -> Self {
        // SAFETY: the caller guarantees `ptr` is valid for eight reads.
        unsafe { _mm256_loadu_si256(ptr.cast()) }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut i32) {
        // SAFETY: the caller guarantees `ptr` is valid for eight writes.
        unsafe { _mm256_storeu_si256(ptr.cast(), self) }
    }

    #[inline(always)]
    unsafe fn load_i8(ptr: *const i8) -> Self {
        // SAFETY: the caller guarantees `ptr` is valid for eight reads.
        unsafe { _mm256_cvtepi8_epi32(_mm_loadl_epi64(ptr.cast())) }
    }

    #[inline(always)]
    unsafe fn load_i16(ptr: *const i16) -> Self {
        // SAFETY: the caller guarantees `ptr` is valid for eight reads.
        unsafe { _mm256_cvtepi16_epi32(_mm_loadu_si128(ptr.cast())) }
    }

    #[inline(always)]
    unsafe fn splat(value: i32) -> Self {
        _mm256_set1_epi32(value)
    }

    #[inline(always)]
    unsafe fn add(self, rhs: Self) -> Self {
        _mm256_add_epi32(self, rhs)
    }

    #[inline(always)]
    unsafe fn sub(self, rhs: Self) -> Self {
        _mm256_sub_epi32(self, rhs)
    }

    #[inline(always)]
    unsafe fn min_unsigned(self, rhs: Self) -> Self {
        _mm256_min_epu32(self, rhs)
    }
}

impl I32Lanes for __m256i {
    #[inline(always)]
    unsafe fn mul_even_i64(self, rhs: Self) -> Self {
        _mm256_mul_epi32(self, rhs)
    }

    #[inline(always)]
    unsafe fn mul_even_u64(self, rhs: Self) -> Self {
        _mm256_mul_epu32(self, rhs)
    }

    #[inline(always)]
    unsafe fn odd_to_even(self) -> Self {
        _mm256_shuffle_epi32::<0xF5>(self)
    }

    #[inline(always)]
    unsafe fn merge_high_halves(even: Self, odd: Self) -> Self {
        _mm256_blend_epi32::<0xAA>(_mm256_shuffle_epi32::<0xF5>(even), odd)
    }
}

impl Lanes for __m512i {
    type Elem = i32;
    type Multiplier = I32Multiplier<Self>;
    const LANES: usize = 16;

    #[inline(always)]
    unsafe fn load(ptr: *const i32) -> Self {
        // SAFETY: the caller guarantees `ptr` is valid for sixteen reads.
        unsafe { _mm512_loadu_si512(ptr.cast()) }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut i32) {
        // SAFETY: the caller guarantees `ptr` is valid for sixteen writes.
        unsafe { _mm512_storeu_si512(ptr.cast(), self) }
    }

    #[inline(always)]
    unsafe fn load_i8(ptr: *const i8) -> Self {
        // SAFETY: the caller guarantees `ptr` is valid for sixteen reads.
        unsafe { _mm512_cvtepi8_epi32(_mm_loadu_si128(ptr.cast())) }
    }

    #[inline(always)]
    unsafe fn load_i16(ptr: *const i16) -> Self {
        // SAFETY: the caller guarantees `ptr` is valid for sixteen reads.
        unsafe { _mm512_cvtepi16_epi32(_mm256_loadu_si256(ptr.cast())) }
    }

    #[inline(always)]
    unsafe fn splat(value: i32) -> Self {
        _mm512_set1_epi32(value)
    }

    #[inline(always)]
    unsafe fn add(self, rhs: Self) -> Self {
        _mm512_add_epi32(self, rhs)
    }

    #[inline(always)]
    unsafe fn sub(self, rhs: Self) -> Self {
        _mm512_sub_epi32(self, rhs)
    }

    #[inline(always)]
    unsafe fn min_unsigned(self, rhs: Self) -> Self {
        _mm512_min_epu32(self, rhs)
    }
}

impl I32Lanes for __m512i {
    #[inline(always)]
    unsafe fn mul_even_i64(self, rhs: Self) -> Self {
        _mm512_mul_epi32(self, rhs)
    }

    #[inline(always)]
    unsafe fn mul_even_u64(self, rhs: Self) -> Self {
        _mm512_mul_epu32(self, rhs)
    }

    #[inline(always)]
    unsafe fn odd_to_even(self) -> Self {
        _mm512_shuffle_epi32::<0xF5>(self)
    }

    #[inline(always)]
    unsafe fn merge_high_halves(even: Self, odd: Self) -> Self {
        // Even lanes take the shuffled `even` high halves; odd lanes keep `odd`.
        _mm512_mask_shuffle_epi32::<0xF5>(odd, 0x5555, even)
    }
}

/// `i32` multipliers with their odd lanes copied down for the odd-lane
/// products.
#[derive(Clone, Copy)]
pub(super) struct I32Multiplier<V> {
    value: V,
    value_odd: V,
    quotient: V,
    quotient_odd: V,
}

impl<V: I32Lanes> Multiply<V> for I32Multiplier<V> {
    #[inline(always)]
    unsafe fn new(value: V, quotient: V) -> Self {
        // SAFETY: the caller's target features cover `V`.
        unsafe {
            Self {
                value,
                value_odd: value.odd_to_even(),
                quotient,
                quotient_odd: quotient.odd_to_even(),
            }
        }
    }

    #[inline(always)]
    unsafe fn mul(self, x: V, m: Modulus<V>) -> V {
        // SAFETY: the caller's target features cover `V`.
        unsafe {
            let x_odd = x.odd_to_even();
            let even = x.mul_even_i64(self.value);
            let odd = x_odd.mul_even_i64(self.value_odd);
            let even_t = x.mul_even_u64(self.quotient);
            let odd_t = x_odd.mul_even_u64(self.quotient_odd);
            // `x*w - t*p` is 0 mod 2^32, so a 32-bit subtract leaves the exact
            // high halves in the odd `i32` lanes.
            let even = even.sub(even_t.mul_even_i64(m.p));
            let odd = odd.sub(odd_t.mul_even_i64(m.p));
            V::merge_high_halves(even, odd)
        }
    }
}

/// Montgomery product of two variable vectors, bit-identical to
/// `NttPrime::mont_mul_raw`, with `p` and `pinv = p^{-1} mod 2^32` broadcast.
#[inline(always)]
pub(super) unsafe fn mont_mul<V: I32Lanes>(a: V, b: V, p: V, pinv: V) -> V {
    // SAFETY: the caller's target features cover `V`.
    unsafe {
        let even = a.mul_even_i64(b);
        let odd = a.odd_to_even().mul_even_i64(b.odd_to_even());
        let even = even.sub(even.mul_even_u64(pinv).mul_even_i64(p));
        let odd = odd.sub(odd.mul_even_u64(pinv).mul_even_i64(p));
        V::merge_high_halves(even, odd)
    }
}

/// Sixteen `i16` lanes of a 256-bit vector.
#[derive(Clone, Copy)]
pub(super) struct I16x16(pub(super) __m256i);

impl Lanes for I16x16 {
    type Elem = i16;
    type Multiplier = I16Multiplier;
    const LANES: usize = 16;

    #[inline(always)]
    unsafe fn load(ptr: *const i16) -> Self {
        // SAFETY: the caller guarantees `ptr` is valid for sixteen reads.
        unsafe { Self(_mm256_loadu_si256(ptr.cast())) }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut i16) {
        // SAFETY: the caller guarantees `ptr` is valid for sixteen writes.
        unsafe { _mm256_storeu_si256(ptr.cast(), self.0) }
    }

    #[inline(always)]
    unsafe fn load_i8(ptr: *const i8) -> Self {
        // SAFETY: the caller guarantees `ptr` is valid for sixteen reads.
        unsafe { Self(_mm256_cvtepi8_epi16(_mm_loadu_si128(ptr.cast()))) }
    }

    #[inline(always)]
    unsafe fn load_i16(ptr: *const i16) -> Self {
        // SAFETY: forwarded from the caller.
        unsafe { Self::load(ptr) }
    }

    #[inline(always)]
    unsafe fn splat(value: i16) -> Self {
        Self(_mm256_set1_epi16(value))
    }

    #[inline(always)]
    unsafe fn add(self, rhs: Self) -> Self {
        Self(_mm256_add_epi16(self.0, rhs.0))
    }

    #[inline(always)]
    unsafe fn sub(self, rhs: Self) -> Self {
        Self(_mm256_sub_epi16(self.0, rhs.0))
    }

    #[inline(always)]
    unsafe fn min_unsigned(self, rhs: Self) -> Self {
        Self(_mm256_min_epu16(self.0, rhs.0))
    }
}

/// `i16` multipliers and their quotients.
#[derive(Clone, Copy)]
pub(super) struct I16Multiplier {
    value: __m256i,
    quotient: __m256i,
}

impl Multiply<I16x16> for I16Multiplier {
    #[inline(always)]
    unsafe fn new(value: I16x16, quotient: I16x16) -> Self {
        Self {
            value: value.0,
            quotient: quotient.0,
        }
    }

    #[inline(always)]
    unsafe fn mul(self, x: I16x16, m: Modulus<I16x16>) -> I16x16 {
        // `x*w - t*p` is 0 mod 2^16 for `t = lo(x*q)`, so its high half is
        // `hi(x*w) - hi(t*p)` with no borrow from the low halves.
        I16x16(_mm256_sub_epi16(
            _mm256_mulhi_epi16(x.0, self.value),
            _mm256_mulhi_epi16(_mm256_mullo_epi16(x.0, self.quotient), m.p.0),
        ))
    }
}
