use std::arch::x86_64::{
    __m128i, _mm_clmulepi64_si128, _mm_cvtsi128_si64, _mm_cvtsi64_si128, _mm_shuffle_epi32,
    _mm_slli_epi64, _mm_slli_si128, _mm_srli_epi64, _mm_srli_si128, _mm_xor_si128,
};

#[derive(Clone, Copy)]
pub(super) struct Word(__m128i);

pub(super) type Unreduced64 = Word;
pub(super) const SCALAR_ACCUMULATOR64: bool = false;
pub(super) const KARATSUBA128: bool = true;
pub(super) const SHIFT_SQUARE128: bool = true;

impl Word {
    #[inline]
    pub(super) fn from_unreduced64(value: Unreduced64) -> Self {
        value
    }

    #[inline]
    pub(super) fn into_unreduced64(self) -> Unreduced64 {
        self
    }

    #[inline]
    pub(super) fn from_u64(value: u64) -> Self {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { Self(_mm_cvtsi64_si128(value as i64)) }
    }

    #[inline]
    pub(super) fn from_u128(value: u128) -> Self {
        // Separate lane loads avoid a 16-byte load spanning two scalar stores.
        // SAFETY: SSE2 is baseline on x86_64; unpack interleaves the low lanes
        // into low/high order, touches no memory, and preserves flags.
        unsafe {
            let mut low = _mm_cvtsi64_si128(value as i64);
            let high = _mm_cvtsi64_si128((value >> 64) as i64);
            std::arch::asm!(
                "punpcklqdq {low}, {high}",
                low = inout(xmm_reg) low,
                high = in(xmm_reg) high,
                options(pure, nomem, nostack, preserves_flags),
            );
            Self(low)
        }
    }

    #[inline]
    pub(super) fn from_accumulator_u128(value: u128) -> Self {
        // SAFETY: both types have 128 bits, all bit patterns are valid, and x86_64 is little-endian.
        unsafe { Self(std::mem::transmute::<u128, __m128i>(value)) }
    }

    #[inline]
    pub(super) fn swap64(self) -> Self {
        // SAFETY: SSE2 is baseline on x86_64; 0x4e exchanges the two 64-bit lanes.
        unsafe { Self(_mm_shuffle_epi32::<0x4e>(self.0)) }
    }

    #[inline]
    pub(super) fn to_u128(self) -> u128 {
        // SAFETY: both types have 128 bits, all bit patterns are valid, and x86_64 is little-endian.
        unsafe { std::mem::transmute::<__m128i, u128>(self.0) }
    }

    #[inline]
    pub(super) fn low(self) -> u64 {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { _mm_cvtsi128_si64(self.0) as u64 }
    }

    #[inline]
    pub(super) fn xor(self, rhs: Self) -> Self {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { Self(_mm_xor_si128(self.0, rhs.0)) }
    }

    #[inline]
    pub(super) fn mul_ll(self, rhs: Self) -> Self {
        // SAFETY: the module cfg guarantees pclmulqdq.
        unsafe { Self(_mm_clmulepi64_si128::<0x00>(self.0, rhs.0)) }
    }

    #[inline]
    pub(super) fn mul_lh(self, rhs: Self) -> Self {
        // SAFETY: the module cfg guarantees pclmulqdq.
        unsafe { Self(_mm_clmulepi64_si128::<0x10>(self.0, rhs.0)) }
    }

    #[inline]
    pub(super) fn mul_hl(self, rhs: Self) -> Self {
        // SAFETY: the module cfg guarantees pclmulqdq.
        unsafe { Self(_mm_clmulepi64_si128::<0x01>(self.0, rhs.0)) }
    }

    #[inline]
    pub(super) fn mul_hh(self, rhs: Self) -> Self {
        // SAFETY: the module cfg guarantees pclmulqdq.
        unsafe { Self(_mm_clmulepi64_si128::<0x11>(self.0, rhs.0)) }
    }

    #[inline]
    pub(super) fn high_to_low(self) -> Self {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { Self(_mm_srli_si128::<8>(self.0)) }
    }

    #[inline]
    pub(super) fn shl<const N: i32>(self) -> Self {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { Self(_mm_slli_epi64::<N>(self.0)) }
    }

    #[inline]
    pub(super) fn shr<const N: i32>(self) -> Self {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { Self(_mm_srli_epi64::<N>(self.0)) }
    }

    #[inline]
    pub(super) fn low_to_high(self) -> Self {
        // SAFETY: SSE2 is baseline on x86_64.
        unsafe { Self(_mm_slli_si128::<8>(self.0)) }
    }
}
