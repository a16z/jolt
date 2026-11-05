use std::arch::aarch64::{
    uint64x2_t, vcombine_u64, vcreate_u64, vdupq_n_u64, veorq_u64, vextq_u64, vgetq_lane_u64,
    vmull_high_p64, vmull_p64, vreinterpretq_p128_u64, vreinterpretq_p64_u64,
    vreinterpretq_u64_p128, vshlq_n_u64, vshrq_n_u64,
};

#[derive(Clone, Copy)]
pub(super) struct Word(uint64x2_t);

// Scalar XOR has lower recurrence latency than NEON XOR on M4.
pub(super) type Unreduced64 = u128;
pub(super) const SCALAR_ACCUMULATOR64: bool = true;
pub(super) const KARATSUBA128: bool = false;
pub(super) const SHIFT_SQUARE128: bool = false;

impl Word {
    #[inline]
    pub(super) fn from_unreduced64(value: Unreduced64) -> Self {
        Self::from_u128(value)
    }

    #[inline]
    pub(super) fn into_unreduced64(self) -> Unreduced64 {
        self.to_u128()
    }

    #[inline]
    pub(super) fn from_u64(value: u64) -> Self {
        // SAFETY: NEON is baseline on AArch64.
        unsafe { Self(vcombine_u64(vcreate_u64(value), vcreate_u64(0))) }
    }

    #[inline]
    pub(super) fn from_u128(value: u128) -> Self {
        // SAFETY: the module cfg guarantees aes; the reinterpret intrinsic preserves lane order.
        unsafe { Self(vreinterpretq_u64_p128(value)) }
    }

    #[inline]
    pub(super) fn from_accumulator_u128(value: u128) -> Self {
        Self::from_u128(value)
    }

    #[inline]
    pub(super) fn swap64(self) -> Self {
        // SAFETY: NEON is baseline on AArch64; the one-lane offset exchanges the lanes.
        unsafe { Self(vextq_u64::<1>(self.0, self.0)) }
    }

    #[inline]
    pub(super) fn to_u128(self) -> u128 {
        // SAFETY: the module cfg guarantees aes; the reinterpret intrinsic preserves lane order.
        unsafe { vreinterpretq_p128_u64(self.0) }
    }

    #[inline]
    pub(super) fn low(self) -> u64 {
        // SAFETY: NEON is baseline on AArch64, and lane 0 exists.
        unsafe { vgetq_lane_u64::<0>(self.0) }
    }

    #[inline]
    pub(super) fn xor(self, rhs: Self) -> Self {
        // SAFETY: NEON is baseline on AArch64.
        unsafe { Self(veorq_u64(self.0, rhs.0)) }
    }

    #[inline]
    pub(super) fn mul_ll(self, rhs: Self) -> Self {
        // SAFETY: the module cfg guarantees aes, enabling polynomial multiplication; both lanes exist.
        unsafe {
            Self(vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64::<0>(self.0),
                vgetq_lane_u64::<0>(rhs.0),
            )))
        }
    }

    #[inline]
    pub(super) fn mul_lh(self, rhs: Self) -> Self {
        // SAFETY: the module cfg guarantees aes, enabling polynomial multiplication; both lanes exist.
        unsafe {
            Self(vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64::<0>(self.0),
                vgetq_lane_u64::<1>(rhs.0),
            )))
        }
    }

    #[inline]
    pub(super) fn mul_hl(self, rhs: Self) -> Self {
        // SAFETY: the module cfg guarantees aes, enabling polynomial multiplication; both lanes exist.
        unsafe {
            Self(vreinterpretq_u64_p128(vmull_p64(
                vgetq_lane_u64::<1>(self.0),
                vgetq_lane_u64::<0>(rhs.0),
            )))
        }
    }

    #[inline]
    pub(super) fn mul_hh(self, rhs: Self) -> Self {
        // SAFETY: the module cfg guarantees aes, enabling polynomial multiplication.
        unsafe {
            Self(vreinterpretq_u64_p128(vmull_high_p64(
                vreinterpretq_p64_u64(self.0),
                vreinterpretq_p64_u64(rhs.0),
            )))
        }
    }

    #[inline]
    pub(super) fn high_to_low(self) -> Self {
        // SAFETY: NEON is baseline on AArch64, and the one-lane offset is in range.
        unsafe { Self(vextq_u64::<1>(self.0, vdupq_n_u64(0))) }
    }

    #[inline]
    pub(super) fn shl<const N: i32>(self) -> Self {
        // SAFETY: NEON is baseline on AArch64; callers use shifts in 1..64.
        unsafe { Self(vshlq_n_u64::<N>(self.0)) }
    }

    #[inline]
    pub(super) fn shr<const N: i32>(self) -> Self {
        // SAFETY: NEON is baseline on AArch64; callers use shifts in 1..64.
        unsafe { Self(vshrq_n_u64::<N>(self.0)) }
    }

    #[inline]
    pub(super) fn low_to_high(self) -> Self {
        // SAFETY: NEON is baseline on AArch64, and lane 0 exists.
        unsafe {
            Self(vcombine_u64(
                vcreate_u64(0),
                vcreate_u64(vgetq_lane_u64::<0>(self.0)),
            ))
        }
    }
}
