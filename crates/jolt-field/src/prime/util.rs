//! Shared helpers for field arithmetic backends.

#![cfg_attr(
    all(target_arch = "x86_64", target_feature = "bmi2"),
    expect(
        clippy::undocumented_unsafe_blocks,
        reason = "the BMI2 intrinsic is gated by its required target feature"
    )
)]

#[inline(always)]
pub(crate) const fn is_pow2_u64(x: u64) -> bool {
    x.is_power_of_two()
}

#[inline(always)]
pub(crate) const fn log2_pow2_u64(mut x: u64) -> u32 {
    let mut k = 0u32;
    while x > 1 {
        x >>= 1;
        k += 1;
    }
    k
}

/// `a * b` widening to 128 bits; returns `(lo64, hi64)`.
#[inline(always)]
pub(crate) fn mul64_wide(a: u64, b: u64) -> (u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        unsafe { mul64_wide_bmi2(a, b) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let prod = (a as u128) * (b as u128);
        (prod as u64, (prod >> 64) as u64)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
unsafe fn mul64_wide_bmi2(a: u64, b: u64) -> (u64, u64) {
    let mut hi = 0;
    let lo = unsafe { std::arch::x86_64::_mulx_u64(a, b, &mut hi) };
    (lo, hi)
}
