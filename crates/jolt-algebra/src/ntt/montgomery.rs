//! Signed Montgomery reduction shared by centered coefficient converters.

/// Multiplicative inverse of odd `p` modulo `2^32`.
#[inline(always)]
pub(crate) fn inverse_i32(p: i32) -> i32 {
    debug_assert!(p > 1 && p & 1 == 1);
    // An odd number is its own inverse modulo eight. Four Newton steps
    // double the correct low bits from three to at least 32.
    let p = p as u32;
    let mut inverse = p;
    for _ in 0..4 {
        inverse = inverse.wrapping_mul(2u32.wrapping_sub(p.wrapping_mul(inverse)));
    }
    inverse as i32
}

/// Signed `t * 2^-32 mod p`, in `(-p, p)` when `|t| < 2^31 p`.
#[inline(always)]
pub(crate) fn reduce_i32(t: i64, p: i32, pinv: i32) -> i32 {
    let m = (t as i32).wrapping_mul(pinv);
    ((t - i64::from(m) * i64::from(p)) >> 32) as i32
}
