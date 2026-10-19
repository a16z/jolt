//! Exact `usize` arithmetic for sizes, offsets, ranges, and allocation bounds.
//!
//! These functions never wrap, saturate, or panic. They return [`None`] when
//! the requested value cannot be represented or an operation is undefined.
//! They do not allocate or collect iterator inputs.
//! The caller maps that failure to the protocol error that matches its trust
//! boundary, such as invalid setup, invalid input, or an invalid proof.
//!
//! Direct calls to standard library operations such as [`usize::checked_add`]
//! remain appropriate for a single local formula. This module owns formulas
//! that otherwise tend to be copied into helpers throughout the workspace.

use core::ops::Range;

/// Return the exact sum of all `values`, or [`None`] on overflow.
///
/// An empty input has sum zero. Arrays work directly, and callers can pass a
/// dynamic slice with `values.iter().copied()`.
#[inline]
#[must_use]
pub fn sum(values: impl IntoIterator<Item = usize>) -> Option<usize> {
    values.into_iter().try_fold(0usize, usize::checked_add)
}

/// Return the exact product of all `factors`, or [`None`] on overflow.
///
/// This function does not allocate or collect its input. A fixed array such as
/// `[a, b, c]` is consumed directly. A borrowed slice can use
/// `factors.iter().copied()` without allocating. An empty input has product
/// one.
#[inline]
#[must_use]
pub fn product(factors: impl IntoIterator<Item = usize>) -> Option<usize> {
    factors.into_iter().try_fold(1usize, usize::checked_mul)
}

/// Return `lhs * rhs + addend`, or [`None`] if either operation overflows.
#[inline]
#[must_use]
pub fn mul_add(lhs: usize, rhs: usize, addend: usize) -> Option<usize> {
    lhs.checked_mul(rhs)?.checked_add(addend)
}

/// Return `2^exponent`, or [`None`] when the result does not fit in `usize`.
#[inline]
#[must_use]
pub fn pow2(exponent: usize) -> Option<usize> {
    let shift = u32::try_from(exponent).ok()?;
    1usize.checked_shl(shift)
}

/// Return the least number of bits needed for a nonzero `value`, rounding the
/// value up to its next power of two first.
///
/// Returns [`None`] for zero or when that power of two does not fit in `usize`.
#[inline]
#[must_use]
pub fn ceil_log2(value: usize) -> Option<usize> {
    if value == 0 {
        return None;
    }
    Some(value.checked_next_power_of_two()?.trailing_zeros() as usize)
}

/// Return the half-open range `start..start + len`, or [`None`] on overflow.
#[inline]
#[must_use]
pub fn range(start: usize, len: usize) -> Option<Range<usize>> {
    Some(start..start.checked_add(len)?)
}

/// Round `value` up to `alignment`, or return [`None`] if the alignment is not
/// a nonzero power of two or the rounded value overflows.
///
/// An already aligned value is returned unchanged, including `usize::MAX`
/// when the alignment is one.
#[inline]
#[must_use]
pub fn align_up(value: usize, alignment: usize) -> Option<usize> {
    if !alignment.is_power_of_two() {
        return None;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - remainder)
    }
}

/// Return `ceil(value / divisor)`, or [`None`] when `divisor` is zero.
///
/// This implementation divides before adding the remainder bit, so it does not
/// overflow for large values as the common `(value + divisor - 1) / divisor`
/// formula can.
#[inline]
#[must_use]
pub fn div_ceil(value: usize, divisor: usize) -> Option<usize> {
    let quotient = value.checked_div(divisor)?;
    quotient.checked_add(usize::from(!value.is_multiple_of(divisor)))
}

/// Return `value / divisor` when the division is defined and exact.
#[inline]
#[must_use]
pub fn exact_div(value: usize, divisor: usize) -> Option<usize> {
    let quotient = value.checked_div(divisor)?;
    value.is_multiple_of(divisor).then_some(quotient)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum_accepts_arrays_iterators_and_empty_inputs() {
        assert_eq!(sum([2, 3, 5]), Some(10));
        assert_eq!(sum([2, 3, 5].iter().copied()), Some(10));
        assert_eq!(sum([]), Some(0));
        assert_eq!(sum([usize::MAX, 1]), None);
    }

    #[test]
    fn product_replaces_fixed_arity_multiplication_helpers() {
        assert_eq!(product([2, 3, 5]), Some(30));
        assert_eq!(product([2, 3, 5, 7]), Some(210));
        assert_eq!(product([]), Some(1));
        assert_eq!(product([usize::MAX, 2]), None);
    }

    #[test]
    fn mul_add_checks_both_operations() {
        assert_eq!(mul_add(4, 5, 3), Some(23));
        assert_eq!(mul_add(usize::MAX, 2, 0), None);
        assert_eq!(mul_add(usize::MAX, 1, 1), None);
    }

    #[test]
    fn pow2_checks_the_shift_and_result_width() {
        assert_eq!(pow2(0), Some(1));
        assert_eq!(
            pow2(usize::BITS as usize - 1),
            Some(1usize << (usize::BITS - 1))
        );
        assert_eq!(pow2(usize::BITS as usize), None);
        assert_eq!(pow2(usize::MAX), None);
    }

    #[test]
    fn ceil_log2_checks_zero_and_padding_overflow() {
        assert_eq!(ceil_log2(1), Some(0));
        assert_eq!(ceil_log2(8), Some(3));
        assert_eq!(ceil_log2(9), Some(4));
        assert_eq!(ceil_log2(0), None);
        assert_eq!(ceil_log2(usize::MAX), None);
    }

    #[test]
    fn range_checks_the_end() {
        assert_eq!(range(4, 3), Some(4..7));
        assert_eq!(range(usize::MAX, 0), Some(usize::MAX..usize::MAX));
        assert_eq!(range(usize::MAX, 1), None);
    }

    #[test]
    fn align_up_checks_alignment_and_overflow() {
        assert_eq!(align_up(9, 8), Some(16));
        assert_eq!(align_up(16, 8), Some(16));
        assert_eq!(align_up(usize::MAX, 1), Some(usize::MAX));
        assert_eq!(align_up(usize::MAX, 2), None);
        assert_eq!(align_up(8, 0), None);
        assert_eq!(align_up(8, 3), None);
    }

    #[test]
    fn div_ceil_avoids_predivision_overflow() {
        assert_eq!(div_ceil(0, 3), Some(0));
        assert_eq!(div_ceil(9, 3), Some(3));
        assert_eq!(div_ceil(10, 3), Some(4));
        assert_eq!(div_ceil(usize::MAX, 2), Some(usize::MAX / 2 + 1));
        assert_eq!(div_ceil(1, 0), None);
    }

    #[test]
    fn exact_div_rejects_zero_and_remainders() {
        assert_eq!(exact_div(12, 3), Some(4));
        assert_eq!(exact_div(0, 3), Some(0));
        assert_eq!(exact_div(10, 3), None);
        assert_eq!(exact_div(1, 0), None);
    }
}
