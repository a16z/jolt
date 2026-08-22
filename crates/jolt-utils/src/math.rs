//! Bit-manipulation utilities on `usize`.

pub trait Math {
    /// Returns `2^self`.
    fn pow2(self) -> usize;
    /// Returns `ceil(log2(self))` — exactly `log2(self)` for powers of two.
    fn log_2(self) -> usize;
}

impl Math for usize {
    #[inline]
    fn pow2(self) -> usize {
        1usize << self
    }

    fn log_2(self) -> usize {
        assert_ne!(self, 0);
        if self.is_power_of_two() {
            (1usize.leading_zeros() - self.leading_zeros()) as usize
        } else {
            (0usize.leading_zeros() - self.leading_zeros()) as usize
        }
    }
}

/// Returns `log2(value)`, panicking unless `value` is a power of two.
pub fn log2_power_of_two(value: usize) -> usize {
    assert!(
        value.is_power_of_two(),
        "expected a power-of-two dimension, got {value}"
    );
    value.trailing_zeros() as usize
}

/// Asserts that a point dimension is below the `usize` shift width, so
/// `1usize << dim` (the eval-table size) cannot overflow the shift.
#[inline]
pub fn assert_shiftable_dim(dim: usize) {
    assert!(
        dim < usize::BITS as usize,
        "point dimension {dim} exceeds usize shift width"
    );
}

/// Returns `log2(value)` if `value` is a power of two, `None` otherwise
/// (zero included).
pub fn checked_log2_power_of_two(value: usize) -> Option<usize> {
    value
        .is_power_of_two()
        .then_some(value.trailing_zeros() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow2_values() {
        assert_eq!(0.pow2(), 1);
        assert_eq!(1.pow2(), 2);
        assert_eq!(10.pow2(), 1024);
        assert_eq!(20.pow2(), 1_048_576);
    }

    #[test]
    fn log_2_powers_of_two() {
        assert_eq!(1.log_2(), 0);
        assert_eq!(2.log_2(), 1);
        assert_eq!(4.log_2(), 2);
        assert_eq!(1024.log_2(), 10);
    }

    #[test]
    fn log_2_non_powers() {
        assert_eq!(3.log_2(), 2);
        assert_eq!(5.log_2(), 3);
        assert_eq!(1023.log_2(), 10);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn log_2_zero_panics() {
        let _ = 0usize.log_2();
    }

    #[test]
    fn log2_power_of_two_values() {
        assert_eq!(log2_power_of_two(1), 0);
        assert_eq!(log2_power_of_two(1024), 10);
    }

    #[test]
    #[should_panic(expected = "expected a power-of-two dimension")]
    fn log2_power_of_two_rejects_non_powers() {
        let _ = log2_power_of_two(6);
    }

    #[test]
    fn checked_log2_power_of_two_values() {
        assert_eq!(checked_log2_power_of_two(0), None);
        assert_eq!(checked_log2_power_of_two(1), Some(0));
        assert_eq!(checked_log2_power_of_two(6), None);
        assert_eq!(checked_log2_power_of_two(1 << 20), Some(20));
    }
}
