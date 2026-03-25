//! Bit-manipulation utilities on `usize`.

pub trait Math {
    /// Returns `2^self`.
    fn pow2(self) -> usize;
    /// Returns `floor(log2(self))`.
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
}
