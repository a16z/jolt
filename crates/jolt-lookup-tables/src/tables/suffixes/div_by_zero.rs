use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// 1 if divisor (x) is all zeros AND quotient (y) is all ones; 0 otherwise.
pub enum DivByZeroSuffix {}

impl SparseDenseSuffix for DivByZeroSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (divisor, quotient) = b.uninterleave();
        let divisor_is_zero = u64::from(divisor) == 0;
        let quotient_is_all_ones = u64::from(quotient) == (1 << quotient.len()) - 1;
        (divisor_is_zero && quotient_is_all_ones).into()
    }
}
