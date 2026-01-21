use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// 1 if the divisor (first operand) is all 0s and the quotient (second operand) is
/// all 1s; 0 otherwise. This is how the expected behavior for division-by-zero,
/// according to the RISC-V spec.
pub enum DivByZeroSuffix {}

impl SparseDenseSuffix for DivByZeroSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (divisor, quotient) = b.uninterleave();
        let divisor_is_zero = u64::from(divisor) == 0;
        let quotient_is_all_ones = u64::from(quotient) == (1 << quotient.len()) - 1;
        (divisor_is_zero && quotient_is_all_ones).into()
    }
}
