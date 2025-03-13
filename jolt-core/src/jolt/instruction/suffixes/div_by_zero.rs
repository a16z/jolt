use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

pub enum DivByZeroSuffix {}

impl SparseDenseSuffix for DivByZeroSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (divisor, quotient) = b.uninterleave();
        let divisor_is_zero = u64::from(divisor) == 0;
        let quotient_is_all_ones = u64::from(quotient) == (1 << quotient.len()) - 1;
        (divisor_is_zero && quotient_is_all_ones).into()
    }
}
