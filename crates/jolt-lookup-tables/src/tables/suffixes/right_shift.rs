use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// Packs the masked bits of the left operand into the low bits.
pub enum RightShiftSuffix {}

impl SparseDenseSuffix for RightShiftSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (mut x, mut y) = b.uninterleave();
        let mut result = 0;
        while !x.is_empty() {
            let x_i = x.pop_msb();
            let y_i = y.pop_msb();
            result *= 1 + u64::from(y_i);
            result += u64::from(x_i * y_i);
        }
        result
    }
}
