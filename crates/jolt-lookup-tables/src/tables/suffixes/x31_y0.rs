use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

pub enum X31Y0Suffix {}

impl SparseDenseSuffix for X31Y0Suffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() < XLEN {
            return 0;
        }

        let (x, y) = b.uninterleave();
        ((u128::from(x) >> (XLEN / 2 - 1)) & 1) as u64 * (u128::from(y) & 1) as u64
    }
}
