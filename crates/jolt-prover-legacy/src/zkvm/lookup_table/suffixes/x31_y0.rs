use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

pub enum X31Y0Suffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for X31Y0Suffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() < XLEN {
            return 0;
        }
        let (x, y) = b.uninterleave();
        (((u128::from(x) >> (XLEN / 2 - 1)) & 1) * (u128::from(y) & 1)) as u64
    }
}
