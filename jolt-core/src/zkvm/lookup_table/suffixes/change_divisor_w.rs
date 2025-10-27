use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

pub enum ChangeDivisorWSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for ChangeDivisorWSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        let y_len = y.len().min(XLEN / 2);
        let (x, y) = (u64::from(x) as u32 as u64, u64::from(y) as u32 as u64);
        (((1u64 << y_len) - 1 == y) && x == 0).into()
    }
}
