use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// Left-shift W variant: processes lower 32 bits.
pub enum LeftShiftWSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for LeftShiftWSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        let y = LookupBits::new(u128::from(y), y.len().min(XLEN / 2));
        let (x, y_u) = (u64::from(x) as u32, u32::from(y));
        let x = x & !y_u;
        x.unbounded_shl(y.leading_ones()) as u64
    }
}
