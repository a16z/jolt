use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::tables::address_offset::AddressOffsetTable;
use crate::XLEN;

/// Suffix-window value of the width's offset bits (suffix windows start at
/// index bit 0).
pub enum AddressOffsetSuffix<const WIDTH: usize> {}

impl<const WIDTH: usize> SparseDenseSuffix for AddressOffsetSuffix<WIDTH> {
    fn suffix_mle(b: LookupBits) -> u64 {
        (u128::from(b) & AddressOffsetTable::<XLEN, WIDTH>::OFFSET_MASK) as u64
    }
}
