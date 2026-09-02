use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

pub enum XorRotL1PairsSuffix {}

impl SparseDenseSuffix for XorRotL1PairsSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let pairs = b.len() / 2;
        let (x, y) = b.uninterleave();
        let x = u64::from(x);
        let y = u64::from(y);

        if pairs == 64 {
            x ^ y.rotate_left(1)
        } else {
            (x ^ y.wrapping_shl(1)) & ((1 << pairs) - 1) & !1
        }
    }
}

pub enum TopYBitSuffix {}

impl SparseDenseSuffix for TopYBitSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let pairs = b.len() / 2;
        if pairs == 0 {
            return 0;
        }
        let (_, y) = b.uninterleave();
        (u64::from(y) >> (pairs - 1)) & 1
    }
}

pub enum BottomXBitSuffix {}

impl SparseDenseSuffix for BottomXBitSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() == 0 {
            return 0;
        }
        let (x, _) = b.uninterleave();
        u64::from(x) & 1
    }
}
