use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// Partial offset indicator on suffix-owned bits. It is always zero or one.
pub enum ByteOffsetSuffix<const LANE: usize> {}

impl<const LANE: usize> SparseDenseSuffix for ByteOffsetSuffix<LANE> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let bits: u128 = b.into();
        for bit in 0..3 {
            let pos = 2 * bit;
            if pos < b.len() && ((bits >> pos) & 1) as usize != (LANE >> bit) & 1 {
                return 0;
            }
        }
        1
    }
}

/// Sum of suffix-owned lane bits times their partial offset indicators.
/// When any offset bit is absent, only lane zero can have remaining x bits.
pub enum ExtractByteUnsignedSuffix {}

impl SparseDenseSuffix for ExtractByteUnsignedSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        (u64::from(x) >> (8 * (u64::from(y) & 7))) & 0xff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::extract_byte_unsigned::ExtractByteUnsignedTable;
    use crate::tables::PrefixSuffixDecomposition;

    #[test]
    fn indicator_accumulator_contract() {
        let suffixes = ExtractByteUnsignedTable::<64>.suffixes();
        assert!(!suffixes[8].is_01_valued());
        for suffix in &suffixes[..8] {
            assert!(suffix.is_01_valued());
            // All relevant Boolean assignments, at every supported length.
            // Higher bits never enter the indicator; test them both clear/set.
            for len in 0..=128 {
                let mask = u128::MAX.checked_shr(128 - len as u32).unwrap_or(0);
                for low in 0..1u128 << len.min(6) {
                    for bits in [low, (mask & !63) | low] {
                        assert!(suffix.suffix_mle(LookupBits::new(bits, len)) <= 1);
                    }
                }
            }
        }
    }
}
