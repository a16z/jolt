use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

/// Suffix-owned SRLW sign-extension predicate `x_{w-1} * y_0`.
///
/// With `w = XLEN / 2` and `y = 2^w - 2^s`, this product is one exactly when
/// the source word has its sign bit set and `s = 0`. It is evaluated here
/// while both bits remain in the suffix. Once `x_{w-1}` moves into the prefix,
/// this suffix returns zero and `SrlwSextPrefix * LsbSuffix` carries the same
/// product across the split.
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
