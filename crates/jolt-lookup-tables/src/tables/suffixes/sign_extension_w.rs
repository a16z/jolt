use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

/// Suffix-owned portion of the SRAW sign-fill term.
///
/// Let `w = XLEN / 2`, `a = x_{w-1}`, and let `y` be the bitmask
/// `2^w - 2^s`. Before any relevant bit leaves the suffix, this evaluates
///
/// `a * ((2^XLEN - 2^w) + sum_{i=1}^{w-1} 2^i * (1 - y_{w-1-i}))`.
///
/// If `a` is still present, this suffix applies it directly. After `a` moves
/// into the prefix, the suffix returns the remaining mask-dependent fill and
/// `VirtualSRAWTable::combine` multiplies it by `WordMsbPrefix`. Terms whose
/// `y` bits have also moved into the prefix are carried by
/// `SignExtensionWPrefix`.
pub enum SignExtensionWSuffix {}

impl SparseDenseSuffix for SignExtensionWSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.is_empty() {
            return 0;
        }

        let half = XLEN / 2;
        let (x, y) = b.uninterleave();
        let y_len = y.len();
        let mut fill = 0u128;

        if b.len() >= XLEN {
            let sign_bit = (u128::from(x) >> (half - 1)) & 1;
            if sign_bit == 0 {
                return 0;
            }
            fill = (1u128 << XLEN) - (1u128 << half);
        }

        let y_bits = u128::from(y);
        let count = y_len.min(half);
        let start_offset = y_len - count;
        let first_position = half - count;
        for offset in 0..count {
            let position = first_position + offset;
            if position == 0 {
                continue;
            }
            let y_bit = (y_bits >> (y_len - 1 - start_offset - offset)) & 1;
            fill += (1 - y_bit) << position;
        }

        fill as u64
    }
}
