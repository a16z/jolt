use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// Sign extension suffix that handles the upper word based on the sign of
/// the lower word right operand
/// If suffix length >= XLEN, it computes sign extension based on bit XLEN - 2
/// Otherwise returns 1
pub enum SignExtensionRightOperandSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for SignExtensionRightOperandSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() >= XLEN {
            let bits = u128::from(b);
            let sign_bit_position = XLEN - 2;
            let sign_bit = (bits >> sign_bit_position) & 1;

            if sign_bit == 1 {
                // Return all 1s in the upper half
                ((1u128 << XLEN) - (1u128 << (XLEN / 2))) as u64
            } else {
                // Return all 0s in the upper half
                0
            }
        } else {
            // Suffix is too small, return 1 (prefix will handle)
            1
        }
    }
}
