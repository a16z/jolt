use jolt_field::JoltField;

use super::{PrefixEval, Prefixes, SparseDensePrefix};
use crate::lookup_bits::LookupBits;

/// Offset indicator (WEIGHTED=false) or indicator times the prefix's byte
/// contribution (WEIGHTED=true). Checkpoints carry E and Q=E*L respectively.
pub enum ExtractBytePrefix<const LANE: usize, const WEIGHTED: bool> {}

impl<const LANE: usize, const WEIGHTED: bool> ExtractBytePrefix<LANE, WEIGHTED> {
    const OFFSETS: [Prefixes; 8] = [
        Prefixes::ByteOffset0,
        Prefixes::ByteOffset1,
        Prefixes::ByteOffset2,
        Prefixes::ByteOffset3,
        Prefixes::ByteOffset4,
        Prefixes::ByteOffset5,
        Prefixes::ByteOffset6,
        Prefixes::ByteOffset7,
    ];
    const VALUES: [Prefixes; 8] = [
        Prefixes::ExtractByte0,
        Prefixes::ExtractByte1,
        Prefixes::ExtractByte2,
        Prefixes::ExtractByte3,
        Prefixes::ExtractByte4,
        Prefixes::ExtractByte5,
        Prefixes::ExtractByte6,
        Prefixes::ExtractByte7,
    ];
}

impl<F: JoltField, const LANE: usize, const WEIGHTED: bool> SparseDensePrefix<F>
    for ExtractBytePrefix<LANE, WEIGHTED>
{
    fn default_checkpoint() -> F {
        if WEIGHTED {
            F::zero()
        } else {
            F::one()
        }
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let bits: u128 = b.into();
        for bit in 0..3 {
            let pos = 2 * bit;
            if pos >= suffix_len
                && pos < suffix_len + b.len()
                && ((bits >> (pos - suffix_len)) & 1) as usize != (LANE >> bit) & 1
            {
                return F::zero();
            }
        }
        let indicator = checkpoints[Self::OFFSETS[LANE]];
        if !WEIGHTED {
            return indicator;
        }
        let mut lane = 0u64;
        for bit in 0..8 {
            let pos = 2 * (8 * LANE + bit) + 1;
            if pos >= suffix_len && pos < suffix_len + b.len() {
                lane |= (((bits >> (pos - suffix_len)) & 1) as u64) << bit;
            }
        }
        checkpoints[Self::VALUES[LANE]] + indicator * F::from_u64(lane)
    }
}
