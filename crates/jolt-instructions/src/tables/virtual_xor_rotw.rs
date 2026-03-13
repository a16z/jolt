use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualXORROTWTable<const XLEN: usize, const ROTATION: u32>;

impl<const XLEN: usize, const ROTATION: u32> LookupTable<XLEN>
    for VirtualXORROTWTable<XLEN, ROTATION>
{
    fn materialize_entry(&self, index: u128) -> u64 {
        match XLEN {
            #[cfg(test)]
            8 => {
                let rotation = ROTATION as usize % (XLEN / 2);
                let (x_bits, y_bits) = uninterleave_bits(index);
                let x_lower = x_bits as u8 & 0x0F;
                let y_lower = y_bits as u8 & 0x0F;
                let xor_result = x_lower ^ y_lower;
                let rotated =
                    ((xor_result >> rotation) | (xor_result << (XLEN / 2 - rotation))) & 0x0F;
                rotated as u64
            }
            64 => {
                let (x, y) = uninterleave_bits(index);
                let x_32 = x as u32;
                let y_32 = y as u32;
                let xor_result = x_32 ^ y_32;
                xor_result.rotate_right(ROTATION) as u64
            }
            _ => panic!("{XLEN}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        for (idx, chunk) in r.chunks_exact(2).enumerate().skip(XLEN / 2) {
            let r_x = chunk[0];
            let r_y = chunk[1];
            let xor_bit = (F::one() - r_x) * r_y + r_x * (F::one() - r_y);
            let position = idx - (XLEN / 2);
            let mut rotated_position = (position + ROTATION as usize) % (XLEN / 2);
            rotated_position = (XLEN / 2) - 1 - rotated_position;
            result += F::from_u64(1u64 << rotated_position) * xor_bit;
        }
        result
    }
}

impl<const XLEN: usize, const ROTATION: u32> PrefixSuffixDecomposition<XLEN>
    for VirtualXORROTWTable<XLEN, ROTATION>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        debug_assert_eq!(XLEN, 64);
        match ROTATION {
            7 => vec![Suffixes::One, Suffixes::XorRotW7],
            8 => vec![Suffixes::One, Suffixes::XorRotW8],
            12 => vec![Suffixes::One, Suffixes::XorRotW12],
            16 => vec![Suffixes::One, Suffixes::XorRotW16],
            _ => unreachable!("unsupported rotation {ROTATION}"),
        }
    }

    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(XLEN, 64);
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, xor_rot] = suffixes.try_into().unwrap();
        match ROTATION {
            7 => prefixes[Prefixes::XorRotW7] * one + xor_rot,
            8 => prefixes[Prefixes::XorRotW8] * one + xor_rot,
            12 => prefixes[Prefixes::XorRotW12] * one + xor_rot,
            16 => prefixes[Prefixes::XorRotW16] * one + xor_rot,
            _ => unreachable!("unsupported rotation {ROTATION}"),
        }
    }
}
