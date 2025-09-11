use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::utils::uninterleave_bits;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualXORROTTable<const XLEN: usize, const ROTATION: u32>;

impl<const XLEN: usize, const ROTATION: u32> JoltLookupTable for VirtualXORROTTable<XLEN, ROTATION> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let xor_result = x ^ y;
        xor_result.rotate_right(ROTATION)
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * XLEN);

        let mut result = F::zero();
        for i in 0..XLEN {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];

            let rotated_position = (i + ROTATION as usize) % XLEN;
            let bit_position = XLEN - 1 - rotated_position;

            result += F::from_u64(1u64 << bit_position)
                * ((F::one() - x_i) * y_i + x_i * (F::one() - y_i));
        }
        result
    }
}

impl<const XLEN: usize, const ROTATION: u32> PrefixSuffixDecomposition<XLEN> for VirtualXORROTTable<XLEN, ROTATION> {
    fn suffixes(&self) -> Vec<Suffixes> {
        match ROTATION {
            16 => vec![Suffixes::One, Suffixes::XorRot16],
            24 => vec![Suffixes::One, Suffixes::XorRot24],
            32 => vec![Suffixes::One, Suffixes::XorRot32],
            63 => vec![Suffixes::One, Suffixes::XorRot63],
            _ => unimplemented!()
        }
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, xor_rot] = suffixes.try_into().unwrap();
        match ROTATION {
            16 => {
                prefixes[Prefixes::XorRot16] * one + xor_rot
            }
            24 => {
                prefixes[Prefixes::XorRot24] * one + xor_rot
            }
            32 => {
                prefixes[Prefixes::XorRot32] * one + xor_rot
            }
            63 => {
                prefixes[Prefixes::XorRot63] * one + xor_rot
            }
            _ => unimplemented!()
        }
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_64_xlen_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::VirtualXORROTTable;

    // Type alias for the common case of rotation by 32
    type VirtualXORROT32Table<const XLEN: usize> = VirtualXORROTTable<XLEN, 32>;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, VirtualXORROT32Table<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_64_xlen_test::<Fr, VirtualXORROT32Table<XLEN>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, VirtualXORROT32Table<XLEN>>();
    }
}
