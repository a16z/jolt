use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::jolt::lookup_table::prefixes::Prefixes;
use crate::subprotocols::sparse_dense_shout::LookupBits;
use crate::utils::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct VirtualRotrTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for VirtualRotrTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        let mut x_bits = LookupBits::new(x as u64, WORD_SIZE);
        let mut y_bits = LookupBits::new(y as u64, WORD_SIZE);

        // First collect all bits to determine rotation amount
        let mut x_arr = [0u8; 32]; // Max WORD_SIZE
        let mut y_arr = [0u8; 32];
        for i in 0..WORD_SIZE {
            x_arr[i] = x_bits.pop_msb();
            y_arr[i] = y_bits.pop_msb();
        }

        // Count trailing zeros in y (from LSB side)
        let mut rotation = 0;
        for i in (0..WORD_SIZE).rev() {
            if y_arr[i] == 0 {
                rotation += 1;
            } else {
                break;
            }
        }

        // Build rotated result bit by bit from MSB
        let mut entry = 0;
        for i in 0..WORD_SIZE {
            entry <<= 1;
            // For ROTR by k: bit at position i comes from position (i + k) % WORD_SIZE
            let src_idx = (i + rotation) % WORD_SIZE;
            entry |= x_arr[src_idx] as u64;
        }

        entry
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        
        let mut result = F::zero();
        
        // For ROTR, we need to compute the multilinear extension that captures rotation
        // Since rotation amount is determined by trailing zeros in y, we sum over all
        // possible rotation amounts weighted by their probability
        
        // Special case: when y = 0 (all bits are 0), rotation is 0
        // We handle rotations from 0 to WORD_SIZE
        for rotation in 0..=WORD_SIZE {
            // Compute probability that y has exactly 'rotation' trailing zeros
            let mut rotation_indicator = F::one();
            
            if rotation == WORD_SIZE {
                // All bits of y should be 0
                for i in 0..WORD_SIZE {
                    let y_bit = r[2 * i + 1];
                    rotation_indicator *= F::one() - y_bit;
                }
            } else {
                // Check each bit position from LSB to MSB
                for bit_pos in 0..WORD_SIZE {
                    let y_idx = WORD_SIZE - 1 - bit_pos;
                    let y_bit = r[2 * y_idx + 1];
                    
                    if bit_pos < rotation {
                        // This bit should be 0 for trailing zeros
                        rotation_indicator *= F::one() - y_bit;
                    } else if bit_pos == rotation {
                        // First non-zero bit
                        rotation_indicator *= y_bit;
                        break;
                    }
                }
            }
            
            // Compute rotated value for this rotation amount
            // Note: rotation by WORD_SIZE is same as rotation by 0
            let effective_rotation = rotation % WORD_SIZE;
            let mut rotated_value = F::zero();
            for out_bit in 0..WORD_SIZE {
                // Source bit for this output position
                let src_bit = (out_bit + effective_rotation) % WORD_SIZE;
                let x_bit = r[2 * src_bit];
                
                // Add contribution of this bit (MSB first)
                rotated_value = rotated_value * F::from_u64(2u64) + x_bit;
            }
            
            result += rotation_indicator * rotated_value;
        }
        
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for VirtualRotrTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::RightShiftHelper, Suffixes::Rotr]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [helper, rotr] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Rotr] * helper + prefixes[Prefixes::RotrHelper] * rotr
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::VirtualRotrTable;
    use crate::jolt::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, VirtualRotrTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, VirtualRotrTable<32>>();
    }

    #[test]
    #[ignore = "Cannot generate lookup_index at random"]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, VirtualRotrTable<32>>();
    }
}
