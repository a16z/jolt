use crate::{
    field::JoltField, utils::lookup_bits::LookupBits,
    zkvm::instruction_lookups::read_raf_checking::current_suffix_len,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Prefix for the second term: r[0]*r[1]*...*r[j-1]
/// This processes overflow bits and possibly the sign bit, checking they are all 1
pub enum SignedOverflowBitsOnePrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for SignedOverflowBitsOnePrefix<XLEN> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        // j represents the number of bits already processed (0-indexed)
        // For 2*XLEN bits total:
        // - Bits 0 to XLEN-1 are overflow bits (processed when j < XLEN)
        // - Bit XLEN is the sign bit (processed when j == XLEN)
        // - Bits XLEN+1 to 2*XLEN-1 are the remaining lower bits
        
        // If j > XLEN, we've already processed overflow bits and sign bit
        if j > XLEN + 1 {
            return checkpoints[Prefixes::SignedOverflowBitsOne].unwrap_or(F::one());
        }

        
        let mut result = checkpoints[Prefixes::SignedOverflowBitsOne].unwrap_or(F::one());
        eprintln!("prefix_mle(): Initial checkpoint(One): {}", result);

        // 1671915034647199753218813458258621223137859435978673653030635308023746501760
        // Process the current bit - all overflow bits and sign bit should be 1
        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            // Current bit should be 1: r_x * y
            if j == XLEN + 1 {
                result *= r_x;
            }
            else {
                result *= r_x * y;
            }  
        } else {
            let x = F::from_u32(c);
            let y = F::from_u8(b.pop_msb());
            // Both bits should be 1: x * y
            if j == XLEN {
                result *= x;
            }
            else {
                result *= x * y;
            }
        }
        eprintln!("prefix_mle(): Updated result(One): {}", result);
        
        // Check the remaining unprocessed bits in b
        // We only need to check remaining overflow bits if they haven't all been processed yet
        // When j == XLEN, we're at the sign bit, and b contains lower bits that don't affect overflow
        if j < XLEN {
            while b.len() > 0 {
                result *= F::from_u8(b.pop_msb());
            }
            // let rest = u128::from(b);
            // if b.len() == 0 {
            //     return result;
            // }
            // let suffix_len = current_suffix_len(j);
            // // 0100111001010001
            
            // // We need to check that the overflow bits and sign bit (XLEN+1 bits total) are all 1s
            // // Shift left by suffix_len to align, then check if the top XLEN+1 bits are all 1s
            // let shifted = rest << suffix_len;
            // let mask = ((1u128 << (XLEN + 1)) - 1) << (2 * XLEN - XLEN - 1);
            // let temp = F::from_u64(((shifted & mask) == mask) as u64);
            // // println!("j in one is: {}", j);
            // // println!("Temp in one is: {}", temp);
            // // println!("b in one is: {}", b);
            // // println!("suffix_len in one is: {}", suffix_len);
            // result *= temp;
        }
        eprintln!("final prefix_mle(): Updated result(One): {}", result);
        
        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        // We update checkpoints every two rounds (on odd j):
        // - For j < XLEN, we're still in the overflow region → include two bits.
        // - For j == XLEN + 1, the pair contains the sign bit and one lower bit → include only the sign bit.
        // - For j > XLEN + 1, no further updates are needed for this prefix.
        let updated;
        if j < XLEN {
            updated = checkpoints[Prefixes::SignedOverflowBitsOne]
                .unwrap_or(F::one())
                * r_x
                * r_y;
        } 
        else if j == XLEN + 1 {
            updated = checkpoints[Prefixes::SignedOverflowBitsOne]
                .unwrap_or(F::one())
                * r_x;
        } 
        else {
            return checkpoints[Prefixes::SignedOverflowBitsOne].into();
        }
        eprintln!("update_prefix_checkpoint(): Updated checkpoint(One): {}", updated);
        
        
        Some(updated).into()
    }
}
