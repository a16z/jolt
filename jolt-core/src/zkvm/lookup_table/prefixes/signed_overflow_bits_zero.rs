use crate::{
    field::JoltField, utils::lookup_bits::LookupBits,
    zkvm::instruction_lookups::read_raf_checking::current_suffix_len,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

/// Prefix for the first term: (1-r[0])(1-r[1])...(1-r[j-1])
/// This processes overflow bits and possibly the sign bit, checking they are all 0
pub enum SignedOverflowBitsZeroPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: JoltField> SparseDensePrefix<F> for SignedOverflowBitsZeroPrefix<XLEN> {
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
            return checkpoints[Prefixes::SignedOverflowBitsZero].unwrap_or(F::one());
        }
        // else if j == XLEN + 1 {
        //     return checkpoints[Prefixes::SignedOverflowBitsZero].unwrap_or(F::one()) * ;
        // }
        
        let mut result = checkpoints[Prefixes::SignedOverflowBitsZero].unwrap_or(F::one());
        eprintln!("prefix_mle(): Initial checkpoint(Zero): {}", result);
        
        // Process the current bit - all overflow bits and sign bit should be 0
        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            // Current bit should be 0: (1 - r_x) * (1 - y)
            if j == XLEN + 1 {
                result *= F::one() - r_x
            }
            else {
                result *= (F::one() - r_x) * (F::one() - y);
            }
        } else {
            let x = F::from_u32(c);
            let y = F::from_u8(b.pop_msb());
            if j == XLEN {
                result *= F::one() - x;
            }
            else {
                // Current bit should be 0: (1 - x) * (1 - y)
                result *= (F::one() - x) * (F::one() - y);
            }
        }
        eprintln!("prefix_mle(): Updated result(Zero): {}", result);
        
        // Check the remaining unprocessed bits in b
        // We only need to check remaining overflow bits if they haven't all been processed yet
        // When j == XLEN, we're at the sign bit, and b contains lower bits that don't affect overflow
        if j < XLEN {
            let rest = u128::from(b);
            let suffix_len = current_suffix_len(j);
            
            // Shift left by suffix_len to align the bits, then right by XLEN-1 
            // to extract just the overflow bits and sign bit
            let temp = F::from_u64((((rest << suffix_len) >> (XLEN - 1)) == 0) as u64);
            // println!("j in zero is: {}", j);
            // println!("Temp in zero is: {}", temp);
            // println!("b in zero is: {}", b);
            // println!("suffix_len in zero is: {}", suffix_len);
            result *= temp;
        }
        
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
        // - For j == XLEN + 1, the pair contains the sign bit (first element of the pair) and one lower bit → include only the sign bit.
        // - For j > XLEN + 1, no further updates are needed for this prefix.
        let updated;
        if j < XLEN {
            updated = checkpoints[Prefixes::SignedOverflowBitsZero]
                .unwrap_or(F::one())
                * (F::one() - r_x)
                * (F::one() - r_y);
        } 
        else if j == XLEN + 1 {
            updated = checkpoints[Prefixes::SignedOverflowBitsZero]
                .unwrap_or(F::one())
                * (F::one() - r_x);
        } 
        else {
            return checkpoints[Prefixes::SignedOverflowBitsZero].into();
        }
        eprintln!("update_prefix_checkpoint(): Updated checkpoint(Zero): {}", updated);
        
        Some(updated).into()
    }
}
