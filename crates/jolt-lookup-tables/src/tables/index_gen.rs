//! In-domain random lookup index generators for tables with constrained
//! operand shapes (shift/rotate tables whose right operand is a bitmask
//! produced earlier in the virtual sequence). Compiled for this crate's own
//! tests and for downstream fixtures via the `test-utils` feature.

use rand::prelude::*;

use crate::interleave::interleave_bits;

pub fn gen_bitmask_lookup_index<const XLEN: usize>(rng: &mut StdRng) -> u128 {
    let mask = ((1u128 << XLEN) - 1) as u64;
    let x = rng.next_u64() & mask;
    let zeros = rng.gen_range(0..=XLEN);
    let y_full = (!0u64).wrapping_shl(zeros as u32);
    let y = y_full & mask;
    interleave_bits(x, y)
}
