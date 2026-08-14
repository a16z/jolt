#[cfg(feature = "parallel")]
use rayon::prelude::*;

use jolt_witness::witnesses::LookupIndex;
use jolt_witness::WitnessBundle;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct InstructionRaVirtualizationWitness {
    pub lookup_index: LookupIndex,
}

fn split(row: &InstructionRaVirtualizationWitness, slot: &mut [u64]) {
    slot[0] = row.lookup_index.0 as u64;
    slot[1] = (row.lookup_index.0 >> 64) as u64;
}

pub fn packed_words(rows: &[InstructionRaVirtualizationWitness]) -> Vec<u64> {
    let mut words = vec![0u64; rows.len() * 2];
    #[cfg(feature = "parallel")]
    words
        .par_chunks_exact_mut(2)
        .zip(rows.par_iter())
        .for_each(|(slot, row)| split(row, slot));
    #[cfg(not(feature = "parallel"))]
    for (slot, row) in words.chunks_exact_mut(2).zip(rows.iter()) {
        split(row, slot);
    }
    words
}
