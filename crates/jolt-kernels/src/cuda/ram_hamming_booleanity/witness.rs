use jolt_witness::witnesses::RamHammingWeight;
use jolt_witness::WitnessBundle;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct RamHammingBooleanityWitness {
    #[opening(RamHammingWeight)]
    pub weight: RamHammingWeight,
}

const PACK_CHUNK: usize = 1 << 14;

pub fn packed_weights(rows: &[RamHammingBooleanityWitness]) -> Vec<u64> {
    let mut weights = vec![0u64; rows.len()];
    #[cfg(feature = "parallel")]
    weights
        .par_chunks_mut(PACK_CHUNK)
        .zip(rows.par_chunks(PACK_CHUNK))
        .for_each(|(slots, rows)| fill(slots, rows));
    #[cfg(not(feature = "parallel"))]
    weights
        .chunks_mut(PACK_CHUNK)
        .zip(rows.chunks(PACK_CHUNK))
        .for_each(|(slots, rows)| fill(slots, rows));
    weights
}

fn fill(slots: &mut [u64], rows: &[RamHammingBooleanityWitness]) {
    for (slot, row) in slots.iter_mut().zip(rows) {
        *slot = u64::from(row.weight.0);
    }
}
