use jolt_witness::witnesses::RemappedRamAddress;
use jolt_witness::WitnessBundle;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub const COLD: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct RamRaVirtualizationWitness {
    pub address: RemappedRamAddress,
}

const PACK_CHUNK: usize = 1 << 14;

pub fn packed_words(rows: &[RamRaVirtualizationWitness]) -> Result<Vec<u32>, u64> {
    let mut words = vec![COLD; rows.len()];
    #[cfg(feature = "parallel")]
    let rejected = words
        .par_chunks_mut(PACK_CHUNK)
        .zip(rows.par_chunks(PACK_CHUNK))
        .filter_map(|(slots, rows)| fill(slots, rows))
        .min();
    #[cfg(not(feature = "parallel"))]
    let rejected = words
        .chunks_mut(PACK_CHUNK)
        .zip(rows.chunks(PACK_CHUNK))
        .filter_map(|(slots, rows)| fill(slots, rows))
        .min();
    match rejected {
        Some(address) => Err(address),
        None => Ok(words),
    }
}

fn fill(slots: &mut [u32], rows: &[RamRaVirtualizationWitness]) -> Option<u64> {
    let mut rejected = None;
    for (slot, row) in slots.iter_mut().zip(rows) {
        match row.address.0 {
            None => *slot = COLD,
            Some(address) => match u32::try_from(address) {
                Ok(packed) if packed != COLD => *slot = packed,
                _ => rejected = Some(rejected.map_or(address, |seen: u64| seen.min(address))),
            },
        }
    }
    rejected
}
