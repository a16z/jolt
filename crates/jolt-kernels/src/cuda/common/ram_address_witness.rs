use jolt_witness::witnesses::RemappedRamAddress;
use jolt_witness::WitnessBundle;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::cuda::common::pack::{encode_address, COLD, PACK_CHUNK};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct RamAddressWitness {
    pub address: RemappedRamAddress,
}

pub fn packed_ram_words(rows: &[RamAddressWitness], addresses: usize) -> Result<Vec<u32>, u64> {
    let mut words = vec![COLD; rows.len()];
    #[cfg(feature = "parallel")]
    let rejected = words
        .par_chunks_mut(PACK_CHUNK)
        .zip(rows.par_chunks(PACK_CHUNK))
        .filter_map(|(slots, rows)| fill(slots, rows, addresses))
        .min();
    #[cfg(not(feature = "parallel"))]
    let rejected = words
        .chunks_mut(PACK_CHUNK)
        .zip(rows.chunks(PACK_CHUNK))
        .filter_map(|(slots, rows)| fill(slots, rows, addresses))
        .min();
    match rejected {
        Some(address) => Err(address),
        None => Ok(words),
    }
}

fn fill(slots: &mut [u32], rows: &[RamAddressWitness], addresses: usize) -> Option<u64> {
    let mut rejected = None;
    for (slot, row) in slots.iter_mut().zip(rows) {
        match encode_address(row.address.0, addresses) {
            Ok(packed) => *slot = packed,
            Err(address) => {
                rejected = Some(rejected.map_or(address, |seen: u64| seen.min(address)));
            }
        }
    }
    rejected
}
