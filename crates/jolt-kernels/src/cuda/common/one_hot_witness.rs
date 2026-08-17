use jolt_witness::witnesses::{LookupIndex, MappedPc, RemappedRamAddress};
use jolt_witness::WitnessBundle;

#[cfg(test)]
use crate::cuda::common::pack::{COLD, PACK_CHUNK};

#[cfg(all(test, feature = "parallel"))]
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct OneHotCycleWitness {
    pub lookup: LookupIndex,
    pub pc: MappedPc,
    pub ram: RemappedRamAddress,
}

#[cfg(test)]
pub struct PackedColumns {
    pub lookup: Vec<u64>,
    pub pc: Vec<u32>,
    pub ram: Vec<u32>,
}

#[cfg(test)]
pub fn packed_columns(rows: &[OneHotCycleWitness]) -> Result<PackedColumns, u128> {
    let mut lookup = vec![0u64; 2 * rows.len()];
    let mut pc = vec![COLD; rows.len()];
    let mut ram = vec![COLD; rows.len()];

    #[cfg(feature = "parallel")]
    let rejected = lookup
        .par_chunks_mut(2 * PACK_CHUNK)
        .zip(pc.par_chunks_mut(PACK_CHUNK))
        .zip(ram.par_chunks_mut(PACK_CHUNK))
        .zip(rows.par_chunks(PACK_CHUNK))
        .filter_map(|(((lookup, pc), ram), rows)| fill(lookup, pc, ram, rows))
        .min();
    #[cfg(not(feature = "parallel"))]
    let rejected = lookup
        .chunks_mut(2 * PACK_CHUNK)
        .zip(pc.chunks_mut(PACK_CHUNK))
        .zip(ram.chunks_mut(PACK_CHUNK))
        .zip(rows.chunks(PACK_CHUNK))
        .filter_map(|(((lookup, pc), ram), rows)| fill(lookup, pc, ram, rows))
        .min();

    match rejected {
        Some(value) => Err(value),
        None => Ok(PackedColumns { lookup, pc, ram }),
    }
}

#[cfg(test)]
fn fill(
    lookup: &mut [u64],
    pc: &mut [u32],
    ram: &mut [u32],
    rows: &[OneHotCycleWitness],
) -> Option<u128> {
    let mut rejected: Option<u128> = None;
    let mut reject = |value: u128| {
        rejected = Some(rejected.map_or(value, |seen: u128| seen.min(value)));
    };
    for (index, row) in rows.iter().enumerate() {
        let index128 = row.lookup.0;
        lookup[2 * index] = index128 as u64;
        lookup[2 * index + 1] = (index128 >> 64) as u64;

        match row.pc.0 {
            None => pc[index] = COLD,
            Some(slot) => match u32::try_from(slot) {
                Ok(packed) if packed != COLD => pc[index] = packed,
                _ => reject(slot as u128),
            },
        }

        match row.ram.0 {
            None => ram[index] = COLD,
            Some(address) => match u32::try_from(address) {
                Ok(packed) if packed != COLD => ram[index] = packed,
                _ => reject(u128::from(address)),
            },
        }
    }
    rejected
}
