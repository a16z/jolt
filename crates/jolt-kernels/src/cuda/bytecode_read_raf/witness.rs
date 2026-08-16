use jolt_witness::witnesses::MappedPc;
use jolt_witness::WitnessBundle;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub use crate::cuda::common::pack::COLD;
use crate::cuda::common::pack::PACK_CHUNK;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct BytecodeReadRafCycleWitness {
    pub pc: MappedPc,
}

pub fn packed_column(rows: &[BytecodeReadRafCycleWitness]) -> Result<Vec<u32>, usize> {
    let mut pc = vec![COLD; rows.len()];

    #[cfg(feature = "parallel")]
    let rejected = pc
        .par_chunks_mut(PACK_CHUNK)
        .zip(rows.par_chunks(PACK_CHUNK))
        .filter_map(|(pc, rows)| fill(pc, rows))
        .min();
    #[cfg(not(feature = "parallel"))]
    let rejected = pc
        .chunks_mut(PACK_CHUNK)
        .zip(rows.chunks(PACK_CHUNK))
        .filter_map(|(pc, rows)| fill(pc, rows))
        .min();

    match rejected {
        Some(value) => Err(value),
        None => Ok(pc),
    }
}

fn fill(pc: &mut [u32], rows: &[BytecodeReadRafCycleWitness]) -> Option<usize> {
    let mut rejected: Option<usize> = None;
    let mut reject = |value: usize| {
        rejected = Some(rejected.map_or(value, |seen: usize| seen.min(value)));
    };
    for (index, row) in rows.iter().enumerate() {
        match row.pc.0 {
            None => pc[index] = COLD,
            Some(slot) => match u32::try_from(slot) {
                Ok(packed) if packed != COLD => pc[index] = packed,
                _ => reject(slot),
            },
        }
    }
    rejected
}
