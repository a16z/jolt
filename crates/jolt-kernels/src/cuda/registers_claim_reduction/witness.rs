#[cfg(test)]
use jolt_witness::witnesses::{RdWriteValue, Rs1Value, Rs2Value};
#[cfg(test)]
use jolt_witness::WitnessBundle;

#[cfg(test)]
use crate::cuda::common::context::CudaKernelContext;
#[cfg(test)]
use crate::cuda::common::device::DeviceFrVec;
#[cfg(test)]
use crate::cuda::common::error::CudaError;

#[cfg(all(test, feature = "parallel"))]
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
#[cfg(test)]
pub struct RegistersClaimReductionWitness {
    #[opening(RdWriteValue)]
    pub rd_write: RdWriteValue,
    #[opening(Rs1Value)]
    pub rs1: Rs1Value,
    #[opening(Rs2Value)]
    pub rs2: Rs2Value,
}

#[cfg(test)]
pub struct Packed {
    pub rd_write: Vec<u64>,
    pub rs1: Vec<u64>,
    pub rs2: Vec<u64>,
}

#[cfg(test)]
const PACK_CHUNK: usize = 1 << 14;

#[cfg(test)]
pub fn device_columns(
    context: &CudaKernelContext,
    rows: &[RegistersClaimReductionWitness],
) -> Result<Vec<DeviceFrVec>, CudaError> {
    let packed = packed_columns(rows);
    Ok(vec![
        context.u64_to_montgomery(&packed.rd_write)?,
        context.u64_to_montgomery(&packed.rs1)?,
        context.u64_to_montgomery(&packed.rs2)?,
    ])
}

#[cfg(test)]
pub fn packed_columns(rows: &[RegistersClaimReductionWitness]) -> Packed {
    let mut rd_write = vec![0u64; rows.len()];
    let mut rs1 = vec![0u64; rows.len()];
    let mut rs2 = vec![0u64; rows.len()];

    #[cfg(feature = "parallel")]
    rd_write
        .par_chunks_mut(PACK_CHUNK)
        .zip(rs1.par_chunks_mut(PACK_CHUNK))
        .zip(rs2.par_chunks_mut(PACK_CHUNK))
        .zip(rows.par_chunks(PACK_CHUNK))
        .for_each(|(((rd, one), two), rows)| fill(rd, one, two, rows));
    #[cfg(not(feature = "parallel"))]
    rd_write
        .chunks_mut(PACK_CHUNK)
        .zip(rs1.chunks_mut(PACK_CHUNK))
        .zip(rs2.chunks_mut(PACK_CHUNK))
        .zip(rows.chunks(PACK_CHUNK))
        .for_each(|(((rd, one), two), rows)| fill(rd, one, two, rows));

    Packed { rd_write, rs1, rs2 }
}

#[cfg(test)]
fn fill(rd: &mut [u64], one: &mut [u64], two: &mut [u64], rows: &[RegistersClaimReductionWitness]) {
    for (index, row) in rows.iter().enumerate() {
        rd[index] = row.rd_write.0;
        one[index] = row.rs1.0;
        two[index] = row.rs2.0;
    }
}
