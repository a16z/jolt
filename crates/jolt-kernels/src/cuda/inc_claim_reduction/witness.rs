use jolt_witness::witnesses::{RamInc, RdInc};
use jolt_witness::WitnessBundle;

use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::DeviceFrVec;
use crate::cuda::common::error::CudaError;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct IncClaimReductionWitness {
    #[opening(committed = RamInc)]
    pub ram: RamInc,
    #[opening(committed = RdInc)]
    pub rd: RdInc,
}

pub struct Packed {
    pub ram: Vec<i128>,
    pub rd: Vec<i128>,
}

const PACK_CHUNK: usize = 1 << 14;

pub fn device_columns(
    context: &CudaKernelContext,
    rows: &[IncClaimReductionWitness],
) -> Result<Vec<DeviceFrVec>, CudaError> {
    let packed = packed_columns(rows);
    Ok(vec![
        context.i128_to_montgomery(&packed.ram)?,
        context.i128_to_montgomery(&packed.rd)?,
    ])
}

pub fn packed_columns(rows: &[IncClaimReductionWitness]) -> Packed {
    let mut ram = vec![0i128; rows.len()];
    let mut rd = vec![0i128; rows.len()];

    #[cfg(feature = "parallel")]
    ram.par_chunks_mut(PACK_CHUNK)
        .zip(rd.par_chunks_mut(PACK_CHUNK))
        .zip(rows.par_chunks(PACK_CHUNK))
        .for_each(|((ram, rd), rows)| fill(ram, rd, rows));
    #[cfg(not(feature = "parallel"))]
    ram.chunks_mut(PACK_CHUNK)
        .zip(rd.chunks_mut(PACK_CHUNK))
        .zip(rows.chunks(PACK_CHUNK))
        .for_each(|((ram, rd), rows)| fill(ram, rd, rows));

    Packed { ram, rd }
}

fn fill(ram: &mut [i128], rd: &mut [i128], rows: &[IncClaimReductionWitness]) {
    for (index, row) in rows.iter().enumerate() {
        ram[index] = row.ram.0;
        rd[index] = row.rd.0;
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_witness::witnesses::{RamInc, RdInc};

    use super::{device_columns, IncClaimReductionWitness};
    use crate::cuda::common::context::shared_context;

    fn sample_rows() -> Vec<IncClaimReductionWitness> {
        let increments: [(i128, i128); 6] = [
            (0, 0),
            (1, -1),
            (-1, 1),
            (u64::MAX as i128, -(u64::MAX as i128)),
            (-(u64::MAX as i128), u64::MAX as i128),
            (0, -(1i128 << 63)),
        ];
        increments
            .into_iter()
            .map(|(ram, rd)| IncClaimReductionWitness {
                ram: RamInc(ram),
                rd: RdInc(rd),
            })
            .collect()
    }

    #[test]
    fn sample_rows_exercise_both_increment_signs() {
        let rows = sample_rows();
        for (name, values) in [
            ("ram", rows.iter().map(|row| row.ram.0).collect::<Vec<_>>()),
            ("rd", rows.iter().map(|row| row.rd.0).collect::<Vec<_>>()),
        ] {
            assert!(
                values.iter().any(|value| *value < 0),
                "no synthetic row carries a negative {name} increment",
            );
            assert!(
                values.iter().any(|value| *value > 0),
                "no synthetic row carries a positive {name} increment",
            );
            assert!(
                values.contains(&0),
                "no synthetic row carries an idle {name} cycle",
            );
        }
    }

    #[test]
    fn synthetic_device_columns_match_the_host_conversion() {
        let Some(context) = shared_context() else {
            return;
        };
        let rows = sample_rows();
        let expected: Vec<Vec<Fr>> = vec![
            rows.iter().map(|row| Fr::from_i128(row.ram.0)).collect(),
            rows.iter().map(|row| Fr::from_i128(row.rd.0)).collect(),
        ];
        let got: Vec<Vec<Fr>> = device_columns(context, &rows)
            .expect("device columns")
            .iter()
            .map(|column| column.to_host().expect("download"))
            .collect();
        assert_eq!(got, expected, "packed device columns diverged");
    }
}
