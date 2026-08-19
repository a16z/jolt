#[cfg(test)]
use jolt_witness::witnesses::{RamInc, RemappedRamAddress};
#[cfg(test)]
use jolt_witness::WitnessBundle;

#[cfg(test)]
use crate::cuda::common::context::CudaKernelContext;
#[cfg(test)]
use crate::cuda::common::device::DeviceFrVec;
#[cfg(test)]
use crate::cuda::common::error::CudaError;
#[cfg(test)]
use crate::cuda::common::pack::device_inc_and_address;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct RamValCheckWitness {
    #[opening(committed = RamInc)]
    pub inc: RamInc,
    pub address: RemappedRamAddress,
}

#[cfg(test)]
pub fn device_columns(
    context: &CudaKernelContext,
    rows: &[RamValCheckWitness],
    addresses: usize,
) -> Result<(DeviceFrVec, Vec<u32>), CudaError> {
    device_inc_and_address(context, rows, addresses, |row| (row.inc.0, row.address.0))
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_witness::witnesses::{RamInc, RemappedRamAddress};

    use super::{device_columns, RamValCheckWitness};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::pack::COLD;

    const ADDRESSES: usize = 1 << 8;

    fn sample_rows() -> Vec<RamValCheckWitness> {
        let rows: [(i128, Option<u64>); 6] = [
            (0, None),
            (1, Some(0)),
            (-1, Some(ADDRESSES as u64 - 1)),
            (i128::from(u64::MAX), Some(7)),
            (-i128::from(u64::MAX), None),
            (-(1i128 << 63), Some(1)),
        ];
        rows.into_iter()
            .map(|(inc, address)| RamValCheckWitness {
                inc: RamInc(inc),
                address: RemappedRamAddress(address),
            })
            .collect()
    }

    #[test]
    fn sample_rows_cover_cold_cycles_and_both_increment_signs() {
        let rows = sample_rows();
        assert!(
            rows.iter().any(|row| row.address.0.is_none()),
            "no synthetic row is a cold cycle",
        );
        assert!(
            rows.iter().any(|row| row.address.0.is_some()),
            "no synthetic row touches RAM",
        );
        assert!(
            rows.iter().any(|row| row.inc.0 < 0),
            "no synthetic row carries a negative increment",
        );
        assert!(
            rows.iter().any(|row| row.inc.0 > 0),
            "no synthetic row carries a positive increment",
        );
        assert!(
            rows.iter().any(|row| row.inc.0 == 0),
            "no synthetic row carries an idle increment",
        );
    }

    #[test]
    fn synthetic_device_columns_match_the_host_conversion() {
        let Some(context) = shared_context() else {
            return;
        };
        let rows = sample_rows();
        let expected_inc: Vec<Fr> = rows.iter().map(|row| Fr::from_i128(row.inc.0)).collect();
        let expected_address: Vec<u32> = rows
            .iter()
            .map(|row| row.address.0.map_or(COLD, |address| address as u32))
            .collect();
        let (inc, address) = device_columns(context, &rows, ADDRESSES).expect("device columns");
        assert_eq!(
            inc.to_host().expect("download"),
            expected_inc,
            "packed increment column diverged"
        );
        assert_eq!(address, expected_address, "packed address column diverged");
    }

    #[test]
    fn an_address_beyond_the_ram_space_is_rejected() {
        let Some(context) = shared_context() else {
            return;
        };
        let rows = vec![RamValCheckWitness {
            inc: RamInc(0),
            address: RemappedRamAddress(Some(ADDRESSES as u64)),
        }];
        assert!(
            device_columns(context, &rows, ADDRESSES).is_err(),
            "an out-of-range RAM address was packed instead of rejected",
        );
    }
}
