use jolt_witness::witnesses::{RdAddress, RdInc};
use jolt_witness::WitnessBundle;

use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::DeviceFrVec;
use crate::cuda::common::error::CudaError;
use crate::cuda::common::pack::device_inc_and_address;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct RegistersValEvaluationWitness {
    #[opening(committed = RdInc)]
    pub inc: RdInc,
    pub address: RdAddress,
}

pub fn device_columns(
    context: &CudaKernelContext,
    rows: &[RegistersValEvaluationWitness],
    registers: usize,
) -> Result<(DeviceFrVec, Vec<u32>), CudaError> {
    device_inc_and_address(context, rows, registers, |row| {
        (row.inc.0, row.address.0.map(u64::from))
    })
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_witness::witnesses::{RdAddress, RdInc};

    use super::{device_columns, RegistersValEvaluationWitness};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::pack::COLD;

    const REGISTERS: usize = 1 << 7;

    fn sample_rows() -> Vec<RegistersValEvaluationWitness> {
        let rows: [(i128, Option<u8>); 6] = [
            (0, None),
            (1, Some(0)),
            (-1, Some(REGISTERS as u8 - 1)),
            (i128::from(u64::MAX), Some(31)),
            (-i128::from(u64::MAX), None),
            (-(1i128 << 63), Some(1)),
        ];
        rows.into_iter()
            .map(|(inc, address)| RegistersValEvaluationWitness {
                inc: RdInc(inc),
                address: RdAddress(address),
            })
            .collect()
    }

    #[test]
    fn sample_rows_cover_idle_cycles_and_both_increment_signs() {
        let rows = sample_rows();
        assert!(
            rows.iter().any(|row| row.address.0.is_none()),
            "no synthetic row leaves rd unwritten",
        );
        assert!(
            rows.iter().any(|row| row.address.0.is_some()),
            "no synthetic row writes rd",
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
            .map(|row| row.address.0.map_or(COLD, u32::from))
            .collect();
        let (inc, address) = device_columns(context, &rows, REGISTERS).expect("device columns");
        assert_eq!(
            inc.to_host().expect("download"),
            expected_inc,
            "packed increment column diverged"
        );
        assert_eq!(address, expected_address, "packed address column diverged");
    }

    #[test]
    fn a_register_beyond_the_address_space_is_rejected() {
        let Some(context) = shared_context() else {
            return;
        };
        let rows = vec![RegistersValEvaluationWitness {
            inc: RdInc(0),
            address: RdAddress(Some(REGISTERS as u8)),
        }];
        assert!(
            device_columns(context, &rows, REGISTERS).is_err(),
            "an out-of-range register was packed instead of rejected",
        );
    }
}
