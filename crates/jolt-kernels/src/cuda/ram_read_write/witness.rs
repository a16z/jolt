use jolt_field::{Fr, FromPrimitiveInt};
use jolt_witness::witnesses::{RamReadValue, RamWriteValue, RemappedRamAddress};
use jolt_witness::WitnessBundle;

use crate::cuda::common::read_write_matrix::MatrixEntry;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct RamReadWriteWitness {
    pub address: RemappedRamAddress,
    pub read_value: RamReadValue,
    pub write_value: RamWriteValue,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the host construction is retained as the oracle for the device-side one"
    )
)]
pub fn matrix_entries(rows: &[RamReadWriteWitness]) -> Vec<MatrixEntry> {
    let one = Fr::from_u64(1);
    let zero = Fr::from_u64(0);
    let mut entries = Vec::with_capacity(rows.len());
    for (cycle, row) in rows.iter().enumerate() {
        let Some(address) = row.address.0 else {
            continue;
        };
        entries.push(MatrixEntry {
            row: cycle as u32,
            col: address as u32,
            val_coeff: Fr::from_u64(row.read_value.0),
            prev_val: row.read_value.0,
            next_val: row.write_value.0,
            coeffs: [one, zero],
        });
    }
    entries
}
