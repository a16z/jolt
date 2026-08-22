use jolt_field::{Fr, FromPrimitiveInt};
use jolt_witness::__private::TraceRow;
use jolt_witness::witnesses::{Extract, WitnessEnv};
use jolt_witness::witnesses::{RdAddress, RdWriteValue, Rs1Value, Rs2Value};
use jolt_witness::{WitnessBundle, WitnessError};

use crate::cuda::common::read_write_matrix::MatrixEntry;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RdPreValue(pub u64);

impl Extract for RdPreValue {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.registers.rd.map_or(0, |write| write.pre_value)))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rs1Address(pub Option<u8>);

impl Extract for Rs1Address {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.registers.rs1.map(|read| read.register)))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rs2Address(pub Option<u8>);

impl Extract for Rs2Address {
    fn extract(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self(row.registers.rs2.map(|read| read.register)))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, WitnessBundle)]
pub struct RegistersReadWriteWitness {
    pub rs1_address: Rs1Address,
    pub rs1_value: Rs1Value,
    pub rs2_address: Rs2Address,
    pub rs2_value: Rs2Value,
    pub rd_address: RdAddress,
    pub rd_pre_value: RdPreValue,
    pub rd_post_value: RdWriteValue,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the host construction is retained as the oracle for the device-side one"
    )
)]
pub fn matrix_entries(rows: &[RegistersReadWriteWitness], gamma: Fr) -> Vec<MatrixEntry> {
    let gamma_squared = gamma * gamma;
    let zero = Fr::from_u64(0);
    let mut entries = Vec::with_capacity(rows.len() * 3);
    let mut slots: [Option<MatrixEntry>; 3] = [None, None, None];

    for (cycle, row) in rows.iter().enumerate() {
        let mut len = 0usize;

        if let Some(rs1) = row.rs1_address.0 {
            slots[len] = Some(MatrixEntry {
                row: cycle as u32,
                col: u32::from(rs1),
                val_coeff: Fr::from_u64(row.rs1_value.0),
                prev_val: row.rs1_value.0,
                next_val: row.rs1_value.0,
                coeffs: [gamma, zero],
            });
            len += 1;
        }

        if let Some(rs2) = row.rs2_address.0 {
            if let Some(entry) = slots[..len]
                .iter_mut()
                .flatten()
                .find(|entry| entry.col == u32::from(rs2))
            {
                entry.coeffs[0] = gamma + gamma_squared;
            } else {
                {
                    slots[len] = Some(MatrixEntry {
                        row: cycle as u32,
                        col: u32::from(rs2),
                        val_coeff: Fr::from_u64(row.rs2_value.0),
                        prev_val: row.rs2_value.0,
                        next_val: row.rs2_value.0,
                        coeffs: [gamma_squared, zero],
                    });
                    len += 1;
                }
            }
        }

        if let Some(rd) = row.rd_address.0 {
            if let Some(entry) = slots[..len]
                .iter_mut()
                .flatten()
                .find(|entry| entry.col == u32::from(rd))
            {
                entry.coeffs[1] = Fr::from_u64(1);
                entry.next_val = row.rd_post_value.0;
            } else {
                {
                    slots[len] = Some(MatrixEntry {
                        row: cycle as u32,
                        col: u32::from(rd),
                        val_coeff: Fr::from_u64(row.rd_pre_value.0),
                        prev_val: row.rd_pre_value.0,
                        next_val: row.rd_post_value.0,
                        coeffs: [zero, Fr::from_u64(1)],
                    });
                    len += 1;
                }
            }
        }

        let mut present: Vec<MatrixEntry> = slots[..len].iter().flatten().copied().collect();
        present.sort_by_key(|entry| entry.col);
        entries.extend(present);
        slots = [None, None, None];
    }
    entries
}

#[cfg(test)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::{matrix_entries, RdPreValue, RegistersReadWriteWitness, Rs1Address, Rs2Address};

    #[test]
    fn collision_cases_use_the_documented_coefficients() {
        let gamma = Fr::from_u64(101);
        let gamma_squared = gamma * gamma;
        let one = Fr::from_u64(1);
        let zero = Fr::from_u64(0);

        let row = |rs1: Option<u8>, rs2: Option<u8>, rd: Option<u8>| RegistersReadWriteWitness {
            rs1_address: Rs1Address(rs1),
            rs1_value: jolt_witness::witnesses::Rs1Value(7),
            rs2_address: Rs2Address(rs2),
            rs2_value: jolt_witness::witnesses::Rs2Value(9),
            rd_address: jolt_witness::witnesses::RdAddress(rd),
            rd_pre_value: RdPreValue(11),
            rd_post_value: jolt_witness::witnesses::RdWriteValue(13),
        };

        let entries = matrix_entries(&[row(Some(5), Some(5), None)], gamma);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].coeffs, [gamma + gamma_squared, zero]);

        let entries = matrix_entries(&[row(Some(5), None, Some(5))], gamma);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].coeffs, [gamma, one]);
        assert_eq!(entries[0].prev_val, 7);
        assert_eq!(entries[0].next_val, 13);

        let entries = matrix_entries(&[row(Some(9), Some(4), Some(2))], gamma);
        assert_eq!(entries.len(), 3);
        assert_eq!(
            entries.iter().map(|e| e.col).collect::<Vec<_>>(),
            vec![2, 4, 9]
        );
        assert_eq!(entries[0].coeffs, [zero, one]);
        assert_eq!(entries[1].coeffs, [gamma_squared, zero]);
        assert_eq!(entries[2].coeffs, [gamma, zero]);
    }
}
