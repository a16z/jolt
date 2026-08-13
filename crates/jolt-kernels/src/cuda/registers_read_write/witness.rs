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
#[expect(clippy::panic, reason = "test module: fixture errors fail loudly")]
mod tests {
    use ark_bn254::Fr as LegacyFr;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_legacy::field::JoltField as LegacyJoltField;
    use jolt_prover_legacy::subprotocols::read_write_matrix::{
        CycleMajorMatrixEntry, ReadWriteMatrixCycleMajor, RegistersCycleMajorEntry,
    };
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use strum::IntoEnumIterator;
    use tracer::instruction::Cycle;

    use super::{matrix_entries, RdPreValue, RegistersReadWriteWitness, Rs1Address, Rs2Address};

    const LOG_T: usize = 6;

    fn random_cycle(rng: &mut StdRng) -> Cycle {
        let variants: Vec<Cycle> = Cycle::iter().collect();
        for _ in 0..10_000 {
            let index = rng.next_u64() as usize % variants.len();
            let candidate = variants[index].random(rng);
            if jolt_prover_legacy::zkvm::instruction::JoltTraceCycle::try_new(&candidate).is_ok() {
                return candidate;
            }
        }
        panic!("no convertible cycle variant found");
    }

    #[test]
    fn matrix_entries_match_legacy_construction() {
        let mut rng = StdRng::seed_from_u64(13);
        let trace: Vec<Cycle> = (0..1usize << LOG_T)
            .map(|_| random_cycle(&mut rng))
            .collect();
        let gamma_raw = 101u64;
        let gamma = <LegacyFr as LegacyJoltField>::from_u64(gamma_raw);

        let legacy =
            ReadWriteMatrixCycleMajor::<LegacyFr, RegistersCycleMajorEntry<LegacyFr, _>>::new(
                &trace, gamma,
            )
            .deref_coeffs();

        let rows: Vec<RegistersReadWriteWitness> = trace
            .iter()
            .map(|cycle| RegistersReadWriteWitness {
                rs1_address: Rs1Address(cycle.rs1_read().map(|(r, _)| r)),
                rs1_value: jolt_witness::witnesses::Rs1Value(
                    cycle.rs1_read().map_or(0, |(_, v)| v),
                ),
                rs2_address: Rs2Address(cycle.rs2_read().map(|(r, _)| r)),
                rs2_value: jolt_witness::witnesses::Rs2Value(
                    cycle.rs2_read().map_or(0, |(_, v)| v),
                ),
                rd_address: jolt_witness::witnesses::RdAddress(cycle.rd_write().map(|(r, ..)| r)),
                rd_pre_value: RdPreValue(cycle.rd_write().map_or(0, |(_, pre, _)| pre)),
                rd_post_value: jolt_witness::witnesses::RdWriteValue(
                    cycle.rd_write().map_or(0, |(_, _, post)| post),
                ),
            })
            .collect();

        let got = matrix_entries(&rows, Fr::from(gamma));

        let expected: Vec<(u32, u32, Fr, u64, u64, Fr, Fr)> = legacy
            .entries
            .iter()
            .map(|entry| {
                (
                    CycleMajorMatrixEntry::row(entry) as u32,
                    CycleMajorMatrixEntry::column(entry) as u32,
                    Fr::from(entry.val_coeff),
                    entry.prev_val,
                    entry.next_val,
                    Fr::from(entry.ra_coeff),
                    Fr::from(entry.wa_coeff),
                )
            })
            .collect();
        let mine: Vec<(u32, u32, Fr, u64, u64, Fr, Fr)> = got
            .iter()
            .map(|e| {
                (
                    e.row,
                    e.col,
                    e.val_coeff,
                    e.prev_val,
                    e.next_val,
                    e.coeffs[0],
                    e.coeffs[1],
                )
            })
            .collect();
        assert_eq!(mine, expected);
    }

    #[test]
    fn collision_cases_use_legacy_lookup_table_values() {
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
