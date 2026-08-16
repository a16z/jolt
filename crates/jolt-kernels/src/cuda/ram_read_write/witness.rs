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

#[cfg(test)]
mod tests {
    use ark_bn254::Fr as LegacyFr;
    use common::jolt_device::MemoryLayout;
    use jolt_field::Fr;
    use jolt_prover_legacy::subprotocols::read_write_matrix::{
        CycleMajorMatrixEntry, RamCycleMajorEntry, ReadWriteMatrixCycleMajor,
    };
    use jolt_witness::witnesses::{RamReadValue, RamWriteValue, RemappedRamAddress};
    use tracer::instruction::lw::LW;
    use tracer::instruction::sw::SW;
    use tracer::instruction::{Cycle, RAMAccess, RAMRead, RAMWrite, RISCVCycle};

    use super::{matrix_entries, RamReadWriteWitness};

    const LOG_T: usize = 6;
    const RAM_K: usize = 32;

    fn read_cycle(address: u64, value: u64) -> Cycle {
        Cycle::LW(RISCVCycle::<LW> {
            ram_access: RAMRead { address, value },
            ..Default::default()
        })
    }

    fn write_cycle(address: u64, pre_value: u64, post_value: u64) -> Cycle {
        Cycle::SW(RISCVCycle::<SW> {
            ram_access: RAMWrite {
                address,
                pre_value,
                post_value,
            },
            ..Default::default()
        })
    }

    fn trace() -> Vec<Cycle> {
        (0..1usize << LOG_T)
            .map(|cycle| {
                let word = 1 + (cycle as u64 * 5) % (RAM_K as u64 - 1);
                let address = 8 * word;
                match cycle % 4 {
                    0 => Cycle::NoOp,
                    1 => read_cycle(address, 900 + cycle as u64),
                    2 => write_cycle(address, 100 + cycle as u64, 700 + cycle as u64),
                    _ => write_cycle(address, 400 + cycle as u64, 400 + cycle as u64),
                }
            })
            .collect()
    }

    fn rows(trace: &[Cycle], layout: &MemoryLayout) -> Vec<RamReadWriteWitness> {
        trace
            .iter()
            .map(|cycle| {
                let access = cycle.ram_access();
                let address = match access {
                    RAMAccess::Read(read) => Some(read.address),
                    RAMAccess::Write(write) => Some(write.address),
                    RAMAccess::NoOp => None,
                };
                let (read_value, write_value) = match access {
                    RAMAccess::Read(read) => (read.value, read.value),
                    RAMAccess::Write(write) => (write.pre_value, write.post_value),
                    RAMAccess::NoOp => (0, 0),
                };
                RamReadWriteWitness {
                    address: RemappedRamAddress(
                        address
                            .and_then(|address| layout.remap_word_address(address).ok().flatten()),
                    ),
                    read_value: RamReadValue(read_value),
                    write_value: RamWriteValue(write_value),
                }
            })
            .collect()
    }

    #[test]
    fn matrix_entries_match_legacy_construction() {
        let layout = MemoryLayout::default();
        let trace = trace();
        let val_init = vec![<LegacyFr as jolt_prover_legacy::field::JoltField>::from_u64(0); RAM_K];
        let legacy = ReadWriteMatrixCycleMajor::<LegacyFr, RamCycleMajorEntry<LegacyFr>>::new(
            &trace, val_init, &layout,
        );

        let got = matrix_entries(&rows(&trace, &layout));

        let expected: Vec<(u32, u32, Fr, u64, u64, Fr)> = legacy
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
                )
            })
            .collect();
        let got: Vec<(u32, u32, Fr, u64, u64, Fr)> = got
            .iter()
            .map(|entry| {
                (
                    entry.row,
                    entry.col,
                    entry.val_coeff,
                    entry.prev_val,
                    entry.next_val,
                    entry.coeffs[0],
                )
            })
            .collect();

        assert!(
            !expected.is_empty(),
            "fixture produced no RAM accesses to compare",
        );
        assert_eq!(got, expected);
    }
}
