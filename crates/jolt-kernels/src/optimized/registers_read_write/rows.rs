//! Typed per-cycle register rows and sparse-entry collection off the shared
//! trace record's register lanes.

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::JoltPolynomialId;
use jolt_field::JoltField;
use jolt_witness::__private::TraceRow;
use jolt_witness::witnesses::WitnessEnv;
use jolt_witness::{WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::super::trace_record::{RegisterLanes, NO_REGISTER};
use super::sparse::IndexedSparseEntry;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RegisterCycleRow {
    /// `(register, read value)`.
    pub rs1: Option<(u8, u64)>,
    /// `(register, read value)`.
    pub rs2: Option<(u8, u64)>,
    /// `(register, pre-write value, post-write value)`.
    pub rd: Option<(u8, u64, u64)>,
}

impl WitnessBundle for RegisterCycleRow {
    // `TraceRow` is nameable from this crate only through the doc-hidden
    // re-export the bundle derive uses; jolt-kernels deliberately has no
    // jolt-program dependency.
    fn from_row(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        let cycle = Self {
            rs1: row.rs1_index().map(|register| (register, row.rs1_value())),
            rs2: row.rs2_index().map(|register| (register, row.rs2_value())),
            rd: row
                .rd_index()
                .map(|register| (register, row.rd_pre_value(), row.rd_write_value())),
        };
        for register in [
            cycle.rs1.map(|(register, _)| register),
            cycle.rs2.map(|(register, _)| register),
            cycle.rd.map(|(register, ..)| register),
        ]
        .into_iter()
        .flatten()
        {
            if usize::from(register) >= 1usize << REGISTER_ADDRESS_BITS {
                return Err(WitnessError::InvalidWitnessData {
                    label: "jolt_vm",
                    reason: format!(
                        "register index {register} exceeds {REGISTER_ADDRESS_BITS}-bit register read-write domain"
                    ),
                });
            }
        }
        Ok(cycle)
    }

    fn annotated_ids() -> Vec<JoltPolynomialId> {
        Vec::new()
    }
}

impl RegisterCycleRow {
    /// The row from the record's register lanes — held alone by this kernel
    /// (the record's other lanes free before the sparse-entry build).
    #[inline]
    fn from_lanes(registers: &RegisterLanes, t: usize) -> Self {
        Self {
            rs1: (registers.rs1_index[t] != NO_REGISTER)
                .then(|| (registers.rs1_index[t], registers.rs1_value[t])),
            rs2: (registers.rs2_index[t] != NO_REGISTER)
                .then(|| (registers.rs2_index[t], registers.rs2_value[t])),
            rd: (registers.rd_index[t] != NO_REGISTER).then(|| {
                (
                    registers.rd_index[t],
                    registers.rd_pre_value[t],
                    registers.rd_post_value[t],
                )
            }),
        }
    }
}

/// Cross-member carry: the per-cycle `rd` hot indices, parked by this kernel's
/// `prepare` for the stage-5 val-evaluation kernel (which otherwise re-walks
/// the trace to collect them).
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct SharedRdIndices(pub Vec<Option<u8>>);

/// The sparse entries and companion per-cycle columns, built in one pass over
/// the record's register lanes.
pub(super) struct RegisterTables<F: JoltField> {
    pub(super) entries: Vec<IndexedSparseEntry<F>>,
    pub(super) inc: Vec<F>,
    pub(super) rs1_indices: Vec<Option<u8>>,
    pub(super) rs2_indices: Vec<Option<u8>>,
    pub(super) rd_indices: Vec<Option<u8>>,
}

/// The entry count [`RegisterCycleRow::entries`] produces for one cycle,
/// straight off the index lanes — the counting pass's cheap twin (kept
/// adjacent to [`RegisterCycleRow::from_lanes`] so the merge rules stay in
/// sync: rs2 folds into rs1's entry, rd into either read's).
#[cfg(feature = "parallel")]
#[inline]
fn cycle_entry_count(registers: &RegisterLanes, t: usize) -> usize {
    let rs1 = registers.rs1_index[t];
    let rs2 = registers.rs2_index[t];
    let rd = registers.rd_index[t];
    usize::from(rs1 != NO_REGISTER)
        + usize::from(rs2 != NO_REGISTER && rs2 != rs1)
        + usize::from(rd != NO_REGISTER && rd != rs1 && rd != rs2)
}

pub(super) fn build_register_tables_serial<F: JoltField>(
    registers: &RegisterLanes,
) -> RegisterTables<F> {
    let cycles = registers.rd_index.len();
    debug_assert!(cycles == 0 || u32::try_from(cycles - 1).is_ok());
    let rd_inc = |t: usize| {
        F::from_i128(i128::from(registers.rd_post_value[t]) - i128::from(registers.rd_pre_value[t]))
    };
    #[cfg(feature = "parallel")]
    let inc = (0..cycles).into_par_iter().map(rd_inc).collect();
    #[cfg(not(feature = "parallel"))]
    let inc = (0..cycles).map(rd_inc).collect();
    let mut tables = RegisterTables {
        entries: Vec::with_capacity(cycles * 3),
        inc,
        rs1_indices: Vec::with_capacity(cycles),
        rs2_indices: Vec::with_capacity(cycles),
        rd_indices: Vec::with_capacity(cycles),
    };
    for t in 0..cycles {
        let cycle = RegisterCycleRow::from_lanes(registers, t);
        let (cells, len) = cycle.entries::<F>(t as u32);
        tables.entries.extend_from_slice(&cells[..len]);
        tables.rs1_indices.push(cycle.rs1.map(|(k, _)| k));
        tables.rs2_indices.push(cycle.rs2.map(|(k, _)| k));
        tables.rd_indices.push(cycle.rd.map(|(k, ..)| k));
    }
    tables
}

#[cfg(feature = "parallel")]
pub(super) fn build_register_tables_parallel<F: JoltField>(
    registers: &RegisterLanes,
    chunk_size: usize,
) -> RegisterTables<F> {
    let cycles = registers.rd_index.len();
    debug_assert!(cycles == 0 || u32::try_from(cycles - 1).is_ok());
    let num_chunks = cycles.div_ceil(chunk_size);

    let chunk_counts: Vec<usize> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk| {
            let start = chunk * chunk_size;
            let end = (start + chunk_size).min(cycles);
            (start..end).map(|t| cycle_entry_count(registers, t)).sum()
        })
        .collect();

    let mut chunk_offsets = Vec::with_capacity(num_chunks + 1);
    chunk_offsets.push(0);
    for count in chunk_counts {
        let next = chunk_offsets[chunk_offsets.len() - 1] + count;
        chunk_offsets.push(next);
    }
    let entries_len = chunk_offsets[num_chunks];

    let mut tables = RegisterTables {
        entries: Vec::with_capacity(cycles * 3),
        inc: Vec::with_capacity(cycles),
        rs1_indices: Vec::with_capacity(cycles),
        rs2_indices: Vec::with_capacity(cycles),
        rd_indices: Vec::with_capacity(cycles),
    };

    let mut entry_chunks = Vec::with_capacity(num_chunks);
    let mut entries_rest = tables.entries.spare_capacity_mut();
    for offsets in chunk_offsets.windows(2) {
        let len = offsets[1] - offsets[0];
        let (chunk, rest) = entries_rest.split_at_mut(len);
        entry_chunks.push(chunk);
        entries_rest = rest;
    }

    entry_chunks
        .into_par_iter()
        .zip(tables.inc.spare_capacity_mut().par_chunks_mut(chunk_size))
        .zip(
            tables
                .rs1_indices
                .spare_capacity_mut()
                .par_chunks_mut(chunk_size),
        )
        .zip(
            tables
                .rs2_indices
                .spare_capacity_mut()
                .par_chunks_mut(chunk_size),
        )
        .zip(
            tables
                .rd_indices
                .spare_capacity_mut()
                .par_chunks_mut(chunk_size),
        )
        .enumerate()
        .for_each(
            |(chunk_index, ((((entries, inc), rs1_indices), rs2_indices), rd_indices))| {
                let start = chunk_index * chunk_size;
                let mut entry_index = 0;
                for local_t in 0..inc.len() {
                    let t = start + local_t;
                    let cycle = RegisterCycleRow::from_lanes(registers, t);
                    let (cells, len) = cycle.entries::<F>(t as u32);
                    debug_assert_eq!(len, cycle_entry_count(registers, t));
                    for cell in &cells[..len] {
                        let _ = entries[entry_index].write(*cell);
                        entry_index += 1;
                    }
                    let _ = inc[local_t].write(F::from_i128(
                        i128::from(registers.rd_post_value[t])
                            - i128::from(registers.rd_pre_value[t]),
                    ));
                    let _ = rs1_indices[local_t].write(cycle.rs1.map(|(k, _)| k));
                    let _ = rs2_indices[local_t].write(cycle.rs2.map(|(k, _)| k));
                    let _ = rd_indices[local_t].write(cycle.rd.map(|(k, ..)| k));
                }
                debug_assert_eq!(entry_index, entries.len());
            },
        );

    // SAFETY: every spare-capacity slot below the new lengths is partitioned
    // into one parallel chunk and initialized exactly once above.
    unsafe {
        tables.entries.set_len(entries_len);
        tables.inc.set_len(cycles);
        tables.rs1_indices.set_len(cycles);
        tables.rs2_indices.set_len(cycles);
        tables.rd_indices.set_len(cycles);
    }
    tables
}

#[cfg(feature = "parallel")]
pub(crate) fn register_build_chunk_size(cycles: usize) -> usize {
    const MAX_WORKERS: usize = 4;
    cycles.div_ceil(MAX_WORKERS).max(1)
}

pub(super) fn build_register_tables<F: JoltField>(registers: &RegisterLanes) -> RegisterTables<F> {
    #[cfg(feature = "parallel")]
    {
        if std::env::var_os("JOLT_REGISTERS_PREPARE_SERIAL").is_some() {
            build_register_tables_serial(registers)
        } else {
            build_register_tables_parallel(
                registers,
                register_build_chunk_size(registers.rd_index.len()),
            )
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        build_register_tables_serial(registers)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use common::jolt_device::MemoryLayout;
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_riscv::{
        CapturedState, JoltInstructionKind, JoltInstructionRow, JoltTraceRow, NonMemoryState,
        NormalizedOperands,
    };

    use super::*;

    #[test]
    fn rejects_register_outside_protocol_domain() {
        let instruction = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADDI,
            operands: NormalizedOperands {
                rs1: Some(200),
                ..Default::default()
            },
            ..Default::default()
        };
        let row = JoltTraceRow::from_components(
            CapturedState::NonMemory(NonMemoryState::default()),
            &instruction,
            0,
        )
        .unwrap();
        let preprocessing = JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::default(),
            ram: RAMPreprocessing::default(),
            memory_layout: MemoryLayout::default(),
            max_padded_trace_length: 1,
        };
        let env = WitnessEnv::new(&preprocessing);

        let error = RegisterCycleRow::from_row(&row, None, &env).unwrap_err();
        assert!(matches!(
            error,
            WitnessError::InvalidWitnessData { reason, .. }
                if reason.contains("register index 200")
        ));
    }
}
