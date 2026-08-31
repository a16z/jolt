#![expect(clippy::unwrap_used, reason = "tests use checked fixtures")]

use std::mem::size_of;

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_field::{Ring as _, Zero as _};

use super::{
    carrier::AddressMajorShape,
    worklist::{BYTECODE_ADDRESS_BASE_STAGES, BYTECODE_ADDRESS_PUSHFORWARD_STAGES},
    BytecodeAddressChunkDescriptor, BytecodeAddressFusedScatterRequest,
    BytecodeAddressStage1TopologyScratch,
};
use crate::metal::solinas::{
    BooleanityRow, InstructionReadRafCompatibilityScatterConfig, MetalError, SolinasMetal,
    INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS, INSTRUCTION_READ_RAF_SEGMENTS,
};

#[derive(Clone, Copy)]
struct TestRow {
    mapped_pc: Option<usize>,
    fused_inc_magnitude: u64,
    fused_inc_negative: bool,
}

impl TestRow {
    fn push_pc(self) -> usize {
        self.mapped_pc.unwrap_or(0)
    }
}

fn tables(shape: AddressMajorShape) -> (Vec<Vec<AkitaField>>, Vec<Vec<AkitaField>>) {
    const MODULUS: u128 = 0xffff_ffff_ffff_ffff_ffff_ffff_0000_5809;
    let e_lo = (0..BYTECODE_ADDRESS_PUSHFORWARD_STAGES)
        .map(|stage| {
            (0..shape.inner_length().unwrap())
                .map(|inner| {
                    let delta = (17 * stage + 5 * inner + 1) as u128;
                    if (stage + inner).is_multiple_of(3) {
                        AkitaField::from_u128(MODULUS - delta)
                    } else {
                        AkitaField::from_u128((1_u128 << 127) + delta)
                    }
                })
                .collect()
        })
        .collect();
    let e_hi = (0..BYTECODE_ADDRESS_PUSHFORWARD_STAGES)
        .map(|stage| {
            (0..shape.outer_length().unwrap())
                .map(|outer| {
                    let delta = (19 * stage + 7 * outer + 3) as u128;
                    if (stage + outer).is_multiple_of(2) {
                        AkitaField::from_u128(MODULUS - delta)
                    } else {
                        AkitaField::from_u128((1_u128 << 126) + delta)
                    }
                })
                .collect()
        })
        .collect();
    (e_lo, e_hi)
}

fn direct_oracle(
    shape: AddressMajorShape,
    rows: &[TestRow],
    e_lo: &[Vec<AkitaField>],
    e_hi: &[Vec<AkitaField>],
) -> Vec<AkitaField> {
    let addresses = shape.addresses().unwrap();
    let inner_length = shape.inner_length().unwrap();
    let mut output = vec![AkitaField::zero(); BYTECODE_ADDRESS_PUSHFORWARD_STAGES * addresses];
    for (index, row) in rows.iter().copied().enumerate() {
        let outer = index / inner_length;
        let inner = index % inner_length;
        let address = row.push_pc();
        let increment = if row.fused_inc_negative {
            AkitaField::zero() - AkitaField::from_u64(row.fused_inc_magnitude)
        } else {
            AkitaField::from_u64(row.fused_inc_magnitude)
        };
        for stage in 0..BYTECODE_ADDRESS_PUSHFORWARD_STAGES {
            let mut term = e_lo[stage][inner] * e_hi[stage][outer];
            if stage >= BYTECODE_ADDRESS_BASE_STAGES {
                term *= increment;
            }
            output[stage * addresses + address] += term;
        }
    }
    output
}

#[test]
#[cfg(target_os = "macos")]
fn fused_stage1_scatter_matches_padded_domain_oracle_across_rank_wraps() {
    let context = SolinasMetal::for_akita().unwrap();
    let shape = AddressMajorShape::production(16).unwrap();
    let physical_rows = 50_123;
    let rows = (0..shape.rows().unwrap())
        .map(|index| {
            if index >= physical_rows {
                return TestRow {
                    mapped_pc: None,
                    fused_inc_magnitude: 0,
                    fused_inc_negative: false,
                };
            }
            let mapped_pc = if index < INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS {
                Some(7)
            } else {
                match index % 5 {
                    0 => None,
                    1 => Some(1),
                    2 => Some(7),
                    3 => Some(8191),
                    _ => Some(31),
                }
            };
            TestRow {
                mapped_pc,
                fused_inc_magnitude: if [255, 256, 4095, 4096, physical_rows - 1].contains(&index) {
                    u64::MAX
                } else {
                    (17 * index + 11) as u64
                },
                fused_inc_negative: index.is_multiple_of(3),
            }
        })
        .collect::<Vec<_>>();
    let resident_rows = rows
        .iter()
        .map(|row| {
            let magnitude = i128::from(row.fused_inc_magnitude);
            BooleanityRow::new(
                0,
                row.mapped_pc.map(|pc| pc as u64),
                None,
                if row.fused_inc_negative {
                    -magnitude
                } else {
                    magnitude
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut source = context
        .prepare_instruction_read_raf_stage1_storage(shape.rows().unwrap())
        .unwrap();
    let mut topology = context
        .prepare_bytecode_address_stage1_topology_storage(shape.rows().unwrap(), physical_rows)
        .unwrap();
    topology
        .with_chunk_writers(|topology_chunks| {
            source.with_chunk_writers(|source_chunks| {
                let mut scratch = BytecodeAddressStage1TopologyScratch::new();
                for (chunk, (source, topology)) in source_chunks
                    .iter_mut()
                    .zip(topology_chunks.iter_mut())
                    .enumerate()
                {
                    let start = chunk * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
                    for offset in 0..source.len() {
                        let row = start + offset;
                        let rank = if row < physical_rows {
                            topology.record(&mut scratch, rows[row].mapped_pc.unwrap_or(0))?
                        } else {
                            0
                        };
                        source.push_with_bytecode_chunk_rank(resident_rows[row], 0, false, rank)?;
                    }
                    topology.finish(&mut scratch)?;
                }
                Ok(())
            })
        })
        .unwrap();
    let owner = source.seal().unwrap();
    let source_receipt = owner.receipt();
    let topology_owner = topology.seal(&owner).unwrap();
    let topology_receipt = topology_owner.receipt();
    assert_eq!(topology_receipt.max_pivots_per_chunk(), 15);
    assert_eq!(topology_receipt.max_descriptors_per_chunk(), 5);

    let scatter_source = owner
        .lease(shape.rows().unwrap(), context.device_registry_id())
        .unwrap();
    let topology_source = owner
        .lease(shape.rows().unwrap(), context.device_registry_id())
        .unwrap();
    let request =
        BytecodeAddressFusedScatterRequest::new(topology_owner.lease(topology_source).unwrap())
            .unwrap();
    let mut planes = context
        .prepare_instruction_read_raf_compatibility_scatter(
            scatter_source,
            &vec![AkitaField::zero(); shape.log_rows() as usize],
            InstructionReadRafCompatibilityScatterConfig {
                threads_per_threadgroup: 256,
            },
            Some(request),
        )
        .unwrap();
    let fused = planes.receipt().bytecode().unwrap();
    assert_eq!(fused.physical_rows(), physical_rows);
    assert_eq!(fused.max_pivots_per_chunk(), 15);
    let count_bytes = (INSTRUCTION_READ_RAF_SEGMENTS * size_of::<u32>()).div_ceil(16) * 16;
    let pivot_bytes = (15 * size_of::<u16>()).div_ceil(16) * 16;
    let descriptor_capacity =
        ((fused.threadgroup_memory_limit_bytes() - count_bytes as u64 - pivot_bytes as u64) / 16
            * 2)
        .saturating_sub(1)
        .min(4096) as usize;
    assert_eq!(
        fused.max_admitted_descriptors_per_chunk(),
        descriptor_capacity
    );
    assert_eq!(fused.max_admitted_pivots_per_chunk(), 15);
    assert_eq!(fused.additional_source_row_scans(), 0);
    assert_eq!(fused.member_upload_bytes(), 0);
    let carrier = planes.take_bytecode_carrier().unwrap();
    let carrier_receipt = carrier.receipt();
    assert_eq!(carrier_receipt.covered_rows(), physical_rows);
    assert_eq!(
        carrier_receipt.source_generation(),
        source_receipt.source_generation()
    );
    assert_eq!(
        carrier_receipt.source_completion_serial(),
        source_receipt.completion_serial()
    );

    let (e_lo, e_hi) = tables(shape);
    let expected = direct_oracle(shape, &rows, &e_lo, &e_hi);
    let invocation = context
        .prepare_bytecode_address_sparse_resident(carrier, &e_lo, &e_hi)
        .unwrap();
    assert_eq!(invocation.execute_timed().unwrap().output, expected);
}

#[test]
#[cfg(target_os = "macos")]
fn fused_stage1_scatter_rejects_rank_past_the_exact_chunk_cell() {
    let context = SolinasMetal::for_akita().unwrap();
    let shape = AddressMajorShape::production(15).unwrap();
    let padded_rows = shape.rows().unwrap();
    let physical_rows = 2 * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
    let resident_rows = (0..padded_rows)
        .map(|row| {
            let mapped_pc = if row >= physical_rows {
                None
            } else if row.is_multiple_of(INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS) {
                Some(7)
            } else {
                Some(8)
            };
            BooleanityRow::new(0, mapped_pc.map(|pc| pc as u64), None, row as i128).unwrap()
        })
        .collect::<Vec<_>>();
    let mut source = context
        .prepare_instruction_read_raf_stage1_storage(padded_rows)
        .unwrap();
    let mut topology = context
        .prepare_bytecode_address_stage1_topology_storage(padded_rows, physical_rows)
        .unwrap();
    topology
        .with_chunk_writers(|topology_chunks| {
            source.with_chunk_writers(|source_chunks| {
                let mut scratch = BytecodeAddressStage1TopologyScratch::new();
                for (chunk, (source, topology)) in source_chunks
                    .iter_mut()
                    .zip(topology_chunks.iter_mut())
                    .enumerate()
                {
                    let start = chunk * INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
                    for offset in 0..source.len() {
                        let row = start + offset;
                        let rank = if row < physical_rows {
                            topology
                                .record(&mut scratch, resident_rows[row].mapped_pc().unwrap_or(0))?
                        } else {
                            0
                        };
                        source.push_with_bytecode_chunk_rank(resident_rows[row], 0, false, rank)?;
                    }
                    topology.finish(&mut scratch)?;
                }
                Ok(())
            })
        })
        .unwrap();
    let owner = source.seal().unwrap();
    let topology_owner = topology.seal(&owner).unwrap();

    let inspection_source = owner
        .lease(padded_rows, context.device_registry_id())
        .unwrap();
    let inspection = topology_owner.lease(inspection_source).unwrap();
    let topology_receipt = inspection.receipt();
    // SAFETY: the lease keeps the fully initialized descriptor allocation alive,
    // and the receipt records its exact element count.
    let descriptors = unsafe {
        std::slice::from_raw_parts(
            inspection
                .descriptors_buffer()
                .contents()
                .cast::<BytecodeAddressChunkDescriptor>(),
            topology_receipt.descriptor_elements(),
        )
    };
    assert_eq!(descriptors[0].address, 7);
    assert_eq!(descriptors[0].base, 0);
    assert_eq!(descriptors[0].count(), 1);
    assert_eq!(descriptors[1].address, 8);
    assert_eq!(descriptors[1].base, 2);
    let corrupted_rank = 1u8;
    assert!(
        usize::from(descriptors[0].base) + usize::from(corrupted_rank)
            < usize::from(descriptors[1].base)
    );
    drop(inspection);

    let corrupt_source = owner
        .lease(padded_rows, context.device_registry_id())
        .unwrap();
    // SAFETY: no command has been submitted, the shared allocation contains
    // four initialized columns of `padded_rows`, and the lease keeps it alive.
    unsafe {
        let words = corrupt_source.row_buffer().contents().cast::<u64>();
        let first = BooleanityRow::from_instruction_source_words([
            *words,
            *words.add(padded_rows),
            *words.add(2 * padded_rows),
            *words.add(3 * padded_rows),
        ])
        .with_bytecode_chunk_rank_low7(corrupted_rank);
        let encoded = first.instruction_source_words(None).unwrap();
        words.add(3 * padded_rows).write(encoded[3]);
    }
    drop(corrupt_source);

    let scatter_source = owner
        .lease(padded_rows, context.device_registry_id())
        .unwrap();
    let topology_source = owner
        .lease(padded_rows, context.device_registry_id())
        .unwrap();
    let request =
        BytecodeAddressFusedScatterRequest::new(topology_owner.lease(topology_source).unwrap())
            .unwrap();
    let result = context.prepare_instruction_read_raf_compatibility_scatter(
        scatter_source,
        &vec![AkitaField::zero(); shape.log_rows() as usize],
        InstructionReadRafCompatibilityScatterConfig {
            threads_per_threadgroup: 256,
        },
        Some(request),
    );

    assert!(matches!(
        result,
        Err(MetalError::InvalidInstructionReadRafGrouped(_))
    ));
}
