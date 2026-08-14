#![expect(clippy::unwrap_used, reason = "tests use checked fixtures")]

use jolt_field::AkitaField;

use super::{
    carrier::AddressMajorShape,
    oracle::{HostAddressMajorCarrier, Row},
    worklist::{SparseAddressRow, SparseAddressWorklist},
    BytecodeAddressChunkDescriptor, BytecodeAddressFusedScatterRequest, BytecodeAddressMajorConfig,
    BytecodeAddressStage1TopologyScratch, BYTECODE_ADDRESS_MAJOR_BASE_STAGES,
    BYTECODE_ADDRESS_MAJOR_SIMD_WIDTH, BYTECODE_ADDRESS_MAJOR_STAGES,
};
use crate::metal::solinas::{
    BooleanityRow, InstructionReadRafCompatibilityScatterConfig, MetalError, SolinasMetal,
    INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS,
};

fn fixture() -> (AddressMajorShape, Vec<Row>) {
    let shape = AddressMajorShape::new(12, 5, 8).unwrap();
    let inner_length = shape.inner_length().unwrap();
    let outer_length = shape.outer_length().unwrap();
    let mut rows = Vec::with_capacity(shape.rows().unwrap());
    for outer in 0..outer_length {
        for inner in 0..inner_length {
            let mapped_pc = if outer == 0 {
                Some(0)
            } else if outer == 1 {
                Some(1 + usize::from(inner >= inner_length / 2))
            } else if outer == 2 {
                Some(match inner {
                    0..31 => 3,
                    31..63 => 4,
                    63..96 => 5,
                    _ => 6,
                })
            } else if (inner + 3 * outer).is_multiple_of(17) {
                None
            } else {
                Some((7 * inner + 3 * outer) % shape.addresses().unwrap())
            };
            rows.push(Row {
                mapped_pc,
                fused_inc_magnitude: if inner == 1 && outer == 3 {
                    0
                } else if inner == 0 && outer.is_multiple_of(2) {
                    u64::MAX
                } else {
                    (13 * inner + 11 * outer) as u64
                },
                fused_inc_negative: (inner + outer).is_multiple_of(5) || (inner == 1 && outer == 3),
            });
        }
    }
    (shape, rows)
}

fn tables(shape: AddressMajorShape) -> (Vec<Vec<AkitaField>>, Vec<Vec<AkitaField>>) {
    const MODULUS: u128 = 0xffff_ffff_ffff_ffff_ffff_ffff_0000_5809;
    let e_lo = (0..BYTECODE_ADDRESS_MAJOR_STAGES)
        .map(|stage| {
            (0..shape.inner_length().unwrap())
                .map(|inner| {
                    let delta = (17 * stage + 5 * inner + 1) as u128;
                    if (stage + inner).is_multiple_of(3) {
                        AkitaField::from_canonical_u128(MODULUS - delta)
                    } else {
                        AkitaField::from_canonical_u128((1_u128 << 127) + delta)
                    }
                })
                .collect()
        })
        .collect();
    let e_hi = (0..BYTECODE_ADDRESS_MAJOR_STAGES)
        .map(|stage| {
            (0..shape.outer_length().unwrap())
                .map(|outer| {
                    let delta = (19 * stage + 7 * outer + 3) as u128;
                    if (stage + outer).is_multiple_of(2) {
                        AkitaField::from_canonical_u128(MODULUS - delta)
                    } else {
                        AkitaField::from_canonical_u128((1_u128 << 126) + delta)
                    }
                })
                .collect()
        })
        .collect();
    (e_lo, e_hi)
}

fn direct_oracle(
    shape: AddressMajorShape,
    rows: &[Row],
    e_lo: &[Vec<AkitaField>],
    e_hi: &[Vec<AkitaField>],
) -> Vec<AkitaField> {
    let addresses = shape.addresses().unwrap();
    let inner_length = shape.inner_length().unwrap();
    let mut output = vec![AkitaField::zero(); BYTECODE_ADDRESS_MAJOR_STAGES * addresses];
    for (index, row) in rows.iter().copied().enumerate() {
        let outer = index / inner_length;
        let inner = index % inner_length;
        let address = row.push_pc();
        let increment = if row.fused_inc_negative {
            AkitaField::zero() - AkitaField::from_u64(row.fused_inc_magnitude)
        } else {
            AkitaField::from_u64(row.fused_inc_magnitude)
        };
        for stage in 0..BYTECODE_ADDRESS_MAJOR_STAGES {
            let mut term = e_lo[stage][inner] * e_hi[stage][outer];
            if stage >= BYTECODE_ADDRESS_MAJOR_BASE_STAGES {
                term *= increment;
            }
            output[stage * addresses + address] += term;
        }
    }
    output
}

#[test]
#[cfg(target_os = "macos")]
fn address_major_worker_matches_independent_direct_oracle() {
    let context = SolinasMetal::for_akita().unwrap();
    let (shape, rows) = fixture();
    let carrier = HostAddressMajorCarrier::build(&rows, shape).unwrap();
    let (e_lo, e_hi) = tables(shape);
    let expected = direct_oracle(shape, &rows, &e_lo, &e_hi);
    let invocation = context
        .prepare_bytecode_address_major_probe(
            &carrier,
            &e_lo,
            &e_hi,
            BytecodeAddressMajorConfig { outer_tiles: 7 },
        )
        .unwrap();

    assert_eq!(
        invocation.worker_pipeline_limits().thread_execution_width,
        32
    );
    assert_eq!(
        invocation.threadgroup_memory_bytes(),
        8 * BYTECODE_ADDRESS_MAJOR_BASE_STAGES * 16
    );
    assert_eq!(BYTECODE_ADDRESS_MAJOR_SIMD_WIDTH, 32);
    assert_eq!(invocation.execute().unwrap(), expected);
}

#[test]
#[cfg(target_os = "macos")]
fn address_major_worker_handles_a_full_u16_cell() {
    let context = SolinasMetal::for_akita().unwrap();
    let shape = AddressMajorShape::new(15, 1, 15).unwrap();
    let rows = (0..shape.rows().unwrap())
        .map(|inner| Row {
            mapped_pc: Some(0),
            fused_inc_magnitude: if inner == 0 { 0 } else { u64::MAX },
            fused_inc_negative: inner.is_multiple_of(2),
        })
        .collect::<Vec<_>>();
    let carrier = HostAddressMajorCarrier::build(&rows, shape).unwrap();
    let (e_lo, e_hi) = tables(shape);
    let expected = direct_oracle(shape, &rows, &e_lo, &e_hi);
    let invocation = context
        .prepare_bytecode_address_major_probe(
            &carrier,
            &e_lo,
            &e_hi,
            BytecodeAddressMajorConfig { outer_tiles: 1 },
        )
        .unwrap();

    assert_eq!(carrier.cell(0, 0).unwrap().count(), 1 << 15);
    assert_eq!(invocation.execute().unwrap(), expected);
}

#[test]
#[cfg(target_os = "macos")]
fn resident_producer_matches_the_independent_row_oracle() {
    let context = SolinasMetal::for_akita().unwrap();
    let shape = AddressMajorShape::production(16).unwrap();
    let rows = (0..shape.rows().unwrap())
        .map(|inner| Row {
            mapped_pc: if inner.is_multiple_of(257) {
                None
            } else {
                Some((17 * inner + inner / 31) % 10)
            },
            fused_inc_magnitude: if inner.is_multiple_of(509) {
                u64::MAX
            } else {
                (13 * inner + 7) as u64
            },
            fused_inc_negative: inner.is_multiple_of(5),
        })
        .collect::<Vec<_>>();
    let resident_words = rows
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
    let mut support_offsets = vec![0];
    let mut active_addresses = Vec::new();
    for outer_rows in rows.chunks(shape.inner_length().unwrap()) {
        let mut outer_addresses = outer_rows
            .iter()
            .map(|row| row.push_pc() as u32)
            .collect::<Vec<_>>();
        outer_addresses.sort_unstable();
        outer_addresses.dedup();
        active_addresses.extend(outer_addresses);
        support_offsets.push(active_addresses.len() as u32);
    }
    let max_active_addresses = support_offsets
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as usize)
        .max()
        .unwrap();
    let resident = context
        .prepare_booleanity_rows_with_bytecode_support(
            &resident_words,
            &support_offsets,
            &active_addresses,
        )
        .unwrap();
    let source_id = resident.allocation_identity();
    let device_id = resident.device_registry_id();
    let (e_lo, e_hi) = tables(shape);
    let expected = direct_oracle(shape, &rows, &e_lo, &e_hi);
    let pending = context
        .prepare_bytecode_address_major_resident_shadow(
            resident,
            &e_lo,
            &e_hi,
            BytecodeAddressMajorConfig { outer_tiles: 1 },
        )
        .unwrap()
        .submit()
        .unwrap();
    let (_, observation) = pending.join().unwrap();

    assert_eq!(observation.source_rows_storage_id, Some(source_id));
    assert_eq!(observation.source_rows_device_registry_id, Some(device_id));
    assert_eq!(observation.max_active_addresses, Some(max_active_addresses));
    assert!(observation.producer_threadgroup_bytes.unwrap() < 6 * 1024);
    assert_eq!(
        observation.producer_status.unwrap().emitted_rows as usize,
        rows.len()
    );
    assert_eq!(observation.output, expected);
}

#[test]
#[cfg(target_os = "macos")]
fn sparse_worklist_worker_matches_the_padded_domain_oracle() {
    let context = SolinasMetal::for_akita().unwrap();
    let shape = AddressMajorShape::production(16).unwrap();
    let physical_rows = 50_123;
    let mut rows = (0..shape.rows().unwrap())
        .map(|index| {
            if index >= physical_rows {
                return Row {
                    mapped_pc: None,
                    fused_inc_magnitude: 0,
                    fused_inc_negative: false,
                };
            }
            Row {
                mapped_pc: if index < 10_000 {
                    Some(0)
                } else if index.is_multiple_of(257) {
                    None
                } else {
                    Some(1 + (17 * index + index / 31) % 37)
                },
                fused_inc_magnitude: if index.is_multiple_of(509) {
                    u64::MAX
                } else {
                    (13 * index + 7) as u64
                },
                fused_inc_negative: index.is_multiple_of(5),
            }
        })
        .collect::<Vec<_>>();
    rows[physical_rows - 1] = Row {
        mapped_pc: Some(8191),
        fused_inc_magnitude: u64::MAX,
        fused_inc_negative: true,
    };
    let resident_words = rows
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
    let resident = context.prepare_booleanity_rows(&resident_words).unwrap();
    let worklist = SparseAddressWorklist::build_with(physical_rows, shape, |index| {
        SparseAddressRow::with_magnitude(
            rows[index].mapped_pc,
            rows[index].fused_inc_magnitude,
            rows[index].fused_inc_negative,
        )
        .unwrap()
    })
    .unwrap();
    let (e_lo, e_hi) = tables(shape);
    let expected = direct_oracle(shape, &rows, &e_lo, &e_hi);

    let invocation = context
        .prepare_bytecode_address_sparse_probe(resident, &worklist, &e_lo, &e_hi)
        .unwrap();
    let storage = invocation.storage();

    assert_eq!(
        invocation.worker_pipeline_limits().thread_execution_width,
        32
    );
    assert_eq!(invocation.threadgroup_memory_bytes(), 0);
    assert_eq!(storage.occurrence_bytes, 2 * physical_rows);
    assert_eq!(storage.magnitude_bytes, 8 * physical_rows);
    assert_eq!(storage.work_item_bytes, 8 * worklist.work_items());
    assert!(worklist.work_items() > 2);
    let observation = invocation.execute_timed().unwrap();
    assert_eq!(observation.output, expected);
    assert_eq!(observation.worker_variant, "packed4_halfwidth_v1");
    assert_eq!(observation.worker_simd_width, 32);
    assert_eq!(observation.worker_threads, 128);
    assert_eq!(observation.worker_items_per_threadgroup, 4);
    assert_eq!(
        observation.worker_threadgroups,
        worklist.work_items().div_ceil(4)
    );
    assert_eq!(
        observation.worker_tail_slots,
        (4 - worklist.work_items() % 4) % 4
    );
    assert_eq!(observation.worker_dynamic_threadgroup_bytes, 0);
    assert_eq!(observation.worker_static_threadgroup_bytes, 0);
    assert_eq!(observation.worker_threadgroup_bytes, 0);
    assert_eq!(observation.reducer_threads, 256);
    assert_eq!(observation.reducer_threadgroups, 288);
    assert_eq!(observation.reducer_static_threadgroup_bytes, 0);
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
                return Row {
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
            Row {
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
    let descriptor_capacity = ((fused.threadgroup_memory_limit_bytes() - 336 - 32) / 16 * 2)
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
    // five initialized columns of `padded_rows`, and the lease keeps it alive.
    unsafe {
        let words = corrupt_source.row_buffer().contents().cast::<u64>();
        let packed = words.add(4 * padded_rows);
        let first = BooleanityRow::from_words([
            *words,
            *words.add(padded_rows),
            *words.add(2 * padded_rows),
            *words.add(3 * padded_rows),
            *packed,
        ])
        .with_bytecode_chunk_rank_low7(corrupted_rank);
        packed.write(first.words()[4]);
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
