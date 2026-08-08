#![expect(clippy::unwrap_used, reason = "tests use checked fixtures")]

use jolt_field::AkitaField;

use super::{
    carrier::AddressMajorShape,
    oracle::{HostAddressMajorCarrier, Row},
    worklist::{SparseAddressRow, SparseAddressWorklist},
    BytecodeAddressMajorConfig, BytecodeAddressMajorSourceRow, BYTECODE_ADDRESS_MAJOR_BASE_STAGES,
    BYTECODE_ADDRESS_MAJOR_SIMD_WIDTH, BYTECODE_ADDRESS_MAJOR_STAGES,
};
use crate::metal::solinas::{BooleanityRow, MetalError, SolinasMetal};

fn stage1_owner(
    context: &SolinasMetal,
    rows: &[BooleanityRow],
) -> crate::metal::solinas::InstructionReadRafStage1Owner {
    let mut storage = context
        .prepare_instruction_read_raf_stage1_storage(rows.len())
        .unwrap();
    storage
        .with_chunk_writers(|chunks| {
            for (chunk_index, chunk) in chunks.iter_mut().enumerate() {
                let start =
                    chunk_index * crate::metal::solinas::INSTRUCTION_READ_RAF_PRODUCER_CHUNK_ROWS;
                for row in &rows[start..start + chunk.len()] {
                    chunk.push(*row, 0, false)?;
                }
            }
            Ok(())
        })
        .unwrap();
    storage.seal().unwrap()
}

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
fn stage1_resident_carrier_matches_the_independent_row_oracle() {
    let context = SolinasMetal::for_akita().unwrap();
    let shape = AddressMajorShape::production(16).unwrap();
    let rows = (0..shape.rows().unwrap())
        .map(|index| {
            let outer = index >> 15;
            let inner = index & ((1 << 15) - 1);
            Row {
                mapped_pc: if index == 0 {
                    Some(7)
                } else if inner.is_multiple_of(257) {
                    None
                } else {
                    Some(if outer == 0 {
                        (17 * inner + inner / 31) % 23
                    } else {
                        97 + (11 * inner + inner / 19) % 29
                    })
                },
                fused_inc_magnitude: if inner.is_multiple_of(509) {
                    u64::MAX
                } else {
                    (13 * inner + 7 * outer + 7) as u64
                },
                fused_inc_negative: (inner + outer).is_multiple_of(5),
            }
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
    let owner = stage1_owner(&context, &resident_words);
    let resident = owner.booleanity_rows();
    let mut storage = context
        .prepare_bytecode_address_major_resident_storage(rows.len())
        .unwrap();
    storage
        .with_outer_writers(|writers| {
            for (outer, writer) in writers.iter_mut().enumerate() {
                let outer_rows = &rows[outer * (1 << 15)..(outer + 1) * (1 << 15)];
                let selectors = outer_rows
                    .iter()
                    .map(|row| {
                        BytecodeAddressMajorSourceRow {
                            mapped_pc: row.mapped_pc,
                            fused_inc_negative: row.fused_inc_negative,
                        }
                        .selector()
                        .unwrap()
                    })
                    .collect::<Vec<_>>();
                let magnitudes = outer_rows
                    .iter()
                    .map(|row| row.fused_inc_magnitude)
                    .collect::<Vec<_>>();
                writer.publish(&selectors, &magnitudes).map_err(|error| {
                    MetalError::InvalidInstructionReadRafGrouped(error.to_string())
                })?;
            }
            Ok(())
        })
        .unwrap();
    let source_id = resident.allocation_identity();
    let source_device = resident.device_registry_id();
    let carrier = storage.seal(&owner).unwrap();
    let receipt = carrier.receipt();
    let host_carrier = HostAddressMajorCarrier::build(&rows, shape).unwrap();
    assert_eq!(receipt.first_push_pc(), 7);
    assert_eq!(receipt.topology(), host_carrier.topology());
    assert_eq!(receipt.producer().source_allocation_identity(), source_id);
    assert_eq!(receipt.producer().device_registry_id(), source_device);

    let (e_lo, e_hi) = tables(shape);
    let expected = direct_oracle(shape, &rows, &e_lo, &e_hi);
    let pending = context
        .prepare_bytecode_address_major_resident_carrier(
            carrier,
            &e_lo,
            &e_hi,
            BytecodeAddressMajorConfig { outer_tiles: 1 },
        )
        .unwrap()
        .submit()
        .unwrap();
    let (_, observation) = pending.join().unwrap();
    assert_eq!(observation.producer_status, None);
    assert_eq!(observation.source_rows_storage_id, Some(source_id));
    assert_eq!(
        observation.source_rows_device_registry_id,
        Some(source_device)
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
    assert_eq!(invocation.threadgroup_memory_bytes(), 640);
    assert_eq!(storage.occurrence_bytes, 2 * physical_rows);
    assert_eq!(storage.magnitude_bytes, 8 * physical_rows);
    assert_eq!(storage.work_item_bytes, 8 * worklist.work_items());
    assert!(worklist.work_items() > 2);
    assert_eq!(invocation.execute_timed().unwrap().output, expected);
}
