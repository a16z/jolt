#![expect(clippy::unwrap_used, reason = "tests use checked fixtures")]

use jolt_field::AkitaField;

use super::{
    carrier::AddressMajorShape,
    oracle::{HostAddressMajorCarrier, Row},
    BytecodeAddressMajorConfig, BYTECODE_ADDRESS_MAJOR_BASE_STAGES,
    BYTECODE_ADDRESS_MAJOR_SIMD_WIDTH, BYTECODE_ADDRESS_MAJOR_STAGES,
};
use crate::metal::solinas::{BooleanityRow, SolinasMetal};

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
    let shape = AddressMajorShape::production(15).unwrap();
    let rows = (0..shape.rows().unwrap())
        .map(|inner| Row {
            mapped_pc: if inner.is_multiple_of(257) {
                None
            } else {
                Some((17 * inner + inner / 31) % shape.addresses().unwrap())
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
    let resident = context.prepare_booleanity_rows(&resident_words).unwrap();
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
    assert_eq!(
        observation.producer_status.unwrap().emitted_rows as usize,
        rows.len()
    );
    assert_eq!(observation.output, expected);
}
