use core::mem::size_of;

use jolt_field::{AkitaField, FromPrimitiveInt};

use super::super::Fp128;
use super::*;

#[derive(Clone, Debug)]
struct DenseFixture {
    rows: Vec<RegisterRow>,
    initial_values: [u64; REGISTER_CSR_COLUMNS],
    checkpoints: Vec<[u64; REGISTER_CSR_COLUMNS]>,
}

fn dense_fixture(cycles: usize) -> DenseFixture {
    let initial_values = core::array::from_fn(|register| register as u64 * 1_003 + 17);
    let mut state = initial_values;
    let mut rows = Vec::with_capacity(cycles);
    let mut checkpoints = Vec::with_capacity(cycles + 1);
    checkpoints.push(state);

    for cycle in 0..cycles {
        let rs1_register = ((3 * cycle + 7) % REGISTER_CSR_COLUMNS) as u8;
        let rs2_register = if cycle.is_multiple_of(6) {
            rs1_register
        } else {
            ((5 * cycle + 11) % REGISTER_CSR_COLUMNS) as u8
        };
        let rs1 = (cycle.is_multiple_of(2) || cycle == 255)
            .then(|| RegisterRead::new(rs1_register, state[usize::from(rs1_register)]));
        let rs2 = (cycle.is_multiple_of(3) || cycle == 256)
            .then(|| RegisterRead::new(rs2_register, state[usize::from(rs2_register)]));

        let write_present = cycle.is_multiple_of(5) || cycle == 255 || cycle == 256;
        let rd_register = if cycle == 255 || cycle == 256 {
            7
        } else {
            ((7 * cycle + 13) % REGISTER_CSR_COLUMNS) as u8
        };
        let rd = write_present.then(|| {
            let register = usize::from(rd_register);
            let pre_value = state[register];
            let post_value = if cycle.is_multiple_of(17) {
                pre_value
            } else {
                pre_value + (cycle % 23 + 1) as u64
            };
            state[register] = post_value;
            RegisterWrite::new(rd_register, pre_value, post_value)
        });
        rows.push(RegisterRow::new(rs1, rs2, rd));
        checkpoints.push(state);
    }

    DenseFixture {
        rows,
        initial_values,
        checkpoints,
    }
}

fn reconstructed_checkpoints(
    rows: &[RegisterRow],
    initial_values: [u64; REGISTER_CSR_COLUMNS],
) -> Vec<[u64; REGISTER_CSR_COLUMNS]> {
    let mut state = initial_values;
    let mut checkpoints = Vec::with_capacity(rows.len() + 1);
    checkpoints.push(state);
    for row in rows {
        if let Some(write) = row.rd() {
            assert_eq!(state[usize::from(write.register())], write.pre_value());
            state[usize::from(write.register())] = write.post_value();
        }
        checkpoints.push(state);
    }
    checkpoints
}

fn event_counts(rows: &[RegisterRow]) -> RegisterEventCounts {
    RegisterEventCounts::new(
        rows.iter().filter(|row| row.rs1().is_some()).count(),
        rows.iter().filter(|row| row.rs2().is_some()).count(),
        rows.iter().filter(|row| row.rd().is_some()).count(),
    )
}

fn expected_read_events(
    rows: &[RegisterRow],
    select: impl Fn(RegisterRow) -> Option<RegisterRead>,
) -> Vec<(usize, u8)> {
    rows.iter()
        .copied()
        .enumerate()
        .filter_map(|(cycle, row)| select(row).map(|read| (cycle, read.register())))
        .collect()
}

fn expected_write_events(rows: &[RegisterRow]) -> Vec<(usize, u8, u64)> {
    rows.iter()
        .enumerate()
        .filter_map(|(cycle, row)| {
            row.rd()
                .map(|write| (cycle, write.register(), write.post_value()))
        })
        .collect()
}

fn expected_rd_inc(rows: &[RegisterRow]) -> Vec<Fp128> {
    rows.iter()
        .map(|row| {
            row.rd().map_or(Fp128::ZERO, |write| {
                let increment = AkitaField::from_i128(
                    i128::from(write.post_value()) - i128::from(write.pre_value()),
                );
                Fp128::from_jolt_field(&increment)
            })
        })
        .collect()
}

fn digest(seed: u64) -> OrderedPrefixDigest {
    OrderedPrefixDigest::new([seed, seed + 1, seed + 2, seed + 3]).unwrap()
}

fn source(cycles: usize, source_identity: usize) -> RegisterBcsrSourceProvenance {
    let geometry = RegisterBcsrGeometry::new(cycles).unwrap();
    let source_bytes = RegisterBcsrLayout::new(geometry)
        .unwrap()
        .producer_bytes()
        .unwrap();
    RegisterBcsrSourceProvenance::new(9, source_identity, source_bytes, 7, cycles, digest(101))
        .unwrap()
}

fn plane(
    identity: usize,
    shape: RegisterBcsrPlaneShape,
    device: u64,
    generation: u64,
) -> RegisterBcsrPlaneProvenance {
    RegisterBcsrPlaneProvenance::new(
        device,
        identity,
        generation,
        1_000 + identity as u64,
        shape.elements(),
        shape.bytes(),
    )
    .unwrap()
}

fn provenances(
    layout: RegisterBcsrLayout,
    identities: [usize; 10],
    device: u64,
    generation: u64,
) -> RegisterBcsrPlaneProvenances {
    RegisterBcsrPlaneProvenances::new(
        plane(identities[0], layout.start_values(), device, generation),
        plane(identities[1], layout.offsets(), device, generation),
        plane(identities[2], layout.offsets(), device, generation),
        plane(identities[3], layout.offsets(), device, generation),
        plane(identities[4], layout.positions(), device, generation),
        plane(identities[5], layout.positions(), device, generation),
        plane(identities[6], layout.positions(), device, generation),
        plane(identities[7], layout.rd_post_values(), device, generation),
        plane(identities[8], layout.rd_index(), device, generation),
        plane(identities[9], layout.rd_inc(), device, generation),
    )
}

fn checked_fixture(
    cycles: usize,
) -> (
    DenseFixture,
    RegisterBcsr256,
    RegisterBcsrStateFlowCertificate,
) {
    let dense = dense_fixture(cycles);
    let (bcsr, certificate) =
        RegisterBcsr256::from_rows(&dense.rows, &dense.initial_values).unwrap();
    (dense, bcsr, certificate)
}

#[test]
fn bcsr_reconstructs_every_event_and_checkpoint_across_block_shapes() {
    for cycles in [
        REGISTER_BCSR_POSITION_SLOTS,
        2 * REGISTER_BCSR_POSITION_SLOTS,
        513,
    ] {
        let (dense, bcsr, certificate) = checked_fixture(cycles);
        let expected_blocks = cycles.div_ceil(REGISTER_BCSR_POSITION_SLOTS);
        assert_eq!(bcsr.geometry().blocks(), expected_blocks);
        assert_eq!(certificate.geometry(), bcsr.geometry());
        assert_eq!(certificate.events(), event_counts(&dense.rows));
        assert_eq!(certificate.initial_values(), &dense.initial_values);
        assert_eq!(
            certificate.final_values(),
            dense.checkpoints.last().unwrap()
        );

        let reconstructed = bcsr.reconstruct_rows().unwrap();
        assert_eq!(reconstructed, dense.rows);
        assert_eq!(
            reconstructed_checkpoints(&reconstructed, dense.initial_values),
            dense.checkpoints
        );

        let rs1 = bcsr
            .rs1_events()
            .unwrap()
            .into_iter()
            .map(|event| (event.cycle(), event.register()))
            .collect::<Vec<_>>();
        let rs2 = bcsr
            .rs2_events()
            .unwrap()
            .into_iter()
            .map(|event| (event.cycle(), event.register()))
            .collect::<Vec<_>>();
        let rd = bcsr
            .rd_events()
            .unwrap()
            .into_iter()
            .map(|event| (event.cycle(), event.register(), event.post_value()))
            .collect::<Vec<_>>();
        assert_eq!(rs1, expected_read_events(&dense.rows, RegisterRow::rs1));
        assert_eq!(rs2, expected_read_events(&dense.rows, RegisterRow::rs2));
        assert_eq!(rd, expected_write_events(&dense.rows));

        for block in 0..expected_blocks {
            let checkpoint = block * REGISTER_BCSR_POSITION_SLOTS;
            assert_eq!(
                bcsr.block_start_values(block).unwrap(),
                &dense.checkpoints[checkpoint]
            );
            let parts = bcsr.parts();
            for (offsets, positions) in [
                (&parts.rs1_offsets[block], &parts.rs1_positions[block]),
                (&parts.rs2_offsets[block], &parts.rs2_positions[block]),
                (&parts.rd_offsets[block], &parts.rd_positions[block]),
            ] {
                let terminal = usize::from(offsets[REGISTER_CSR_COLUMNS]);
                assert!(positions[terminal..].iter().all(|&value| value == 0));
            }
            let rd_terminal = usize::from(parts.rd_offsets[block][REGISTER_CSR_COLUMNS]);
            assert!(parts.rd_post_values[block][rd_terminal..]
                .iter()
                .all(|&value| value == 0));
        }

        let expected_rd_index = dense
            .rows
            .iter()
            .map(|row| {
                row.rd()
                    .map_or(REGISTER_ABSENT_INDEX, RegisterWrite::register)
            })
            .collect::<Vec<_>>();
        assert_eq!(bcsr.rd_index(), expected_rd_index.as_slice());
        assert_eq!(bcsr.rd_inc(), expected_rd_inc(&dense.rows).as_slice());
    }
}

#[test]
fn bcsr_keeps_alias_reads_and_present_zero_delta_writes() {
    let (dense, bcsr, certificate) = checked_fixture(513);
    let alias_cycles = dense
        .rows
        .iter()
        .enumerate()
        .filter_map(|(cycle, row)| match (row.rs1(), row.rs2()) {
            (Some(rs1), Some(rs2)) if rs1.register() == rs2.register() => Some(cycle),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(!alias_cycles.is_empty());
    let rs1 = bcsr.rs1_events().unwrap();
    let rs2 = bcsr.rs2_events().unwrap();
    for cycle in alias_cycles {
        let register = dense.rows[cycle].rs1().unwrap().register();
        assert!(rs1
            .iter()
            .any(|event| event.cycle() == cycle && event.register() == register));
        assert!(rs2
            .iter()
            .any(|event| event.cycle() == cycle && event.register() == register));
    }

    let zero_delta_cycles = dense
        .rows
        .iter()
        .enumerate()
        .filter_map(|(cycle, row)| {
            row.rd()
                .filter(|write| write.pre_value() == write.post_value())
                .map(|_| cycle)
        })
        .collect::<Vec<_>>();
    assert!(!zero_delta_cycles.is_empty());
    assert_eq!(certificate.zero_delta_writes(), zero_delta_cycles.len());
    let rd_events = bcsr.rd_events().unwrap();
    for cycle in zero_delta_cycles {
        let write = dense.rows[cycle].rd().unwrap();
        assert_eq!(bcsr.rd_index()[cycle], write.register());
        assert!(rd_events.iter().any(|event| {
            event.cycle() == cycle
                && event.register() == write.register()
                && event.post_value() == write.post_value()
        }));
    }
}

#[test]
fn full_block_uses_the_u16_256_terminal_and_local_position_255() {
    let initial_values = core::array::from_fn(|register| register as u64 + 41);
    let register = 3u8;
    let read = RegisterRead::new(register, initial_values[usize::from(register)]);
    let rows = vec![RegisterRow::new(Some(read), Some(read), None); 256];
    let (bcsr, certificate) = RegisterBcsr256::from_rows(&rows, &initial_values).unwrap();
    let parts = bcsr.parts();
    assert_eq!(parts.rs1_offsets[0][REGISTER_CSR_COLUMNS], 256u16);
    assert_eq!(parts.rs2_offsets[0][REGISTER_CSR_COLUMNS], 256u16);
    assert_eq!(parts.rs1_positions[0][255], 255u8);
    assert_eq!(parts.rs2_positions[0][255], 255u8);
    assert_eq!(certificate.events(), RegisterEventCounts::new(256, 256, 0));
    assert!(bcsr
        .rd_index()
        .iter()
        .all(|&index| index == REGISTER_ABSENT_INDEX));
    assert!(bcsr
        .rd_inc()
        .iter()
        .all(|&increment| increment == Fp128::ZERO));
}

#[test]
fn bcsr_layout_is_derived_from_geometry() {
    let geometry = RegisterBcsrGeometry::new(513).unwrap();
    let layout = RegisterBcsrLayout::new(geometry).unwrap();
    let blocks = geometry.blocks();
    assert_eq!(blocks, 3);
    assert_eq!(
        layout.start_values().elements(),
        blocks * REGISTER_CSR_COLUMNS
    );
    assert_eq!(
        layout.offsets().elements(),
        blocks * REGISTER_BCSR_OFFSET_ENTRIES
    );
    assert_eq!(
        layout.positions().elements(),
        blocks * REGISTER_BCSR_POSITION_SLOTS
    );
    assert_eq!(layout.rd_index().elements(), geometry.cycles());
    assert_eq!(layout.rd_inc().elements(), geometry.cycles());
    assert_eq!(
        layout.topology_bytes().unwrap(),
        blocks * REGISTER_CSR_COLUMNS * size_of::<u64>()
            + 3 * blocks * REGISTER_BCSR_OFFSET_ENTRIES * size_of::<u16>()
            + 3 * blocks * REGISTER_BCSR_POSITION_SLOTS * size_of::<u8>()
            + blocks * REGISTER_BCSR_POSITION_SLOTS * size_of::<u64>()
    );
    assert_eq!(
        layout.registers_val_bytes().unwrap(),
        geometry.cycles() * (size_of::<u8>() + REGISTER_FP128_BYTES)
    );
    assert_eq!(
        layout.producer_bytes().unwrap(),
        layout.topology_bytes().unwrap() + layout.registers_val_bytes().unwrap()
    );
}

#[test]
fn bcsr_rejects_bad_capacity_offsets_padding_and_indices() {
    assert_eq!(
        RegisterBcsrGeometry::new(0),
        Err(RegistersRwV3Error::InvalidBcsrCycleCount(0))
    );
    if let Ok(too_many) = usize::try_from(u64::from(u32::MAX) + 1) {
        assert_eq!(
            RegisterBcsrGeometry::new(too_many),
            Err(RegistersRwV3Error::InvalidBcsrCycleCount(too_many))
        );
    }

    let (_, bcsr, _) = checked_fixture(513);
    let mut parts = bcsr.clone().into_parts();
    parts.rs1_offsets[0][0] = 1;
    assert!(matches!(
        RegisterBcsr256::from_parts(parts),
        Err(RegistersRwV3Error::OffsetStart { plane: "rs1", .. })
    ));

    let mut parts = bcsr.clone().into_parts();
    parts.rs2_offsets[0][1] = 2;
    parts.rs2_offsets[0][2] = 1;
    assert!(matches!(
        RegisterBcsr256::from_parts(parts),
        Err(RegistersRwV3Error::OffsetOrder { plane: "rs2", .. })
    ));

    let mut parts = bcsr.clone().into_parts();
    parts.rd_offsets[2][REGISTER_CSR_COLUMNS] = 2;
    assert_eq!(
        RegisterBcsr256::from_parts(parts),
        Err(RegistersRwV3Error::BcsrOffsetTerminal {
            plane: "rd",
            block: 2,
            maximum: 1,
            got: 2,
        })
    );

    let mut parts = bcsr.clone().into_parts();
    parts.rs2_offsets[2][1..].fill(1);
    parts.rs2_positions[2][0] = 1;
    assert_eq!(
        RegisterBcsr256::from_parts(parts),
        Err(RegistersRwV3Error::PositionOutOfBlock {
            plane: "rs2",
            block: 2,
            block_len: 1,
            position: 1,
        })
    );

    let mut parts = bcsr.clone().into_parts();
    let terminal = usize::from(parts.rs1_offsets[0][REGISTER_CSR_COLUMNS]);
    parts.rs1_positions[0][terminal] = 1;
    assert_eq!(
        RegisterBcsr256::from_parts(parts),
        Err(RegistersRwV3Error::BcsrNonzeroPadding {
            plane: "rs1",
            block: 0,
            slot: terminal,
        })
    );

    let mut parts = bcsr.clone().into_parts();
    let terminal = usize::from(parts.rd_offsets[0][REGISTER_CSR_COLUMNS]);
    parts.rd_post_values[0][terminal] = 1;
    assert_eq!(
        RegisterBcsr256::from_parts(parts),
        Err(RegistersRwV3Error::BcsrNonzeroPadding {
            plane: "rd post values",
            block: 0,
            slot: terminal,
        })
    );

    let mut parts = bcsr.clone().into_parts();
    parts.rd_index[0] = REGISTER_CSR_COLUMNS as u8;
    assert!(matches!(
        RegisterBcsr256::from_parts(parts),
        Err(RegistersRwV3Error::InvalidRegister {
            cycle: 0,
            access: "rd index",
            ..
        })
    ));

    let mut parts = bcsr.clone().into_parts();
    parts.rd_index[0] = REGISTER_ABSENT_INDEX;
    assert!(matches!(
        RegisterBcsr256::from_parts(parts),
        Err(RegistersRwV3Error::RdIndexMismatch { cycle: 0, .. })
    ));

    let mut parts = bcsr.clone().into_parts();
    parts.rd_inc[1] = Fp128::ONE;
    assert_eq!(
        RegisterBcsr256::from_parts(parts),
        Err(RegistersRwV3Error::IncrementMismatch { cycle: 1 })
    );

    let mut parts = bcsr.clone().into_parts();
    parts.rd_inc[1] = Fp128::from_u128(u128::MAX);
    assert_eq!(
        RegisterBcsr256::from_parts(parts),
        Err(RegistersRwV3Error::IncrementMismatch { cycle: 1 })
    );

    let mut parts = bcsr.clone().into_parts();
    parts.start_values[1][7] += 1;
    assert!(matches!(
        RegisterBcsr256::from_parts(parts),
        Err(RegistersRwV3Error::BlockStateMismatch {
            block: 1,
            register: 7,
            ..
        })
    ));

    let mut parts = bcsr.into_parts();
    let _ = parts.rs1_positions.pop();
    assert!(matches!(
        RegisterBcsr256::from_parts(parts),
        Err(RegistersRwV3Error::PlaneLength {
            plane: "BCSR rs1 positions",
            ..
        })
    ));
}

#[test]
fn scalar_constructor_rejects_uncertified_rows() {
    let fixture = dense_fixture(256);

    let mut rows = fixture.rows.clone();
    let row = rows[0];
    rows[0] = RegisterRow::new(Some(RegisterRead::new(128, 0)), row.rs2(), row.rd());
    assert!(matches!(
        RegisterBcsr256::from_rows(&rows, &fixture.initial_values),
        Err(RegistersRwV3Error::InvalidRegister {
            cycle: 0,
            access: "rs1",
            register: 128,
        })
    ));

    let mut rows = fixture.rows.clone();
    let row = rows[1];
    rows[1] = RegisterRow::new(row.rs1(), row.rs2(), Some(RegisterWrite::new(128, 0, 0)));
    assert!(matches!(
        RegisterBcsr256::from_rows(&rows, &fixture.initial_values),
        Err(RegistersRwV3Error::InvalidRegister {
            cycle: 1,
            access: "rd",
            register: 128,
        })
    ));

    let mut rows = fixture.rows.clone();
    let row = rows[0];
    let read = row.rs1().unwrap();
    rows[0] = RegisterRow::new(
        Some(RegisterRead::new(read.register(), read.value() + 1)),
        row.rs2(),
        row.rd(),
    );
    assert!(matches!(
        RegisterBcsr256::from_rows(&rows, &fixture.initial_values),
        Err(RegistersRwV3Error::ReadValueMismatch {
            cycle: 0,
            access: "rs1",
            ..
        })
    ));

    let write_cycle = fixture
        .rows
        .iter()
        .position(|row| row.rd().is_some())
        .unwrap();
    let mut rows = fixture.rows.clone();
    let row = rows[write_cycle];
    let write = row.rd().unwrap();
    rows[write_cycle] = RegisterRow::new(
        row.rs1(),
        row.rs2(),
        Some(RegisterWrite::new(
            write.register(),
            write.pre_value() + 1,
            write.post_value(),
        )),
    );
    assert!(matches!(
        RegisterBcsr256::from_rows(&rows, &fixture.initial_values),
        Err(RegistersRwV3Error::WritePreValueMismatch { .. })
    ));
}

#[test]
fn provenance_receipt_publishes_exact_registers_val_planes() {
    let geometry = RegisterBcsrGeometry::new(512).unwrap();
    let layout = RegisterBcsrLayout::new(geometry).unwrap();
    let identities = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    let receipt = RegisterBcsrReceipt::admit(
        source(geometry.cycles(), 1),
        layout,
        &provenances(layout, identities, 9, 7),
    )
    .unwrap();
    assert_eq!(receipt.allocation_identities(), identities);
    assert_eq!(
        receipt.source().source_allocation_bytes(),
        layout.producer_bytes().unwrap()
    );
    assert_eq!(receipt.layout(), layout);
    receipt.verify_binding(9, 7, digest(101)).unwrap();

    let input = receipt.registers_val_input().unwrap();
    assert_eq!(input.cycles(), geometry.cycles());
    assert_eq!(input.device_registry_id(), 9);
    assert_eq!(input.generation(), 7);
    assert_eq!(input.ordered_prefix_digest(), digest(101));
    assert_eq!(input.rd_index().elements(), geometry.cycles());
    assert_eq!(input.rd_index().bytes(), geometry.cycles());
    assert_eq!(input.rd_inc().elements(), geometry.cycles());
    assert_eq!(
        input.rd_inc().bytes(),
        geometry.cycles() * REGISTER_FP128_BYTES
    );
    assert_ne!(
        input.rd_index().allocation_identity(),
        input.rd_inc().allocation_identity()
    );
    assert!(input.rd_index().initialization_serial() > 0);
    assert!(input.rd_inc().initialization_serial() > 0);

    let abi = input.resident_abi().unwrap();
    assert_eq!(abi.rows, geometry.cycles() as u64);
    assert_eq!(abi.rd_index_bytes, geometry.cycles() as u64);
    assert_eq!(
        abi.rd_inc_bytes,
        (geometry.cycles() * REGISTER_FP128_BYTES) as u64
    );
    assert_eq!(abi.rd_index_allocation_id, identities[8] as u64);
    assert_eq!(abi.rd_inc_allocation_id, identities[9] as u64);
    assert_eq!(abi.device_registry_id, 9);
    assert_eq!(abi.generation, 7);
    abi.validate().unwrap();
}

#[test]
fn provenance_receipt_rejects_wrong_binding_and_aliases() {
    let geometry = RegisterBcsrGeometry::new(512).unwrap();
    let layout = RegisterBcsrLayout::new(geometry).unwrap();
    let identities = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    let valid_source = source(geometry.cycles(), 1);

    assert!(matches!(
        RegisterBcsrPlaneProvenance::new(
            9,
            identities[0],
            7,
            0,
            layout.start_values().elements(),
            layout.start_values().bytes(),
        ),
        Err(RegistersRwV3Error::MissingIdentity(
            "BCSR plane initialization serial"
        ))
    ));

    let wrong_cycles_source = source(256, 1);
    assert_eq!(
        RegisterBcsrReceipt::admit(
            wrong_cycles_source,
            layout,
            &provenances(layout, identities, 9, 7),
        ),
        Err(RegistersRwV3Error::ProducerCycleMismatch {
            expected: 512,
            got: 256,
        })
    );

    let wrong_device = provenances(layout, identities, 10, 7);
    assert!(matches!(
        RegisterBcsrReceipt::admit(valid_source, layout, &wrong_device),
        Err(RegistersRwV3Error::PlaneDeviceMismatch { .. })
    ));

    let wrong_generation = provenances(layout, identities, 9, 8);
    assert!(matches!(
        RegisterBcsrReceipt::admit(valid_source, layout, &wrong_generation),
        Err(RegistersRwV3Error::PlaneGenerationMismatch { .. })
    ));

    let mut wrong_shape = provenances(layout, identities, 9, 7);
    wrong_shape.start_values = RegisterBcsrPlaneProvenance::new(
        9,
        identities[0],
        7,
        1_001,
        layout.start_values().elements(),
        layout.start_values().bytes() + 1,
    )
    .unwrap();
    assert!(matches!(
        RegisterBcsrReceipt::admit(valid_source, layout, &wrong_shape),
        Err(RegistersRwV3Error::PlaneShape { .. })
    ));

    let mut duplicate = provenances(layout, identities, 9, 7);
    duplicate.rd_inc = plane(
        duplicate.rd_index.allocation_identity(),
        layout.rd_inc(),
        9,
        7,
    );
    assert_eq!(
        RegisterBcsrReceipt::admit(valid_source, layout, &duplicate),
        Err(RegistersRwV3Error::DuplicateAllocationIdentity {
            identity: identities[8],
        })
    );

    let source_alias = source(geometry.cycles(), identities[0]);
    assert_eq!(
        RegisterBcsrReceipt::admit(source_alias, layout, &provenances(layout, identities, 9, 7),),
        Err(RegistersRwV3Error::DuplicateAllocationIdentity {
            identity: identities[0],
        })
    );

    let receipt =
        RegisterBcsrReceipt::admit(valid_source, layout, &provenances(layout, identities, 9, 7))
            .unwrap();
    assert!(matches!(
        receipt.verify_binding(8, 7, digest(101)),
        Err(RegistersRwV3Error::ReceiptDeviceMismatch { .. })
    ));
    assert!(matches!(
        receipt.verify_binding(9, 8, digest(101)),
        Err(RegistersRwV3Error::ReceiptGenerationMismatch { .. })
    ));
    assert_eq!(
        receipt.verify_binding(9, 7, digest(102)),
        Err(RegistersRwV3Error::ReceiptDigestMismatch)
    );

    let partial_geometry = RegisterBcsrGeometry::new(513).unwrap();
    let partial_layout = RegisterBcsrLayout::new(partial_geometry).unwrap();
    let partial_receipt = RegisterBcsrReceipt::admit(
        source(partial_geometry.cycles(), 1),
        partial_layout,
        &provenances(partial_layout, identities, 9, 7),
    )
    .unwrap();
    assert_eq!(
        partial_receipt.registers_val_input(),
        Err(RegistersRwV3Error::InvalidRegistersValHandoff(513))
    );
}
