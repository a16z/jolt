#![expect(
    clippy::expect_used,
    reason = "tests assert successful witness construction"
)]

use super::*;
use jolt_field::FixedByteSize;
use jolt_openings::{PackedAlphabet, PackedFamilySpec, PackedWitnessSource};
use jolt_riscv::{
    CapturedState, JoltInstructionKind, JoltInstructionRow, LoadState, NormalizedOperands,
    StoreState,
};

fn trace_domain() -> PackedFactDomain {
    PackedFactDomain::TraceRows { log_t: 1 }
}

fn trace_row(
    instruction_kind: JoltInstructionKind,
    operands: NormalizedOperands,
    state: CapturedState,
    bytecode_pc: u32,
) -> JoltTraceRow {
    let instruction = JoltInstructionRow {
        instruction_kind,
        address: 0x8000_0000,
        operands,
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    };
    JoltTraceRow::from_components(state, &instruction, bytecode_pc).expect("trace row should build")
}

fn get(
    witness: &SparsePackedWitness<AkitaField>,
    family: PackedFamilyId,
    row: usize,
    limb: usize,
    symbol: usize,
) -> AkitaField {
    witness
        .eval_direct_fact(&PackedCellAddress {
            family,
            row,
            limb,
            symbol,
        })
        .expect("address should be in layout")
}

#[test]
fn packs_trace_ra_and_unsigned_increment_facts() {
    let layout = PackedWitnessLayout::new([
        PackedFamilySpec::direct(
            PackedFamilyId::InstructionRa { index: 0 },
            trace_domain(),
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::BytecodeRa { index: 0 },
            trace_domain(),
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::RamRa { index: 0 },
            trace_domain(),
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::UnsignedIncChunk { index: 0 },
            trace_domain(),
            1,
            PackedAlphabet::Byte,
        ),
        PackedFamilySpec::direct(
            PackedFamilyId::UnsignedIncMsb,
            trace_domain(),
            1,
            PackedAlphabet::Bit,
        ),
    ])
    .expect("layout should build");
    let rows = [
        trace_row(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: Some(3),
                imm: 0,
            },
            CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                rs1_value: 1,
                rs2_value: 2,
                rd_pre_value: 10,
                rd_write_value: 3,
            }),
            9,
        ),
        trace_row(
            JoltInstructionKind::SD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: None,
                imm: 8,
            },
            CapturedState::Store(StoreState {
                rs1_value: 1,
                rs2_value: 30,
                ram_read_value: 10,
                ram_address: 0x42,
            }),
            11,
        ),
    ];

    let mut builder = JoltPackedWitnessBuilder::new(layout);
    let _ = builder
        .pack_trace_rows(
            &rows,
            8,
            |index, _| [0x7f, 0x80][index],
            |index, _| [None, Some(0x42)][index],
        )
        .expect("trace packing should succeed");
    let witness = builder.finish().expect("source should build");

    assert_eq!(
        get(
            &witness,
            PackedFamilyId::InstructionRa { index: 0 },
            0,
            0,
            0x7f
        ),
        AkitaField::one()
    );
    assert_eq!(
        get(&witness, PackedFamilyId::BytecodeRa { index: 0 }, 1, 0, 11),
        AkitaField::one()
    );
    assert_eq!(
        get(&witness, PackedFamilyId::RamRa { index: 0 }, 1, 0, 0x42),
        AkitaField::one()
    );
    assert_eq!(
        get(
            &witness,
            PackedFamilyId::UnsignedIncChunk { index: 0 },
            0,
            0,
            249
        ),
        AkitaField::one()
    );
    assert!(get(&witness, PackedFamilyId::UnsignedIncMsb, 0, 0, 1).is_zero());
    assert_eq!(
        get(
            &witness,
            PackedFamilyId::UnsignedIncChunk { index: 0 },
            1,
            0,
            20
        ),
        AkitaField::one()
    );
    assert_eq!(
        get(&witness, PackedFamilyId::UnsignedIncMsb, 1, 0, 1),
        AkitaField::one()
    );
}

#[test]
fn zero_increment_emits_zero_lower_chunks_and_set_msb() {
    let layout = increment_layout();
    let rows = [
        trace_row(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: Some(3),
                imm: 0,
            },
            CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                rs1_value: 1,
                rs2_value: 2,
                rd_pre_value: 10,
                rd_write_value: 10,
            }),
            9,
        ),
        trace_row(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: Some(3),
                imm: 0,
            },
            CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                rs1_value: 1,
                rs2_value: 2,
                rd_pre_value: 7,
                rd_write_value: 7,
            }),
            11,
        ),
    ];

    let mut builder = JoltPackedWitnessBuilder::new(layout);
    let _ = builder
        .pack_trace_rows(&rows, 8, |_, _| 0, |_, _| None)
        .expect("trace packing should succeed");
    let witness = builder.finish().expect("source should build");

    for index in 0..8 {
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::UnsignedIncChunk { index },
                0,
                0,
                0
            ),
            AkitaField::one()
        );
    }
    assert_eq!(
        get(&witness, PackedFamilyId::UnsignedIncMsb, 0, 0, 1),
        AkitaField::one()
    );
}

#[test]
fn store_zero_delta_uses_ram_increment_source() {
    let layout = increment_layout();
    let rows = [
        trace_row(
            JoltInstructionKind::SD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: None,
                imm: 8,
            },
            CapturedState::Store(StoreState {
                rs1_value: 1,
                rs2_value: 17,
                ram_read_value: 17,
                ram_address: 0x34,
            }),
            9,
        ),
        JoltTraceRow::no_op(),
    ];

    let mut builder = JoltPackedWitnessBuilder::new(layout);
    let _ = builder
        .pack_trace_rows(&rows, 8, |_, _| 0, |index, _| [Some(0x34), None][index])
        .expect("trace packing should succeed");
    let witness = builder.finish().expect("source should build");

    for index in 0..8 {
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::UnsignedIncChunk { index },
                0,
                0,
                0
            ),
            AkitaField::one()
        );
    }
    assert_eq!(
        get(&witness, PackedFamilyId::UnsignedIncMsb, 0, 0, 1),
        AkitaField::one()
    );
}

#[test]
fn load_row_uses_rd_increment_source_not_ram_delta() {
    let layout = increment_layout();
    let rows = [
        trace_row(
            JoltInstructionKind::LD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: None,
                rd: Some(3),
                imm: 8,
            },
            CapturedState::Load(LoadState {
                rs1_value: 1,
                ram_address: 0x34,
                rd_pre_value: 10,
                rd_write_value: 13,
            }),
            9,
        ),
        JoltTraceRow::no_op(),
    ];

    let mut builder = JoltPackedWitnessBuilder::new(layout);
    let _ = builder
        .pack_trace_rows(&rows, 8, |_, _| 0, |index, _| [Some(0x34), None][index])
        .expect("trace packing should succeed");
    let witness = builder.finish().expect("source should build");

    assert_eq!(
        get(
            &witness,
            PackedFamilyId::UnsignedIncChunk { index: 0 },
            0,
            0,
            3
        ),
        AkitaField::one()
    );
    for index in 1..8 {
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::UnsignedIncChunk { index },
                0,
                0,
                0
            ),
            AkitaField::one()
        );
    }
    assert_eq!(
        get(&witness, PackedFamilyId::UnsignedIncMsb, 0, 0, 1),
        AkitaField::one()
    );
}

#[test]
fn unsigned_increment_ignores_rd_slots_without_rd_destination() {
    let layout = increment_layout();
    let rows = [
        trace_row(
            JoltInstructionKind::BEQ,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: None,
                imm: 4,
            },
            CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                rs1_value: 1,
                rs2_value: 2,
                rd_pre_value: 10,
                rd_write_value: 3,
            }),
            9,
        ),
        JoltTraceRow::no_op(),
    ];

    let mut builder = JoltPackedWitnessBuilder::new(layout);
    let _ = builder
        .pack_trace_rows(&rows, 8, |_, _| 0, |_, _| None)
        .expect("trace packing should succeed");
    let witness = builder.finish().expect("source should build");

    for index in 0..8 {
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::UnsignedIncChunk { index },
                0,
                0,
                0
            ),
            AkitaField::one()
        );
    }
    assert_eq!(
        get(&witness, PackedFamilyId::UnsignedIncMsb, 0, 0, 1),
        AkitaField::one()
    );
}

#[test]
fn unsigned_increment_rejects_store_with_rd_destination() {
    let layout = increment_layout();
    let rows = [
        trace_row(
            JoltInstructionKind::SD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: Some(3),
                imm: 8,
            },
            CapturedState::Store(StoreState {
                rs1_value: 1,
                rs2_value: 11,
                ram_read_value: 10,
                ram_address: 0x34,
            }),
            9,
        ),
        JoltTraceRow::no_op(),
    ];

    let mut builder = JoltPackedWitnessBuilder::new(layout);
    let error = builder
        .pack_trace_rows(&rows, 8, |_, _| 0, |index, _| [Some(0x34), None][index])
        .expect_err("ambiguous increment source should reject");

    assert!(matches!(
        error,
        JoltPackedWitnessError::IncrementSourceConflict { row: 0 }
    ));
}

#[test]
fn negative_increment_emits_offset_lower_chunks_and_clear_msb() {
    let layout = increment_layout();
    let rows = [
        trace_row(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: Some(3),
                imm: 0,
            },
            CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                rs1_value: 1,
                rs2_value: 2,
                rd_pre_value: 10,
                rd_write_value: 3,
            }),
            9,
        ),
        trace_row(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: Some(3),
                imm: 0,
            },
            CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                rs1_value: 1,
                rs2_value: 2,
                rd_pre_value: 7,
                rd_write_value: 7,
            }),
            11,
        ),
    ];

    let mut builder = JoltPackedWitnessBuilder::new(layout);
    let _ = builder
        .pack_trace_rows(&rows, 8, |_, _| 0, |_, _| None)
        .expect("trace packing should succeed");
    let witness = builder.finish().expect("source should build");

    assert_eq!(
        get(
            &witness,
            PackedFamilyId::UnsignedIncChunk { index: 0 },
            0,
            0,
            249
        ),
        AkitaField::one()
    );
    for index in 1..8 {
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::UnsignedIncChunk { index },
                0,
                0,
                255
            ),
            AkitaField::one()
        );
    }
    assert!(get(&witness, PackedFamilyId::UnsignedIncMsb, 0, 0, 1).is_zero());
}

#[test]
fn unsigned_increment_supports_four_bit_lower_chunks() {
    let layout = increment_layout_with_chunk_bits(4);
    let rows = [
        trace_row(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: Some(3),
                imm: 0,
            },
            CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                rs1_value: 1,
                rs2_value: 2,
                rd_pre_value: 0,
                rd_write_value: 0x1234,
            }),
            9,
        ),
        JoltTraceRow::no_op(),
    ];

    let mut builder = JoltPackedWitnessBuilder::new(layout);
    let _ = builder
        .pack_trace_rows(&rows, 4, |_, _| 0, |_, _| None)
        .expect("trace packing should succeed");
    let witness = builder.finish().expect("source should build");

    let expected = [4, 3, 2, 1];
    for (index, symbol) in expected.into_iter().enumerate() {
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::UnsignedIncChunk { index },
                0,
                0,
                symbol
            ),
            AkitaField::one()
        );
    }
    for index in expected.len()..16 {
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::UnsignedIncChunk { index },
                0,
                0,
                0
            ),
            AkitaField::one()
        );
    }
    assert_eq!(
        get(&witness, PackedFamilyId::UnsignedIncMsb, 0, 0, 1),
        AkitaField::one()
    );
}

#[test]
fn field_rd_inc_uses_canonical_field_bytes() {
    let layout = field_rd_inc_layout();
    let rows = [
        trace_row(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: Some(3),
                imm: 0,
            },
            CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                rs1_value: 1,
                rs2_value: 2,
                rd_pre_value: 10,
                rd_write_value: 3,
            }),
            9,
        ),
        trace_row(
            JoltInstructionKind::ADD,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: Some(3),
                imm: 0,
            },
            CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                rs1_value: 1,
                rs2_value: 2,
                rd_pre_value: 7,
                rd_write_value: 7,
            }),
            11,
        ),
    ];

    let mut builder = JoltPackedWitnessBuilder::new(layout);
    let _ = builder
        .pack_trace_rows(&rows, 8, |_, _| 0, |_, _| None)
        .expect("trace packing should succeed");
    let witness = builder.finish().expect("source should build");
    let encoded = AkitaField::from_i128(-7).to_bytes_le_vec();

    assert_eq!(encoded.len(), AkitaField::NUM_BYTES);
    for (index, byte) in encoded.into_iter().enumerate() {
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::FieldRdIncByte { index },
                0,
                0,
                byte as usize
            ),
            AkitaField::one()
        );
    }
    assert!(witness
        .layout()
        .family(&PackedFamilyId::FieldRdIncSign)
        .is_none());
}

#[test]
fn field_rd_inc_ignores_rd_slots_without_rd_destination() {
    let layout = field_rd_inc_layout();
    let rows = [
        trace_row(
            JoltInstructionKind::BEQ,
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: None,
                imm: 4,
            },
            CapturedState::NonMemory(jolt_riscv::NonMemoryState {
                rs1_value: 1,
                rs2_value: 2,
                rd_pre_value: 10,
                rd_write_value: 3,
            }),
            9,
        ),
        JoltTraceRow::no_op(),
    ];

    let mut builder = JoltPackedWitnessBuilder::new(layout);
    let _ = builder
        .pack_trace_rows(&rows, 8, |_, _| 0, |_, _| None)
        .expect("trace packing should succeed");
    let witness = builder.finish().expect("source should build");

    for index in 0..AkitaField::NUM_BYTES {
        assert_eq!(
            get(&witness, PackedFamilyId::FieldRdIncByte { index }, 0, 0, 0),
            AkitaField::one()
        );
    }
}

#[test]
fn untrusted_advice_encoding_roundtrip() {
    let bytes = [255, 0, 7, 8];
    let layout = advice_layout(PackedAdviceKind::Untrusted);

    let mut builder = JoltPackedWitnessBuilder::new(layout);
    let _ = builder
        .pack_untrusted_advice_bytes(&bytes)
        .expect("advice packing should succeed");
    let witness = builder.finish().expect("source should build");

    for (row, byte) in bytes.iter().copied().enumerate() {
        assert_eq!(
            get(
                &witness,
                PackedFamilyId::AdviceBytes {
                    kind: PackedAdviceKind::Untrusted,
                    index: 0,
                },
                row,
                0,
                byte as usize
            ),
            AkitaField::one()
        );
    }
}

fn advice_layout(kind: PackedAdviceKind) -> PackedWitnessLayout {
    PackedWitnessLayout::new([PackedFamilySpec::direct(
        PackedFamilyId::AdviceBytes { kind, index: 0 },
        PackedFactDomain::AdviceBytes { kind, log_bytes: 2 },
        1,
        PackedAlphabet::Byte,
    )])
    .expect("layout should build")
}

fn field_rd_inc_layout() -> PackedWitnessLayout {
    PackedWitnessLayout::new((0..AkitaField::NUM_BYTES).map(|index| {
        PackedFamilySpec::direct(
            PackedFamilyId::FieldRdIncByte { index },
            trace_domain(),
            1,
            PackedAlphabet::Byte,
        )
    }))
    .expect("layout should build")
}

fn increment_layout() -> PackedWitnessLayout {
    increment_layout_with_chunk_bits(8)
}

fn increment_layout_with_chunk_bits(log_k_chunk: usize) -> PackedWitnessLayout {
    let chunk_count = unsigned_inc_lower_chunk_count(log_k_chunk).expect("valid test chunk size");
    let alphabet = PackedAlphabet::Fixed {
        size: 1 << log_k_chunk,
    };
    let mut specs = (0..chunk_count)
        .map(|index| {
            PackedFamilySpec::direct(
                PackedFamilyId::UnsignedIncChunk { index },
                trace_domain(),
                1,
                alphabet,
            )
        })
        .collect::<Vec<_>>();
    specs.push(PackedFamilySpec::direct(
        PackedFamilyId::UnsignedIncMsb,
        trace_domain(),
        1,
        PackedAlphabet::Bit,
    ));
    PackedWitnessLayout::new(specs).expect("layout should build")
}
