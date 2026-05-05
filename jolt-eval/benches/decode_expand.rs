use common::constants::RAM_START_ADDRESS;
use common::jolt_device::{MemoryConfig, MemoryLayout};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use jolt_program::expand::expand_program;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_riscv::{InstructionKind, NormalizedInstruction, NormalizedOperands};

fn bench_decode_expand(c: &mut Criterion) {
    c.bench_function("decode_expand", |b| {
        b.iter_batched(
            fixture_program,
            |program| {
                let expanded = expand_program(program).expect("fixture expansion should succeed");
                let memory_config = MemoryConfig {
                    program_size: Some(0x1000),
                    ..MemoryConfig::default()
                };
                JoltProgramPreprocessing::new(
                    expanded,
                    vec![(RAM_START_ADDRESS, 0x13), (RAM_START_ADDRESS + 4, 0x37)],
                    MemoryLayout::new(&memory_config),
                    RAM_START_ADDRESS,
                    1 << 12,
                )
                .expect("fixture preprocessing should succeed")
            },
            BatchSize::SmallInput,
        )
    });
}

fn fixture_program() -> Vec<NormalizedInstruction> {
    (0..64)
        .map(|index| {
            let address = RAM_START_ADDRESS as usize + index * 4;
            match index % 6 {
                0 => instruction(InstructionKind::ADDI, address, Some(1), Some(2), None, 3),
                1 => instruction(InstructionKind::ADDIW, address, Some(3), Some(4), None, -7),
                2 => instruction(InstructionKind::ADDW, address, Some(5), Some(6), Some(7), 0),
                3 => instruction(
                    InstructionKind::MULH,
                    address,
                    Some(8),
                    Some(9),
                    Some(10),
                    0,
                ),
                4 => instruction(InstructionKind::LB, address, Some(11), Some(12), None, 5),
                _ => instruction(InstructionKind::SLLI, address, Some(15), Some(16), None, 4),
            }
        })
        .collect()
}

fn instruction(
    instruction_kind: InstructionKind,
    address: usize,
    rd: Option<u8>,
    rs1: Option<u8>,
    rs2: Option<u8>,
    imm: i128,
) -> NormalizedInstruction {
    NormalizedInstruction {
        instruction_kind,
        address,
        operands: NormalizedOperands { rd, rs1, rs2, imm },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    }
}

criterion_group!(benches, bench_decode_expand);
criterion_main!(benches);
