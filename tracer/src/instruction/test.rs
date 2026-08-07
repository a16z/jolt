use core::panic::AssertUnwindSafe;
use std::panic;

use crate::emulator::cpu::Cpu;
#[cfg(test)]
use crate::instruction::format::{format_load::FormatLoad, format_s::FormatS};
use crate::instruction::format::{InstructionFormat, InstructionRegisterState};
#[cfg(test)]
use jolt_riscv::RV64IMAC_JOLT;

#[cfg(test)]
use super::{
    addiw::ADDIW, addw::ADDW, amoaddd::AMOADDD, amoaddw::AMOADDW, amoandd::AMOANDD,
    amoandw::AMOANDW, amomaxd::AMOMAXD, amomaxud::AMOMAXUD, amomaxuw::AMOMAXUW, amomaxw::AMOMAXW,
    amomind::AMOMIND, amominud::AMOMINUD, amominuw::AMOMINUW, amominw::AMOMINW, amoord::AMOORD,
    amoorw::AMOORW, amoswapd::AMOSWAPD, amoswapw::AMOSWAPW, amoxord::AMOXORD, amoxorw::AMOXORW,
    div::DIV, divu::DIVU, divuw::DIVUW, divw::DIVW, lb::LB, lbu::LBU, lh::LH, lhu::LHU, lw::LW,
    lwu::LWU, mulh::MULH, mulhsu::MULHSU, mulw::MULW, rem::REM, remu::REMU, remuw::REMUW,
    remw::REMW, sb::SB, sh::SH, sll::SLL, slli::SLLI, slliw::SLLIW, sllw::SLLW, sra::SRA,
    srai::SRAI, sraiw::SRAIW, sraw::SRAW, srl::SRL, srli::SRLI, srliw::SRLIW, srlw::SRLW,
    subw::SUBW, sw::SW,
};

use super::{Instruction, RISCVInstruction, RISCVTrace};

use crate::emulator::terminal::DummyTerminal;

use common::constants::RISCV_REGISTER_COUNT;

use rand::{rngs::StdRng, SeedableRng};

use super::{Cycle, RISCVCycle};

pub const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024;
pub const DRAM_BASE: u64 = 0x80000000;

macro_rules! test_inline_sequences {
  ($( $instr:ty ),* $(,)?) => {
      $(
          paste::paste! {
              #[test]
              fn [<test_ $instr:lower _inline_sequence>]() {
                  inline_sequence_trace_test::<$instr>();
              }
          }
      )*
  };
}

test_inline_sequences!(
    AMOADDD, AMOADDW, AMOANDD, AMOANDW, AMOMAXD, AMOMAXUD, AMOMAXUW, AMOMAXW, AMOMIND, AMOMINUD,
    AMOMINUW, AMOMINW, AMOORD, AMOORW, AMOSWAPD, AMOSWAPW, AMOXORD, AMOXORW, LB, LBU, LH, LHU, LW,
    LWU, SB, SH, SW, ADDIW, ADDW, DIV, DIVU, DIVUW, DIVW, MULH, MULHSU, MULW, REM, REMU, REMUW,
    REMW, SLL, SLLI, SLLIW, SLLW, SRA, SRAI, SRAIW, SRAW, SRL, SRLI, SRLIW, SRLW, SUBW
);

fn test_rng() -> StdRng {
    let seed = [0u8; 32];
    StdRng::from_seed(seed)
}

#[test]
fn jolt_program_rv64_decode_matches_tracer_normalization() {
    use crate::instruction::{uncompress_instruction, Instruction};

    let address = DRAM_BASE;
    let cases = [
        (0x1234_50b7, false),
        (0x1234_5097, false),
        (0x0080_00ef, false),
        (0x0000_80e7, false),
        (0x0020_8063, false),
        (0x0000_b183, false),
        (0x0030_b023, false),
        (0xfff1_0093, false),
        (0x0010_809b, false),
        (0x0020_81b3, false),
        (0x0220_81b3, false),
        (0x0020_81bb, false),
        (0x0000_000f, false),
        (0x0000_0073, false),
        (0x0010_0073, false),
        (0x3020_0073, false),
        (0x3001_10f3, false),
        (0x0000_10db, false),
        (0x0020_802b, false),
        (uncompress_instruction(0x107a), true),
    ];

    for (word, compressed) in cases {
        let expected = Instruction::decode(word, address, compressed)
            .unwrap()
            .source_instruction();
        let actual = jolt_program::image::decode::decode_instruction(
            word,
            address,
            compressed,
            RV64IMAC_JOLT,
        )
        .unwrap();
        assert_eq!(actual, expected, "word={word:08x} compressed={compressed}");
    }
}

pub fn inline_sequence_trace_test<I: RISCVInstruction + RISCVTrace + Copy>()
where
    Cycle: From<RISCVCycle<I>>,
{
    let mut rng = test_rng();
    let mut non_panic = 0;

    for _ in 0..1000 {
        let instruction = I::random(&mut rng);
        let concrete: Instruction = instruction.into();
        let source = concrete.source_instruction();
        let register_state =
            <<I::Format as InstructionFormat>::RegisterState as InstructionRegisterState>::random(
                &mut rng,
                &source.row().operands,
            );

        let mut original_cpu = Cpu::new(Box::new(DummyTerminal::default()));
        let memory_config = common::jolt_device::MemoryConfig {
            heap_size: TEST_MEMORY_CAPACITY,
            program_size: Some(1024), // Set a small program size for tests
            ..Default::default()
        };
        original_cpu.get_mut_mmu().jolt_device =
            Some(common::jolt_device::JoltDevice::new(&memory_config));
        original_cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        let mut virtual_cpu = Cpu::new(Box::new(DummyTerminal::default()));
        virtual_cpu.get_mut_mmu().jolt_device =
            Some(common::jolt_device::JoltDevice::new(&memory_config));
        virtual_cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        // Initialize memory with test values for AMO operations
        // Write some test values at aligned addresses throughout memory
        for i in 0..100 {
            let offset = (i * 8) as u64; // 8-byte aligned offsets
            if offset < TEST_MEMORY_CAPACITY {
                let test_value = 0x12345678 + i;
                // Store as doubleword for AMO.D instructions
                let addr = DRAM_BASE + offset;
                original_cpu
                    .mmu
                    .store_doubleword(addr, test_value as u64)
                    .ok();
                virtual_cpu
                    .mmu
                    .store_doubleword(addr, test_value as u64)
                    .ok();
            }
        }

        let rs1 = source.row().operands.rs1.unwrap_or(0) as usize;
        if let Some(rs1_val) = register_state.rs1_value() {
            original_cpu.write_register(rs1, rs1_val as i64);
            virtual_cpu.write_register(rs1, rs1_val as i64);
        }
        let rs2 = source.row().operands.rs2.unwrap_or(0) as usize;
        if let Some(rs2_val) = register_state.rs2_value() {
            original_cpu.write_register(rs2, rs2_val as i64);
            virtual_cpu.write_register(rs2, rs2_val as i64);
        }

        let mut ram_access = Default::default();

        let res = panic::catch_unwind(AssertUnwindSafe(|| {
            instruction.execute(&mut original_cpu, &mut ram_access);
        }));
        if res.is_err() {
            continue;
        }
        non_panic += 1;

        let mut trace_vec = Vec::new();
        instruction.trace(&mut virtual_cpu, Some(&mut trace_vec));

        assert_eq!(
            original_cpu.pc, virtual_cpu.pc,
            "PC register has different values after execution"
        );

        for i in 0..RISCV_REGISTER_COUNT {
            assert_eq!(
                original_cpu.x[i as usize], virtual_cpu.x[i as usize],
                "Register {} has different values after execution. Original: {:?}, Virtual: {:?}",
                i, original_cpu.x[i as usize], virtual_cpu.x[i as usize]
            );
        }
    }
    if non_panic == 0 {
        panic!("All of instructions panic at the execute function");
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
fn memory_test_cpu(initial_dword: u64, rs2: u64, memory_address: u64) -> Cpu {
    let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
    let memory_config = common::jolt_device::MemoryConfig {
        heap_size: TEST_MEMORY_CAPACITY,
        program_size: Some(1024),
        ..Default::default()
    };
    cpu.get_mut_mmu().jolt_device = Some(common::jolt_device::JoltDevice::new(&memory_config));
    cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
    cpu.mmu
        .store_doubleword(memory_address, initial_dword)
        .unwrap();
    cpu.write_register(1, memory_address as i64);
    cpu.write_register(2, rs2 as i64);
    cpu
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
fn assert_directed_memory_trace<I>(
    instruction: I,
    initial_dword: u64,
    rs2: u64,
    memory_address: u64,
    expected_rows: usize,
) where
    I: RISCVInstruction + RISCVTrace + Copy,
    Cycle: From<RISCVCycle<I>>,
{
    let mut original_cpu = memory_test_cpu(initial_dword, rs2, memory_address);
    let mut virtual_cpu = memory_test_cpu(initial_dword, rs2, memory_address);
    instruction.execute(&mut original_cpu, &mut Default::default());
    let mut trace = Vec::new();
    instruction.trace(&mut virtual_cpu, Some(&mut trace));

    assert_eq!(trace.len(), expected_rows);
    assert_eq!(original_cpu.pc, virtual_cpu.pc);
    assert_eq!(
        original_cpu.x[..RISCV_REGISTER_COUNT as usize],
        virtual_cpu.x[..RISCV_REGISTER_COUNT as usize]
    );
    assert_eq!(
        original_cpu.mmu.load_doubleword(memory_address).unwrap().0,
        virtual_cpu.mmu.load_doubleword(memory_address).unwrap().0
    );
}

#[cfg(test)]
fn assert_misaligned_trace_panics<I>(instruction: I)
where
    I: RISCVInstruction + RISCVTrace + Copy,
    Cycle: From<RISCVCycle<I>>,
{
    let mut original_cpu = memory_test_cpu(0, 0, DRAM_BASE);
    let mut virtual_cpu = memory_test_cpu(0, 0, DRAM_BASE);
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        instruction.execute(&mut original_cpu, &mut Default::default());
    }))
    .is_err());
    assert!(panic::catch_unwind(AssertUnwindSafe(|| {
        instruction.trace(&mut virtual_cpu, None);
    }))
    .is_err());
}

#[test]
#[cfg(test)]
fn subword_memory_inline_sequences_cover_lanes_and_sign_boundaries() {
    // Trace snapshots at offsets 4..=7 include the following doubleword.
    let memory_boundaries = [DRAM_BASE, DRAM_BASE + TEST_MEMORY_CAPACITY - 16];

    macro_rules! check_load {
        ($instruction:ident, $width:expr, $offsets:expr, $expected_rows:expr) => {
            for memory_address in memory_boundaries {
                for offset in $offsets {
                    let lane_mask = ((1_u64 << $width) - 1) << (offset * 8);
                    for payload in [
                        0,
                        (1_u64 << ($width - 1)) - 1,
                        1_u64 << ($width - 1),
                        u64::MAX,
                    ] {
                        let initial = (0x0123_4567_89ab_cdef & !lane_mask)
                            | ((payload << (offset * 8)) & lane_mask);
                        assert_directed_memory_trace(
                            $instruction {
                                address: DRAM_BASE,
                                operands: FormatLoad {
                                    rd: 3,
                                    rs1: 1,
                                    imm: offset,
                                },
                                virtual_sequence_remaining: None,
                                is_first_in_sequence: false,
                                is_compressed: false,
                            },
                            initial,
                            0,
                            memory_address,
                            $expected_rows,
                        );
                    }
                }
            }
        };
    }

    macro_rules! check_store {
        ($instruction:ident, $width:expr, $offsets:expr, $expected_rows:expr) => {
            for memory_address in memory_boundaries {
                for offset in $offsets {
                    for payload in [
                        0,
                        (1_u64 << ($width - 1)) - 1,
                        1_u64 << ($width - 1),
                        u64::MAX,
                    ] {
                        assert_directed_memory_trace(
                            $instruction {
                                address: DRAM_BASE,
                                operands: FormatS {
                                    rs1: 1,
                                    rs2: 2,
                                    imm: offset,
                                },
                                virtual_sequence_remaining: None,
                                is_first_in_sequence: false,
                                is_compressed: false,
                            },
                            0x0123_4567_89ab_cdef,
                            payload,
                            memory_address,
                            $expected_rows,
                        );
                    }
                }
            }
        };
    }

    check_load!(LB, 8, 0_i64..8, 4);
    check_load!(LBU, 8, 0_i64..8, 4);
    check_store!(SB, 8, 0_i64..8, 9);
    check_load!(LH, 16, [0_i64, 2, 4, 6], 5);
    check_load!(LHU, 16, [0_i64, 2, 4, 6], 5);
    check_store!(SH, 16, [0_i64, 2, 4, 6], 10);
    check_load!(LW, 32, [0_i64, 4], 5);
    check_load!(LWU, 32, [0_i64, 4], 5);
    check_store!(SW, 32, [0_i64, 4], 10);

    macro_rules! check_misaligned_load {
        ($instruction:ident, $offset:expr) => {
            assert_misaligned_trace_panics($instruction {
                address: DRAM_BASE,
                operands: FormatLoad {
                    rd: 3,
                    rs1: 1,
                    imm: $offset,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            });
        };
    }
    macro_rules! check_misaligned_store {
        ($instruction:ident, $offset:expr) => {
            assert_misaligned_trace_panics($instruction {
                address: DRAM_BASE,
                operands: FormatS {
                    rs1: 1,
                    rs2: 2,
                    imm: $offset,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            });
        };
    }

    for offset in [1, 3, 5, 7] {
        check_misaligned_load!(LH, offset);
        check_misaligned_load!(LHU, offset);
        check_misaligned_store!(SH, offset);
    }
    for offset in [1, 2, 3, 5, 6, 7] {
        check_misaligned_load!(LW, offset);
        check_misaligned_load!(LWU, offset);
        check_misaligned_store!(SW, offset);
    }
}
