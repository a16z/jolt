use core::panic::AssertUnwindSafe;
#[cfg(test)]
use std::any::Any;
use std::panic;

use crate::emulator::cpu::Cpu;
use crate::instruction::registers::InstructionRegisterState;
use jolt_riscv::RamAccess;
#[cfg(test)]
use jolt_riscv::RV64IMAC_JOLT;

#[cfg(test)]
use super::format::{format_load::FormatLoad, format_s::FormatS};
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
use common::jolt_device::{JoltDevice, MemoryConfig};

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
    LWU, SB, SH, SW, ADDIW, ADDW, DIVU, DIVUW, MULH, MULHSU, MULW, REMU, REMUW, SLL, SLLI, SLLIW,
    SLLW, SRA, SRAI, SRAIW, SRAW, SRL, SRLI, SRLIW, SRLW, SUBW
);

#[cfg(test)]
const SIGNED_DIVISION_EDGE_CASES: [(u64, u64); 5] = [
    ((-37i64) as u64, 0),
    (i64::MIN as u64, (-1i64) as u64),
    (i64::MIN as u64, 1),
    (37, i64::MIN as u64),
    (0, (-7i64) as u64),
];

#[cfg(test)]
const SIGNED_WORD_DIVISION_EDGE_CASES: [(u64, u64); 5] = [
    (0xdead_beef_ffff_ffdb, 0x1234_5678_0000_0000),
    (0xdead_beef_8000_0000, u64::MAX),
    (0xdead_beef_8000_0000, 0x1234_5678_0000_0001),
    (0xdead_beef_0000_0025, 0x1234_5678_8000_0000),
    (0xabcd_ef01_0000_0000, 0x1234_5678_ffff_fff9),
];

macro_rules! test_signed_division_inline_sequence {
    ($(($name:ident, $instr:ty, $cases:expr)),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                inline_sequence_trace_test::<$instr>();
                inline_sequence_edge_trace_test::<$instr>($cases);
            }
        )*
    };
}

test_signed_division_inline_sequence!(
    (test_div_inline_sequence, DIV, &SIGNED_DIVISION_EDGE_CASES),
    (test_rem_inline_sequence, REM, &SIGNED_DIVISION_EDGE_CASES),
    (
        test_divw_inline_sequence,
        DIVW,
        &SIGNED_WORD_DIVISION_EDGE_CASES
    ),
    (
        test_remw_inline_sequence,
        REMW,
        &SIGNED_WORD_DIVISION_EDGE_CASES
    ),
);

fn test_rng() -> StdRng {
    let seed = [0u8; 32];
    StdRng::from_seed(seed)
}

fn test_cpu() -> Cpu {
    let memory_config = MemoryConfig {
        heap_size: TEST_MEMORY_CAPACITY,
        program_size: Some(1024),
        ..Default::default()
    };
    let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
    cpu.get_mut_mmu().jolt_device = Some(JoltDevice::new(&memory_config));
    cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);
    cpu
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
        let register_state = <I::RegisterState as InstructionRegisterState>::random(
            &mut rng,
            &source.row().operands,
        );

        let mut original_cpu = test_cpu();
        let mut virtual_cpu = test_cpu();

        for i in 0..100 {
            let offset = (i * 8) as u64;
            if offset < TEST_MEMORY_CAPACITY {
                let test_value = 0x12345678 + i;
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

        // Random memory operands address [DRAM_BASE, DRAM_BASE + 0x10000) plus
        // a signed 8-bit offset.
        if trace_vec
            .iter()
            .any(|cycle| matches!(cycle.ram_access(), RamAccess::Write(_)))
        {
            for address in (DRAM_BASE..DRAM_BASE + 0x10100).step_by(8) {
                assert_eq!(
                    original_cpu.mmu.memory.read_doubleword(address),
                    virtual_cpu.mmu.memory.read_doubleword(address),
                    "memory at {address:#x} differs after execution"
                );
            }
        }
    }
    if non_panic == 0 {
        panic!("All of instructions panic at the execute function");
    }
}

#[cfg(test)]
fn inline_sequence_edge_trace_test<I: RISCVInstruction + RISCVTrace + Copy>(cases: &[(u64, u64)])
where
    Cycle: From<RISCVCycle<I>>,
{
    let word = I::MATCH | (3 << 7) | (1 << 15) | (2 << 20);
    let instruction = I::new(word, DRAM_BASE, true, false);
    for &(rs1, rs2) in cases {
        let mut original_cpu = test_cpu();
        original_cpu.write_register(1, rs1 as i64);
        original_cpu.write_register(2, rs2 as i64);

        let mut virtual_cpu = test_cpu();
        virtual_cpu.write_register(1, rs1 as i64);
        virtual_cpu.write_register(2, rs2 as i64);

        instruction.execute(&mut original_cpu, &mut Default::default());
        instruction.trace(&mut virtual_cpu, Some(&mut Vec::new()));

        assert_eq!(original_cpu.pc, virtual_cpu.pc);
        for register in 0..RISCV_REGISTER_COUNT {
            assert_eq!(
                original_cpu.x[register as usize], virtual_cpu.x[register as usize],
                "register={register}, rs1={rs1:#018x}, rs2={rs2:#018x}"
            );
        }
    }
}

/// Traces every sub-word load and store at each offset within a doubleword,
/// reaching the address with a non-negative and with a negative immediate.
/// Against a byte-array model of RISC-V semantics, an access aligned to its
/// width reads or writes only the containing doubleword, whose address its RAM
/// records carry; a misaligned one aborts.
#[test]
fn sub_word_accesses_touch_only_the_containing_doubleword() {
    const DOUBLEWORD: u64 = DRAM_BASE + 0x100;
    const OLD: u64 = 0x8899_aabb_ccdd_eeff;
    const VALUE: u64 = 0xf1e2_d3c4_b5a6_9788;
    const RD: u8 = 10;
    const RS2: u8 = 11;

    macro_rules! access {
        ($instr:ident, $operands:expr) => {
            Instruction::from($instr {
                address: DRAM_BASE,
                operands: $operands,
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            })
        };
    }

    for offset in 0..8u64 {
        for (base, imm) in [
            (DOUBLEWORD, offset as i64),
            (DOUBLEWORD + 16, offset as i64 - 16),
        ] {
            let load = FormatLoad {
                rd: RD,
                rs1: 1,
                imm,
            };
            let store = FormatS {
                rs1: 1,
                rs2: RS2,
                imm,
            };
            let accesses = [
                (access!(LB, load), 1, Some(true)),
                (access!(LBU, load), 1, Some(false)),
                (access!(LH, load), 2, Some(true)),
                (access!(LHU, load), 2, Some(false)),
                (access!(LW, load), 4, Some(true)),
                (access!(LWU, load), 4, Some(false)),
                (access!(SB, store), 1, None),
                (access!(SH, store), 2, None),
                (access!(SW, store), 4, None),
            ];
            for (instruction, width, signed_load) in accesses {
                let lane = offset as usize..(offset + width) as usize;
                let mut cpu = test_cpu();
                cpu.mmu.store_doubleword(DOUBLEWORD, OLD).unwrap();
                cpu.write_register(1, base as i64);
                cpu.write_register(RS2 as usize, VALUE as i64);

                let mut trace = Vec::new();
                let result = panic::catch_unwind(AssertUnwindSafe(|| {
                    instruction.trace(&mut cpu, Some(&mut trace));
                }));
                if !offset.is_multiple_of(width) {
                    let message = panic_message(result.expect_err("misaligned access traced"));
                    let class = if width == 2 { "halfword" } else { "word" };
                    assert!(
                        message.contains(&format!("not {class} aligned")),
                        "{instruction:?}: {message}"
                    );
                    continue;
                }
                assert!(result.is_ok(), "{instruction:?} at offset {offset}");

                let mut expected = OLD.to_le_bytes();
                match signed_load {
                    Some(signed) => {
                        let mut extended = [0u8; 8];
                        extended[..lane.len()].copy_from_slice(&expected[lane.clone()]);
                        if signed && expected[lane.end - 1] & 0x80 != 0 {
                            extended[lane.len()..].fill(0xff);
                        }
                        assert_eq!(
                            cpu.x[RD as usize] as u64,
                            u64::from_le_bytes(extended),
                            "{instruction:?} at offset {offset}"
                        );
                    }
                    None => {
                        expected[lane.clone()].copy_from_slice(&VALUE.to_le_bytes()[..lane.len()])
                    }
                }
                let expected = u64::from_le_bytes(expected);
                assert_eq!(cpu.mmu.memory.read_doubleword(DOUBLEWORD), expected);

                let records: Vec<_> = trace
                    .iter()
                    .map(Cycle::ram_access)
                    .filter(|access| !matches!(access, RamAccess::NoOp))
                    .collect();
                assert_eq!(records.len(), if signed_load.is_some() { 1 } else { 2 });
                for record in records {
                    match record {
                        RamAccess::Read(read) => {
                            assert_eq!((read.address, read.value), (DOUBLEWORD, OLD));
                        }
                        RamAccess::Write(write) => assert_eq!(
                            (write.address, write.pre_value, write.post_value),
                            (DOUBLEWORD, OLD, expected)
                        ),
                        RamAccess::NoOp => unreachable!(),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
fn panic_message(payload: Box<dyn Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => payload
            .downcast_ref::<&str>()
            .map_or_else(String::new, |message| (*message).to_string()),
    }
}
