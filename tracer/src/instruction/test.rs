use core::panic::AssertUnwindSafe;
use std::panic;

use crate::emulator::cpu::Cpu;
use crate::instruction::format::{InstructionFormat, InstructionRegisterState};
use crate::instruction::NormalizedInstruction;

#[cfg(test)]
use super::{
    add::ADD, addiw::ADDIW, addw::ADDW, advice_lb::AdviceLB, advice_ld::AdviceLD,
    advice_lh::AdviceLH, advice_lw::AdviceLW, amoaddd::AMOADDD, amoaddw::AMOADDW, amoandd::AMOANDD,
    amoandw::AMOANDW, amomaxd::AMOMAXD, amomaxud::AMOMAXUD, amomaxuw::AMOMAXUW, amomaxw::AMOMAXW,
    amomind::AMOMIND, amominud::AMOMINUD, amominuw::AMOMINUW, amominw::AMOMINW, amoord::AMOORD,
    amoorw::AMOORW, amoswapd::AMOSWAPD, amoswapw::AMOSWAPW, amoxord::AMOXORD, amoxorw::AMOXORW,
    csrrs::CSRRS, csrrw::CSRRW, div::DIV, divu::DIVU, divuw::DIVUW, divw::DIVW, ebreak::EBREAK,
    ecall::ECALL, jal::JAL, lb::LB, lbu::LBU, lh::LH, lhu::LHU, lrd::LRD, lrw::LRW, lw::LW,
    lwu::LWU, mret::MRET, mulh::MULH, mulhsu::MULHSU, mulw::MULW, rem::REM, remu::REMU,
    remuw::REMUW, remw::REMW, sb::SB, scd::SCD, scw::SCW, sh::SH, sll::SLL, slli::SLLI,
    slliw::SLLIW, sllw::SLLW, sra::SRA, srai::SRAI, sraiw::SRAIW, sraw::SRAW, srl::SRL, srli::SRLI,
    srliw::SRLIW, srlw::SRLW, subw::SUBW, sw::SW,
};

use super::{RISCVInstruction, RISCVTrace};

use crate::emulator::terminal::DummyTerminal;

use common::constants::RISCV_REGISTER_COUNT;
use jolt_program::expand::ExpansionError;

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
fn jolt_program_expansion_matches_tracer_bridge() -> Result<(), ExpansionError> {
    use crate::{
        emulator::cpu::Xlen, instruction::Instruction,
        utils::virtual_registers::VirtualRegisterAllocator,
    };
    use jolt_program::expand::{expand_instruction, ExpansionAllocator};
    use jolt_riscv::{InstructionKind, NormalizedOperands};

    fn row(
        instruction_kind: InstructionKind,
        operands: NormalizedOperands,
    ) -> NormalizedInstruction {
        NormalizedInstruction {
            instruction_kind,
            address: DRAM_BASE as usize,
            operands,
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        }
    }

    let add_operands = NormalizedOperands {
        rd: Some(0),
        rs1: Some(1),
        rs2: Some(2),
        imm: 0,
    };
    let jal_operands = NormalizedOperands {
        rd: Some(0),
        rs1: None,
        rs2: None,
        imm: 16,
    };

    let format_i = NormalizedOperands {
        rd: Some(3),
        rs1: Some(1),
        rs2: None,
        imm: 4,
    };
    let format_r = NormalizedOperands {
        rd: Some(3),
        rs1: Some(1),
        rs2: Some(2),
        imm: 0,
    };
    let format_s = NormalizedOperands {
        rd: None,
        rs1: Some(1),
        rs2: Some(2),
        imm: 4,
    };
    let advice_operands = NormalizedOperands {
        rd: Some(3),
        rs1: None,
        rs2: None,
        imm: 0,
    };
    let csr_operands = NormalizedOperands {
        rd: Some(3),
        rs1: Some(1),
        rs2: None,
        imm: 0x300,
    };

    macro_rules! assert_expands_like_tracer {
        ($ty:ty, $kind:ident, $operands:expr) => {{
            let normalized = row(InstructionKind::$kind, $operands);
            let tracer_instruction: Instruction = <$ty>::from(normalized).into();
            let tracer_expanded: Vec<NormalizedInstruction> = tracer_instruction
                .inline_sequence(&VirtualRegisterAllocator::new(), Xlen::Bit64)
                .iter()
                .map(Instruction::normalize)
                .collect();
            let program_expanded = expand_instruction(&normalized, &mut ExpansionAllocator::new())?;

            assert_eq!(program_expanded, tracer_expanded, "{}", stringify!($kind));
        }};
    }

    assert_expands_like_tracer!(ADD, ADD, add_operands);
    assert_expands_like_tracer!(JAL, JAL, jal_operands);
    assert_expands_like_tracer!(ADDIW, ADDIW, format_i);
    assert_expands_like_tracer!(ADDW, ADDW, format_r);
    assert_expands_like_tracer!(SUBW, SUBW, format_r);
    assert_expands_like_tracer!(MULH, MULH, format_r);
    assert_expands_like_tracer!(MULHSU, MULHSU, format_r);
    assert_expands_like_tracer!(MULW, MULW, format_r);
    assert_expands_like_tracer!(AMOADDD, AMOADDD, format_r);
    assert_expands_like_tracer!(AMOADDW, AMOADDW, format_r);
    assert_expands_like_tracer!(AMOANDD, AMOANDD, format_r);
    assert_expands_like_tracer!(AMOANDW, AMOANDW, format_r);
    assert_expands_like_tracer!(AMOMAXD, AMOMAXD, format_r);
    assert_expands_like_tracer!(AMOMAXUD, AMOMAXUD, format_r);
    assert_expands_like_tracer!(AMOMAXUW, AMOMAXUW, format_r);
    assert_expands_like_tracer!(AMOMAXW, AMOMAXW, format_r);
    assert_expands_like_tracer!(AMOMIND, AMOMIND, format_r);
    assert_expands_like_tracer!(AMOMINUD, AMOMINUD, format_r);
    assert_expands_like_tracer!(AMOMINUW, AMOMINUW, format_r);
    assert_expands_like_tracer!(AMOMINW, AMOMINW, format_r);
    assert_expands_like_tracer!(AMOORD, AMOORD, format_r);
    assert_expands_like_tracer!(AMOORW, AMOORW, format_r);
    assert_expands_like_tracer!(AMOSWAPD, AMOSWAPD, format_r);
    assert_expands_like_tracer!(AMOSWAPW, AMOSWAPW, format_r);
    assert_expands_like_tracer!(AMOXORD, AMOXORD, format_r);
    assert_expands_like_tracer!(AMOXORW, AMOXORW, format_r);
    assert_expands_like_tracer!(LB, LB, format_i);
    assert_expands_like_tracer!(LBU, LBU, format_i);
    assert_expands_like_tracer!(LH, LH, format_i);
    assert_expands_like_tracer!(LHU, LHU, format_i);
    assert_expands_like_tracer!(LW, LW, format_i);
    assert_expands_like_tracer!(LWU, LWU, format_i);
    assert_expands_like_tracer!(AdviceLB, AdviceLB, advice_operands);
    assert_expands_like_tracer!(AdviceLH, AdviceLH, advice_operands);
    assert_expands_like_tracer!(AdviceLW, AdviceLW, advice_operands);
    assert_expands_like_tracer!(AdviceLD, AdviceLD, advice_operands);
    assert_expands_like_tracer!(LRD, LRD, format_r);
    assert_expands_like_tracer!(LRW, LRW, format_r);
    assert_expands_like_tracer!(DIV, DIV, format_r);
    assert_expands_like_tracer!(DIVU, DIVU, format_r);
    assert_expands_like_tracer!(DIVW, DIVW, format_r);
    assert_expands_like_tracer!(DIVUW, DIVUW, format_r);
    assert_expands_like_tracer!(REM, REM, format_r);
    assert_expands_like_tracer!(REMU, REMU, format_r);
    assert_expands_like_tracer!(REMW, REMW, format_r);
    assert_expands_like_tracer!(REMUW, REMUW, format_r);
    assert_expands_like_tracer!(SB, SB, format_s);
    assert_expands_like_tracer!(SCD, SCD, format_r);
    assert_expands_like_tracer!(SCW, SCW, format_r);
    assert_expands_like_tracer!(CSRRW, CSRRW, csr_operands);
    assert_expands_like_tracer!(CSRRS, CSRRS, csr_operands);
    assert_expands_like_tracer!(EBREAK, EBREAK, format_i);
    assert_expands_like_tracer!(ECALL, ECALL, format_i);
    assert_expands_like_tracer!(MRET, MRET, format_i);
    assert_expands_like_tracer!(SH, SH, format_s);
    assert_expands_like_tracer!(SW, SW, format_s);
    assert_expands_like_tracer!(SLL, SLL, format_r);
    assert_expands_like_tracer!(SLLI, SLLI, format_i);
    assert_expands_like_tracer!(SLLW, SLLW, format_r);
    assert_expands_like_tracer!(SLLIW, SLLIW, format_i);
    assert_expands_like_tracer!(SRL, SRL, format_r);
    assert_expands_like_tracer!(SRLI, SRLI, format_i);
    assert_expands_like_tracer!(SRA, SRA, format_r);
    assert_expands_like_tracer!(SRAI, SRAI, format_i);
    assert_expands_like_tracer!(SRLIW, SRLIW, format_i);
    assert_expands_like_tracer!(SRAIW, SRAIW, format_i);
    assert_expands_like_tracer!(SRLW, SRLW, format_r);
    assert_expands_like_tracer!(SRAW, SRAW, format_r);

    Ok(())
}

#[test]
fn jolt_program_rv64_decode_matches_tracer_normalization() {
    use crate::{
        emulator::cpu::Xlen,
        instruction::{uncompress_instruction, Instruction},
    };

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
        (0x0000_50db, false),
        (0x0020_802b, false),
        (uncompress_instruction(0x107a, Xlen::Bit64), true),
    ];

    for (word, compressed) in cases {
        let expected = Instruction::decode(word, address, compressed)
            .unwrap()
            .normalize();
        let actual =
            jolt_program::image::decode::decode_instruction(word, address, compressed).unwrap();
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
        let instr: NormalizedInstruction = instruction.into();
        let register_state =
            <<I::Format as InstructionFormat>::RegisterState as InstructionRegisterState>::random(
                &mut rng,
                &instr.operands,
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

        let rs1 = instr.operands.rs1.unwrap_or(0) as usize;
        if let Some(rs1_val) = register_state.rs1_value() {
            original_cpu.write_register(rs1, rs1_val as i64);
            virtual_cpu.write_register(rs1, rs1_val as i64);
        }
        let rs2 = instr.operands.rs2.unwrap_or(0) as usize;
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
