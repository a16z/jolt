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
fn jolt_program_rd_zero_expansion_matches_tracer(
) -> Result<(), jolt_program::expand::ExpansionError> {
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

    let cases: [(NormalizedInstruction, Instruction); 68] = [
        (
            row(InstructionKind::ADD, add_operands),
            ADD::from(row(InstructionKind::ADD, add_operands)).into(),
        ),
        (
            row(InstructionKind::JAL, jal_operands),
            JAL::from(row(InstructionKind::JAL, jal_operands)).into(),
        ),
        (
            row(InstructionKind::ADDIW, format_i),
            ADDIW::from(row(InstructionKind::ADDIW, format_i)).into(),
        ),
        (
            row(InstructionKind::ADDW, format_r),
            ADDW::from(row(InstructionKind::ADDW, format_r)).into(),
        ),
        (
            row(InstructionKind::SUBW, format_r),
            SUBW::from(row(InstructionKind::SUBW, format_r)).into(),
        ),
        (
            row(InstructionKind::MULH, format_r),
            MULH::from(row(InstructionKind::MULH, format_r)).into(),
        ),
        (
            row(InstructionKind::MULHSU, format_r),
            MULHSU::from(row(InstructionKind::MULHSU, format_r)).into(),
        ),
        (
            row(InstructionKind::MULW, format_r),
            MULW::from(row(InstructionKind::MULW, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOADDD, format_r),
            AMOADDD::from(row(InstructionKind::AMOADDD, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOADDW, format_r),
            AMOADDW::from(row(InstructionKind::AMOADDW, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOANDD, format_r),
            AMOANDD::from(row(InstructionKind::AMOANDD, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOANDW, format_r),
            AMOANDW::from(row(InstructionKind::AMOANDW, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOMAXD, format_r),
            AMOMAXD::from(row(InstructionKind::AMOMAXD, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOMAXUD, format_r),
            AMOMAXUD::from(row(InstructionKind::AMOMAXUD, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOMAXUW, format_r),
            AMOMAXUW::from(row(InstructionKind::AMOMAXUW, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOMAXW, format_r),
            AMOMAXW::from(row(InstructionKind::AMOMAXW, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOMIND, format_r),
            AMOMIND::from(row(InstructionKind::AMOMIND, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOMINUD, format_r),
            AMOMINUD::from(row(InstructionKind::AMOMINUD, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOMINUW, format_r),
            AMOMINUW::from(row(InstructionKind::AMOMINUW, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOMINW, format_r),
            AMOMINW::from(row(InstructionKind::AMOMINW, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOORD, format_r),
            AMOORD::from(row(InstructionKind::AMOORD, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOORW, format_r),
            AMOORW::from(row(InstructionKind::AMOORW, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOSWAPD, format_r),
            AMOSWAPD::from(row(InstructionKind::AMOSWAPD, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOSWAPW, format_r),
            AMOSWAPW::from(row(InstructionKind::AMOSWAPW, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOXORD, format_r),
            AMOXORD::from(row(InstructionKind::AMOXORD, format_r)).into(),
        ),
        (
            row(InstructionKind::AMOXORW, format_r),
            AMOXORW::from(row(InstructionKind::AMOXORW, format_r)).into(),
        ),
        (
            row(InstructionKind::LB, format_i),
            LB::from(row(InstructionKind::LB, format_i)).into(),
        ),
        (
            row(InstructionKind::LBU, format_i),
            LBU::from(row(InstructionKind::LBU, format_i)).into(),
        ),
        (
            row(InstructionKind::LH, format_i),
            LH::from(row(InstructionKind::LH, format_i)).into(),
        ),
        (
            row(InstructionKind::LHU, format_i),
            LHU::from(row(InstructionKind::LHU, format_i)).into(),
        ),
        (
            row(InstructionKind::LW, format_i),
            LW::from(row(InstructionKind::LW, format_i)).into(),
        ),
        (
            row(InstructionKind::LWU, format_i),
            LWU::from(row(InstructionKind::LWU, format_i)).into(),
        ),
        (
            row(InstructionKind::AdviceLB, advice_operands),
            AdviceLB::from(row(InstructionKind::AdviceLB, advice_operands)).into(),
        ),
        (
            row(InstructionKind::AdviceLH, advice_operands),
            AdviceLH::from(row(InstructionKind::AdviceLH, advice_operands)).into(),
        ),
        (
            row(InstructionKind::AdviceLW, advice_operands),
            AdviceLW::from(row(InstructionKind::AdviceLW, advice_operands)).into(),
        ),
        (
            row(InstructionKind::AdviceLD, advice_operands),
            AdviceLD::from(row(InstructionKind::AdviceLD, advice_operands)).into(),
        ),
        (
            row(InstructionKind::LRD, format_r),
            LRD::from(row(InstructionKind::LRD, format_r)).into(),
        ),
        (
            row(InstructionKind::LRW, format_r),
            LRW::from(row(InstructionKind::LRW, format_r)).into(),
        ),
        (
            row(InstructionKind::DIV, format_r),
            DIV::from(row(InstructionKind::DIV, format_r)).into(),
        ),
        (
            row(InstructionKind::DIVU, format_r),
            DIVU::from(row(InstructionKind::DIVU, format_r)).into(),
        ),
        (
            row(InstructionKind::DIVW, format_r),
            DIVW::from(row(InstructionKind::DIVW, format_r)).into(),
        ),
        (
            row(InstructionKind::DIVUW, format_r),
            DIVUW::from(row(InstructionKind::DIVUW, format_r)).into(),
        ),
        (
            row(InstructionKind::REM, format_r),
            REM::from(row(InstructionKind::REM, format_r)).into(),
        ),
        (
            row(InstructionKind::REMU, format_r),
            REMU::from(row(InstructionKind::REMU, format_r)).into(),
        ),
        (
            row(InstructionKind::REMW, format_r),
            REMW::from(row(InstructionKind::REMW, format_r)).into(),
        ),
        (
            row(InstructionKind::REMUW, format_r),
            REMUW::from(row(InstructionKind::REMUW, format_r)).into(),
        ),
        (
            row(InstructionKind::SB, format_s),
            SB::from(row(InstructionKind::SB, format_s)).into(),
        ),
        (
            row(InstructionKind::SCD, format_r),
            SCD::from(row(InstructionKind::SCD, format_r)).into(),
        ),
        (
            row(InstructionKind::SCW, format_r),
            SCW::from(row(InstructionKind::SCW, format_r)).into(),
        ),
        (
            row(InstructionKind::CSRRW, csr_operands),
            CSRRW::from(row(InstructionKind::CSRRW, csr_operands)).into(),
        ),
        (
            row(InstructionKind::CSRRS, csr_operands),
            CSRRS::from(row(InstructionKind::CSRRS, csr_operands)).into(),
        ),
        (
            row(InstructionKind::EBREAK, format_i),
            EBREAK::from(row(InstructionKind::EBREAK, format_i)).into(),
        ),
        (
            row(InstructionKind::ECALL, format_i),
            ECALL::from(row(InstructionKind::ECALL, format_i)).into(),
        ),
        (
            row(InstructionKind::MRET, format_i),
            MRET::from(row(InstructionKind::MRET, format_i)).into(),
        ),
        (
            row(InstructionKind::SH, format_s),
            SH::from(row(InstructionKind::SH, format_s)).into(),
        ),
        (
            row(InstructionKind::SW, format_s),
            SW::from(row(InstructionKind::SW, format_s)).into(),
        ),
        (
            row(InstructionKind::SLL, format_r),
            SLL::from(row(InstructionKind::SLL, format_r)).into(),
        ),
        (
            row(InstructionKind::SLLI, format_i),
            SLLI::from(row(InstructionKind::SLLI, format_i)).into(),
        ),
        (
            row(InstructionKind::SLLW, format_r),
            SLLW::from(row(InstructionKind::SLLW, format_r)).into(),
        ),
        (
            row(InstructionKind::SLLIW, format_i),
            SLLIW::from(row(InstructionKind::SLLIW, format_i)).into(),
        ),
        (
            row(InstructionKind::SRL, format_r),
            SRL::from(row(InstructionKind::SRL, format_r)).into(),
        ),
        (
            row(InstructionKind::SRLI, format_i),
            SRLI::from(row(InstructionKind::SRLI, format_i)).into(),
        ),
        (
            row(InstructionKind::SRA, format_r),
            SRA::from(row(InstructionKind::SRA, format_r)).into(),
        ),
        (
            row(InstructionKind::SRAI, format_i),
            SRAI::from(row(InstructionKind::SRAI, format_i)).into(),
        ),
        (
            row(InstructionKind::SRLIW, format_i),
            SRLIW::from(row(InstructionKind::SRLIW, format_i)).into(),
        ),
        (
            row(InstructionKind::SRAIW, format_i),
            SRAIW::from(row(InstructionKind::SRAIW, format_i)).into(),
        ),
        (
            row(InstructionKind::SRLW, format_r),
            SRLW::from(row(InstructionKind::SRLW, format_r)).into(),
        ),
        (
            row(InstructionKind::SRAW, format_r),
            SRAW::from(row(InstructionKind::SRAW, format_r)).into(),
        ),
    ];

    for (normalized, tracer_instruction) in cases {
        let tracer_expanded: Vec<NormalizedInstruction> = tracer_instruction
            .legacy_inline_sequence(&VirtualRegisterAllocator::new(), Xlen::Bit64)
            .iter()
            .map(Instruction::normalize)
            .collect();
        let program_expanded = expand_instruction(&normalized, &mut ExpansionAllocator::new())?;

        assert_eq!(program_expanded, tracer_expanded);
    }

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
