use core::panic::AssertUnwindSafe;
use std::panic;

use crate::emulator::cpu::Cpu;
use crate::instruction::format::{InstructionFormat, InstructionRegisterState};
use crate::instruction::NormalizedInstruction;

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
    REMW, SLL, SLLI, SLLIW, SLLW, SRA, SRAI, SRAIW, SRAW, SRL, SRLI, SRLIW, SRLW, SUBW,
);

fn test_rng() -> StdRng {
    let seed = [0u8; 32];
    StdRng::from_seed(seed)
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
            memory_size: TEST_MEMORY_CAPACITY,
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
        if rs1 != 0 {
            if let Some(rs1_val) = register_state.rs1_value() {
                original_cpu.x[rs1] = rs1_val as i64;
                virtual_cpu.x[rs1] = rs1_val as i64;
            }
        }
        let rs2 = instr.operands.rs2.unwrap_or(0) as usize;
        if rs2 != 0 {
            if let Some(rs2_val) = register_state.rs2_value() {
                original_cpu.x[rs2] = rs2_val as i64;
                virtual_cpu.x[rs2] = rs2_val as i64;
            }
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
