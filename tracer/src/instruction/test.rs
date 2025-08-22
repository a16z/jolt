use core::panic::AssertUnwindSafe;
use std::panic;

use crate::emulator::cpu::Cpu;
use crate::instruction::format::{InstructionFormat, InstructionRegisterState};
use crate::instruction::NormalizedInstruction;
use crate::utils::test_harness::TEST_MEMORY_CAPACITY;

use super::{
    // amoaddd::AMOADDD, amoaddw::AMOADDW, amoandd::AMOANDD,
    // amoandw::AMOANDW, amomaxd::AMOMAXD, amomaxud::AMOMAXUD, amomaxuw::AMOMAXUW, amomaxw::AMOMAXW,
    // amomind::AMOMIND, amominud::AMOMINUD, amominuw::AMOMINUW, amominw::AMOMINW, amoord::AMOORD,
    // amoorw::AMOORW, amoswapd::AMOSWAPD, amoswapw::AMOSWAPW, amoxord::AMOXORD, amoxorw::AMOXORW,
    // lb::LB, lbu::LBU, lh::LH, lhu::LHU, lw::LW, lwu::LWU,
    // sb::SB, sh::SH, sw::SW,
    addiw::ADDIW,
    addw::ADDW,
    div::DIV,
    divu::DIVU,
    divuw::DIVUW,
    divw::DIVW,
    mulh::MULH,
    mulhsu::MULHSU,
    mulw::MULW,
    rem::REM,
    remu::REMU,
    remuw::REMUW,
    remw::REMW,
    sll::SLL,
    slli::SLLI,
    slliw::SLLIW,
    sllw::SLLW,
    sra::SRA,
    srai::SRAI,
    sraiw::SRAIW,
    sraw::SRAW,
    srl::SRL,
    srli::SRLI,
    srliw::SRLIW,
    srlw::SRLW,
    subw::SUBW,
    RISCVInstruction,
    RISCVTrace,
};

use crate::emulator::terminal::DummyTerminal;

use common::constants::RISCV_REGISTER_COUNT;

use rand::{rngs::StdRng, SeedableRng};

use super::{RISCVCycle, RV32IMCycle};

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
    // NOTE: AMO instructinos panic on all cases, because `random` generates invalid
    // memory accessses. Same with store and load instructions.
    //
    // AMOADDD, AMOADDW, AMOANDD, AMOANDW, AMOMAXD, AMOMAXUD, AMOMAXUW, AMOMAXW, AMOMIND,
    // AMOMINUD, AMOMINUW, AMOMINW, AMOORD, AMOORW, AMOSWAPD, AMOSWAPW, AMOXORD, AMOXORW,
    // LB, LBU, LH, LHU, LW, LWU, SB, SH, SW
    ADDIW, ADDW, DIV, DIVU, DIVUW, DIVW, MULH, MULHSU, MULW, REM, REMU, REMUW, REMW, SLL, SLLI,
    SLLIW, SLLW, SRA, SRAI, SRAIW, SRAW, SRL, SRLI, SRLIW, SRLW, SUBW,
);

fn test_rng() -> StdRng {
    let seed = [0u8; 32];
    StdRng::from_seed(seed)
}

pub fn inline_sequence_trace_test<I: RISCVInstruction + RISCVTrace + Copy>()
where
    RV32IMCycle: From<RISCVCycle<I>>,
{
    let mut rng = test_rng();
    let mut non_panic = 0;

    for _ in 0..1000 {
        let instruction = I::random(&mut rng);
        let instr: NormalizedInstruction = instruction.into();
        let register_state =
            <<I::Format as InstructionFormat>::RegisterState as InstructionRegisterState>::random(
                &mut rng,
            );

        let mut original_cpu = Cpu::new(Box::new(DummyTerminal::default()));
        original_cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        let mut virtual_cpu = Cpu::new(Box::new(DummyTerminal::default()));
        virtual_cpu.get_mut_mmu().init_memory(TEST_MEMORY_CAPACITY);

        if instr.operands.rs1 != 0 {
            original_cpu.x[instr.operands.rs1 as usize] = register_state.rs1_value() as i64;
            virtual_cpu.x[instr.operands.rs1 as usize] = register_state.rs1_value() as i64;
        }
        if instr.operands.rs2 != 0 {
            original_cpu.x[instr.operands.rs2 as usize] = register_state.rs2_value() as i64;
            virtual_cpu.x[instr.operands.rs2 as usize] = register_state.rs2_value() as i64;
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
