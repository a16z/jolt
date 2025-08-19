use core::panic::AssertUnwindSafe;
use std::panic;

use crate::emulator::cpu::Cpu;
use crate::instruction::format::{InstructionFormat, InstructionRegisterState};
use crate::instruction::NormalizedInstruction;

use super::{
    addiw::ADDIW, addw::ADDW, amoaddd::AMOADDD, amoaddw::AMOADDW, amoandd::AMOANDD,
    amoandw::AMOANDW, amomaxd::AMOMAXD, amomaxud::AMOMAXUD, amomaxuw::AMOMAXUW, amomaxw::AMOMAXW,
    amomind::AMOMIND, amominud::AMOMINUD, amominuw::AMOMINUW, amominw::AMOMINW, amoord::AMOORD,
    amoorw::AMOORW, amoswapd::AMOSWAPD, amoswapw::AMOSWAPW, amoxord::AMOXORD, amoxorw::AMOXORW,
    div::DIV, divu::DIVU, divuw::DIVUW, divw::DIVW, lb::LB, lbu::LBU, lh::LH, lhu::LHU, lw::LW,
    lwu::LWU, mulh::MULH, mulhsu::MULHSU, mulw::MULW, rem::REM, remu::REMU, remuw::REMUW,
    remw::REMW, sb::SB, sh::SH, sll::SLL, slli::SLLI, slliw::SLLIW, sllw::SLLW, sra::SRA,
    srai::SRAI, sraiw::SRAIW, sraw::SRAW, srl::SRL, srli::SRLI, srliw::SRLIW, srlw::SRLW,
    subw::SUBW, sw::SW, RISCVInstruction, RISCVTrace,
};

use crate::emulator::terminal::DummyTerminal;

use common::constants::RISCV_REGISTER_COUNT;
use rand::rngs::OsRng;

use rand::{rngs::StdRng, RngCore, SeedableRng};

use super::{RISCVCycle, RV32IMCycle};

macro_rules! test_virtual_sequences {
  ($( $instr:ty ),* $(,)?) => {
      $(
          paste::paste! {
              #[test]
              fn [<test_ $instr:lower _virtual_sequence>]() {
                  virtual_sequence_trace_test::<$instr>();
              }
          }
      )*
  };
}

test_virtual_sequences!(
    ADDIW, ADDW, AMOADDD, AMOADDW, AMOANDD, AMOANDW, AMOMAXD, AMOMAXUD, AMOMAXUW, AMOMAXW, AMOMIND,
    AMOMINUD, AMOMINUW, AMOMINW, AMOORD, AMOORW, AMOSWAPD, AMOSWAPW, AMOXORD, AMOXORW, DIV, DIVU,
    DIVUW, DIVW, LB, LBU, LH, LHU, LW, LWU, MULH, MULHSU, MULW, REM, REMU, REMUW, REMW, SB, SH,
    SLL, SLLI, SLLIW, SLLW, SRA, SRAI, SRAIW, SRAW, SRL, SRLI, SRLIW, SRLW, SUBW, SW,
);

fn test_rng() -> StdRng {
    let mut seed = [0u8; 32];

    OsRng.fill_bytes(&mut seed);

    StdRng::from_seed(seed)
}

const TEST_MEMORY_CAPACITY: u64 = 1024 * 1024;

pub fn virtual_sequence_trace_test<I: RISCVInstruction + RISCVTrace + Copy>()
where
    RV32IMCycle: From<RISCVCycle<I>>,
{
    let mut rng = test_rng();

    for _ in 0..100 {
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
}
