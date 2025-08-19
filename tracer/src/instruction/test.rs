use crate::emulator::cpu::Cpu;
use crate::instruction::format::{InstructionFormat, InstructionRegisterState};
use crate::instruction::NormalizedInstruction;

use super::{
    sll::SLL, slli::SLLI, sra::SRA, srai::SRAI, srl::SRL, srli::SRLI, RISCVInstruction, RISCVTrace,
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

test_virtual_sequences!(SLL, SLLI, SRA, SRAI, SRL, SRLI,);

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

        instruction.execute(&mut original_cpu, &mut ram_access);

        instruction.trace(&mut virtual_cpu, None);

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
