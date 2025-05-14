use crate::emulator::cpu::{Cpu, Xlen};

use crate::emulator::terminal::DummyTerminal;

use rand::rngs::OsRng;

use rand::{rngs::StdRng, RngCore, SeedableRng};

use super::{RISCVInstruction, RISCVTrace, VirtualInstructionSequence};

use super::{RISCVCycle, RV32IMCycle};


macro_rules! test_virtual_sequences {
  ($( $instr:ty ),* $(,)?) => {
      $(
          paste::paste! {
              #[test]
              fn [<test_ $instr:snake _virtual_sequence>]() {
                  virtual_sequence_trace_test::<$instr>();
              }
          }
      )*
  };
}

// List of instruction types to test.
// Each must implement `VirtualInstructionSequence`.
test_virtual_sequences!(
  DIV, DIVU, LB, LBU, LH, LHU, MULH, MULHSU, REM, REMU,
  SB, SH, SLL, SLLI, SRA, SRAI, SRL, SRLI,
);


fn test_rng() -> StdRng {
    let mut seed = [0u8; 32];

    OsRng.fill_bytes(&mut seed);

    StdRng::from_seed(seed)
}

pub fn virtual_sequence_trace_test<
    I: RISCVInstruction + VirtualInstructionSequence + RISCVTrace + Copy,
>()
where
    RV32IMCycle: From<RISCVCycle<I>>,
{
    let mut rng = test_rng();

    const STANDARD_REGISTER_COUNT: usize = 32; //@TODO support 64 bit?

    for _ in 0..100 {
        let rs1 = rng.next_u64() % STANDARD_REGISTER_COUNT as u64;

        let rs2 = rng.next_u64() % STANDARD_REGISTER_COUNT as u64;

        let mut rd = rng.next_u64() % STANDARD_REGISTER_COUNT as u64;

        while rd == 0 {
            rd = rng.next_u64() % STANDARD_REGISTER_COUNT as u64;
        }

        let rs1_val = if rs1 == 0 { 0 } else { rng.next_u64() };

        let rs2_val = if rs2 == 0 {
            0
        } else if rs2 == rs1 {
            rs1_val
        } else {
            rng.next_u64()
        };

        let instruction = I::random(&mut rng);

        let mut original_cpu = Cpu::new(Box::new(DummyTerminal::new()));

        let mut virtual_cpu = Cpu::new(Box::new(DummyTerminal::new()));

        original_cpu.x[rs1 as usize] = rs1_val as i64;

        original_cpu.x[rs2 as usize] = rs2_val as i64;

        virtual_cpu.x[rs1 as usize] = rs1_val as i64;

        virtual_cpu.x[rs2 as usize] = rs2_val as i64;

        let mut ram_access = Default::default();

        instruction.execute(&mut original_cpu, &mut ram_access);

        instruction.trace(&mut virtual_cpu);

        assert_eq!(
            original_cpu.pc, virtual_cpu.pc,
            "PC register has different values after execution"
        );

        //@TODO markosg04: load/store/others are failing this check? what is the desired output...

        for i in 0..STANDARD_REGISTER_COUNT {
            assert_eq!(
                original_cpu.x[i], virtual_cpu.x[i],
                "Register {} has different values after execution. Original: {:?}, Virtual: {:?}",
                i, original_cpu.x[i], virtual_cpu.x[i]
            );
        }
    }
}
