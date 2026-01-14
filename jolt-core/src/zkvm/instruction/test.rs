use crate::{field::JoltField, zkvm::instruction::LookupQuery};
use common::constants::XLEN;
use rand::prelude::*;
use tracer::{
    emulator::{cpu::Cpu, terminal::DummyTerminal},
    instruction::{
        self, format::InstructionRegisterState, Cycle, NormalizedInstruction, RISCVCycle,
        RISCVInstruction, RISCVTrace,
    },
};

use super::{CircuitFlags, Flags, InstructionLookup};

pub fn materialize_entry_test<F, T>()
where
    RISCVCycle<T>: LookupQuery<XLEN>,
    T: InstructionLookup<XLEN> + RISCVInstruction + Default,
    F: JoltField,
{
    let cycle: RISCVCycle<T> = Default::default();
    let table = cycle.instruction.lookup_table().unwrap();
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10000 {
        let random_cycle = cycle.random(&mut rng);
        assert_eq!(
            random_cycle.to_lookup_output(),
            table.materialize_entry(random_cycle.to_lookup_index()),
            "{:?}",
            random_cycle.register_state
        );
    }
}

/// Test that certain combinations of circuit flags are exclusive.
mod flags {
    use std::panic;

    use crate::zkvm::instruction::{Flags, InstructionFlags};

    use super::CircuitFlags;
    use strum::IntoEnumIterator;
    use tracer::instruction::Cycle;

    #[test]
    fn left_operand_exclusive() {
        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            if let Ok(flags) = panic::catch_unwind(|| instr.instruction_flags()) {
                assert!(
                    !(flags[InstructionFlags::LeftOperandIsPC]
                        && flags[InstructionFlags::LeftOperandIsRs1Value]),
                    "Left operand flags not exclusive for {instr:?}",
                );
            }
        }
    }

    #[test]
    fn right_operand_exclusive() {
        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            if let Ok(flags) = panic::catch_unwind(|| instr.instruction_flags()) {
                assert!(
                    !(flags[InstructionFlags::RightOperandIsRs2Value]
                        && flags[InstructionFlags::RightOperandIsImm]),
                    "Right operand flags not exclusive for {instr:?}",
                );
            }
        }
    }

    #[test]
    fn lookup_shape_exclusive() {
        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            if let Ok(flags) = panic::catch_unwind(|| instr.circuit_flags()) {
                let num_true = [
                    flags[CircuitFlags::AddOperands],
                    flags[CircuitFlags::SubtractOperands],
                    flags[CircuitFlags::MultiplyOperands],
                    flags[CircuitFlags::Advice],
                ]
                .iter()
                .filter(|&&b| b)
                .count();
                assert!(
                    num_true <= 1,
                    "Lookup shaping flags not exclusive for {instr:?}",
                );
            }
        }
    }

    #[test]
    fn load_store_exclusive() {
        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            if let Ok(flags) = panic::catch_unwind(|| instr.circuit_flags()) {
                assert!(
                    !(flags[CircuitFlags::Load] && flags[CircuitFlags::Store]),
                    "Load/Store flags not exclusive for {instr:?}",
                );
            }
        }
    }

    #[test]
    fn branch_lookup_output_is_boolean() {
        use crate::zkvm::instruction::LookupQuery;
        use common::constants::XLEN;

        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            if let Ok(instr_flags) = panic::catch_unwind(|| instr.instruction_flags()) {
                if instr_flags[InstructionFlags::Branch] {
                    let out = LookupQuery::<XLEN>::to_lookup_output(&cycle);
                    assert!(
                        out == 0 || out == 1,
                        "Branch lookup output not boolean for {instr:?}: got {out}",
                    );
                }
            }
        }
    }
}

pub fn lookup_output_matches_trace_test<T>()
where
    T: InstructionLookup<XLEN> + RISCVInstruction + RISCVTrace + Default + Flags + 'static,
    RISCVCycle<T>: LookupQuery<XLEN> + Into<Cycle>,
{
    let cycle: RISCVCycle<T> = Default::default();
    let mut rng = StdRng::seed_from_u64(12345);
    for _ in 0..10000 {
        let random_cycle = cycle.random(&mut rng);
        let normalized_instr: NormalizedInstruction = random_cycle.instruction.into();
        let normalized_operands = normalized_instr.operands;

        let mut cpu = Cpu::new(Box::new(DummyTerminal::default()));
        if let Some(rs1_val) = random_cycle.register_state.rs1_value() {
            cpu.x[normalized_operands.rs1.unwrap() as usize] = rs1_val as i64;
        }
        if let Some(rs2_val) = random_cycle.register_state.rs2_value() {
            cpu.x[normalized_operands.rs2.unwrap() as usize] = rs2_val as i64;
        }

        random_cycle.instruction.trace(&mut cpu, None);
        let lookup_result = LookupQuery::<XLEN>::to_lookup_output(&random_cycle);

        use std::any::TypeId;
        let is_jal = TypeId::of::<T>() == TypeId::of::<instruction::jal::JAL>();
        let is_jalr = TypeId::of::<T>() == TypeId::of::<instruction::jalr::JALR>();
        if is_jal || is_jalr {
            let cpu_pc = cpu.read_pc();
            assert_eq!(cpu_pc, lookup_result, "{random_cycle:?}");
        } else if let Some(rd) = normalized_operands.rd {
            let cpu_result = cpu.x[rd as usize] as u64;
            assert_eq!(cpu_result, lookup_result, "{random_cycle:?}");
        }
    }
}
