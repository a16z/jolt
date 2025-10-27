use common::constants::XLEN;
use rand::prelude::*;
use tracer::instruction::{RISCVCycle, RISCVInstruction};

use super::{CircuitFlags, InstructionLookup};
use crate::{field::JoltField, zkvm::instruction::LookupQuery};

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

    use strum::IntoEnumIterator;
    use tracer::instruction::Cycle;

    use super::CircuitFlags;
    use crate::zkvm::instruction::{Flags, InstructionFlags};

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
        use common::constants::XLEN;

        use crate::zkvm::instruction::LookupQuery;

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
