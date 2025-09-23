use crate::{field::JoltField, zkvm::instruction::LookupQuery};
use common::constants::XLEN;
use rand::prelude::*;
use tracer::instruction::{RISCVCycle, RISCVInstruction};

use super::{CircuitFlags, InstructionLookup};

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
    use super::CircuitFlags;
    use crate::zkvm::instruction::InstructionFlags;
    use strum::IntoEnumIterator;
    use tracer::instruction::Cycle;

    #[test]
    fn left_operand_exclusive() {
        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            let flags = instr.circuit_flags();
            assert!(
                !(flags[CircuitFlags::LeftOperandIsPC]
                    && flags[CircuitFlags::LeftOperandIsRs1Value]),
                "Left operand flags not exclusive for {instr:?}",
            );
        }
    }

    #[test]
    fn right_operand_exclusive() {
        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            let flags = instr.circuit_flags();
            assert!(
                !(flags[CircuitFlags::RightOperandIsRs2Value]
                    && flags[CircuitFlags::RightOperandIsImm]),
                "Right operand flags not exclusive for {instr:?}",
            );
        }
    }

    #[test]
    fn lookup_shape_exclusive() {
        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            let flags = instr.circuit_flags();
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

    #[test]
    fn load_store_exclusive() {
        for cycle in Cycle::iter() {
            if let Cycle::INLINE(_) = cycle {
                continue;
            }
            let instr = cycle.instruction();
            let flags = instr.circuit_flags();
            assert!(
                !(flags[CircuitFlags::Load] && flags[CircuitFlags::Store]),
                "Load/Store flags not exclusive for {instr:?}",
            );
        }
    }
}
