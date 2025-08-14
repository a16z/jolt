use crate::{field::JoltField, zkvm::instruction::LookupQuery};
use rand::prelude::*;
use tracer::instruction::{RISCVCycle, RISCVInstruction};

use super::InstructionLookup;

pub fn materialize_entry_test<F: JoltField, T: RISCVInstruction + Default>()
where
    RISCVCycle<T>: LookupQuery<32>,
    T: InstructionLookup<32>,
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
