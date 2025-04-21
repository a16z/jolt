use tracer::instruction::{virtual_assert_valid_div0::VirtualAssertValidDiv0, RISCVCycle};

use crate::jolt::lookup_table::{valid_div0::ValidDiv0Table, LookupTables};

use super::InstructionLookup;

impl<const WORD_SIZE: usize> InstructionLookup<WORD_SIZE> for VirtualAssertValidDiv0 {
    fn lookup_table() -> Option<LookupTables<WORD_SIZE>> {
        Some(ValidDiv0Table.into())
    }

    fn lookup_query(cycle: &RISCVCycle<Self>) -> (u64, u64) {
        (cycle.register_state.rs1, cycle.register_state.rs2)
    }

    fn lookup_entry(cycle: &RISCVCycle<Self>) -> u64 {
        let (divisor, quotient) = InstructionLookup::<WORD_SIZE>::lookup_query(cycle);
        if divisor == 0 {
            match WORD_SIZE {
                32 => (quotient == u32::MAX as u64).into(),
                64 => (quotient == u64::MAX).into(),
                _ => panic!("Unsupported WORD_SIZE: {}", WORD_SIZE),
            }
        } else {
            1
        }
    }
}
